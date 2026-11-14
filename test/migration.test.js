import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { diagnose } from '../src/migration.js';
import { normalizePolicy } from '../src/policy.js';

const policy = normalizePolicy({
  classes: ['A', 'B'], classifier: 'f1', asOf: 20, conditionLimit: 1000, seed: 17,
  shift: { bootstrapReplicates: 200 }
});

function scenario(name) {
  return readFile(`data/scenarios/${name}.json`, 'utf8').then(text => JSON.parse(text).records);
}

let counter = 0;
function row(partition, truth, scoreA, scoreB = 1 - scoreA) {
  counter += 1;
  return {
    id: `x${String(counter).padStart(4, '0')}`,
    sample: `s${counter}`,
    partition, truth, scoreA, scoreB, visible: 1, classifier: 'f1'
  };
}

test('可手算二分类：单位算子与 0.7/0.3 目标边缘应还原 0.7/0.3', async () => {
  const result = diagnose(await scenario('hand-computable'), policy);
  assert.equal(result.estimation.status, 'estimated');
  assert.deepEqual(result.estimation.prior, { A: 0.7, B: 0.3 });
  assert.deepEqual(result.correction.appliedWeights, { A: 1.4, B: 0.6 });
  assert.equal(result.estimation.fit.residual, 0);
  assert.equal(result.decision.recommendation, 'use-corrected-proportions');
});

test('校正逐行归一，权重改变后总后验质量与硬计数一并记录', () => {
  const records = [
    // 源条件分布：A 类分数在 .6/.8，B 类在 .2/.4，软算子为 .7/.3。
    ...Array.from({ length: 10 }, () => row('source', 'A', 0.6)),
    ...Array.from({ length: 10 }, () => row('source', 'A', 0.8)),
    ...Array.from({ length: 10 }, () => row('source', 'B', 0.2)),
    ...Array.from({ length: 10 }, () => row('source', 'B', 0.4)),
    // 目标按 75% A-like、25% B-like 混合（源条件分布各取一半），先验应为 0.75/0.25。
    ...Array.from({ length: 75 }, () => row('target', '', 0.6)),
    ...Array.from({ length: 75 }, () => row('target', '', 0.8)),
    ...Array.from({ length: 25 }, () => row('target', '', 0.2)),
    ...Array.from({ length: 25 }, () => row('target', '', 0.4))
  ];
  const result = diagnose(records, policy);
  assert.equal(result.estimation.status, 'estimated');
  assert.ok(Math.abs(result.estimation.prior.A - 0.75) < 1e-9);
  assert.ok(Math.abs(result.estimation.prior.B - 0.25) < 1e-9);
  for (const entry of result.correction.archive) {
    const total = entry.corrected.A + entry.corrected.B;
    assert.ok(Math.abs(total - 1) < 1e-9, `校正后应为概率：${total}`);
    assert.ok(entry.corrected.A >= 0 && entry.corrected.B >= 0);
  }
  const before = result.correction.totals.posteriorMassBefore;
  const after = result.correction.totals.posteriorMassAfter;
  assert.ok(Math.abs(after.A - before.A) > 1e-6);
  assert.ok(Math.abs((after.A + after.B) - (before.A + before.B)) < 1e-9);
  assert.equal(result.correction.totals.changedRecords, 200);
  assert.match(result.correction.totals.interpretation, /不表示准确率/);
});

test('几乎相同的类别分数被判为病态并停用校正', async () => {
  const result = diagnose(await scenario('near-identical'), policy);
  assert.equal(result.estimation.status, 'not-identifiable');
  assert.equal(result.correction, null);
  assert.equal(result.uncertainty, null);
  assert.ok(result.estimation.operator.condition > 1000);
  const codes = result.diagnostics.map(entry => entry.code);
  assert.ok(codes.includes('operator-ill-conditioned'));
  assert.equal(result.decision.recommendation, 'manual-sampling');
});

test('目标新类别落在源条件支撑之外时不可识别', async () => {
  const result = diagnose(await scenario('new-class'), policy);
  assert.equal(result.estimation.status, 'not-identifiable');
  assert.ok(result.estimation.target.support.outsideCount > 0);
  assert.ok(result.diagnostics.some(entry => entry.code === 'target-outside-source-support'));
});

test('目标条件分布变化：边缘残差为零仍被重加权 KS 识别', async () => {
  const result = diagnose(await scenario('conditional-shift'), policy);
  assert.equal(result.estimation.status, 'not-identifiable');
  assert.equal(result.estimation.fit.residual, 0);
  assert.ok(result.estimation.fit.reweightedKs > 0.2);
  assert.ok(result.diagnostics.some(entry => entry.code === 'conditional-shift-suspected'));
});

test('零先验不伪装成正比例，无约束负值原样报告', async () => {
  const result = diagnose(await scenario('zero-prior'), policy);
  assert.deepEqual(result.estimation.prior, { A: 1, B: 0 });
  assert.equal(result.correction.appliedWeights.B, 0);
  assert.deepEqual(result.correction.zeroPriorClasses, ['B']);
  assert.ok(result.estimation.unconstrained.minimum < 0);
  assert.ok(result.estimation.unconstrained.hasNegative);
  assert.ok(result.diagnostics.some(entry => entry.code === 'zero-prior'));
  assert.ok(result.diagnostics.some(entry => entry.code === 'unconstrained-negative'));
  assert.equal(result.decision.recommendation, 'manual-sampling');
});

test('源类别为空直接不可识别，不输出校正', () => {
  const records = [
    ...Array.from({ length: 10 }, () => row('source', 'A', 1)),
    ...Array.from({ length: 10 }, () => row('target', '', 1)),
    ...Array.from({ length: 10 }, () => row('target', '', 0))
  ];
  const result = diagnose(records, policy);
  assert.equal(result.estimation.status, 'not-identifiable');
  assert.equal(result.correction, null);
  assert.ok(result.diagnostics.some(entry => entry.code === 'source-class-empty'));
});

test('源、目标与训练必须为不同样本，重复样本被拒绝', () => {
  const shared = row('source', 'A', 1);
  const records = [
    shared,
    { ...shared, id: 'other', partition: 'target', truth: '' },
    ...Array.from({ length: 10 }, () => row('target', '', 0))
  ];
  assert.throws(() => diagnose(records, policy), /必须分开/);
});

test('空 sample 不构成跨角色共享样本，不误报', () => {
  const records = [
    ...Array.from({ length: 10 }, () => row('source', 'A', 1)),
    ...Array.from({ length: 10 }, () => row('source', 'B', 0)),
    ...Array.from({ length: 20 }, () => row('target', '', 1))
  ].map(entry => ({ ...entry, sample: '' }));
  const result = diagnose(records, policy);
  assert.equal(result.estimation.status, 'estimated');
});

test('非冻结分类器不得混入源验证或目标批次', () => {
  const records = [
    ...Array.from({ length: 10 }, () => row('source', 'A', 1)),
    ...Array.from({ length: 10 }, () => row('source', 'B', 0)),
    ...Array.from({ length: 20 }, () => ({ ...row('target', '', 1), classifier: 'f2' }))
  ];
  assert.throws(() => diagnose(records, policy), /非冻结分类器/);
});

test('目标真值存在也不参与估计，且被标记为忽略', () => {
  const base = [
    ...Array.from({ length: 10 }, () => row('source', 'A', 1)),
    ...Array.from({ length: 10 }, () => row('source', 'B', 0))
  ];
  const withoutLabels = [
    ...base,
    ...Array.from({ length: 20 }, () => row('target', '', 1))
  ];
  const withLabels = [
    ...base,
    ...Array.from({ length: 20 }, () => row('target', 'B', 1))
  ];
  const first = diagnose(withoutLabels, policy);
  const second = diagnose(withLabels, policy);
  assert.deepEqual(first.estimation.prior, second.estimation.prior);
  assert.ok(second.diagnostics.some(entry => entry.code === 'target-labels-ignored'));
});

test('目标无标签时不声称准确率提高', async () => {
  const result = diagnose(await scenario('hand-computable'), policy);
  assert.equal(result.assumptions.accuracyClaim, false);
  assert.equal(result.assumptions.targetLabelsUsed, false);
  assert.match(result.assumptions.accuracyNote, /不构成准确率提高/);
});

test('不可识别时保留可复核诊断且不给可执行校正', async () => {
  for (const name of ['near-identical', 'new-class', 'conditional-shift']) {
    const result = diagnose(await scenario(name), policy);
    assert.equal(result.estimation.status, 'not-identifiable', name);
    assert.equal(result.estimation.prior, null);
    assert.equal(result.correction, null);
    assert.equal(result.decision.status, 'not-identifiable');
    assert.ok(result.decision.reasons.length > 0);
    assert.equal(result.decision.recommendation, 'manual-sampling');
    assert.ok(result.partitions.source.fingerprint);
  }
});

test('先验区间使用同一种子可复现，且源与目标分别重采样', async () => {
  const records = await scenario('hand-computable');
  const first = diagnose(records, policy);
  const second = diagnose(records, policy);
  assert.deepEqual(first.uncertainty, second.uncertainty);
  assert.equal(first.uncertainty.seed, 17);
  assert.match(first.uncertainty.method, /独立重采样/);
  for (const [, interval] of Object.entries(first.uncertainty.intervals)) {
    assert.ok(interval.lower <= interval.upper);
    assert.ok(interval.lower >= 0 && interval.upper <= 1);
  }
});

test('延迟审阅集在公开时点前不参与任何估计', async () => {
  const records = await scenario('hand-computable');
  const early = diagnose(records, policy);
  assert.equal(early.review.usedForTuning, false);
  assert.equal(early.review.usedForPostHoc, false);
  assert.match(early.review.reason, /尚未到公开时点/);

  const later = normalizePolicy({
    classes: ['A', 'B'], classifier: 'f1', asOf: 60, conditionLimit: 1000, seed: 17,
    reviewRelease: 60, shift: { bootstrapReplicates: 50 }
  });
  const post = diagnose(records, later);
  assert.equal(post.review.usedForTuning, false);
  assert.equal(post.review.usedForPostHoc, true);
  assert.deepEqual(post.estimation.prior, early.estimation.prior);
  assert.deepEqual(post.review.check.observedPrior, { A: 0.7, B: 0.3 });

  // 翻转审阅集真值只改变事后观察，不改变先验、校正权重或总体量。
  const flipped = records.map(entry => entry.partition === 'review'
    ? { ...entry, truth: entry.truth === 'A' ? 'B' : 'A' } : entry);
  const after = diagnose(flipped, later);
  assert.deepEqual(after.estimation.prior, post.estimation.prior);
  assert.deepEqual(after.correction.appliedWeights, post.correction.appliedWeights);
  assert.deepEqual(after.correction.totals, post.correction.totals);
  assert.deepEqual(after.review.check.observedPrior, { A: 0.3, B: 0.7 });
});

test('训练分区即使带真值也不进入估计', async () => {
  const records = await scenario('hand-computable');
  const result = diagnose(records, policy);
  assert.equal(result.partitions.training.used, false);
  assert.ok(result.partitions.training.count > 0);
  const tampered = records.map(entry => entry.partition === 'training'
    ? { ...entry, truth: 'B', scoreA: 0, scoreB: 1 } : entry);
  const after = diagnose(tampered, policy);
  assert.deepEqual(after.estimation, result.estimation);
});

test('非法分数行被排除且记录在诊断中', () => {
  const records = [
    ...Array.from({ length: 10 }, () => row('source', 'A', 1)),
    ...Array.from({ length: 10 }, () => row('source', 'B', 0)),
    ...Array.from({ length: 20 }, () => row('target', '', 1)),
    { ...row('target', '', 0.5), scoreB: 0.9 }
  ];
  const result = diagnose(records, policy);
  assert.equal(result.partitions.target.scored, 20);
  assert.ok(result.diagnostics.some(entry => entry.code === 'invalid-score-rows'));
});
