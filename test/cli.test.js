import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { parseArguments } from '../src/cli.js';
import { defaultShiftOptions, normalizePolicy } from '../src/policy.js';

test('migrate 默认使用 data/policy.json，且可显式覆盖', () => {
  assert.equal(parseArguments(['migrate']).policy, 'data/policy.json');
  assert.equal(parseArguments(['migrate', '--policy', 'p.json']).policy, 'p.json');
  assert.throws(() => parseArguments(['migrate', '--policy', 'a', '--policy', 'b']), /重复/);
  assert.throws(() => parseArguments(['summary', '--policy', 'p.json']), /参数/);
  assert.throws(() => parseArguments(['migrate', '--policy']), /参数/);
});

test('政策默认值覆盖全部诊断阈值', () => {
  const policy = normalizePolicy({
    classes: ['B', 'A'], classifier: 'f1', asOf: 20, conditionLimit: 1000, seed: 17
  });
  assert.deepEqual(policy.classes, ['B', 'A']);
  assert.deepEqual(policy.partitionRoles, {
    training: ['training'], source: ['source'], target: ['target'], review: ['review']
  });
  assert.deepEqual(policy.options, defaultShiftOptions);
});

test('政策拒绝重复类别、未知角色和未知阈值', () => {
  const base = { classes: ['A', 'B'], classifier: 'f1', asOf: 20, conditionLimit: 1000, seed: 17 };
  assert.throws(() => normalizePolicy({ ...base, classes: ['A', 'A'] }), /不重复/);
  assert.throws(() => normalizePolicy({ ...base, classes: ['A', 'B', 'C'] }), /两个类别/);
  assert.throws(() => normalizePolicy({
    ...base, partitionRoles: { source: ['source'], target: ['target'], mystery: ['x'] }
  }), /未知角色/);
  assert.throws(() => normalizePolicy({
    ...base, shift: { priorResidualLimit: -1 }
  }), /不合法/);
  assert.throws(() => normalizePolicy({
    ...base, partitionRoles: { source: ['source'], target: ['source'] }
  }), /同时映射/);
});

test('migrate 落盘结果与标准输出一致，且不改变输入数据', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'migrate-'));
  try {
    const outPath = join(folder, 'migration.json');
    const before = await readFile('data/scenarios/hand-computable.json', 'utf8');
    const run = spawnSync(process.execPath,
      ['src/cli.js', 'migrate', '--input', 'data/scenarios/hand-computable.json', '--out', outPath],
      { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr);
    const document = JSON.parse(await readFile(outPath, 'utf8'));
    assert.deepEqual(JSON.parse(run.stdout), document);
    assert.equal(document.estimation.prior.A, 0.7);
    assert.equal(document.assumptions.accuracyClaim, false);
    assert.equal(await readFile('data/scenarios/hand-computable.json', 'utf8'), before);
  } finally { await rm(folder, { recursive: true, force: true }); }
});

test('migrate 在输入不可用时以非零退出码结束', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'migrate-bad-'));
  try {
    const path = join(folder, 'records.json');
    const records = JSON.parse(await readFile('data/scenarios/near-identical.json', 'utf8')).records;
    await writeFile(path, JSON.stringify({ version: 1, records: [
      { ...records[0], classifier: 'f2' }
    ] }));
    const run = spawnSync(process.execPath, ['src/cli.js', 'migrate', '--input', path],
      { encoding: 'utf8' });
    assert.equal(run.status, 1);
    assert.match(run.stderr, /非冻结分类器/);
  } finally { await rm(folder, { recursive: true, force: true }); }
});

test('既有 summary 与 inspect 命令保持不变', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'legacy-'));
  try {
    const path = join(folder, 'summary.json');
    const run = spawnSync(process.execPath,
      ['src/cli.js', 'summary', '--input', 'data/records.json', '--out', path], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr);
    const document = JSON.parse(run.stdout);
    assert.equal(document.count, 96);
    assert.equal(document.groups.length, 2);
    assert.ok(document.inputFingerprint);
  } finally { await rm(folder, { recursive: true, force: true }); }
});
