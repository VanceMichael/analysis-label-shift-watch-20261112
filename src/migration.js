import { fingerprint, validateRecords } from './dataset.js';
import { matrixCondition, solveLinearSystem } from './linalg.js';
import {
  bootstrapPrior, conditionalSupport, confusionOperator, constrainedPrior,
  correctedPosterior, emPrior, inspectScores, marginal, mixtureResidual,
  reweightedKs, roleIndex, scoresOf, scoresToObject, targetSupport
} from './diagnostics.js';

const EPS = 1e-12;

function round(value, digits = 12) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function rounded(value) {
  if (Array.isArray(value)) return value.map(rounded);
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, rounded(entry)]));
  }
  return round(value);
}

function byId(left, right) {
  return left.id.localeCompare(right.id, 'en');
}

function diagnostic(code, severity, message, detail) {
  const entry = { code, severity, message };
  if (detail !== undefined) entry.detail = detail;
  return entry;
}

// 软混淆算子对角与非对角之差的最小值；接近零说明两个类别的分数几乎相同。
function classSeparation(matrix) {
  let separation = Infinity;
  for (let column = 0; column < matrix.length; column += 1) {
    for (let row = 0; row < matrix.length; row += 1) {
      if (row === column) continue;
      separation = Math.min(separation, Math.abs(matrix[column][column] - matrix[row][column]));
    }
  }
  return Number.isFinite(separation) ? separation : null;
}

function collect(records, policy) {
  const index = roleIndex(policy);
  const buckets = { training: [], source: [], target: [], review: [], unmapped: [] };
  for (const row of [...records].sort(byId)) {
    const role = index.get(row.partition);
    if (role) buckets[role].push(row);
    else buckets.unmapped.push(row);
  }
  return buckets;
}

function partitionReport(rows, extra = {}) {
  return { count: rows.length, fingerprint: fingerprint(rows), ...extra };
}

// 三个估计角色必须来自不同记录：同一个 sample 不能同时出现在训练、源验证或目标批次。
// 空 sample 不构成可比较的样本键，跳过而不误报。
function assertSeparateSamples(buckets) {
  const owners = new Map();
  for (const role of ['training', 'source', 'target']) {
    for (const row of buckets[role]) {
      if (row.sample === '') continue;
      const previous = owners.get(row.sample);
      if (previous && previous !== role) {
        throw new Error(`样本 ${row.sample} 同时出现在 ${previous} 与 ${role}，训练、源验证和目标批次必须分开`);
      }
      owners.set(row.sample, role);
    }
  }
}

export function diagnose(records, policy) {
  validateRecords(records);
  const classes = policy.classes;
  const options = policy.options;
  const buckets = collect(records, policy);
  const diagnostics = [];
  assertSeparateSamples(buckets);

  // 冻结分类器：源验证与目标批次必须来自同一个已冻结分类器，否则分数不可比。
  const mismatched = [...buckets.source, ...buckets.target]
    .filter(row => row.classifier !== policy.classifier);
  if (mismatched.length) {
    throw new Error(`源验证或目标记录使用了非冻结分类器 ${mismatched[0].classifier}，期望 ${policy.classifier}`);
  }

  const sourceScored = inspectScores(buckets.source, classes, options.scoreRangeTolerance, options.scoreSumTolerance);
  const targetScored = inspectScores(buckets.target, classes, options.scoreRangeTolerance, options.scoreSumTolerance);
  for (const [name, report] of [['source', sourceScored], ['target', targetScored]]) {
    if (report.invalid.length) {
      diagnostics.push(diagnostic('invalid-score-rows', 'warning',
        `${name} 有分数不满足有限、落在 [0,1] 且两类和为一的记录，已排除`,
        { count: report.invalid.length, examples: report.invalid.slice(0, 10) }));
    }
  }

  const sourceRows = sourceScored.valid.filter(row => row.truth !== '');
  const sourceUnlabelled = sourceScored.valid.length - sourceRows.length;
  const targetRows = targetScored.valid;
  const targetLabelled = targetRows.filter(row => row.truth !== '').length;
  if (sourceUnlabelled > 0) {
    diagnostics.push(diagnostic('source-unlabelled-excluded', 'info',
      '源验证集中无真值记录不参与混淆算子估计', { count: sourceUnlabelled }));
  }
  if (targetLabelled > 0) {
    diagnostics.push(diagnostic('target-labels-ignored', 'warning',
      '目标批次含真值，但标签迁移口径只用目标分数分布；这些真值不参与估计、校正或参数选择',
      { count: targetLabelled }));
  }
  if (buckets.unmapped.length) {
    diagnostics.push(diagnostic('unmapped-partitions', 'warning',
      '存在未映射到任何角色的分区，已排除在估计之外',
      { count: buckets.unmapped.length, names: [...new Set(buckets.unmapped.map(row => row.partition))] }));
  }

  const sourceCounts = Object.fromEntries(classes.map(className =>
    [className, sourceRows.filter(row => row.truth === className).length]));
  const unknownTruth = sourceRows.filter(row => !classes.includes(row.truth));
  if (unknownTruth.length) {
    throw new Error(`源验证集含类别之外的 truth（例如 ${unknownTruth[0].id}），无法建立混淆算子`);
  }
  const emptyClasses = classes.filter(className => sourceCounts[className] === 0);
  const sourcePriors = Object.fromEntries(classes.map(className =>
    [className, sourceRows.length ? sourceCounts[className] / sourceRows.length : null]));

  const assumptions = {
    labelShiftOnly: true,
    statement: '只假设 p(score|class) 在源地区与目标地区保持不变、类别先验可以变化；不假设任意协变量迁移。',
    frozenClassifier: policy.classifier,
    sourceLabelsUsedFor: '估计混淆/软混淆算子',
    targetScoresUsedFor: '目标分数边缘分布',
    targetLabelsUsed: false,
    scoreVectorsInterpretedAs: '冻结分类器输出的软后验分数',
    classOrder: classes,
    trainingSeparate: true,
    reviewUsedForTuning: false,
    accuracyClaim: false,
    accuracyNote: '目标批次没有可用真值；校正只改变类别比例与分数的相对权重，不构成准确率提高的证据。'
  };

  const base = {
    operation: 'migration',
    asOf: policy.asOf,
    classOrder: classes,
    assumptions,
    partitions: {
      training: partitionReport(buckets.training, {
        used: false, reason: '训练分区与源验证、目标批次分开，不进入估计'
      }),
      source: partitionReport(buckets.source, {
        labelled: sourceRows.length,
        unlabelled: sourceUnlabelled,
        invalidScores: sourceScored.invalid.length
      }),
      target: partitionReport(buckets.target, {
        scored: targetRows.length,
        invalidScores: targetScored.invalid.length
      }),
      review: partitionReport(buckets.review, {
        usedForTuning: false,
        usedForPostHoc: false,
        release: policy.reviewRelease
      })
    },
    diagnostics
  };

  if (emptyClasses.length) {
    diagnostics.push(diagnostic('source-class-empty', 'error',
      '源验证集存在没有真值样本的类别，混淆算子不可识别',
      { classes: emptyClasses, counts: sourceCounts }));
  }
  if (!sourceRows.length) {
    diagnostics.push(diagnostic('source-empty', 'error', '源验证集没有带真值且分数有效的记录'));
  }
  if (!targetRows.length) {
    diagnostics.push(diagnostic('target-empty', 'error', '目标批次没有分数有效的记录'));
  }
  if (sourceRows.length > 0 && sourceRows.length < options.minSourceTotal) {
    diagnostics.push(diagnostic('source-too-small', 'warning',
      '源验证集样本量低于估计下限', { count: sourceRows.length, minimum: options.minSourceTotal }));
  }
  for (const className of classes) {
    if (sourceCounts[className] > 0 && sourceCounts[className] < options.minSourcePerClass) {
      diagnostics.push(diagnostic('source-class-small', 'warning',
        `源验证集类别 ${className} 样本量低于下限`,
        { count: sourceCounts[className], minimum: options.minSourcePerClass }));
    }
  }
  if (targetRows.length > 0 && targetRows.length < options.minTargetTotal) {
    diagnostics.push(diagnostic('target-too-small', 'warning',
      '目标批次样本量低于估计下限', { count: targetRows.length, minimum: options.minTargetTotal }));
  }

  const hardReasons = diagnostics.filter(entry => entry.severity === 'error').map(entry => entry.code);
  if (hardReasons.length) {
    const review = reviewReport(buckets, policy, classes, null, options);
    if (review) {
      review.usedForPostHoc = false;
      review.reason = '本次估计不可识别，未启用审阅集事后检查';
      base.partitions.review.usedForPostHoc = false;
      base.partitions.review.reason = review.reason;
    }
    return {
      ...base,
      estimation: { status: 'not-identifiable', prior: null, reasons: hardReasons },
      uncertainty: null,
      correction: null,
      review,
      decision: {
        status: 'not-identifiable',
        recommendation: 'manual-sampling',
        reasons: hardReasons,
        guidance: '本轮停用比例校正；先补齐可识别的输入，或改由人工抽样确认目标类别构成。'
      }
    };
  }

  const softOperator = confusionOperator(sourceRows, classes, 'soft');
  const hardOperator = confusionOperator(sourceRows, classes, 'hard');
  const targetMarginalSoft = marginal(targetRows, classes, 'soft');
  const targetMarginalHard = marginal(targetRows, classes, 'hard');
  const condition = matrixCondition(softOperator);
  const separation = classSeparation(softOperator);

  // 无约束逆矩阵解只作为诊断呈现：负值原样保留，绝不裁零后当作可行解。
  const unconstrainedSolution = solveLinearSystem(softOperator, targetMarginalSoft);
  const unconstrained = unconstrainedSolution
    ? {
      prior: scoresToObject(unconstrainedSolution, classes),
      minimum: Math.min(...unconstrainedSolution),
      hasNegative: unconstrainedSolution.some(value => value < -options.zeroPriorTolerance)
    }
    : { prior: null, minimum: null, hasNegative: null };

  const constrained = constrainedPrior(softOperator, targetMarginalSoft, options.zeroPriorTolerance);
  const crossCheck = emPrior(softOperator, targetMarginalSoft);
  const support = targetSupport(targetRows, sourceRows, classes,
    conditionalSupport(sourceRows, classes), options.outsideSupportTolerance);
  const residual = mixtureResidual(softOperator, constrained.prior, targetMarginalSoft);
  const priorObject = scoresToObject(constrained.prior, classes);
  const ks = reweightedKs(sourceRows, targetRows, sourcePriors, priorObject,
    classes, options.zeroPriorTolerance);

  const warnings = [];
  if (condition.rankDeficient) {
    hardReasons.push('operator-rank-deficient');
    diagnostics.push(diagnostic('operator-rank-deficient', 'error',
      '软混淆算子秩亏损，无法从目标分数唯一还原类别先验',
      { rank: condition.rank, classCount: classes.length }));
  } else if (condition.value === null || condition.value > policy.conditionLimit) {
    hardReasons.push('operator-ill-conditioned');
    diagnostics.push(diagnostic('operator-ill-conditioned', 'error',
      '软混淆算子病态：类别分数几乎相同，先验估计对分数误差极敏感',
      { condition: condition.value, limit: policy.conditionLimit, separation }));
  }
  if (unconstrained.hasNegative) {
    warnings.push('unconstrained-negative');
    diagnostics.push(diagnostic('unconstrained-negative', 'warning',
      '无约束最小二乘解含负值；已改用非负且和为一的约束解，负值未被裁剪成零',
      { unconstrained: rounded(unconstrained.prior), minimum: round(unconstrained.minimum) }));
  }
  if (!constrained.solved) {
    hardReasons.push('constrained-solve-failed');
    diagnostics.push(diagnostic('constrained-solve-failed', 'error',
      '未能求得非负且和为一的类别先验'));
  }
  if (residual.residual > options.priorResidualLimit) {
    hardReasons.push('prior-fit-residual-exceeds-limit');
    diagnostics.push(diagnostic('prior-fit-residual-exceeds-limit', 'error',
      '目标分数边缘无法由源混淆算子与任何类别先验组合解释，标签迁移假设不成立',
      { residual: round(residual.residual), limit: options.priorResidualLimit }));
  } else if (residual.residual > options.priorResidualLimit * 0.75) {
    warnings.push('prior-fit-residual-near-limit');
    diagnostics.push(diagnostic('prior-fit-residual-near-limit', 'warning',
      '目标分数边缘的可解释程度接近阈值，建议人工抽样复核',
      { residual: round(residual.residual), limit: options.priorResidualLimit }));
  }
  if (support.outsideFraction > options.supportOutsideLimit) {
    hardReasons.push('target-outside-source-support');
    diagnostics.push(diagnostic('target-outside-source-support', 'error',
      '目标分数落在源类别条件支撑之外，疑似出现目标新类别或分数机制变化',
      { fraction: round(support.outsideFraction), count: support.outsideCount,
        limit: options.supportOutsideLimit, examples: support.outsideIds }));
  } else if (support.outsideCount > 0) {
    warnings.push('target-outside-source-support');
    diagnostics.push(diagnostic('target-outside-source-support', 'warning',
      '少量目标分数落在源条件支撑之外，可能是新类别或分数噪声',
      { fraction: round(support.outsideFraction), count: support.outsideCount,
        examples: support.outsideIds }));
  }
  if (ks.maximum !== null && ks.maximum > options.conditionalShiftLimit) {
    hardReasons.push('conditional-shift-suspected');
    diagnostics.push(diagnostic('conditional-shift-suspected', 'error',
      '按估计先验给源验证集重加权后，分数分布仍与目标明显不同，提示条件分布也在变化',
      { statistic: round(ks.maximum), limit: options.conditionalShiftLimit,
        byScore: rounded(ks.byScore) }));
  } else if (ks.maximum !== null && ks.maximum > options.conditionalShiftLimit * 0.75) {
    warnings.push('conditional-shift-near-limit');
    diagnostics.push(diagnostic('conditional-shift-near-limit', 'warning',
      '重加权后的分数分布接近条件漂移阈值，建议人工抽样复核',
      { statistic: round(ks.maximum), limit: options.conditionalShiftLimit }));
  }

  const zeroClasses = classes.filter(className =>
    priorObject[className] <= options.zeroPriorTolerance);
  const agreement = crossCheck.prior
    ? Math.max(...classes.map((className, index) =>
      Math.abs(crossCheck.prior[index] - constrained.prior[index])))
    : null;
  if (crossCheck.prior === null) {
    warnings.push('cross-check-unavailable');
    diagnostics.push(diagnostic('cross-check-unavailable', 'warning',
      'EM 交叉核对未得到可用解，只采用约束最小二乘解', { reason: crossCheck.reason }));
  } else if (agreement > options.estimatorAgreementLimit) {
    warnings.push('estimator-disagreement');
    diagnostics.push(diagnostic('estimator-disagreement', 'warning',
      '两条估计路径差异较大，先验估计不稳定',
      { agreement: round(agreement), limit: options.estimatorAgreementLimit,
        leastSquares: rounded(priorObject),
        em: rounded(scoresToObject(crossCheck.prior, classes)) }));
  }
  if (zeroClasses.length) {
    warnings.push('zero-prior');
    diagnostics.push(diagnostic('zero-prior', 'warning',
      '约束解把部分类别先验固定在零；校正时该类别权重为零，不伪造正比例',
      { classes: zeroClasses, fixedZeroClasses: constrained.fixedZero.map(index => classes[index]) }));
  }

  const identified = hardReasons.length === 0;

  // 只有可识别时才给区间与可执行校正；否则输出诊断和停用理由。
  let uncertainty = null;
  if (identified) {
    uncertainty = bootstrapPrior(sourceRows, targetRows, classes, options, policy.seed);
    if (uncertainty.validFraction < options.bootstrapMinValidFraction) {
      warnings.push('bootstrap-unstable');
      diagnostics.push(diagnostic('bootstrap-unstable', 'warning',
        '重采样中可估计比例偏低，先验区间不可靠',
        { valid: uncertainty.valid, replicates: uncertainty.replicates,
          limit: options.bootstrapMinValidFraction, invalid: uncertainty.invalid }));
    }
    const wide = classes.filter(className =>
      uncertainty.intervals[className].width !== null
      && uncertainty.intervals[className].width > options.priorIntervalWidthLimit);
    if (wide.length) {
      warnings.push('prior-interval-wide');
      diagnostics.push(diagnostic('prior-interval-wide', 'warning',
        '先验区间宽度超过阈值，自动调比例的依据不足',
        { classes: wide.map(className => ({
          class: className,
          width: round(uncertainty.intervals[className].width),
          limit: options.priorIntervalWidthLimit
        })) }));
    }
    for (const [reason, count] of Object.entries(uncertainty.invalid)) {
      if (count > 0) {
        diagnostics.push(diagnostic('bootstrap-invalid-replicates', 'info',
          '重采样中有不可估计的重复，已从区间计算中剔除', { reason, count }));
      }
    }
  }

  const correction = identified
    ? buildCorrection(targetRows, classes, priorObject, sourcePriors, options)
    : null;

  const decisionReasons = [...new Set(identified ? warnings : hardReasons)];
  const decision = {
    status: !identified ? 'not-identifiable' : warnings.length ? 'estimated-with-caveats' : 'estimated',
    recommendation: !identified || warnings.length ? 'manual-sampling' : 'use-corrected-proportions',
    reasons: decisionReasons,
    guidance: !identified
      ? '本轮停用比例校正；先补充源验证覆盖，或改由人工抽样确认目标类别构成。'
      : warnings.length
        ? '可得到校正比例，但诊断不够稳定；建议先人工抽样核对，再决定是否自动调比例。'
        : '在标签迁移假设下可启用校正比例；仍建议保留常规抽样复核。'
  };

  const estimation = {
    status: identified ? 'estimated' : 'not-identifiable',
    method: 'constrained-nonnegative-simplex-least-squares',
    prior: identified ? rounded(priorObject) : null,
    sourcePriors: rounded(sourcePriors),
    sourceCounts,
    operator: {
      kind: 'soft-confusion',
      matrix: rounded(softOperator),
      hardMatrix: rounded(hardOperator),
      condition: condition.value === null ? null : round(condition.value),
      rank: condition.rank,
      classSeparation: separation === null ? null : round(separation),
      conditionLimit: policy.conditionLimit
    },
    target: {
      marginalSoft: rounded(scoresToObject(targetMarginalSoft, classes)),
      marginalHard: rounded(scoresToObject(targetMarginalHard, classes)),
      support
    },
    unconstrained: rounded(unconstrained),
    crossCheck: {
      method: 'expectation-maximization on marginal likelihood',
      prior: crossCheck.prior ? rounded(scoresToObject(crossCheck.prior, classes)) : null,
      converged: crossCheck.converged,
      iterations: crossCheck.iterations,
      reason: crossCheck.reason,
      maxAgreementGap: agreement === null ? null : round(agreement),
      agreementLimit: options.estimatorAgreementLimit
    },
    fit: {
      impliedMarginal: rounded(scoresToObject(residual.fit, classes)),
      residual: round(residual.residual),
      residualLimit: options.priorResidualLimit,
      reweightedKs: ks.maximum === null ? null : round(ks.maximum),
      reweightedKsByScore: rounded(ks.byScore),
      conditionalShiftLimit: options.conditionalShiftLimit
    }
  };

  const review = reviewReport(buckets, policy, classes, identified ? priorObject : null, options);
  if (review) {
    base.partitions.review.usedForPostHoc = review.usedForPostHoc === true;
    base.partitions.review.reason = review.reason ?? null;
  }

  return {
    ...base,
    estimation,
    uncertainty: uncertainty ? rounded(uncertainty) : null,
    correction,
    review,
    decision,
    diagnostics
  };
}

function buildCorrection(targetRows, classes, priorObject, sourcePriors, options) {
  const weights = Object.fromEntries(classes.map(className => [className,
    priorObject[className] <= options.zeroPriorTolerance || sourcePriors[className] <= 0
      ? 0
      : priorObject[className] / sourcePriors[className]]));
  const massBefore = Object.fromEntries(classes.map(className => [className, 0]));
  const massAfter = Object.fromEntries(classes.map(className => [className, 0]));
  const hardBefore = Object.fromEntries(classes.map(className => [className, 0]));
  const hardAfter = Object.fromEntries(classes.map(className => [className, 0]));
  const archive = [];
  let changed = 0;
  let identityFallbackRecords = 0;
  let maxRecordDelta = 0;
  for (const row of targetRows) {
    const original = scoresOf(row, classes);
    const adjusted = correctedPosterior(original, priorObject, sourcePriors, classes,
      options.zeroPriorTolerance);
    const originalTop = Math.max(...classes.map(className => original[className]));
    const adjustedTop = Math.max(...classes.map(className => adjusted.scores[className]));
    for (const className of classes) {
      massBefore[className] += original[className];
      massAfter[className] += adjusted.scores[className];
      if (original[className] >= originalTop - EPS) hardBefore[className] += 1;
      if (adjusted.scores[className] >= adjustedTop - EPS) hardAfter[className] += 1;
      maxRecordDelta = Math.max(maxRecordDelta,
        Math.abs(adjusted.scores[className] - original[className]));
    }
    if (adjusted.fallback) identityFallbackRecords += 1;
    if (classes.some(className =>
      Math.abs(adjusted.scores[className] - original[className]) > 1e-12)) changed += 1;
    archive.push({
      id: row.id,
      uncorrected: rounded(original),
      corrected: rounded(adjusted.scores),
      identityFallback: adjusted.fallback !== null
    });
  }
  return {
    method: 'prior-ratio reweighting with per-record renormalization',
    appliedWeights: rounded(weights),
    zeroPriorClasses: classes.filter(className =>
      priorObject[className] <= options.zeroPriorTolerance),
    zeroPriorHandling: classes.some(className => priorObject[className] <= options.zeroPriorTolerance)
      ? `先验为零的类别权重固定为零；所有类别的权重同时为零的行保留原始分数（共 ${identityFallbackRecords} 行）`
      : '无零先验类别',
    identityFallbackRecords,
    totals: {
      posteriorMassBefore: rounded(massBefore),
      posteriorMassAfter: rounded(massAfter),
      posteriorMassDelta: rounded(Object.fromEntries(classes.map(className =>
        [className, massAfter[className] - massBefore[className]]))),
      hardCountBefore: hardBefore,
      hardCountAfter: hardAfter,
      changedRecords: changed,
      unchangedRecords: targetRows.length - changed,
      maxRecordDelta: round(maxRecordDelta),
      interpretation: '总体量按逐行归一后的后验质量与硬计数汇总；不涉及目标真值，不表示准确率变化。'
    },
    archive
  };
}

function reviewReport(buckets, policy, classes, priorObject, options) {
  if (!buckets.review.length) return null;
  const eligible = policy.reviewRelease !== null && policy.asOf >= policy.reviewRelease;
  const labelled = buckets.review.filter(row => row.truth !== '' && classes.includes(row.truth));
  const common = { usedForTuning: false, release: policy.reviewRelease, asOf: policy.asOf };
  if (!eligible) {
    return { ...common, usedForPostHoc: false,
      reason: '审阅集尚未到公开时点，不能用于事后检查，更不能用于选择本次校正参数' };
  }
  if (!labelled.length) {
    return { ...common, usedForPostHoc: false, reason: '审阅集没有可用真值' };
  }
  const scored = inspectScores(labelled, classes, options.scoreRangeTolerance, options.scoreSumTolerance).valid;
  const counts = Object.fromEntries(classes.map(className =>
    [className, scored.filter(row => row.truth === className).length]));
  return {
    ...common,
    usedForPostHoc: true,
    check: {
      labelled: scored.length,
      classCounts: counts,
      estimatedPrior: priorObject ? rounded(priorObject) : null,
      observedPrior: rounded(Object.fromEntries(classes.map(className =>
        [className, scored.length ? counts[className] / scored.length : null]))),
      note: '事后检查只在公开时点之后复盘；未据此反选校正参数。'
    }
  };
}

