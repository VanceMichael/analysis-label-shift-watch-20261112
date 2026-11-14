import { matrixCondition, solveLinearSystem } from './linalg.js';
import { createRandom } from './random.js';

const EPS = 1e-12;

export function sum(values) {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

export function scoresOf(row, classes) {
  return Object.fromEntries(classes.map(className => [className, row[`score${className}`]]));
}

export function scoresToObject(vector, classes) {
  return Object.fromEntries(classes.map((className, index) => [className, vector[index]]));
}

export function roleIndex(policy) {
  const index = new Map();
  for (const [role, names] of Object.entries(policy.partitionRoles)) {
    for (const name of names) index.set(name, role);
  }
  return index;
}

// 分数必须有限、落在 [0,1]，且两类之和为一；不满足的行不进入任何估计。
export function inspectScores(rows, classes, rangeTolerance, sumTolerance = rangeTolerance) {
  const valid = [];
  const invalid = [];
  for (const row of rows) {
    const values = classes.map(className => row[`score${className}`]);
    const inRange = values.every(value => Number.isFinite(value)
      && value >= -rangeTolerance && value <= 1 + rangeTolerance);
    if (!inRange) invalid.push({ id: row.id, reason: 'score-out-of-range' });
    else if (Math.abs(sum(values) - 1) > sumTolerance) {
      invalid.push({ id: row.id, reason: 'score-sum-not-one' });
    } else valid.push(row);
  }
  return { valid, invalid };
}

// 源验证集上的混淆算子。soft 为各类条件分数均值；hard 为多数票比例，并列时均摊。
export function confusionOperator(rows, classes, method = 'soft') {
  const matrix = classes.map(() => classes.map(() => 0));
  for (const [column, actual] of classes.entries()) {
    const members = rows.filter(row => row.truth === actual);
    if (!members.length) continue;
    for (const row of members) {
      const values = classes.map(className => row[`score${className}`]);
      const top = Math.max(...values);
      const winners = values.filter(value => value >= top - EPS).length;
      for (const [index, value] of values.entries()) {
        if (method === 'hard') {
          if (value >= top - EPS) matrix[index][column] += 1 / (winners * members.length);
        } else matrix[index][column] += value / members.length;
      }
    }
  }
  return matrix;
}

// 目标集或任意行集的分数边缘。并列最大时同样均摊，避免硬计数偏袒第一类。
export function marginal(rows, classes, method = 'soft') {
  const vector = classes.map(() => 0);
  if (!rows.length) return vector;
  for (const row of rows) {
    const values = classes.map(className => row[`score${className}`]);
    const top = Math.max(...values);
    const winners = values.filter(value => value >= top - EPS).length;
    for (const [index, value] of values.entries()) {
      if (method === 'soft') vector[index] += value / rows.length;
      else if (value >= top - EPS) vector[index] += 1 / (winners * rows.length);
    }
  }
  return vector;
}

function quadratic(A, b, prior) {
  let total = 0;
  for (let row = 0; row < A.length; row += 1) {
    for (let column = 0; column < A.length; column += 1) {
      total += prior[row] * A[row][column] * prior[column];
    }
    total -= 2 * b[row] * prior[row];
  }
  return total;
}

function normalEquations(matrix, target) {
  const size = matrix[0]?.length ?? 0;
  const A = Array.from({ length: size }, () => Array(size).fill(0));
  const b = Array(size).fill(0);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      let total = 0;
      for (let observation = 0; observation < matrix.length; observation += 1) {
        total += matrix[observation][row] * matrix[observation][column];
      }
      A[row][column] = total;
    }
    let total = 0;
    for (let observation = 0; observation < matrix.length; observation += 1) {
      total += matrix[observation][row] * target[observation];
    }
    b[row] = total;
  }
  return { A, b };
}

// 在单纯形上最小化 ||C pi - q||^2：枚举有效集，求精确非负且和为一的解。
// 不使用“逆矩阵后把负值裁零”；若最优解落在边界，会显式记录被固定为零的类别。
export function constrainedPrior(matrix, target, tolerance = 1e-9) {
  const size = target.length;
  const { A, b } = normalEquations(matrix, target);
  const subsetCount = 2 ** size;
  let best = null;
  let feasibleSubsets = 0;
  for (let mask = 1; mask < subsetCount; mask += 1) {
    const free = [];
    for (let index = 0; index < size; index += 1) if (mask & (1 << index)) free.push(index);
    const dimension = free.length;
    const system = Array.from({ length: dimension + 1 }, () => Array(dimension + 1).fill(0));
    const rhs = Array(dimension + 1).fill(0);
    for (let row = 0; row < dimension; row += 1) {
      for (let column = 0; column < dimension; column += 1) {
        system[row][column] = 2 * A[free[row]][free[column]];
      }
      system[row][dimension] = 1;
      system[dimension][row] = 1;
      rhs[row] = 2 * b[free[row]];
    }
    rhs[dimension] = 1;
    const solution = solveLinearSystem(system, rhs);
    if (!solution) continue;
    const values = solution.slice(0, dimension);
    if (values.some(value => value < -tolerance)) continue;
    feasibleSubsets += 1;
    const prior = Array(size).fill(0);
    free.forEach((index, position) => { prior[index] = Math.max(0, values[position]); });
    const total = sum(prior);
    if (!(total > 0)) continue;
    const normalized = prior.map(value => value / total);
    const objective = quadratic(A, b, normalized);
    const fixedZero = [...Array(size).keys()].filter(index => !free.includes(index));
    if (!best || objective < best.objective - 1e-15
      || (Math.abs(objective - best.objective) <= 1e-15
        && fixedZero.length < best.fixedZero.length)) {
      best = { prior: normalized, objective, fixedZero, free, fallback: false };
    }
  }
  if (!best) {
    let vertex = null;
    for (let index = 0; index < size; index += 1) {
      const prior = Array(size).fill(0);
      prior[index] = 1;
      const objective = quadratic(A, b, prior);
      if (!vertex || objective < vertex.objective) {
        vertex = { prior, objective, fixedZero: [...Array(size).keys()].filter(other => other !== index),
          free: [index], fallback: true };
      }
    }
    return { ...vertex, feasibleSubsets, solved: false };
  }
  return { ...best, feasibleSubsets, solved: true };
}

// 独立数值路径的交叉核对：在目标边缘上迭代更新先验。
export function emPrior(matrix, target, { maxIterations = 500, tolerance = 1e-11 } = {}) {
  const size = target.length;
  let prior = Array(size).fill(1 / size);
  let iterations = 0;
  let converged = false;
  let finalChange = null;
  for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
    const fit = matrix.map(row => row.reduce((total, value, index) => total + value * prior[index], 0));
    if (fit.some(value => !(value > EPS))) return { prior: null, iterations, converged: false,
      finalChange: null, reason: 'em-zero-fit' };
    const updated = prior.map((value, index) => value * (target[index] / fit[index]));
    const total = sum(updated);
    if (!(total > 0)) return { prior: null, iterations, converged: false, finalChange: null,
      reason: 'em-non-normalizable' };
    const next = updated.map(value => value / total);
    finalChange = Math.max(...next.map((value, index) => Math.abs(value - prior[index])));
    prior = next;
    iterations = iteration;
    if (finalChange <= tolerance) { converged = true; break; }
  }
  return { prior, iterations, converged, finalChange, reason: converged ? null : 'em-max-iterations' };
}

export function mixtureResidual(matrix, prior, target) {
  const fit = matrix.map(row => row.reduce((total, value, index) => total + value * prior[index], 0));
  const residual = Math.max(...fit.map((value, index) => Math.abs(value - target[index])));
  return { fit, residual };
}

// 每个源真值类别的条件分数支撑，用于检查目标分数是否落在标签迁移可解释的范围内。
export function conditionalSupport(sourceRows, classes) {
  return Object.fromEntries(classes.map(actual => {
    const members = sourceRows.filter(row => row.truth === actual);
    return [actual, Object.fromEntries(classes.map(className => {
      const values = members.map(row => row[`score${className}`]);
      return [className, values.length
        ? { min: Math.min(...values), max: Math.max(...values) }
        : { min: null, max: null }];
    }))];
  }));
}

// 目标行若不属于任何源类别的条件支撑盒，则计为超出源支持（疑似目标新类别）。
export function targetSupport(rows, sourceRows, classes, support, tolerance) {
  let outside = 0;
  let maxNearest = 0;
  const outsideIds = [];
  for (const row of rows) {
    const inside = classes.some(actual => classes.every(className => {
      const bounds = support[actual][className];
      if (bounds.min === null) return false;
      const value = row[`score${className}`];
      return value >= bounds.min - tolerance && value <= bounds.max + tolerance;
    }));
    if (!inside) { outside += 1; outsideIds.push(row.id); }
    let nearest = Infinity;
    for (const source of sourceRows) {
      let distance = 0;
      for (const className of classes) {
        distance += (row[`score${className}`] - source[`score${className}`]) ** 2;
      }
      nearest = Math.min(nearest, Math.sqrt(distance));
    }
    maxNearest = Math.max(maxNearest, nearest);
  }
  return {
    outsideCount: outside,
    outsideFraction: rows.length ? outside / rows.length : 0,
    outsideIds: outsideIds.slice(0, 20),
    maxNearestSourceDistance: Number.isFinite(maxNearest) ? maxNearest : null
  };
}

// 按先验比修正分数并逐行归一；零先验类别的权重固定为零，没有正比例可伪造。
export function correctedPosterior(scores, priors, sourcePriors, classes, zeroTolerance) {
  const zeroPriorClasses = classes.filter(className => priors[className] <= zeroTolerance);
  const weights = Object.fromEntries(classes.map(className => [
    className,
    priors[className] <= zeroTolerance || !(sourcePriors[className] > 0)
      ? 0
      : priors[className] / sourcePriors[className]
  ]));
  const unnormalized = classes.map(className => scores[className] * weights[className]);
  const total = sum(unnormalized);
  if (!(total > 0)) {
    return { scores: { ...scores }, weights, zeroPriorClasses, fallback: 'identity' };
  }
  return {
    scores: Object.fromEntries(classes.map((className, index) =>
      [className, unnormalized[index] / total])),
    weights,
    zeroPriorClasses,
    fallback: null
  };
}

function weightedCdf(pairs, values) {
  const total = sum(pairs.map(([, weight]) => weight));
  const output = [];
  let cumulative = 0;
  let index = 0;
  for (const value of values) {
    while (index < pairs.length && pairs[index][0] <= value) {
      cumulative += pairs[index][1] / total;
      index += 1;
    }
    output.push(cumulative);
  }
  return output;
}

// 加权 KS：按目标/源先验比给源验证集重加权后，其分数分布是否与目标一致。
// 一致支持“标签迁移”解释；明显偏离则提示条件分布也在变化。
export function reweightedKs(sourceRows, targetRows, sourcePriors, priors, classes, zeroTolerance) {
  if (!sourceRows.length || !targetRows.length) return { maximum: null, byScore: {} };
  const weights = sourceRows.map(row => {
    const sourceWeight = sourcePriors[row.truth] ?? 0;
    const targetWeight = priors[row.truth] ?? 0;
    return sourceWeight > 0 && targetWeight > zeroTolerance ? targetWeight / sourceWeight : 0;
  });
  if (!(sum(weights) > 0)) return { maximum: null, byScore: {} };
  const targetCount = targetRows.length;
  const byScore = {};
  let maximum = 0;
  for (const className of classes) {
    const field = `score${className}`;
    const pairs = sourceRows.map((row, index) => [row[field], weights[index]])
      .sort((left, right) => left[0] - right[0]);
    const targetValues = targetRows.map(row => row[field]).sort((left, right) => left - right);
    const pooled = [...new Set([...pairs.map(([value]) => value), ...targetValues])]
      .sort((left, right) => left - right);
    const sourceCdf = weightedCdf(pairs, pooled);
    let targetIndex = 0;
    const targetCdf = pooled.map(value => {
      while (targetIndex < targetCount && targetValues[targetIndex] <= value) targetIndex += 1;
      return targetIndex / targetCount;
    });
    const statistic = Math.max(...pooled.map((_, index) =>
      Math.abs(sourceCdf[index] - targetCdf[index])));
    byScore[field] = statistic;
    maximum = Math.max(maximum, statistic);
  }
  return { maximum, byScore };
}

// 对源（按真值分层）与目标各自独立重采样，给出先验区间；只使用配置中的 seed。
export function bootstrapPrior(sourceRows, targetRows, classes, options, seed) {
  const random = createRandom(seed);
  const replicates = [];
  const invalid = { 'rank-deficient': 0, 'infeasible': 0, 'empty-class': 0 };
  for (let replicate = 0; replicate < options.bootstrapReplicates; replicate += 1) {
    const sampledSource = [];
    let empty = false;
    for (const actual of classes) {
      const members = sourceRows.filter(row => row.truth === actual);
      if (!members.length) { empty = true; break; }
      for (let draw = 0; draw < members.length; draw += 1) {
        sampledSource.push(members[Math.floor(random() * members.length)]);
      }
    }
    if (empty) { invalid['empty-class'] += 1; continue; }
    const sampledTarget = targetRows.map(() => targetRows[Math.floor(random() * targetRows.length)]);
    const matrix = confusionOperator(sampledSource, classes, 'soft');
    const condition = matrixCondition(matrix);
    if (condition.rankDeficient || condition.value === null) { invalid['rank-deficient'] += 1; continue; }
    const target = marginal(sampledTarget, classes, 'soft');
    const solved = constrainedPrior(matrix, target, options.zeroPriorTolerance);
    if (!solved.solved) { invalid.infeasible += 1; continue; }
    replicates.push(solved.prior);
  }
  const alpha = (1 - options.confidence) / 2;
  const intervals = {};
  for (const [index, className] of classes.entries()) {
    const values = replicates.map(prior => prior[index]).sort((left, right) => left - right);
    const at = probability => quantileAt(values, probability);
    const lower = at(alpha);
    const upper = at(1 - alpha);
    intervals[className] = {
      lower,
      upper,
      width: lower === null ? null : upper - lower
    };
  }
  return {
    seed,
    replicates: options.bootstrapReplicates,
    valid: replicates.length,
    validFraction: options.bootstrapReplicates ? replicates.length / options.bootstrapReplicates : 0,
    invalid,
    intervals,
    method: '源（按真值分层）与目标各自独立重采样；每次重采样重新估计算子与先验'
  };
}

function quantileAt(sortedValues, probability) {
  if (!sortedValues.length) return null;
  if (sortedValues.length === 1) return sortedValues[0];
  const position = (sortedValues.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sortedValues[lower];
  const weight = position - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}
