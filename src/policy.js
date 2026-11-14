import { readFile } from 'node:fs/promises';

const ROLES = ['training', 'source', 'target', 'review'];

export const defaultPartitionRoles = {
  training: ['training'],
  source: ['source'],
  target: ['target'],
  review: ['review']
};

export const defaultShiftOptions = {
  scoreSumTolerance: 1e-6,
  scoreRangeTolerance: 1e-9,
  outsideSupportTolerance: 1e-9,
  supportOutsideLimit: 0.05,
  priorResidualLimit: 0.05,
  conditionalShiftLimit: 0.2,
  minSourcePerClass: 5,
  minSourceTotal: 20,
  minTargetTotal: 20,
  bootstrapReplicates: 200,
  bootstrapMinValidFraction: 0.5,
  confidence: 0.95,
  priorIntervalWidthLimit: 0.5,
  zeroPriorTolerance: 1e-9
};

const optionRules = {
  scoreSumTolerance: value => typeof value === 'number' && Number.isFinite(value) && value >= 0,
  scoreRangeTolerance: value => typeof value === 'number' && Number.isFinite(value) && value >= 0,
  outsideSupportTolerance: value => typeof value === 'number' && Number.isFinite(value) && value >= 0,
  supportOutsideLimit: value => typeof value === 'number' && Number.isFinite(value) && value > 0 && value <= 1,
  priorResidualLimit: value => typeof value === 'number' && Number.isFinite(value) && value > 0,
  conditionalShiftLimit: value => typeof value === 'number' && Number.isFinite(value) && value > 0 && value <= 1,
  minSourcePerClass: value => Number.isSafeInteger(value) && value >= 0,
  minSourceTotal: value => Number.isSafeInteger(value) && value >= 0,
  minTargetTotal: value => Number.isSafeInteger(value) && value >= 0,
  bootstrapReplicates: value => Number.isSafeInteger(value) && value >= 1,
  bootstrapMinValidFraction: value => typeof value === 'number' && Number.isFinite(value) && value > 0 && value <= 1,
  confidence: value => typeof value === 'number' && Number.isFinite(value) && value > 0 && value < 1,
  priorIntervalWidthLimit: value => typeof value === 'number' && Number.isFinite(value) && value > 0,
  zeroPriorTolerance: value => typeof value === 'number' && Number.isFinite(value) && value >= 0
};

function normalizeRoles(input) {
  if (input === undefined) return structuredClone(defaultPartitionRoles);
  if (input === null || typeof input !== 'object' || Array.isArray(input)) {
    throw new Error('partitionRoles 应为对象');
  }
  for (const key of Object.keys(input)) {
    if (!ROLES.includes(key)) throw new Error(`partitionRoles 含未知角色 ${key}`);
  }
  const roles = {};
  const owners = new Map();
  for (const role of ROLES) {
    const names = input[role] === undefined ? defaultPartitionRoles[role] : input[role];
    if (!Array.isArray(names)) throw new Error(`partitionRoles.${role} 应为数组`);
    for (const name of names) {
      if (typeof name !== 'string' || name.length === 0) {
        throw new Error(`partitionRoles.${role} 应包含非空字符串`);
      }
      if (owners.has(name)) throw new Error(`分区 ${name} 同时映射到 ${owners.get(name)} 与 ${role}`);
      owners.set(name, role);
    }
    roles[role] = [...new Set(names)];
  }
  if (roles.source.length === 0) throw new Error('partitionRoles.source 不能为空');
  if (roles.target.length === 0) throw new Error('partitionRoles.target 不能为空');
  return roles;
}

function normalizeClasses(input) {
  if (!Array.isArray(input) || input.length !== 2) {
    throw new Error('classes 应包含两个类别（当前分数字段为 scoreA 与 scoreB）');
  }
  const classes = [...new Set(input)];
  if (classes.length !== 2 || classes.some(name => typeof name !== 'string' || name.length === 0)) {
    throw new Error('classes 应包含两个不重复的非空类别名');
  }
  if (!classes.includes('A') || !classes.includes('B')) {
    throw new Error('classes 必须为 A 与 B（当前分数字段为 scoreA 与 scoreB）');
  }
  return classes;
}

function normalizeOptions(input) {
  if (input === undefined) return { ...defaultShiftOptions };
  if (input === null || typeof input !== 'object' || Array.isArray(input)) {
    throw new Error('shift 配置应为对象');
  }
  for (const key of Object.keys(input)) {
    if (!(key in defaultShiftOptions)) throw new Error(`shift 配置含未知字段 ${key}`);
  }
  const options = { ...defaultShiftOptions };
  for (const [key, value] of Object.entries(input)) {
    if (!optionRules[key](value)) throw new Error(`shift.${key} 取值不合法`);
    options[key] = value;
  }
  return options;
}

export function normalizePolicy(document) {
  if (document === null || typeof document !== 'object' || Array.isArray(document)) {
    throw new Error('政策配置应为对象');
  }
  const classes = normalizeClasses(document.classes);
  if (typeof document.classifier !== 'string' || document.classifier.length === 0) {
    throw new Error('classifier 应为非空字符串');
  }
  if (typeof document.asOf !== 'number' || !Number.isFinite(document.asOf)) {
    throw new Error('asOf 应为有限数值');
  }
  if (typeof document.conditionLimit !== 'number' || !Number.isFinite(document.conditionLimit)
    || document.conditionLimit <= 0) {
    throw new Error('conditionLimit 应为正数');
  }
  if (!Number.isSafeInteger(document.seed) || document.seed < 0) {
    throw new Error('seed 应为非负整数');
  }
  const reviewRelease = document.reviewRelease === undefined ? null : document.reviewRelease;
  if (reviewRelease !== null && (typeof reviewRelease !== 'number' || !Number.isFinite(reviewRelease))) {
    throw new Error('reviewRelease 应为有限数值或省略');
  }
  return {
    classes,
    classifier: document.classifier,
    asOf: document.asOf,
    conditionLimit: document.conditionLimit,
    seed: document.seed,
    reviewRelease,
    partitionRoles: normalizeRoles(document.partitionRoles),
    options: normalizeOptions(document.shift)
  };
}

export async function loadPolicy(path) {
  return normalizePolicy(JSON.parse(await readFile(path, 'utf8')));
}
