import { main } from '../src/cli.js';
import { readFile } from 'node:fs/promises';

const started = performance.now();
console.log('读取本地演练记录并核对输入指纹');
console.log(JSON.stringify(await main(['inspect']), null, 2));
console.log('现有领域核算');
console.log(JSON.stringify(await main(['summary', '--out', 'var/summary.json']), null, 2));
console.log('迁移诊断：冻结分类器、源验证集算子与目标未标注分数');
const migration = await main(['migrate', '--out', 'var/migration.json']);
console.log(JSON.stringify({
  assumptions: migration.assumptions,
  partitions: migration.partitions,
  estimation: {
    status: migration.estimation.status,
    prior: migration.estimation.prior,
    condition: migration.estimation.operator?.condition,
    residual: migration.estimation.fit?.residual
  },
  uncertainty: migration.uncertainty,
  totals: migration.correction?.totals ?? null,
  decision: migration.decision,
  diagnostics: migration.diagnostics
}, null, 2));
console.log('迁移诊断覆盖的合成情形');
for (const name of ['hand-computable', 'near-identical', 'new-class', 'conditional-shift', 'zero-prior']) {
  const result = await main(['migrate', '--input', `data/scenarios/${name}.json`]);
  console.log(JSON.stringify({
    scenario: name,
    status: result.estimation.status,
    prior: result.estimation.prior,
    recommendation: result.decision.recommendation,
    reasons: result.decision.reasons
  }));
}
console.log('业务配置');
console.log(JSON.stringify(JSON.parse(await readFile('data/policy.json', 'utf8')), null, 2));
console.log(`完成，用时 ${Math.ceil(performance.now() - started)} 毫秒`);
