# label-shift-watch

truth 允许空字符串表示无标签；scoreA+scoreB 为一，classifier 标识冻结分类器。

项目目前提供本地记录校验、输入指纹、按 `partition` 的原始核算，以及在标签迁移假设下的迁移诊断。分析人员可以保留完整输入并复查每次导出的 JSON；原始数据不被命令改写。业务配置位于 `data/policy.json`，字段约定在 `docs/data.md`。

## 运行

使用 Node.js 22 LTS（>=22.22.1 且 <23）和 npm，无第三方运行依赖。
`.nvmrc` 选择 22 主版本，清单与锁文件记录相同的兼容范围；Dockerfile 仍固定基础镜像摘要，容器构建可精确复现。

```sh
npm ci --ignore-scripts --no-audit --no-fund --offline
npm run build
npm test
npm run summary
node src/cli.js inspect --input data/records.json
node src/cli.js summary --input data/records.json --out var/summary.json
node src/cli.js migrate --out var/migration.json
node src/cli.js migrate --input data/scenarios/hand-computable.json
npm run demo
```

`npm run demo` 在本地打印输入、现有核算、迁移诊断及配置，并生成 `var/summary.json` 与 `var/migration.json`，不访问网络、不等待真实业务时间，通常几秒内完成。输出目录可删除后重新运行，测试只使用临时目录。

## 迁移诊断

`migrate` 使用冻结分类器、源地区带真值的验证集和目标地区未标注分数，估计目标类别比例。核心假设是标签迁移而非任意协变量迁移：`p(score|class)` 不变，只有类别先验可以变化。训练、源验证和目标批次必须是不同记录；训练分区从不进入估计。

可估计时，它在非负且和为一的约束下求解类别先验（不使用逆矩阵后裁零），按先验比修正后验分数并逐行归一，记录零先验的处置，并用源、目标分别重采样给出先验区间。不可估计时停用校正并保留理由：源类别为空、算子病态、目标分数超出源条件支撑、边缘残差超限或条件分布漂移。目标没有标签，因此只报告校正前后的总体量，不声称准确率提高。延迟审阅集只在公开时点之后作事后检查，不用于反选本次参数。详见 `docs/data.md`。

`data/scenarios/` 覆盖可手算二分类、几乎相同的类别分数、目标新类别、目标真实条件分布变化与零先验等合成情形。


## 容器

```sh
docker build -t label-shift-watch:local .
docker run --rm --network none label-shift-watch:local
```

Dockerfile 固定 Node.js 基础镜像摘要，构建时执行语法检查和测试。镜像构建完成后演示无需联网；可覆盖命令运行 `npm test`。构建不需要真实账号、邮件服务、数据库或线上数据。

## 目录

- `src/dataset.js`：记录校验、规范序列化和数据指纹。
- `src/domain.js`：现有业务分组核算。
- `src/policy.js`：角色映射、冻结分类器与诊断阈值的校验。
- `src/diagnostics.js`：混淆算子、约束先验、重采样区间与支撑检查的数值实现。
- `src/migration.js`：迁移诊断编排、可识别性判定与校正档案。
- `src/linalg.js`、`src/random.js`：小型线性代数与可复现随机数，均无第三方依赖。
- `src/cli.js`：本地读取、检查、诊断和导出入口。
- `src/output.js`：以临时文件写入后替换完整 JSON。
- `test/`：输入约束、领域核对、迁移诊断与命令输出回归。
- `data/`：合成记录、独立核对值、本地业务配置与迁移情形。
