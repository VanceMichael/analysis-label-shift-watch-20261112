import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { loadDataset } from './dataset.js';
import { summarize } from './domain.js';
import { diagnose } from './migration.js';
import { loadPolicy } from './policy.js';
import { writeResult } from './output.js';

const commands = ['summary', 'inspect', 'migrate'];

export function parseArguments(args) {
  const [command = 'summary', ...options] = args;
  if (!commands.includes(command)) throw new Error('命令应为 summary、inspect 或 migrate');
  const result = { command, input: 'data/records.json' };
  const seen = new Set();
  const allowed = command === 'migrate'
    ? ['--input', '--out', '--policy'] : ['--input', '--out'];
  for (let index = 0; index < options.length; index += 2) {
    const name = options[index];
    const value = options[index + 1];
    if (!allowed.includes(name) || !value || value.startsWith('--')) {
      throw new Error(`参数应为 ${allowed.join(' 或 ')}，且需要取值`);
    }
    if (seen.has(name)) throw new Error('参数不可重复');
    seen.add(name);
    result[name.slice(2)] = value;
  }
  if (command === 'migrate' && !seen.has('--policy')) result.policy = 'data/policy.json';
  return result;
}

export async function main(args) {
  const options = parseArguments(args);
  const dataset = await loadDataset(options.input);
  let result;
  if (options.command === 'summary') result = summarize(dataset.records);
  else if (options.command === 'migrate') {
    const policy = await loadPolicy(options.policy);
    result = diagnose(dataset.records, policy);
  } else {
    result = { records: dataset.records.length, fingerprint: dataset.fingerprint,
      fields: dataset.records.length ? Object.keys(dataset.records[0]).sort() : [] };
  }
  const document = { ...result, inputFingerprint: dataset.fingerprint };
  if (options.out) await writeResult(options.out, document);
  return document;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).then(result => {
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  }).catch(error => {
    process.stderr.write(`处理失败：${error.message}\n`);
    process.exitCode = 1;
  });
}
