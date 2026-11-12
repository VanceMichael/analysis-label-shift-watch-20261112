import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { summarize } from '../src/domain.js';

test('领域核算与独立金额及分组清单一致', async () => {
  const data = JSON.parse(await readFile('data/records.json','utf8'));
  const expected = JSON.parse(await readFile('data/expected-summary.json','utf8'));
  const actual = summarize(data.records).groups.map(({key,count,sum}) => ({key,count,sum}));
  assert.deepEqual(actual.map(({key,count}) => ({key,count})), expected.map(({key,count}) => ({key,count})));
  for (const [index, group] of actual.entries()) {
    assert.ok(Math.abs(group.sum - expected[index].sum) <= 1e-12 * Math.max(1, Math.abs(expected[index].sum)));
  }
});
