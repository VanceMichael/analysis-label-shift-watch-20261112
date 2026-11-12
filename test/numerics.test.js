import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { summarize } from '../src/domain.js';

const seed = JSON.parse(await readFile('data/records.json', 'utf8')).records[0];
const probe = patches => summarize(patches.map((patch, index) => ({...seed, id: `p${index}`, ...patch}))).groups[0];

test('分数核对保留软概率，不按最大类硬计数', () => {

  assert.equal(probe([{scoreA: 0.8, scoreB: 0.2}, {scoreA: 0.6, scoreB: 0.4}]).meanB, 0.30000000000000004);
  assert.equal(probe([{scoreA: 1, scoreB: 0}]).meanB, 0);
});
