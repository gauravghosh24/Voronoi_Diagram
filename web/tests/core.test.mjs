import test from "node:test";
import assert from "node:assert/strict";
import { bruteFrames, exactLabels, jfaFrames, jumpSchedule, proposedFrames } from "../voronoi-lab/simulation-core.mjs";

test("the three timelines reproduce a known small-grid example", () => {
  const sites = [{ x: 1, y: 1 }, { x: 0, y: 3 }];
  const reference = exactLabels(5, sites);
  assert.equal(reference[3 * 5 + 3], 0); // A: 8 vs B: 9
  assert.equal(reference[3 * 5 + 2], 1); // A: 5 vs B: 4
  assert.deepEqual(Array.from(bruteFrames(5, sites).at(-1).labels), Array.from(reference));

  const proposed = proposedFrames(5, sites, 3).at(-1).labels;
  assert.equal(proposed[3 * 5 + 3], 0);
  assert.equal(proposed[3 * 5 + 2], 1);
  assert.equal(proposed[3 * 5 + 4], -1);
  assert.equal(proposed[4 * 5 + 4], -1);
});

test("a diagonal-sized cutoff agrees with brute force for varied site layouts", () => {
  const size = 9;
  const cases = [
    [{ x: 0, y: 0 }],
    [{ x: 1, y: 1 }, { x: 7, y: 7 }],
    [{ x: 0, y: 8 }, { x: 8, y: 0 }, { x: 4, y: 4 }],
    [{ x: 3, y: 2 }, { x: 4, y: 2 }, { x: 3, y: 3 }, { x: 4, y: 3 }]
  ];
  for (const sites of cases) {
    const expected = exactLabels(size, sites);
    const actual = proposedFrames(size, sites, 12).at(-1).labels;
    assert.deepEqual(Array.from(actual), Array.from(expected));
  }
});

test("JFA carries a seed through shrinking jumps using previous-pass labels", () => {
  const sites = [{ x: 1, y: 1 }, { x: 6, y: 5 }];
  assert.deepEqual(jumpSchedule(8), [4, 2, 1, 1]);
  const frames = jfaFrames(8, sites);
  assert.equal(frames[0].labels[5 * 8 + 6], 1);
  assert.equal(frames[1].labels[1 * 8 + 2], 1); // B reaches (2,1) at jump 4.
  assert.equal(frames[2].labels[3 * 8 + 4], 1); // B reaches (4,3) at jump 2.
  assert.equal(frames.at(-1).labels[3 * 8 + 4], 1); // 8 < 13, so B wins.
});
