export const COLORS = ["#4f6cff", "#f16d5d", "#20a88a", "#ad6bdd", "#df9d2f", "#e6539a", "#328ab3", "#7b9b38"];

export function distanceSquared(x, y, site) {
  const dx = x - site.x;
  const dy = y - site.y;
  return dx * dx + dy * dy;
}

export function roundedRadius(x, y, site) {
  return Math.floor(Math.sqrt(distanceSquared(x, y, site)) + 0.5);
}

export function exactLabels(size, sites) {
  const labels = new Int16Array(size * size).fill(-1);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let best = Infinity;
      for (let s = 0; s < sites.length; s++) {
        const d2 = distanceSquared(x, y, sites[s]);
        if (d2 < best) {
          best = d2;
          labels[y * size + x] = s;
        }
      }
    }
  }
  return labels;
}

export function bruteFrames(size, sites) {
  const frames = [{ labels: new Int16Array(size * size).fill(-1), title: "No sites checked", detail: "Every pixel is still unassigned." }];
  for (let count = 1; count <= sites.length; count++) {
    frames.push({
      labels: exactLabels(size, sites.slice(0, count)),
      title: `After site ${String.fromCharCode(64 + count)}`,
      detail: `Each pixel compares its current best distance with site ${String.fromCharCode(64 + count)}.`
    });
  }
  return frames;
}

export function jumpSchedule(size) {
  let jump = 1;
  while (jump * 2 < size) jump *= 2;
  const schedule = [];
  while (jump >= 1) {
    schedule.push(jump);
    jump = Math.floor(jump / 2);
  }
  schedule.push(1); // This project's extra local correction pass.
  return schedule;
}

export function jfaFrames(size, sites) {
  const seed = new Int16Array(size * size).fill(-1);
  // The Python kernel seeds sites in index order. The UI keeps sites unique.
  sites.forEach((site, s) => { seed[site.y * size + site.x] = s; });
  const frames = [{ labels: seed, title: "Seed sites", detail: "Only the site pixels have labels." }];
  let src = seed;
  const schedule = jumpSchedule(size);
  schedule.forEach((jump, pass) => {
    const dst = new Int16Array(size * size).fill(-1);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const index = y * size + x;
        let bestLabel = src[index];
        let bestDist = bestLabel < 0 ? Infinity : distanceSquared(x, y, sites[bestLabel]);
        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            const nx = x + ox * jump;
            const ny = y + oy * jump;
            if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
            const candidate = src[ny * size + nx];
            if (candidate < 0) continue;
            const d2 = distanceSquared(x, y, sites[candidate]);
            if (d2 < bestDist || (d2 === bestDist && (bestLabel < 0 || candidate < bestLabel))) {
              bestDist = d2;
              bestLabel = candidate;
            }
          }
        }
        dst[index] = bestLabel;
      }
    }
    frames.push({
      labels: dst,
      title: pass === schedule.length - 1 ? "Extra correction · jump 1" : `Jump ${jump}`,
      detail: `Each pixel checks up to 8 neighbors ${jump} cell${jump === 1 ? "" : "s"} away, using labels from the previous pass.`
    });
    src = dst;
  });
  return frames;
}

export function proposedFrames(size, sites, cutoff) {
  const frames = [{ labels: new Int16Array(size * size).fill(-1), title: "Frames initialized", detail: "Every site–pixel frame entry starts at INF." }];
  for (let visibleRadius = 0; visibleRadius <= cutoff; visibleRadius++) {
    const labels = new Int16Array(size * size).fill(-1);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let bestRadius = Infinity;
        let bestDist = Infinity;
        for (let s = 0; s < sites.length; s++) {
          const radius = roundedRadius(x, y, sites[s]);
          if (radius > visibleRadius) continue;
          const d2 = distanceSquared(x, y, sites[s]);
          if (radius < bestRadius || (radius === bestRadius && d2 < bestDist)) {
            bestRadius = radius;
            bestDist = d2;
            labels[y * size + x] = s;
          }
        }
      }
    }
    frames.push({
      labels,
      title: `Frame coverage · r ≤ ${visibleRadius}`,
      detail: "A teaching slice of the completed frames. In the project, all site–offset pairs fill in one parallel kernel."
    });
  }
  frames.push({
    labels: frames.at(-1).labels.slice(),
    title: "Final pixel reduction",
    detail: "Pixels run in parallel; each pixel loops over sites, then compares exact squared distance and site index for ties."
  });
  return frames;
}

export function accuracy(labels, reference) {
  let assigned = 0;
  let correct = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] >= 0) assigned++;
    if (labels[i] === reference[i]) correct++;
  }
  return { assigned, correct, total: labels.length, percent: 100 * correct / labels.length };
}

export function complexity(size, siteCount, cutoff) {
  const pixels = size * size;
  const jfaPasses = jumpSchedule(size).length;
  return {
    bruteDistances: pixels * siteCount,
    jfaPasses,
    jfaProbeUpperBound: pixels * 9 * jfaPasses,
    proposedFrameEntries: pixels * siteCount,
    proposedOffsetChecks: siteCount * (2 * cutoff + 1) ** 2,
    proposedReductionChecks: pixels * siteCount,
    proposedFrameBytes: 4 * pixels * siteCount
  };
}
