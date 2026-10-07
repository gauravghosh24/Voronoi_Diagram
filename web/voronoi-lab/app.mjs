import { COLORS, accuracy, bruteFrames, complexity, distanceSquared, exactLabels, jfaFrames, proposedFrames, roundedRadius } from "./simulation-core.mjs";

const algorithms = ["brute", "jfa", "proposed"];
const state = {
  size: 16,
  cutoff: 7,
  sites: [{ x: 3, y: 4 }, { x: 12, y: 11 }],
  frames: {},
  steps: { brute: 0, jfa: 0, proposed: 0 },
  reference: null,
  hover: { x: 4, y: 3 },
  dragging: null,
  timer: null,
  playing: null,
  randomSeed: 137
};

const $ = (id) => document.getElementById(id);
const fmt = (value) => value.toLocaleString("en-US");
const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
const letter = (i) => String.fromCharCode(65 + i);

function seededRandom() {
  state.randomSeed = (Math.imul(state.randomSeed, 1664525) + 1013904223) >>> 0;
  return state.randomSeed / 4294967296;
}

function uniqueSites(points) {
  const found = new Set();
  return points.map(([rx, ry]) => ({ x: clamp(Math.round(rx * (state.size - 1)), 0, state.size - 1), y: clamp(Math.round(ry * (state.size - 1)), 0, state.size - 1) }))
    .filter((site) => { const key = `${site.x},${site.y}`; if (found.has(key)) return false; found.add(key); return true; });
}

function randomSites() {
  const sites = [];
  while (sites.length < 5) {
    const site = { x: Math.floor(seededRandom() * state.size), y: Math.floor(seededRandom() * state.size) };
    if (!sites.some((other) => other.x === site.x && other.y === site.y)) sites.push(site);
  }
  return sites;
}

function applyPreset(name) {
  if (name === "two") state.sites = uniqueSites([[0.2, 0.27], [0.8, 0.73]]);
  if (name === "three") state.sites = uniqueSites([[0.17, 0.22], [0.77, 0.25], [0.49, 0.78]]);
  if (name === "cluster") state.sites = uniqueSites([[0.18, 0.2], [0.25, 0.23], [0.2, 0.33], [0.78, 0.77], [0.87, 0.72]]);
  if (name === "random") state.sites = randomSites();
  document.querySelectorAll("[data-preset]").forEach((button) => button.classList.toggle("is-active", button.dataset.preset === name));
  rebuild(true);
}

function stopPlayback() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  state.playing = null;
  document.querySelectorAll(".play-button").forEach((button) => {
    button.textContent = "▶";
    button.setAttribute("aria-label", `Play ${button.dataset.algo}`);
  });
}

function rebuild(goToEnd = true) {
  stopPlayback();
  state.reference = exactLabels(state.size, state.sites);
  state.frames = {
    brute: bruteFrames(state.size, state.sites),
    jfa: jfaFrames(state.size, state.sites),
    proposed: proposedFrames(state.size, state.sites, state.cutoff)
  };
  for (const algorithm of algorithms) {
    state.steps[algorithm] = goToEnd ? state.frames[algorithm].length - 1 : clamp(state.steps[algorithm], 0, state.frames[algorithm].length - 1);
    $(`${algorithm}-range`).max = state.frames[algorithm].length - 1;
  }
  $("grid-size-value").textContent = `${state.size} × ${state.size}`;
  const maxCutoff = Math.ceil(Math.sqrt(2 * (state.size - 1) ** 2));
  $("cutoff").max = maxCutoff;
  $("cutoff").value = state.cutoff;
  $("cutoff-value").textContent = state.cutoff;
  renderSites();
  renderComplexity();
  render();
}

function renderSites() {
  const list = $("site-list");
  list.replaceChildren();
  state.sites.forEach((site, index) => {
    const chip = document.createElement("span");
    chip.className = "site-chip";
    chip.innerHTML = `<span class="site-swatch" style="--site-color:${COLORS[index]}"></span><span>${letter(index)} (${site.x},${site.y})</span>`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove site ${letter(index)}`);
    remove.addEventListener("click", () => { state.sites.splice(index, 1); clearPreset(); rebuild(true); });
    chip.append(remove);
    list.append(chip);
  });
}

function clearPreset() {
  document.querySelectorAll("[data-preset]").forEach((button) => button.classList.remove("is-active"));
}

function renderComplexity() {
  const cost = complexity(state.size, state.sites.length, state.cutoff);
  $("brute-cost").textContent = fmt(cost.bruteDistances);
  $("jfa-cost").textContent = `${fmt(cost.jfaProbeUpperBound)} (${cost.jfaPasses} passes)`;
  $("proposed-cost").textContent = fmt(cost.proposedOffsetChecks);
  $("proposed-memory").textContent = cost.proposedFrameBytes < 1024 ? `${cost.proposedFrameBytes} B` : `${(cost.proposedFrameBytes / 1024).toFixed(1)} KiB`;
}

function tinted(hex, strength = 0.62) {
  const value = parseInt(hex.slice(1), 16);
  const r = (value >> 16) & 255, g = (value >> 8) & 255, b = value & 255;
  const mix = (channel) => Math.round(channel + (255 - channel) * strength);
  return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
}

function renderCanvas(algorithm) {
  const canvas = $(`${algorithm}-canvas`);
  const context = canvas.getContext("2d");
  const size = state.size;
  const unit = canvas.width / size;
  const stage = state.frames[algorithm][state.steps[algorithm]];
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#fbfcfe";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const isCoverageSlice = algorithm === "proposed" && state.steps.proposed < state.frames.proposed.length - 1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const label = stage.labels[y * size + x];
      if (label < 0) continue;
      context.fillStyle = tinted(COLORS[label % COLORS.length], isCoverageSlice ? 0.77 : 0.58);
      context.fillRect(x * unit + 0.5, y * unit + 0.5, unit, unit);
    }
  }
  context.strokeStyle = "#dce3ec";
  context.lineWidth = 1;
  context.beginPath();
  for (let i = 0; i <= size; i++) {
    const position = Math.round(i * unit) + 0.5;
    context.moveTo(position, 0);
    context.lineTo(position, canvas.height);
    context.moveTo(0, position);
    context.lineTo(canvas.width, position);
  }
  context.stroke();

  if (state.hover && state.hover.x < size && state.hover.y < size) {
    context.strokeStyle = "#1b2541";
    context.lineWidth = 2.5;
    context.strokeRect(state.hover.x * unit + 1.5, state.hover.y * unit + 1.5, unit - 3, unit - 3);
  }
  state.sites.forEach((site, index) => {
    const cx = (site.x + 0.5) * unit;
    const cy = (site.y + 0.5) * unit;
    const radius = clamp(unit * 0.35, 7, 13);
    context.beginPath();
    context.arc(cx, cy, radius, 0, 2 * Math.PI);
    context.fillStyle = COLORS[index];
    context.fill();
    context.lineWidth = 2;
    context.strokeStyle = "#fff";
    context.stroke();
    context.fillStyle = "#fff";
    context.font = `700 ${clamp(unit * 0.35, 10, 13)}px DM Sans, sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(letter(index), cx, cy + 0.5);
  });
  $(`${algorithm}-corner`).textContent = `${size} × ${size} · ${state.sites.length} sites`;
}

function renderCard(algorithm) {
  const frames = state.frames[algorithm];
  const step = state.steps[algorithm];
  const frame = frames[step];
  const result = accuracy(frame.labels, state.reference);
  $(`${algorithm}-title`).textContent = frame.title;
  $(`${algorithm}-detail`).textContent = frame.detail;
  $(`${algorithm}-count`).textContent = `${step} / ${frames.length - 1}`;
  $(`${algorithm}-range`).value = step;
  $(`${algorithm}-assigned`).textContent = `${fmt(result.assigned)} / ${fmt(result.total)}`;
  $(`${algorithm}-accuracy`).textContent = `${result.percent.toFixed(1)}%`;
  renderCanvas(algorithm);
}

function renderInspector() {
  const { x, y } = state.hover ?? { x: 0, y: 0 };
  $("inspector-title").textContent = `Pixel (${x}, ${y})`;
  const nearest = state.reference[y * state.size + x];
  $("inspector-subtitle").textContent = nearest < 0 ? "No sites yet. Click a grid cell to add one." : `Exact nearest site: ${letter(nearest)}. Square roots are unnecessary for comparing distances.`;
  const list = $("inspector-distances");
  list.replaceChildren();
  state.sites.forEach((site, index) => {
    const item = document.createElement("div");
    item.className = `distance-pill${index === nearest ? " is-nearest" : ""}`;
    item.innerHTML = `<span class="site-swatch" style="--site-color:${COLORS[index]}"></span><strong>${letter(index)}</strong><span>d² = ${distanceSquared(x, y, site)}</span><em>r = ${roundedRadius(x, y, site)}</em>`;
    list.append(item);
  });
}

function render() {
  algorithms.forEach(renderCard);
  renderInspector();
}

function setStep(algorithm, step) {
  state.steps[algorithm] = clamp(step, 0, state.frames[algorithm].length - 1);
  renderCard(algorithm);
}

function togglePlayback(algorithm) {
  if (state.playing === algorithm) { stopPlayback(); return; }
  stopPlayback();
  if (state.steps[algorithm] === state.frames[algorithm].length - 1) setStep(algorithm, 0);
  state.playing = algorithm;
  const button = document.querySelector(`.play-button[data-algo="${algorithm}"]`);
  button.textContent = "Ⅱ";
  button.setAttribute("aria-label", `Pause ${algorithm}`);
  state.timer = setInterval(() => {
    if (state.steps[algorithm] >= state.frames[algorithm].length - 1) { stopPlayback(); return; }
    setStep(algorithm, state.steps[algorithm] + 1);
  }, 620);
}

function cellFromEvent(canvas, event) {
  const box = canvas.getBoundingClientRect();
  return {
    x: clamp(Math.floor((event.clientX - box.left) * state.size / box.width), 0, state.size - 1),
    y: clamp(Math.floor((event.clientY - box.top) * state.size / box.height), 0, state.size - 1)
  };
}

function attachCanvas(algorithm) {
  const canvas = $(`${algorithm}-canvas`);
  canvas.addEventListener("keydown", (event) => {
    const movement = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
    if (movement) {
      event.preventDefault();
      state.hover = { x: clamp(state.hover.x + movement[0], 0, state.size - 1), y: clamp(state.hover.y + movement[1], 0, state.size - 1) };
      render();
      return;
    }
    const existing = state.sites.findIndex((site) => site.x === state.hover.x && site.y === state.hover.y);
    if (event.key === "Enter" && existing < 0 && state.sites.length < COLORS.length) {
      event.preventDefault();
      state.sites.push({ ...state.hover });
      clearPreset();
      rebuild(true);
    } else if ((event.key === "Delete" || event.key === "Backspace") && existing >= 0) {
      event.preventDefault();
      state.sites.splice(existing, 1);
      clearPreset();
      rebuild(true);
    }
  });
  canvas.addEventListener("pointerdown", (event) => {
    const cell = cellFromEvent(canvas, event);
    state.hover = cell;
    const existing = state.sites.findIndex((site) => site.x === cell.x && site.y === cell.y);
    if (event.shiftKey) {
      if (existing >= 0) { state.sites.splice(existing, 1); clearPreset(); rebuild(true); }
      return;
    }
    if (existing >= 0) {
      state.dragging = existing;
      canvas.setPointerCapture(event.pointerId);
    } else if (state.sites.length < COLORS.length) {
      state.sites.push(cell);
      clearPreset();
      rebuild(true);
    }
  });
  canvas.addEventListener("pointermove", (event) => {
    const cell = cellFromEvent(canvas, event);
    const hoverChanged = !state.hover || state.hover.x !== cell.x || state.hover.y !== cell.y;
    state.hover = cell;
    if (state.dragging !== null) {
      const occupied = state.sites.some((site, index) => index !== state.dragging && site.x === cell.x && site.y === cell.y);
      const site = state.sites[state.dragging];
      if (!occupied && (site.x !== cell.x || site.y !== cell.y)) {
        state.sites[state.dragging] = cell;
        clearPreset();
        rebuild(true);
      } else if (hoverChanged) render();
    } else if (hoverChanged) render();
  });
  const stopDrag = () => { state.dragging = null; };
  canvas.addEventListener("pointerup", stopDrag);
  canvas.addEventListener("pointercancel", stopDrag);
}

document.querySelectorAll("[data-preset]").forEach((button) => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
$("grid-size").addEventListener("input", (event) => {
  const previous = state.size;
  state.size = Number(event.target.value);
  state.sites = state.sites.map((site) => ({ x: Math.round(site.x * (state.size - 1) / (previous - 1)), y: Math.round(site.y * (state.size - 1) / (previous - 1)) }));
  state.sites = state.sites.filter((site, index, all) => all.findIndex((other) => other.x === site.x && other.y === site.y) === index);
  state.cutoff = Math.min(state.cutoff, Math.ceil(Math.sqrt(2 * (state.size - 1) ** 2)));
  state.hover = { x: Math.min(state.hover.x, state.size - 1), y: Math.min(state.hover.y, state.size - 1) };
  rebuild(true);
});
$("cutoff").addEventListener("input", (event) => { state.cutoff = Number(event.target.value); rebuild(true); });
$("reset-steps").addEventListener("click", () => { stopPlayback(); algorithms.forEach((algorithm) => setStep(algorithm, 0)); });
document.querySelectorAll(".step-button").forEach((button) => button.addEventListener("click", () => {
  stopPlayback();
  setStep(button.dataset.algo, state.steps[button.dataset.algo] + (button.dataset.action === "next" ? 1 : -1));
}));
document.querySelectorAll(".play-button").forEach((button) => button.addEventListener("click", () => togglePlayback(button.dataset.algo)));
algorithms.forEach((algorithm) => {
  $(`${algorithm}-range`).addEventListener("input", (event) => { stopPlayback(); setStep(algorithm, Number(event.target.value)); });
  attachCanvas(algorithm);
});
rebuild(true);

// Optional browser-agent interface. It uses the same state changes as the controls above.
const modelContext = document.modelContext;
if (modelContext?.registerTool) {
  const lifecycle = new AbortController();
  window.addEventListener("beforeunload", () => lifecycle.abort(), { once: true });
  const register = (tool) => {
    try { Promise.resolve(modelContext.registerTool(tool, { signal: lifecycle.signal })).catch((error) => console.warn("WebMCP registration failed", error)); }
    catch (error) { console.warn("WebMCP registration failed", error); }
  };
  register({
    name: "read_simulation_state",
    title: "Read Voronoi simulation",
    description: "Read the current grid, sites, cutoff, algorithm stages, and visible accuracy values.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: false },
    execute() {
      return {
        gridSize: state.size,
        sites: state.sites.map(({ x, y }) => ({ x, y })),
        cutoff: state.cutoff,
        stages: { ...state.steps },
        results: Object.fromEntries(algorithms.map((algorithm) => [algorithm, accuracy(state.frames[algorithm][state.steps[algorithm]].labels, state.reference)]))
      };
    }
  });
  register({
    name: "configure_simulation",
    title: "Configure Voronoi simulation",
    description: "Set the grid size, cutoff radius, and unique integer-coordinate sites, then show final results for all three algorithms.",
    inputSchema: {
      type: "object",
      properties: {
        gridSize: { type: "integer", minimum: 8, maximum: 28 },
        cutoff: { type: "integer", minimum: 0 },
        sites: { type: "array", maxItems: 8, items: { type: "object", properties: { x: { type: "integer" }, y: { type: "integer" } }, required: ["x", "y"], additionalProperties: false } }
      },
      required: ["gridSize", "cutoff", "sites"], additionalProperties: false
    },
    annotations: { readOnlyHint: false, untrustedContentHint: false },
    execute(input) {
      if (!input || !Number.isInteger(input.gridSize) || input.gridSize < 8 || input.gridSize > 28) throw new Error("gridSize must be an integer from 8 to 28");
      const maxCutoff = Math.ceil(Math.sqrt(2 * (input.gridSize - 1) ** 2));
      if (!Number.isInteger(input.cutoff) || input.cutoff < 0 || input.cutoff > maxCutoff) throw new Error(`cutoff must be from 0 to ${maxCutoff}`);
      if (!Array.isArray(input.sites) || input.sites.length > COLORS.length) throw new Error("sites must contain at most 8 positions");
      const seen = new Set();
      for (const site of input.sites) {
        if (!site || !Number.isInteger(site.x) || !Number.isInteger(site.y) || site.x < 0 || site.y < 0 || site.x >= input.gridSize || site.y >= input.gridSize) throw new Error("site coordinates must be integer cells inside the grid");
        const key = `${site.x},${site.y}`;
        if (seen.has(key)) throw new Error("site coordinates must be unique");
        seen.add(key);
      }
      state.size = input.gridSize;
      state.cutoff = input.cutoff;
      state.sites = input.sites.map(({ x, y }) => ({ x, y }));
      state.hover = { x: clamp(state.hover.x, 0, state.size - 1), y: clamp(state.hover.y, 0, state.size - 1) };
      $("grid-size").value = state.size;
      clearPreset();
      rebuild(true);
      return { gridSize: state.size, cutoff: state.cutoff, sites: state.sites.length, stages: { ...state.steps } };
    }
  });
  register({
    name: "set_algorithm_stage",
    title: "Set algorithm stage",
    description: "Move one algorithm timeline to a specified numbered stage, from zero through its final stage.",
    inputSchema: {
      type: "object",
      properties: { algorithm: { type: "string", enum: algorithms }, stage: { type: "integer", minimum: 0 } },
      required: ["algorithm", "stage"], additionalProperties: false
    },
    annotations: { readOnlyHint: false, untrustedContentHint: false },
    execute(input) {
      if (!input || !algorithms.includes(input.algorithm) || !Number.isInteger(input.stage) || input.stage < 0 || input.stage >= state.frames[input.algorithm].length) throw new Error("unknown algorithm or stage outside its timeline");
      stopPlayback();
      setStep(input.algorithm, input.stage);
      return { algorithm: input.algorithm, stage: input.stage, title: state.frames[input.algorithm][input.stage].title, accuracy: accuracy(state.frames[input.algorithm][input.stage].labels, state.reference) };
    }
  });
}
