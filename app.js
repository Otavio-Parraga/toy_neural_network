/* ════════════════════════════════════════════════════════
   Configuration
   ════════════════════════════════════════════════════════ */
const API   = "";
const R     = 20;          // neuron radius (px)
const MAX_N = 6;           // max neurons to draw per layer
const FEAT  = ["Weight (kg)", "Ear Point.", "Meow/Bark", "Agility"];
const FEAT_FULL = ["Weight (kg)", "Ear Pointiness", "Meow/Bark Ratio", "Agility Score"];
const CLS   = ["Cat 🐱", "Dog 🐶"];
const CCOL  = ["#5d59a8", "#9a6130"];   // [cat, dog]; refreshed in place from --cat / --dog by syncThemeColors()

/* ════════════════════════════════════════════════════════
   Theme colors (read from CSS custom properties in style.css)
   ════════════════════════════════════════════════════════ */
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Resolved palette for the active theme. Call at render time (colors change on toggle).
function theme() {
  return {
    bg:          css("--bg"),
    surface:     css("--surface"),
    border:      css("--border"),
    text:        css("--text"),
    textDim:     css("--text-dim"),
    textMuted:   css("--text-muted"),
    textFaint:   css("--text-faint"),
    accent:      css("--accent"),
    fwd:         css("--fwd"),
    bwd:         css("--bwd"),
    bwdSoft:     css("--bwd-soft"),
    pos:         css("--pos"),
    neg:         css("--neg"),
    wInc:        css("--w-inc"),
    wDec:        css("--w-dec"),
    edgePos:     css("--edge-pos"),
    edgeNeg:     css("--edge-neg"),
    edgeIdle:    css("--edge-idle"),
    highlight:   css("--highlight"),
    cat:         css("--cat"),
    dog:         css("--dog"),
    nodeFill:    css("--node-fill"),
    nodeStroke:  css("--node-stroke"),
    nodeStrokeActive: css("--node-stroke-active"),
    inkOnLight:  css("--ink-on-light"),
    inkOnDark:   css("--ink-on-dark"),
    chartGrid:   css("--chart-grid"),
  };
}

// Readable text color on top of an arbitrary fill.
function inkOn(fill, T = theme()) {
  const L = d3.lab(fill).l;   // perceptual lightness 0–100
  return (isNaN(L) || L > 60) ? T.inkOnLight : T.inkOnDark;
}

// Color with alpha, for Chart.js fills etc.
function withAlpha(color, a) {
  const c = d3.color(color);
  if (!c) return color;
  c.opacity = a;
  return c.formatRgb();
}

function syncThemeColors() {
  const T = theme();
  CCOL[0] = T.cat || CCOL[0];
  CCOL[1] = T.dog || CCOL[1];
}
syncThemeColors();

/* ════════════════════════════════════════════════════════
   State
   ════════════════════════════════════════════════════════ */
const st = {
  hiddenLayers: [4],
  yLabels: [], xRaw: [], nSamples: 0,
  sample: 0,
  data: null,         // full /api/compute response
  steps: [],          // step descriptors built from data
  step: 0,
  lossChart: null,
  lossHistory: [],
  batchSize: 1,       // mini-batch size B sent to /api/compute
  focus: 0,           // index into data.batch.samples shown in the network
  showWeights: false, // label edges with their weight values
  selNeuron: null,    // {layer, idx} clicked by the user; null → neuron 1 of the step's layer
  featStats: [],      // per-feature {mu, sd} of the raw data (for x̂ = (x − μ)/σ)
};

/* ════════════════════════════════════════════════════════
   Boot
   ════════════════════════════════════════════════════════ */
window.addEventListener("DOMContentLoaded", async () => {
  buildArchRow();
  wireControls();
  wireBatchControls();
  await apiInit();
});

/* ════════════════════════════════════════════════════════
   Architecture row
   ════════════════════════════════════════════════════════ */
function buildArchRow() {
  const row = document.getElementById("arch-row");
  row.innerHTML = "";

  const fixed = (v, lbl) => {
    const w = document.createElement("div");
    w.className = "layer-wrap";
    w.innerHTML = `<span class="lbl">${lbl}</span><span class="fixed">${v}</span>`;
    return w;
  };
  const sep = () => { const s = document.createElement("span"); s.className = "sep"; s.textContent = "→"; return s; };
  const editable = (v, i) => {
    const w = document.createElement("div");
    w.className = "layer-wrap";
    w.innerHTML = `<span class="lbl">Hidden ${i+1}</span><input type="number" min="1" max="16" value="${v}" data-i="${i}">`;
    w.querySelector("input").addEventListener("change", e => {
      st.hiddenLayers[i] = Math.max(1, Math.min(16, parseInt(e.target.value) || 1));
      e.target.value = st.hiddenLayers[i];
    });
    return w;
  };

  row.appendChild(fixed(4, "Input"));
  st.hiddenLayers.forEach((n, i) => { row.appendChild(sep()); row.appendChild(editable(n, i)); });
  row.appendChild(sep());
  row.appendChild(fixed(1, "Output"));
}

/* ════════════════════════════════════════════════════════
   Controls
   ════════════════════════════════════════════════════════ */
function wireControls() {
  const slLR = document.getElementById("sl-lr");
  const vLR  = document.getElementById("val-lr");
  const sync = () => {
    const v = Math.pow(10, parseFloat(slLR.value)).toPrecision(2);
    vLR.textContent = v;
    document.getElementById("lr-show").textContent = v;
    // The update step previews w − α·∂L/∂w with the slider's α
    const step = st.steps[st.step];
    if (st.data && step && step.type === "update" && !step.updated) renderInfo(step, st.data);
  };
  slLR.addEventListener("input", sync);
  sync();

  document.getElementById("btn-add").addEventListener("click", () => {
    if (st.hiddenLayers.length >= 5) return;
    st.hiddenLayers.push(4);
    buildArchRow();
  });
  document.getElementById("btn-rm").addEventListener("click", () => {
    if (st.hiddenLayers.length <= 1) return;
    st.hiddenLayers.pop();
    buildArchRow();
  });
  document.getElementById("btn-init").addEventListener("click", () => apiInit());
  document.getElementById("btn-compute").addEventListener("click", () => apiCompute());
  document.getElementById("btn-prev").addEventListener("click", () => goStep(st.step - 1));
  document.getElementById("btn-next").addEventListener("click", () => goStep(st.step + 1));
  document.getElementById("btn-update").addEventListener("click", () => apiUpdate());
  document.getElementById("sel-sample").addEventListener("change", e => {
    st.sample = parseInt(e.target.value);
  });
  document.getElementById("chk-weights").addEventListener("change", e => {
    st.showWeights = e.target.checked;
    if (st.steps.length && st.data) drawNetwork(st.steps[st.step], st.data);
  });
}

function getLR() { return Math.pow(10, parseFloat(document.getElementById("sl-lr").value)); }

/* ════════════════════════════════════════════════════════
   API: init
   ════════════════════════════════════════════════════════ */
async function apiInit() {
  const arch = [4, ...st.hiddenLayers, 1];
  let d;
  try {
    const res = await fetch(`${API}/api/init`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ architecture: arch }),
    });
    if (!res.ok) throw new Error(res.status);
    d = await res.json();
  } catch (e) {
    document.getElementById("net-placeholder").innerHTML =
      `<div>⚠ Cannot reach server.<br>Run: <code>uv run python app.py</code> (or <code>./start.sh</code>)<br>Then open <strong>http://127.0.0.1:8080</strong></div>`;
    return;
  }

  st.yLabels  = d.y_labels;
  st.xRaw     = d.X_raw;
  st.nSamples = d.n_samples;
  st.featStats = computeFeatStats(d.X_raw);
  st.selNeuron = null;
  st.lossHistory = [];
  st.data = null;
  st.steps = [];

  buildSampleSelect();
  resetBatchUI();
  updateEpoch(0);
  document.getElementById("step-bar").classList.add("hidden");
  document.getElementById("net-placeholder").style.display = "flex";
  document.getElementById("net-toolbar").classList.add("hidden");
  hideTip();
  document.getElementById("net-container").innerHTML = "";
  document.getElementById("info-body").innerHTML = '<p class="muted">Architecture reset. Click ▶ Compute to run a training step.</p>';
  document.getElementById("formula-box").style.display = "none";
  document.getElementById("btn-update").classList.add("hidden");
  renderLossChart();
}

/* ════════════════════════════════════════════════════════
   API: compute (forward + backward)
   ════════════════════════════════════════════════════════ */
async function apiCompute() {
  document.getElementById("btn-compute").disabled = true;
  document.getElementById("compute-hint").textContent = "Computing…";
  try {
    const res = await fetch(`${API}/api/compute`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ sample_idx: st.sample, batch_size: st.batchSize }),
    });
    if (!res.ok) throw new Error(res.status);
    st.data = await res.json();
    st.focus = 0;
    validateSelection(st.data);
    st.steps = buildSteps(st.data);
    document.getElementById("net-placeholder").style.display = "none";
    document.getElementById("net-toolbar").classList.remove("hidden");
    document.getElementById("step-bar").classList.remove("hidden");
    document.getElementById("compute-hint").textContent =
      batchSizeOf(st.data) > 1
        ? `Batch of ${batchSizeOf(st.data)} computed (starting at sample ${st.sample + 1}). Step through the passes.`
        : `Sample ${st.sample + 1} computed. Step through the passes.`;
    goStep(0);
  } catch(e) {
    document.getElementById("compute-hint").textContent = "Error: " + e.message;
  } finally {
    document.getElementById("btn-compute").disabled = false;
  }
}

/* ════════════════════════════════════════════════════════
   API: apply update
   ════════════════════════════════════════════════════════ */
async function apiUpdate() {
  const lr = getLR();
  const res = await fetch(`${API}/api/update`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ lr }),
  });
  const d = await res.json();
  st.lossHistory = d.loss_history;
  updateEpoch(d.epoch);
  renderLossChart();

  const pct = ((d.loss_before - d.loss_after) / d.loss_before * 100).toFixed(1);
  const improved = d.loss_after < d.loss_before;

  document.getElementById("loss-now").textContent =
    `${batchSizeOf(st.data) > 1 ? "Mean batch loss" : "Loss"}: ${d.loss_before.toFixed(4)} → ${d.loss_after.toFixed(4)}  (${improved ? "▼ −" : "▲ +"}${Math.abs(pct)}%)`;

  // Update the last step info to show the weight deltas
  const last = st.steps[st.steps.length - 1];
  last.updated = true;
  last.updateData = d;
  last.lr = lr;
  goStep(st.steps.length - 1);

  document.getElementById("btn-update").classList.add("hidden");
  document.getElementById("compute-hint").textContent =
    "Weights updated! Run ▶ Compute again to see the next iteration.";
}

/* ════════════════════════════════════════════════════════
   Build step descriptors
   ════════════════════════════════════════════════════════ */
function buildSteps(d) {
  const arch = d.architecture;
  const nW   = arch.length - 1;
  const steps = [];

  // ── Forward ─────────────────────────────────
  steps.push({ phase: "forward", type: "input",   layer: 0 });
  for (let l = 1; l < arch.length; l++) {
    steps.push({ phase: "forward",
      type: l === arch.length - 1 ? "fwd_output" : "fwd_hidden",
      layer: l, wIdx: l - 1 });
  }

  // ── Loss ────────────────────────────────────
  steps.push({ phase: "loss", type: "loss", layer: arch.length - 1 });

  // ── Backward ────────────────────────────────
  for (let l = arch.length - 1; l >= 1; l--) {
    steps.push({ phase: "backward",
      type: l === arch.length - 1 ? "bwd_output" : "bwd_hidden",
      layer: l, wIdx: l - 1 });
  }

  // ── Update ──────────────────────────────────
  steps.push({ phase: "update", type: "update", layer: null });

  return steps;
}

/* ════════════════════════════════════════════════════════
   Navigate steps
   ════════════════════════════════════════════════════════ */
function goStep(n) {
  if (!st.steps.length) return;
  n = Math.max(0, Math.min(st.steps.length - 1, n));
  st.step = n;

  const step = st.steps[n];
  const total = st.steps.length;

  // Step bar UI
  document.getElementById("btn-prev").disabled = n === 0;
  document.getElementById("btn-next").disabled = n === total - 1;
  document.getElementById("step-counter").textContent = `Step ${n + 1} / ${total}`;

  const tag = document.getElementById("phase-tag");
  const titles = {
    input:      ["FORWARD",  "phase-forward",  "Input Layer"],
    fwd_hidden: ["FORWARD",  "phase-forward",  `Hidden Layer ${step.layer}`],
    fwd_output: ["FORWARD",  "phase-forward",  "Output Layer"],
    loss:       ["LOSS",     "phase-loss",     "Loss Computation"],
    bwd_output: ["BACKWARD", "phase-backward", "Output Gradients"],
    bwd_hidden: ["BACKWARD", "phase-backward", `Hidden Layer ${step.layer} Gradients`],
    update:     ["UPDATE",   "phase-update",   "Weight Update"],
  };
  const [phase, cls, title] = titles[step.type];
  tag.textContent  = phase;
  tag.className    = `phase-tag ${cls}`;
  document.getElementById("step-title").textContent = title;

  // Show/hide update button only on last step (before applied)
  const isLast = n === total - 1;
  const already = !!st.steps[total - 1].updated;   // boolean: toggle(cls, undefined) would flip it
  document.getElementById("btn-update").classList.toggle("hidden", !isLast || already);

  drawNetwork(step, st.data);
  renderInfo(step, st.data);
  renderBatchPanel(step, st.data);
}

/* ════════════════════════════════════════════════════════
   Draw network SVG
   ════════════════════════════════════════════════════════ */
function drawNetwork(step, d) {
  const arch = d.architecture;
  const cont = document.getElementById("net-container");
  hideTip();
  cont.innerHTML = "";

  const W  = Math.max(cont.clientWidth || 650, 400);
  const H  = 420;
  const ML = 100, MR = 60, MT = 44, MB = 24;
  const nL = arch.length;
  const T  = theme();

  const svg = d3.select("#net-container").append("svg")
    .attr("class", "net-svg").attr("width", W).attr("height", H);

  // Glow filter
  const defs = svg.append("defs");
  const f = defs.append("filter").attr("id", "glow").attr("x","-50%").attr("y","-50%").attr("width","200%").attr("height","200%");
  f.append("feGaussianBlur").attr("stdDeviation", "5").attr("result", "blur");
  const m = f.append("feMerge");
  m.append("feMergeNode").attr("in","blur");
  m.append("feMergeNode").attr("in","SourceGraphic");

  // Paint order: visible edges → weight labels → invisible hit paths → neurons
  const gEdges  = svg.append("g").attr("class", "edges");
  const gLabels = svg.append("g").attr("class", "edge-labels");
  const gHits   = svg.append("g").attr("class", "edge-hits");
  const gNodes  = svg.append("g").attr("class", "nodes");

  const xOf = l => ML + (l / (nL - 1)) * (W - ML - MR);
  const yOf = (l) => {
    const n  = Math.min(arch[l], MAX_N);
    const sp = Math.min(H / (n + 1), (H - MT - MB) / n, 60);
    const total = (n - 1) * sp;
    return Array.from({length: n}, (_, i) => H / 2 - total / 2 + i * sp);
  };

  // ── Activation value at layer l, neuron n ──
  const actOf = (l, n) => {
    if (!d.activations) return null;
    const a = d.activations[l];
    return a && n < a.length ? a[n] : null;
  };

  const labelled = new Set(weightLabelLayers(step, nL));
  const eff = effNeuron(step, d);   // neuron whose calculation the formula box shows
  const ud = step.phase === "update" && step.updated ? step.updateData : null;

  // ── Draw edges ──────────────────────────────────────────
  for (let l = 0; l < nL - 1; l++) {
    const srcYs = yOf(l);
    const dstYs = yOf(l + 1);

    const isFwdActive = step.phase === "forward"  && step.layer === l + 1;
    const isBwdActive = step.phase === "backward" && step.wIdx  === l;
    const isUpdate    = step.phase === "update";

    // Weight labels: every edge when the layer is small; otherwise only the edges
    // into (or, for an input, out of) the explained neuron, so labels stay legible.
    const dense  = srcYs.length * dstYs.length > 16;
    const inSel  = eff.layer === l + 1;
    const outSel = eff.layer === 0 && l === 0;
    const wantLabel = (si, di) => st.showWeights && labelled.has(l) &&
      (!dense || (inSel && di === eff.idx) || (outSel && si === eff.idx));
    // Near the source end the fan-out keeps labels apart; a single fan-out/fan-in has more room
    const tLbl = !dense ? Math.max(0.14, Math.min(0.45, 1 / (dstYs.length + 1.2)))
               : (outSel ? 0.55 : 0.3);

    srcYs.forEach((sy, si) => {
      dstYs.forEach((dy, di) => {
        let color   = T.edgeIdle;
        let width   = 0.5;
        let opacity = 0.18;
        let extraClass = "";

        if (isUpdate && step.updateData && l < step.updateData.deltas_w.length) {
          const dw = step.updateData.deltas_w[l];
          if (di < dw.length && si < dw[di].length) {
            const delta = dw[di][si];
            const mag   = Math.min(Math.abs(delta) * 8, 1);
            color   = delta > 0 ? T.wInc : T.wDec;
            width   = 0.5 + mag * 3;
            opacity = 0.15 + mag * 0.7;
          }
        } else if (d.weights && l < d.weights.length) {
          const w = d.weights[l];
          if (di < w.length && si < w[di].length) {
            const wv  = w[di][si];
            const mag = Math.min(Math.abs(wv) / 2, 1);
            color   = wv > 0 ? T.edgePos : T.edgeNeg;
            width   = 0.3 + mag * 2;
            opacity = 0.08 + mag * 0.45;
          }
        }

        if (isFwdActive) { color = T.fwd; width = Math.max(width, 1.2); opacity = Math.max(opacity, 0.55); extraClass = "edge-pulse"; }
        if (isBwdActive) { color = T.bwd; width = Math.max(width, 1.2); opacity = Math.max(opacity, 0.55); extraClass = "edge-pulse"; }
        // Emphasize the weights that enter the explained neuron's calculation
        if ((isFwdActive || isBwdActive) &&
            ((eff.layer === l + 1 && di === eff.idx) || (eff.layer === 0 && l === 0 && si === eff.idx))) {
          width = Math.max(width * 1.4, 2.2); opacity = Math.max(opacity, 0.85);
        }

        const x1 = xOf(l), x2 = xOf(l + 1);
        const line = gEdges.append("line")
          .attr("class", "edge")
          .attr("x1", x1).attr("y1", sy)
          .attr("x2", x2).attr("y2", dy)
          .attr("stroke", color)
          .attr("stroke-width", width)
          .attr("opacity", opacity);
        if (extraClass) line.classed(extraClass, true);

        // Optional always-visible weight label
        if (wantLabel(si, di)) {
          const wv = ud ? ud.weights_new[l][di][si] : d.weights[l][di][si];
          const lx = x1 + tLbl * (x2 - x1), ly = sy + tLbl * (dy - sy);
          const txt = fmtNum(wv, 2);
          const lg = gLabels.append("g").attr("transform", `translate(${lx},${ly})`);
          lg.append("rect")
            .attr("x", -txt.length * 2.7 - 3).attr("y", -6.5)
            .attr("width", txt.length * 5.4 + 6).attr("height", 13).attr("rx", 3)
            .attr("fill", withAlpha(T.surface, 0.88));
          lg.append("text")
            .attr("class", "edge-label")
            .attr("text-anchor", "middle").attr("dy", "0.35em")
            .attr("fill", wv >= 0 ? T.pos : T.neg)
            .text(txt);
        }

        // Invisible, wide hit path so thin edges are easy to hover / tap
        const hit = gHits.append("line")
          .attr("class", "edge-hit")
          .attr("x1", x1).attr("y1", sy)
          .attr("x2", x2).attr("y2", dy);
        const baseWidth = width;
        const enter = ev => {
          clearHotEdge();
          svg.classed("edge-hovering", true);
          line.classed("edge-hot", true).attr("stroke-width", Math.max(baseWidth, 2.5));
          hotEdge = { svg, line, baseWidth };
          showTip(edgeTipHTML(step, d, l, di, si), ev);
        };
        hit.on("pointerenter", enter)
           .on("click", enter)
           .on("pointermove", moveTip)
           .on("pointerleave", ev => {
             if (ev.pointerType === "touch") return;   // touch: keep until the next tap elsewhere
             clearHotEdge();
             hideTip();
           });
      });
    });

    // Backward direction arrow (chevron label on active edge set)
    if (isBwdActive) {
      const midX = (xOf(l) + xOf(l+1)) / 2;
      const midY = H / 2;
      gLabels.append("text")
        .attr("x", midX).attr("y", midY - 18)
        .attr("text-anchor", "middle")
        .attr("fill", T.bwd).attr("font-size", "14px")
        .text("← ∇");
    }
    if (isFwdActive) {
      const midX = (xOf(l) + xOf(l+1)) / 2;
      const midY = H / 2;
      gLabels.append("text")
        .attr("x", midX).attr("y", midY - 18)
        .attr("text-anchor", "middle")
        .attr("fill", T.fwd).attr("font-size", "14px")
        .text("→");
    }
  }

  // ── Draw neurons ─────────────────────────────────────────
  arch.forEach((nN, l) => {
    const xs  = xOf(l);
    const ys  = yOf(l);
    const isHL = step.layer === l;
    const isBwd = step.phase === "backward" && step.layer === l;

    // Layer label
    const lname = layerName(l, nL);
    const fnname = l > 0 ? (l === nL - 1 ? "σ(z)" : "ReLU(z)") : "";
    svg.append("text")
      .attr("x", xs).attr("y", MT - 20)
      .attr("text-anchor", "middle")
      .attr("fill", isHL ? T.highlight : T.textMuted)
      .attr("font-size", "11px").attr("font-weight", isHL ? "700" : "400")
      .text(lname);
    if (fnname) {
      svg.append("text")
        .attr("x", xs).attr("y", MT - 8)
        .attr("text-anchor", "middle")
        .attr("fill", isHL ? T.highlight : T.textFaint)
        .attr("font-size", "9px")
        .text(fnname);
    }
    if (nN > MAX_N) {
      svg.append("text")
        .attr("x", xs).attr("y", H - MB + 14)
        .attr("text-anchor", "middle")
        .attr("fill", T.textMuted).attr("font-size", "9px")
        .text(`+ ${nN - MAX_N} more`);
    }

    ys.forEach((y, ni) => {
      const act  = actOf(l, ni);
      const isSel = l === eff.layer && ni === eff.idx;
      const g    = gNodes.append("g")
        .attr("class", `neuron${isSel ? " is-selected" : ""}`)
        .attr("data-l", l).attr("data-i", ni)
        .attr("tabindex", 0).attr("role", "button")
        .attr("aria-pressed", isSel && eff.user ? "true" : "false")
        .attr("aria-label", `${neuronName(l, ni, nL)}: show its calculation`)
        .attr("transform", `translate(${xs},${y})`);

      // Determine fill / stroke based on phase + layer
      let fill   = T.nodeFill;
      let stroke = T.nodeStroke;
      let showVal = null;

      if (fwdComputed(step, l) && act !== null) {
        if (l === nL - 1) {
          fill = d3.interpolateRgb(T.cat, T.dog)(act);
        } else if (l === 0) {
          fill = d3.interpolateRgb(T.nodeFill, T.fwd)(Math.min(Math.abs(act) / 2, 1));
        } else {
          fill = d3.interpolateRgb(T.nodeFill, T.fwd)(Math.min(act, 1));
        }
        stroke = T.nodeStrokeActive;
        showVal = act.toFixed(2);
      }

      // Backward: gradient highlighted on current backward layer
      if (isBwd && d.grad_z && step.wIdx < d.grad_z.length) {
        const gz = d.grad_z[step.wIdx];
        if (gz && ni < gz.length) {
          fill   = d3.interpolateRgb(T.bwdSoft, T.bwd)(Math.min(Math.abs(gz[ni]) * 3, 1));
          stroke = T.bwd;
          showVal = gz[ni].toFixed(3);
        }
      }

      // Selected neuron: accent halo + ring (dashed while it is only the default example)
      if (isSel) {
        g.append("circle").attr("r", R + 10).attr("fill", withAlpha(T.accent, 0.13)).attr("stroke", "none");
        g.append("circle").attr("r", R + 4.5).attr("fill", "none")
          .attr("stroke", T.accent).attr("stroke-width", 2.5)
          .attr("stroke-dasharray", eff.user ? null : "4 3");
      } else if (isHL) {
        g.append("circle")
          .attr("r", R + 5).attr("fill", "none")
          .attr("stroke", T.highlight).attr("stroke-width", 1.5).attr("opacity", 0.5)
          .attr("filter", "url(#glow)");
      }

      g.append("circle").attr("class", "node-focus").attr("r", R + 7.5).attr("fill", "none");
      g.append("circle").attr("class", "node-core").attr("r", R)
        .attr("fill", fill).attr("stroke", stroke)
        .attr("stroke-width", isHL ? 2 : 1);

      if (showVal != null) {
        g.append("text")
          .attr("text-anchor", "middle").attr("dy", "0.35em")
          .attr("fill", inkOn(fill, T)).attr("font-size", "9px").attr("font-weight", "bold")
          .text(showVal);
      }

      g.on("pointerenter", ev => showTip(neuronTipHTML(step, d, l, ni), ev))
       .on("pointermove", moveTip)
       .on("pointerleave", ev => { if (ev.pointerType !== "touch") hideTip(); })
       .on("click", ev => selectNeuron(l, ni, { ev }))
       .on("keydown", ev => {
         if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); selectNeuron(l, ni, { focus: true }); }
       })
       .on("focus", ev => { if (ev.currentTarget.matches(":focus-visible")) showTip(neuronTipHTML(step, d, l, ni), { target: ev.currentTarget }); })
       .on("blur", () => hideTip());

      // Input layer: feature names on the left
      if (l === 0 && FEAT[ni]) {
        svg.append("text")
          .attr("x", xs - R - 6).attr("y", y).attr("dy", "0.35em")
          .attr("text-anchor", "end")
          .attr("fill", isHL ? T.text : T.textMuted).attr("font-size", "10px")
          .text(FEAT[ni]);
      }

      // Output layer: class labels on the right
      if (l === nL - 1 && ni === 0) {
        svg.append("text").attr("x", xs + R + 8).attr("y", y - 8)
          .attr("fill", T.cat).attr("font-size", "10px").attr("font-weight", "600")
          .text("Cat 🐱");
        svg.append("text").attr("x", xs + R + 8).attr("y", y + 8)
          .attr("fill", T.dog).attr("font-size", "10px").attr("font-weight", "600")
          .text("Dog 🐶");
        if (act !== null) {
          svg.append("text").attr("x", xs + R + 8).attr("y", y + 24)
            .attr("fill", T.textDim).attr("font-size", "9px")
            .text(`p=${act.toFixed(3)}`);
        }
      }
    });
  });
}

// Which weight layers get always-visible labels when "Show weights" is on:
// the active layer in forward/backward steps, every layer otherwise.
function weightLabelLayers(step, nL) {
  if ((step.phase === "forward" || step.phase === "backward") && step.wIdx != null) return [step.wIdx];
  return Array.from({length: nL - 1}, (_, l) => l);
}

/* ════════════════════════════════════════════════════════
   Hover tooltips (edges + neurons)
   ════════════════════════════════════════════════════════ */
let hotEdge = null;       // currently highlighted edge {svg, line, baseWidth}

function clearHotEdge() {
  if (!hotEdge) return;
  hotEdge.svg.classed("edge-hovering", false);
  hotEdge.line.classed("edge-hot", false).attr("stroke-width", hotEdge.baseWidth);
  hotEdge = null;
}

function tipEl() { return document.getElementById("net-tip"); }

function showTip(html, ev) {
  const tip = tipEl();
  if (!tip) return;
  tip.innerHTML = html;
  tip.hidden = false;
  moveTip(ev);
}

// Place the tooltip next to the pointer (or an element's box), flipped to stay in the viewport.
function moveTip(ev) {
  const tip = tipEl();
  if (!tip || tip.hidden || !ev) return;
  let x, y;
  if (ev.clientX != null && (ev.clientX || ev.clientY)) { x = ev.clientX; y = ev.clientY; }
  else if (ev.target && ev.target.getBoundingClientRect) {
    const r = ev.target.getBoundingClientRect();
    x = r.right; y = r.top + r.height / 2;
  } else return;
  const pad = 8, off = 14;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
  let left = x + off, top = y + off;
  if (left + w + pad > vw) left = x - off - w;
  if (top + h + pad > vh) top = y - off - h;
  tip.style.left = `${Math.max(pad, Math.min(left, vw - w - pad))}px`;
  tip.style.top  = `${Math.max(pad, Math.min(top,  vh - h - pad))}px`;
}

function hideTip() {
  const tip = tipEl();
  if (tip) tip.hidden = true;
  clearHotEdge();
}

// Tap/click outside the network's interactive parts dismisses a pinned (touch) tooltip.
document.addEventListener("pointerdown", ev => {
  if (!(ev.target.closest && ev.target.closest(".edge-hit, .neuron"))) hideTip();
});
window.addEventListener("scroll", () => hideTip(), { passive: true });

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Signed number with a real minus sign; tiny non-zero values use exponent notation.
function fmtNum(v, p = 4) {
  if (v == null || isNaN(v)) return "—";
  if (v !== 0 && Math.abs(v) < Math.pow(10, -p) / 2 && p >= 3) {
    return v.toExponential(1).replace(/-/g, "−").replace("e+", "e");
  }
  const s = Math.abs(v).toFixed(p);
  return (v < 0 && Number(s) !== 0 ? "−" : "") + s;
}
const signCls = v => (v > 0 ? "vp" : v < 0 ? "vn" : "v0");

const SUBS = "₀₁₂₃₄₅₆₇₈₉";
const sub = n => String(n).split("").map(c => SUBS[+c] || c).join("");
// Weight index subscript w_ji (1-based); comma only when an index has 2 digits.
const sub2 = (j, i) => (j < 10 && i < 10) ? sub(j) + sub(i) : `${sub(j)},${sub(i)}`;

function layerName(l, nL) {
  return l === 0 ? "Input" : (l === nL - 1 ? "Output" : `Hidden ${l}`);
}
// Short name for a neuron, e.g. "Hidden 1 · n2" or the feature name for inputs.
function neuronName(l, i, nL) {
  if (l === 0) return FEAT_FULL[i] || `Input · x${i + 1}`;
  if (l === nL - 1) return "Output · ŷ";
  return `${layerName(l, nL)} · n${i + 1}`;
}

// Has the forward pass reached layer l at this step?
function fwdComputed(step, l) {
  return (step.phase === "forward" && l <= step.layer) || step.phase !== "forward";
}
// Has the backward pass produced δ for layer l (l ≥ 1) at this step?
function gradComputed(step, l) {
  return l >= 1 && ((step.phase === "backward" && l >= step.layer) || step.phase === "update");
}

function tipRow(label, value, cls = "") {
  return `<dt>${label}</dt><dd class="${cls}">${value}</dd>`;
}
const tipSep = () => `<div class="tip-sep"></div>`;

// Weight-update values for W[l][j][i] (or bias when i == null): old, grad, Δ, new, α.
function updateVals(step, d, l, j, i) {
  const bias = i == null;
  const old  = bias ? d.biases[l][j] : d.weights[l][j][i];
  const g    = bias ? d.grad_b[l][j] : d.grad_w[l][j][i];
  const ud   = step.updated ? step.updateData : null;
  const lr   = ud && step.lr != null ? step.lr : getLR();
  const nw   = ud ? (bias ? ud.biases_new[l][j] : ud.weights_new[l][j][i]) : old - lr * g;
  return { old, g, lr, nw, delta: nw - old, applied: !!ud };
}

const fmtLR = lr => String(+lr.toPrecision(2));

function edgeTipHTML(step, d, l, j, i) {
  const nL = d.architecture.length;
  const B  = batchSizeOf(d);
  const w  = d.weights[l][j][i];
  const src = l === 0 ? `x̂${sub(i + 1)}` : `a${sub(i + 1)}`;
  let rows = tipRow(`w${sub2(j + 1, i + 1)}`, fmtNum(w), signCls(w));

  if (step.phase === "forward" || step.phase === "loss") {
    if (fwdComputed(step, l)) {
      const a = d.activations[l][i];
      rows += tipRow(`${src} <span class="tip-dim">(source)</span>`, fmtNum(a), signCls(a));
      rows += tipSep();
      rows += tipRow(`w · ${src}`, fmtNum(w * a), `${signCls(w * a)} tip-strong`);
      rows += `<div class="tip-note">This connection's contribution to z${sub(j + 1)} of ${esc(neuronName(l + 1, j, nL))}.</div>`;
    } else {
      rows += `<div class="tip-note">Source activation not computed yet — step forward to see w · a.</div>`;
    }
  } else if (step.phase === "backward") {
    if (gradComputed(step, l + 1)) {
      const dz = d.grad_z[l][j], a = d.activations[l][i];
      rows += tipRow(`δ${sub(j + 1)} <span class="tip-dim">(target)</span>`, fmtNum(dz), signCls(dz));
      rows += tipRow(`${src} <span class="tip-dim">(source)</span>`, fmtNum(a), signCls(a));
      rows += tipSep();
      rows += tipRow(`∂L/∂w = δ · ${src}`, fmtNum(dz * a), `${signCls(dz * a)} tip-strong`);
      if (B > 1) {
        const gAvg = d.grad_w[l][j][i];
        rows += tipRow(`batch avg ∂L/∂w`, fmtNum(gAvg), signCls(gAvg));
        rows += `<div class="tip-note">δ · a is for the focus sample; the update uses the mean over B = ${B}.</div>`;
      }
    } else {
      rows += `<div class="tip-note">Gradient not computed yet — backprop reaches this layer in a later step.</div>`;
    }
  } else if (step.phase === "update") {
    const u = updateVals(step, d, l, j, i);
    rows = tipRow("old w", fmtNum(u.old), signCls(u.old));
    rows += tipRow(`∂L/∂w${B > 1 ? " (avg)" : ""}`, fmtNum(u.g), signCls(u.g));
    rows += tipRow(`Δw = −α·∂L/∂w`, fmtNum(u.delta, 5), signCls(u.delta));
    rows += tipSep();
    rows += tipRow("new w", fmtNum(u.nw), `${signCls(u.nw)} tip-strong`);
    rows += `<div class="tip-note">${u.applied ? "Applied" : "Preview"} with α = ${fmtLR(u.lr)}.</div>`;
  }

  return `<div class="tip-title">${esc(neuronName(l, i, nL))} → ${esc(neuronName(l + 1, j, nL))}</div>
    <div class="tip-sub">Weight · ${esc(layerName(l, nL))} → ${esc(layerName(l + 1, nL))} layer</div>
    <dl class="tip-grid">${rows}</dl>`;
}

function neuronTipHTML(step, d, l, j) {
  const nL = d.architecture.length;
  const out = l === nL - 1;
  let rows = "";
  if (l === 0) {
    rows += tipRow("raw x", fmtNum(d.features_raw[j], 3));
    rows += tipRow("x̂ = (x − μ)/σ", fmtNum(d.features_norm[j]), `${signCls(d.features_norm[j])} tip-strong`);
  } else {
    const b = d.biases[l - 1][j];
    rows += tipRow(`bias b${sub(j + 1)}`, fmtNum(b), signCls(b));
    if (fwdComputed(step, l)) {
      const z = d.pre_activations[l - 1][j], a = d.activations[l][j];
      rows += tipRow(`z${sub(j + 1)}`, fmtNum(z), signCls(z));
      rows += tipRow(out ? "ŷ = σ(z)" : "a = ReLU(z)", fmtNum(a), `${out ? "vhl" : signCls(a)} tip-strong`);
      if (out) rows += tipRow("true y", `${d.true_label} (${CLS[d.true_label]})`);
    } else {
      rows += `<div class="tip-note">z and a are computed in a later forward step.</div>`;
    }
    if (gradComputed(step, l)) {
      const dz = d.grad_z[l - 1][j];
      rows += tipSep();
      if (!out && d.grad_a_hidden && d.grad_a_hidden[l - 1]) {
        const da = d.grad_a_hidden[l - 1][j];
        rows += tipRow("∂L/∂a", fmtNum(da), signCls(da));
      }
      rows += tipRow(`δ = ∂L/∂z`, fmtNum(dz), `${signCls(dz)} tip-strong`);
    }
    if (step.phase === "update") {
      const u = updateVals(step, d, l - 1, j, null);
      rows += tipSep();
      rows += tipRow("bias old → new", `${fmtNum(u.old)} → ${fmtNum(u.nw)}`, signCls(u.nw));
    }
  }
  return `<div class="tip-title">${esc(neuronName(l, j, nL))}</div>
    <div class="tip-sub">${esc(layerName(l, nL))} layer${l > 0 ? ` · ${out ? "sigmoid" : "ReLU"}` : " · feature"}</div>
    <dl class="tip-grid">${rows}</dl>`;
}

/* ════════════════════════════════════════════════════════
   Neuron selection + worked calculations
   st.selNeuron = {layer, idx} is the neuron the user clicked (null = none).
   The formula box explains the current step's layer: it uses the selected
   neuron when it belongs to that layer, else neuron 1 of the layer.
   Clicking a neuron jumps to the step of the current phase where that
   neuron is computed (forward: its activation; backward: its gradient).
   ════════════════════════════════════════════════════════ */

// The neuron whose calculation is shown (and ringed) at this step.
function effNeuron(step, d) {
  const nL  = d.architecture.length;
  const sel = st.selNeuron;
  if (step.phase === "update") {
    return sel ? { ...sel, user: true } : { layer: 1, idx: 0, user: false };
  }
  const L = step.type === "loss" ? nL - 1 : step.layer;
  if (sel && sel.layer === L) return { ...sel, user: true };
  // Inputs have no δ: in the last backward step they show the gradients of their outgoing weights
  if (sel && sel.layer === 0 && step.phase === "backward" && step.layer === 1) return { ...sel, user: true };
  return { layer: L, idx: 0, user: false };
}

function selectNeuron(l, i, { focus = false, ev = null } = {}) {
  const d = st.data;
  if (!d || !st.steps.length) return;
  const nL = d.architecture.length;
  st.selNeuron = { layer: l, idx: i };

  const step = st.steps[st.step];
  let target = -1;
  if (step.phase === "forward" || (step.phase === "loss" && l !== nL - 1)) {
    target = st.steps.findIndex(s => s.phase === "forward" && s.layer === l);
  } else if (step.phase === "backward") {
    target = st.steps.findIndex(s => s.phase === "backward" && s.layer === Math.max(1, l));
  }
  goStep(target >= 0 ? target : st.step);

  // The SVG was redrawn: restore focus / tooltip on the new element
  const el = document.querySelector(`#net-container .neuron[data-l="${l}"][data-i="${i}"]`);
  if (!el) return;
  if (focus) el.focus();
  else if (ev) showTip(neuronTipHTML(st.steps[st.step], d, l, i), ev);
}

// Drop a selection that no longer exists (architecture change).
function validateSelection(d) {
  const s = st.selNeuron;
  if (!s || !d) return;
  const a = d.architecture;
  if (s.layer >= a.length || s.idx >= Math.min(a[s.layer], MAX_N)) st.selNeuron = null;
}

// Mean / population std of each raw feature (matches the backend's standardization).
function computeFeatStats(X) {
  if (!X || !X.length) return [];
  const n = X.length;
  return X[0].map((_, c) => {
    const mu = X.reduce((s, r) => s + r[c], 0) / n;
    const sd = Math.sqrt(X.reduce((s, r) => s + (r[c] - mu) ** 2, 0) / n);
    return { mu, sd };
  });
}

// ── Monospace layout helpers (combining marks like x̂ take no column) ──
const vlen = s => String(s).replace(/[̀-ͯ]/g, "").length;
const padL = (s, n) => " ".repeat(Math.max(0, n - vlen(s))) + s;
const padR = (s, n) => s + " ".repeat(Math.max(0, n - vlen(s)));

// Aligned "label = a × b = a·b" rows. Returns the lines and the column of the final "=".
function productBlock(rows, p = 3, pProd = p) {
  const f  = v => fmtNum(v, p);
  const lw = Math.max(...rows.map(r => vlen(r.lbl)));
  const aw = Math.max(...rows.map(r => vlen(f(r.a))));
  const bw = Math.max(...rows.map(r => vlen(f(r.b))));
  const col = 2 + lw + 3 + aw + 3 + bw;
  const lines = rows.map(r =>
    `  ${padR(r.lbl, lw)} = ${padL(f(r.a), aw)} × ${padL(f(r.b), bw)} = ${padL(fmtNum(r.a * r.b, pProd), 7)}`);
  return { lines, col };
}
const resultLine = (lbl, val, col) => `${padR("  " + lbl, col)} = ${padL(val, 7)}`;
const ruleLine   = col => " ".repeat(col + 3) + "───────";

function formulaHead(name, user) {
  return `<div class="fx-head"><span class="fx-who">${esc(name)}</span>` +
    `<span class="fx-tag${user ? " is-sel" : ""}">${user ? "selected" : "example · click a neuron"}</span></div>`;
}

function setFormula(head, lines) {
  const fbox = document.getElementById("formula-box");
  const body = lines.map(l => typeof l === "string" ? esc(l) : `<span class="${l.cls}">${esc(l.t)}</span>`).join("\n");
  fbox.innerHTML = `${head}<div class="fx-body">${body}</div>`;
  fbox.style.display = "block";
}

// ── Forward: input feature standardization ──
function formulaInput(d, i, user) {
  const I = sub(i + 1);
  const s = (st.featStats || [])[i];
  const raw = d.features_raw[i], x = d.features_norm[i];
  const lines = [`x̂${I} = (x${I} − μ${I}) / σ${I}`];
  if (s) lines.push(`    = (${fmtNum(raw, 3)} − ${fmtNum(s.mu, 3)}) / ${fmtNum(s.sd, 3)}`);
  lines.push({ t: `    = ${fmtNum(x, 3)}`, cls: "fx-res" }, "",
    `μ, σ: mean and std of "${FEAT_FULL[i]}"`,
    `over all ${st.nSamples || "the"} samples.`,
    `x̂${I} is fed to every neuron of ${layerName(1, d.architecture.length)}.`);
  setFormula(formulaHead(neuronName(0, i, d.architecture.length), user), lines);
}

// ── Forward: z = Σ w·a + b, then activation (and loss for the output) ──
function formulaForward(d, l, j, user) {
  const nL = d.architecture.length, out = l === nL - 1;
  const J = sub(j + 1), src = l === 1 ? "x̂" : "a";
  const W = d.weights[l - 1][j], b = d.biases[l - 1][j], prev = d.activations[l - 1];
  const z = d.pre_activations[l - 1][j], a = d.activations[l][j];

  const { lines: terms, col } = productBlock(
    W.map((w, i) => ({ lbl: `w${sub2(j + 1, i + 1)}·${src}${sub(i + 1)}`, a: w, b: prev[i] })));
  const lines = [
    `z${J} = Σ w · ${src} + b${J}`,
    `(one w · ${src} term per neuron of ${layerName(l - 1, nL)})`, "",
    ...terms,
    resultLine(`+ b${J} (bias)`, fmtNum(b, 3), col),
    ruleLine(col),
    { t: resultLine(`z${J}`, fmtNum(z, 3), col), cls: "fx-res" }, "",
  ];
  if (!out) {
    lines.push({ t: `a${J} = ReLU(z${J}) = max(0, ${fmtNum(z, 3)}) = ${fmtNum(a, 3)}`, cls: "fx-res" });
    if (z <= 0) lines.push("     z ≤ 0 → this neuron outputs 0 (inactive)");
  } else {
    const y = d.true_label;
    lines.push(
      `ŷ = σ(z${J}) = 1 / (1 + e^(−z${J}))`,
      `  = 1 / (1 + e^${fmtNum(-z, 3)})`,
      `  = 1 / (1 + ${fmtNum(Math.exp(-z), 4)})`,
      { t: `  = ${fmtNum(a, 4)}   ← P(Dog)`, cls: "fx-res" }, "",
      `L = −[y·log ŷ + (1−y)·log(1−ŷ)],  y = ${y} (${y ? "Dog" : "Cat"})`,
      y ? `  = −log(${fmtNum(a, 4)})` : `  = −log(1 − ${fmtNum(a, 4)})`,
      { t: `  = ${fmtNum(d.loss, 4)}`, cls: "fx-res" });
  }
  setFormula(formulaHead(neuronName(l, j, nL), user), lines);
}

// Per-weight gradient rows ∂L/∂w = δ·a for a neuron's incoming weights (plus batch average).
function gradLines(d, l, j, dz) {
  const J = sub(j + 1), src = l === 1 ? "x̂" : "a";
  const prev = d.activations[l - 1];
  const { lines: rows } = productBlock(
    prev.map((a, i) => ({ lbl: `∂L/∂w${sub2(j + 1, i + 1)}`, a: dz, b: a })), 4);
  const lines = ["", `∂L/∂w = δ${J} · ${src}   (one per incoming weight)`, ...rows, `  ∂L/∂b${J} = δ${J} = ${fmtNum(dz, 4)}`];
  const B = batchSizeOf(d);
  if (B > 1) {
    lines.push("", `Batch average over B = ${B} (used by the update):`);
    const g = d.grad_w[l - 1][j];
    const lw = vlen(`∂L/∂w${sub2(j + 1, prev.length)}`);
    g.forEach((v, i) => lines.push(`  ${padR(`∂L/∂w${sub2(j + 1, i + 1)}`, lw)} = ${padL(fmtNum(v, 4), 7)}`));
    lines.push(`  ${padR(`∂L/∂b${J}`, lw)} = ${padL(fmtNum(d.grad_b[l - 1][j], 4), 7)}`);
  }
  return lines;
}

// ── Backward: δ for a hidden/output neuron and its weight gradients ──
function formulaBackward(d, l, j, user) {
  const nL = d.architecture.length, out = l === nL - 1;
  const J = sub(j + 1);
  const dz = d.grad_z[l - 1][j];
  let lines;
  if (out) {
    lines = [
      `δ${J} = ∂L/∂z${J} = ŷ − y`,
      `   = ${fmtNum(d.probability, 4)} − ${d.true_label}`,
      { t: `   = ${fmtNum(dz, 4)}`, cls: "fx-res" },
      "(sigmoid + BCE: the derivative simplifies to ŷ − y)",
    ];
  } else {
    const Wn = d.weights[l], dn = d.grad_z[l];
    const z  = d.pre_activations[l - 1][j];
    const da = d.grad_a_hidden && d.grad_a_hidden[l - 1] ? d.grad_a_hidden[l - 1][j]
             : dn.reduce((s, dk, k) => s + Wn[k][j] * dk, 0);
    const relu = z > 0 ? 1 : 0;
    const { lines: terms, col } = productBlock(
      dn.map((dk, k) => ({ lbl: `w${sub2(k + 1, j + 1)}·δ${sub(k + 1)}`, a: Wn[k][j], b: dk })), 4);
    lines = [
      `δ${J} = ∂L/∂z${J} = ∂L/∂a${J} · ReLU'(z${J})`,
      `∂L/∂a${J} = Σ w · δ over ${layerName(l + 1, nL)} (${dn.length} term${dn.length > 1 ? "s" : ""})`, "",
      ...terms,
      ruleLine(col),
      resultLine(`∂L/∂a${J} = Σ`, fmtNum(da, 4), col),
      resultLine(`ReLU'(${fmtNum(z, 3)})`, String(relu), col),
      { t: resultLine(`δ${J} = ∂L/∂a${J}·ReLU'`, fmtNum(dz, 4), col), cls: "fx-res" },
    ];
    if (!relu) lines.push("z ≤ 0 → gradient blocked (ReLU' = 0)");
  }
  lines.push(...gradLines(d, l, j, dz));
  setFormula(formulaHead(neuronName(l, j, nL), user), lines);
}

// ── Backward: an input has no δ; show the gradients of its outgoing weights ──
function formulaBackwardInput(d, i, user) {
  const nL = d.architecture.length, I = sub(i + 1);
  const x = d.activations[0][i], dn = d.grad_z[0];
  const { lines: rows } = productBlock(
    dn.map((dj, j) => ({ lbl: `∂L/∂w${sub2(j + 1, i + 1)}`, a: dj, b: x })), 4);
  const lines = [
    `Inputs are data: no δ to backpropagate.`,
    `Weights leaving x̂${I} (into ${layerName(1, nL)}):`,
    `∂L/∂w = δ · x̂${I}   (one per neuron it feeds)`, "", ...rows,
  ];
  const B = batchSizeOf(d);
  if (B > 1) {
    lines.push("", `Batch average over B = ${B}:`);
    const lw = vlen(`∂L/∂w${sub2(dn.length, i + 1)}`);
    dn.forEach((_, j) => lines.push(`  ${padR(`∂L/∂w${sub2(j + 1, i + 1)}`, lw)} = ${padL(fmtNum(d.grad_w[0][j][i], 4), 7)}`));
  }
  setFormula(formulaHead(neuronName(0, i, nL), user), lines);
}

// ── Update: w_new = w_old − α · ∂L/∂w for the neuron's incoming (or outgoing, for inputs) weights ──
function formulaUpdate(step, d, l, j, user) {
  const nL = d.architecture.length, B = batchSizeOf(d);
  const items = [];   // {lbl, vals}
  if (l === 0) {
    d.weights[0].forEach((_, k) => items.push({ lbl: `w${sub2(k + 1, j + 1)}`, vals: updateVals(step, d, 0, k, j) }));
  } else {
    d.weights[l - 1][j].forEach((_, i) => items.push({ lbl: `w${sub2(j + 1, i + 1)}`, vals: updateVals(step, d, l - 1, j, i) }));
    items.push({ lbl: `b${sub(j + 1)}`, vals: updateVals(step, d, l - 1, j, null) });
  }
  const u0 = items[0].vals;
  const a  = fmtLR(u0.lr);
  const lw = Math.max(...items.map(it => vlen(it.lbl)));
  const ow = Math.max(...items.map(it => vlen(fmtNum(it.vals.old))));
  const gw = Math.max(...items.map(it => vlen(fmtNum(it.vals.g))));
  const lines = [
    `w ← w − α · ∂L/∂w,   α = ${a}`,
    B > 1 ? `∂L/∂w = batch average over B = ${B}` : `∂L/∂w from the backward pass`,
    l === 0 ? `(weights leaving this input)` : `(incoming weights and bias)`, "",
    ...items.map(it => {
      const line = `  ${padR(it.lbl, lw)} = ${padL(fmtNum(it.vals.old), ow)} − ${a} × ${padL(fmtNum(it.vals.g), gw)} = ${padL(fmtNum(it.vals.nw), 7)}`;
      return Math.abs(it.vals.delta) > 1e-12 ? line : { t: line, cls: "fx-dim" };
    }),
    "",
    u0.applied ? { t: "✓ Applied: these are the new values.", cls: "fx-ok" }
               : "Preview: click “Apply Weight Update” to commit.",
  ];
  setFormula(formulaHead(neuronName(l, j, nL), user), lines);
}

// Worked calculation for the current step (formula box), or false if the step has none.
function renderFormula(step, d) {
  const e = effNeuron(step, d);
  switch (step.type) {
    case "input":      formulaInput(d, e.idx, e.user); return true;
    case "fwd_hidden":
    case "fwd_output": formulaForward(d, e.layer, e.idx, e.user); return true;
    case "bwd_output":
    case "bwd_hidden":
      if (e.layer === 0) formulaBackwardInput(d, e.idx, e.user);
      else formulaBackward(d, e.layer, e.idx, e.user);
      return true;
    case "update":     formulaUpdate(step, d, e.layer, e.idx, e.user); return true;
  }
  return false;
}

// Make neuron rows in the info tables select that neuron.
function wireSelectableRows(body, step, d) {
  const e = effNeuron(step, d);
  body.querySelectorAll("tr[data-sel-l]").forEach(tr => {
    const l = +tr.dataset.selL, i = +tr.dataset.selI;
    tr.classList.add("sel-row");
    tr.classList.toggle("is-sel", l === e.layer && i === e.idx);
    tr.addEventListener("click", () => selectNeuron(l, i));
  });
}

/* ════════════════════════════════════════════════════════
   Info panel content per step
   ════════════════════════════════════════════════════════ */
function renderInfo(step, d) {
  const title = document.getElementById("info-title");
  const body  = document.getElementById("info-body");
  const fbox  = document.getElementById("formula-box");
  fbox.style.display = "none";

  const lbl = d.true_label;
  const raw = d.features_raw;
  const norm = d.features_norm;

  switch (step.type) {

    // ── INPUT ──────────────────────────────────────────────
    case "input": {
      title.textContent = `Input — Sample #${d.sample_idx + 1}  (True: ${CLS[lbl]})`;
      body.innerHTML = `
        <table>
          <tr><th>Feature</th><th>Raw value</th><th>Normalized</th></tr>
          ${FEAT_FULL.map((f, i) => `
            <tr data-sel-l="0" data-sel-i="${i}"><td>${f}</td>
                <td>${raw[i].toFixed(3)}</td>
                <td class="${norm[i] >= 0 ? "vp" : "vn"}">${norm[i].toFixed(3)}</td></tr>
          `).join("")}
        </table>
        <p style="margin-top:10px;font-size:12px;color:var(--text-muted)">
          Features are <strong>standardized</strong>: x̂ = (x − μ) / σ<br>
          These 4 values enter the network as input activations.
        </p>`;
      break;
    }

    // ── FORWARD HIDDEN ─────────────────────────────────────
    case "fwd_hidden": {
      const l    = step.layer;
      const acts = d.activations[l];
      const pres = d.pre_activations[l - 1];
      title.textContent = `Forward — Hidden Layer ${l}  (ReLU)`;

      const rows = acts.slice(0, MAX_N).map((a, ni) => {
        const z   = pres[ni];
        const cls = a > 0.01 ? "vp" : "v0";
        return `<tr data-sel-l="${l}" data-sel-i="${ni}"><td>n${ni+1}</td>
          <td class="${z >= 0 ? "vp" : "vn"}">${z.toFixed(3)}</td>
          <td class="${cls}">${a.toFixed(3)}</td>
          ${a < 0.001 && z < 0 ? '<td class="vn" style="font-size:10px">⚡ dead</td>' : '<td></td>'}</tr>`;
      }).join("") + (acts.length > MAX_N ? `<tr><td colspan="4" class="muted">… ${acts.length - MAX_N} more</td></tr>` : "");

      body.innerHTML = `
        <table><tr><th>Neuron</th><th>z (pre-act.)</th><th>a = ReLU(z)</th><th></th></tr>${rows}</table>`;
      break;
    }

    // ── FORWARD OUTPUT ─────────────────────────────────────
    case "fwd_output": {
      const prob = d.probability;
      const pred = d.prediction;
      const z_out = d.pre_activations[d.pre_activations.length - 1][0];
      title.textContent = "Forward — Output Layer  (Sigmoid)";

      body.innerHTML = `
        <table>
          <tr><th>Value</th><th></th></tr>
          <tr><td>z (pre-activation)</td><td class="${z_out >= 0 ? "vp" : "vn"}">${z_out.toFixed(4)}</td></tr>
          <tr><td>ŷ = σ(z)  = P(Dog)</td><td class="vhl">${prob.toFixed(4)}</td></tr>
          <tr><td>Prediction</td><td style="color:${CCOL[pred]};font-weight:700">${CLS[pred]}</td></tr>
          <tr><td>True label</td><td style="color:${CCOL[lbl]}">${CLS[lbl]}</td></tr>
          <tr><td>Correct?</td><td class="${pred === lbl ? "vp" : "vn"}">${pred === lbl ? "✓ Yes" : "✗ No"}</td></tr>
        </table>`;
      break;
    }

    // ── LOSS ───────────────────────────────────────────────
    case "loss": {
      const prob = d.probability;
      title.textContent = "Loss — Binary Cross-Entropy";
      const y = lbl, yhat = prob;
      const termY  = y   === 1 ? `1 × log(${yhat.toFixed(4)})` : `0 × log(${yhat.toFixed(4)})`;
      const term1y = y   === 0 ? `1 × log(${(1 - yhat).toFixed(4)})` : `0 × log(${(1 - yhat).toFixed(4)})`;
      body.innerHTML = `
        <table>
          <tr><th>Value</th><th></th></tr>
          <tr><td>True label  y</td><td>${y}  (${CLS[y]})</td></tr>
          <tr><td>Prediction  ŷ</td><td>${yhat.toFixed(4)}</td></tr>
          <tr><td>Loss  L</td><td class="vhl" style="font-size:16px">${d.loss.toFixed(4)}</td></tr>
        </table>
        <p style="margin-top:10px;font-size:12px;color:var(--text-muted)">
          High loss → the network is wrong (or uncertain).<br>
          The backward pass will compute how to reduce this loss.
        </p>`;
      fbox.textContent =
        `L = −[ y · log(ŷ)  +  (1−y) · log(1−ŷ) ]\n  = −[ ${termY}\n    + ${term1y} ]\n  = ${d.loss.toFixed(4)}`;
      fbox.style.display = "block";
      break;
    }

    // ── BACKWARD OUTPUT ────────────────────────────────────
    case "bwd_output": {
      const gz   = d.grad_z[step.wIdx];
      const dz   = gz[0];
      const prob = d.probability;
      title.textContent = "Backward — Output Layer Gradients";
      body.innerHTML = `
        <p style="font-size:12px;color:var(--text-dim);margin-bottom:8px">
          For <strong>BCE + Sigmoid</strong>, the gradient simplifies elegantly:
        </p>
        <table>
          <tr><th>Quantity</th><th>Value</th><th>Meaning</th></tr>
          <tr><td>dL/dz_out</td>
              <td class="${dz >= 0 ? "vp" : "vn"} vhl">${dz.toFixed(4)}</td>
              <td style="font-size:11px">ŷ − y = ${prob.toFixed(3)} − ${lbl}</td></tr>
          <tr><td>dL/dW_out</td>
              <td class="vn" style="font-size:11px">matrix (${d.grad_w[step.wIdx].length}×${d.grad_w[step.wIdx][0].length})</td>
              <td style="font-size:11px">dz · a_hidden^T</td></tr>
          <tr><td>dL/db_out</td>
              <td class="${dz >= 0 ? "vp" : "vn"}">${dz.toFixed(4)}</td>
              <td style="font-size:11px">= dz</td></tr>
        </table>
        <p style="margin-top:10px;font-size:12px;color:var(--text-muted)">
          ${dz > 0 ? "dz > 0: output was too high (reduce it)" : "dz < 0: output was too low (increase it)"}<br>
          This gradient now propagates backward to the hidden layer.
        </p>`;
      break;
    }

    // ── BACKWARD HIDDEN ────────────────────────────────────
    case "bwd_hidden": {
      const l    = step.layer;
      const gz   = d.grad_z[step.wIdx];
      const ga   = d.grad_a_hidden && d.grad_a_hidden[step.wIdx];  // gradient at hidden activations
      const pre  = d.pre_activations[step.wIdx];
      title.textContent = `Backward — Hidden Layer ${l} Gradients`;
      const rows = gz.slice(0, MAX_N).map((dz, ni) => {
        const z    = pre[ni];
        const relu = z > 0 ? 1 : 0;
        const daV  = ga ? ga[ni].toFixed(4) : "—";
        return `<tr data-sel-l="${l}" data-sel-i="${ni}">
          <td>n${ni+1}</td>
          <td class="${(ga ? ga[ni] : 0) >= 0 ? "vp" : "vn"}">${daV}</td>
          <td>${relu} ${relu === 0 ? '<span class="vn">✗ blocked</span>' : '<span class="vp">✓ pass</span>'}</td>
          <td class="${dz >= 0 ? "vp" : "vn"}">${dz.toFixed(4)}</td>
        </tr>`;
      }).join("") + (gz.length > MAX_N ? `<tr><td colspan="4" class="muted">… ${gz.length - MAX_N} more</td></tr>` : "");

      body.innerHTML = `
        <table>
          <tr><th>Neuron</th><th>da (received)</th><th>ReLU'</th><th>dz = da·ReLU'</th></tr>
          ${rows}
        </table>
        <p style="margin-top:8px;font-size:12px;color:var(--text-muted)">
          Neurons with z≤0 had ReLU output = 0.<br>
          Their gradient is <strong>blocked</strong> (dead neurons).
        </p>`;
      break;
    }

    // ── UPDATE ─────────────────────────────────────────────
    case "update": {
      title.textContent = "Weight Update — Gradient Descent";
      const already = step.updated;

      if (already && step.updateData) {
        const ud = step.updateData;
        const pct = ((ud.loss_before - ud.loss_after) / ud.loss_before * 100).toFixed(1);
        body.innerHTML = `
          <table>
            <tr><th>Quantity</th><th>Before</th><th>After</th></tr>
            <tr><td>Loss</td>
                <td>${ud.loss_before.toFixed(4)}</td>
                <td class="${ud.loss_after < ud.loss_before ? "vp" : "vn"}">${ud.loss_after.toFixed(4)}</td></tr>
          </table>
          <p style="margin-top:10px;font-size:13px;color:var(--pos)">
            ✓ Weights updated! Loss ${ud.loss_after < ud.loss_before ? "decreased" : "changed"} by ${Math.abs(pct)}%.<br>
            Run <strong>▶ Compute</strong> again to do the next training step.
          </p>`;
      } else {
        const lr = getLR();
        body.innerHTML = `
          <p style="font-size:13px;margin-bottom:10px">
            All gradients computed. Gradient descent will update every weight:
          </p>
          <div class="update-rule">W  ← W  − α · dL/dW\nb  ← b  − α · dL/db\nα  = ${fmtLR(lr)}
          </div>
          <p style="margin-top:10px;font-size:12px;color:var(--text-muted)">
            • Connections lit in <span style="color:var(--w-inc)">blue</span>: weight increases (dw &lt; 0)<br>
            • Connections lit in <span style="color:var(--w-dec)">rose</span>: weight decreases (dw &gt; 0)
          </p>`;
      }
      break;
    }
  }

  // Worked calculation for the selected (or default) neuron of this step
  renderFormula(step, d);
  wireSelectableRows(body, step, d);
}

/* ════════════════════════════════════════════════════════
   Sample selector
   ════════════════════════════════════════════════════════ */
function buildSampleSelect() {
  const sel = document.getElementById("sel-sample");
  sel.innerHTML = "";
  st.yLabels.forEach((lbl, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `#${i+1}  ${CLS[lbl]}`;
    sel.appendChild(opt);
  });
}

/* ════════════════════════════════════════════════════════
   Loss chart
   ════════════════════════════════════════════════════════ */
function renderLossChart() {
  const ctx = document.getElementById("loss-chart").getContext("2d");
  if (st.lossChart) st.lossChart.destroy();
  const T = theme();
  st.lossChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: st.lossHistory.map((_, i) => i + 1),
      datasets: [{
        label: "Mean batch loss per update",
        data: st.lossHistory,
        borderColor: T.accent,
        backgroundColor: withAlpha(T.accent, 0.1),
        pointBackgroundColor: T.accent,
        pointBorderColor: T.surface,
        borderWidth: 2,
        pointRadius: st.lossHistory.length < 30 ? 3 : 0,
        fill: true, tension: 0.3,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: { labels: { color: T.textDim, font: { size: 11 } } },
        tooltip: { backgroundColor: T.text, titleColor: T.bg, bodyColor: T.bg, callbacks: { label: c => `Loss: ${c.raw.toFixed(4)}` } },
      },
      scales: {
        x: { ticks: { color: T.textMuted, maxTicksLimit: 8 }, grid: { color: T.chartGrid }, border: { color: T.border } },
        y: { ticks: { color: T.textMuted }, grid: { color: T.chartGrid }, border: { color: T.border }, beginAtZero: false },
      },
    },
  });
}

function updateEpoch(n) {
  document.getElementById("epoch-badge").textContent = `Epoch ${n}`;
}

/* ════════════════════════════════════════════════════════
   Mini-batch
   The backend returns top-level fields for the focus sample
   (first of the batch), except grad_w / grad_b, which are
   batch-averaged: ∂L/∂W = (1/B) Σᵢ ∂Lᵢ/∂W.
   data.batch.samples[k] holds the full per-sample results.
   ════════════════════════════════════════════════════════ */

// Per-sample fields swapped when the focus sample changes.
// grad_w / grad_b are deliberately NOT swapped (they stay batch-averaged).
const SAMPLE_FIELDS = [
  "activations", "pre_activations", "loss", "prediction", "probability",
  "grad_z", "grad_a_hidden", "features_raw", "features_norm", "true_label", "sample_idx",
];

function batchSizeOf(d) {
  return d && d.batch ? d.batch.size : 1;
}

function clampBatch(v) {
  const max = st.nSamples || 1;
  return Math.max(1, Math.min(max, parseInt(v) || 1));
}

function wireBatchControls() {
  const inp = document.getElementById("in-batch");
  inp.addEventListener("change", () => {
    st.batchSize = clampBatch(inp.value);
    inp.value = st.batchSize;
  });
  document.getElementById("btn-full-batch").addEventListener("click", () => {
    st.batchSize = clampBatch(st.nSamples);
    inp.value = st.batchSize;
  });
}

// Called from apiInit: sync the batch input to the dataset size and hide the panel.
function resetBatchUI() {
  const inp = document.getElementById("in-batch");
  inp.max = st.nSamples || 32;
  st.batchSize = clampBatch(st.batchSize);
  inp.value = st.batchSize;
  st.focus = 0;
  document.getElementById("batch-panel").classList.add("hidden");
}

// Show batch.samples[k] in the network / info panel and re-render the current step.
function setFocusSample(k) {
  const d = st.data;
  if (!d || !d.batch || k < 0 || k >= d.batch.samples.length) return;
  const s = d.batch.samples[k];
  SAMPLE_FIELDS.forEach(f => { d[f] = s[f]; });
  d.sample_grad_w = s.grad_w;
  d.sample_grad_b = s.grad_b;
  st.focus = k;
  goStep(st.step);
}

function renderBatchPanel(step, d) {
  const panel = document.getElementById("batch-panel");
  const B = batchSizeOf(d);
  if (B <= 1) { panel.classList.add("hidden"); panel.innerHTML = ""; return; }
  panel.classList.remove("hidden");

  const b   = d.batch;
  const ud  = step.type === "update" && step.updated ? step.updateData : null;
  const cur = b.samples[st.focus];
  const avgNote =
    `Gradients are <strong>batch-averaged</strong>: ∂L/∂W = (1/B) Σᵢ ∂Lᵢ/∂W, with B = ${B}.`;

  let html = `<div class="batch-head">
      <span>Mini-batch · B = ${B}</span>
      <span>focus: <strong>#${cur.sample_idx + 1}</strong> (${st.focus + 1}/${B}) · mean loss
        <strong class="vhl">${b.mean_loss.toFixed(4)}</strong></span>
    </div>`;

  if (step.phase === "backward") {
    html += `<p class="batch-note">The values above are for the focus sample only (per-sample ∂Lᵢ).
      ${avgNote} The update applies the average, not the focus sample's gradient.</p>`;
  } else if (step.phase === "update") {
    html += `<p class="batch-note">${avgNote} Edge colors show Δw = −α · (1/B) Σᵢ ∂Lᵢ/∂W.
      ${ud ? "Loss before/after above is the mean over the batch." : ""}</p>`;
  } else if (step.phase === "loss") {
    html += `<p class="batch-note">Batch loss L = (1/B) Σᵢ Lᵢ = <strong>${b.mean_loss.toFixed(4)}</strong>.
      The panel above shows Lᵢ for the focus sample.</p>`;
  }

  if (step.phase === "loss" || step.phase === "update") {
    // Full table: one row per sample, click a row to change focus
    const rows = b.samples.map((s, k) => {
      const lbl = b.true_labels[k], p = b.probabilities[k], pred = b.predictions[k];
      const after = ud && ud.batch_losses_after ? ud.batch_losses_after[k] : null;
      return `<tr class="batch-row${k === st.focus ? " batch-focus" : ""}" data-k="${k}">
        <td>#${s.sample_idx + 1}</td>
        <td class="${lbl ? "batch-dog" : "batch-cat"}">${CLS[lbl]}</td>
        <td class="${pred === lbl ? "vp" : "vn"}">${p.toFixed(3)}</td>
        <td>${b.losses[k].toFixed(4)}</td>
        ${after !== null ? `<td class="${after < b.losses[k] ? "vp" : "vn"}">${after.toFixed(4)}</td>` : ""}
      </tr>`;
    }).join("");
    html += `<div class="batch-table-wrap"><table class="batch-table">
        <tr><th>Sample</th><th>True</th><th>ŷ</th><th>${ud ? "Lᵢ before" : "Lᵢ"}</th>${ud ? "<th>Lᵢ after</th>" : ""}</tr>
        ${rows}
        <tr class="batch-mean"><td colspan="3">Mean (1/B) Σ Lᵢ</td>
          <td>${b.mean_loss.toFixed(4)}</td>${ud ? `<td>${ud.loss_after.toFixed(4)}</td>` : ""}</tr>
      </table></div>`;
  } else {
    // Compact chip strip for the other steps
    html += `<div class="batch-chips">${b.samples.map((s, k) =>
      `<button class="batch-chip ${b.true_labels[k] ? "batch-dog" : "batch-cat"}${k === st.focus ? " batch-focus" : ""}"
         data-k="${k}" title="Sample #${s.sample_idx + 1} · ${CLS[b.true_labels[k]]} · ŷ=${b.probabilities[k].toFixed(3)} · L=${b.losses[k].toFixed(4)}">#${s.sample_idx + 1}</button>`
    ).join("")}</div>`;
  }
  html += `<div class="batch-hint">Click a sample to show it in the network.</div>`;

  panel.innerHTML = html;
  panel.querySelectorAll("[data-k]").forEach(el =>
    el.addEventListener("click", () => setFocusSample(parseInt(el.dataset.k))));
}

/* ════════════════════════════════════════════════════════
   Theme toggle (light / dark)
   ════════════════════════════════════════════════════════ */
const THEME_KEY = "mlp-theme";

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function updateThemeToggleLabel() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  const next = currentTheme() === "dark" ? "light" : "dark";
  btn.setAttribute("aria-label", `Switch to ${next} theme`);
  btn.setAttribute("title", `Switch to ${next} theme`);
}

// Re-apply theme colors to everything drawn by JS (SVG network, info panel, loss chart).
function refreshThemedViews() {
  syncThemeColors();
  if (st.steps.length && st.data) goStep(st.step);
  if (st.lossChart) renderLossChart();
}

function setTheme(t, persist = true) {
  document.documentElement.setAttribute("data-theme", t);
  if (persist) { try { localStorage.setItem(THEME_KEY, t); } catch (e) {} }
  updateThemeToggleLabel();
  refreshThemedViews();
}

if (window.Chart) Chart.defaults.font.family = '"Inter", ui-sans-serif, system-ui, sans-serif';

window.addEventListener("DOMContentLoaded", () => {
  updateThemeToggleLabel();
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.addEventListener("click", () => setTheme(currentTheme() === "dark" ? "light" : "dark"));

  // Follow OS changes until the user makes an explicit choice.
  const mq = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
  if (mq && mq.addEventListener) {
    mq.addEventListener("change", e => {
      let stored = null;
      try { stored = localStorage.getItem(THEME_KEY); } catch (err) {}
      if (!stored) setTheme(e.matches ? "dark" : "light", false);
    });
  }
});
