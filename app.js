/* ════════════════════════════════════════════════════════
   Configuration
   ════════════════════════════════════════════════════════ */
const API   = "";
const R     = 20;          // neuron radius (px)
const MAX_N = 6;           // max neurons to draw per layer
const FEAT  = ["Weight (kg)", "Ear Point.", "Meow/Bark", "Agility"];
const FEAT_FULL = ["Weight (kg)", "Ear Pointiness", "Meow/Bark Ratio", "Agility Score"];
const CLS   = ["Cat 🐱", "Dog 🐶"];
const CCOL  = ["#818cf8", "#fb923c"];

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
};

/* ════════════════════════════════════════════════════════
   Boot
   ════════════════════════════════════════════════════════ */
window.addEventListener("DOMContentLoaded", async () => {
  buildArchRow();
  wireControls();
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
      `<div>⚠ Cannot reach server.<br>Run: <code>conda run -n python3 python app.py</code><br>Then open <strong>http://127.0.0.1:8080</strong></div>`;
    return;
  }

  st.yLabels  = d.y_labels;
  st.xRaw     = d.X_raw;
  st.nSamples = d.n_samples;
  st.lossHistory = [];
  st.data = null;
  st.steps = [];

  buildSampleSelect();
  updateEpoch(0);
  document.getElementById("step-bar").classList.add("hidden");
  document.getElementById("net-placeholder").style.display = "flex";
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
      body: JSON.stringify({ sample_idx: st.sample }),
    });
    if (!res.ok) throw new Error(res.status);
    st.data = await res.json();
    st.steps = buildSteps(st.data);
    document.getElementById("net-placeholder").style.display = "none";
    document.getElementById("step-bar").classList.remove("hidden");
    document.getElementById("compute-hint").textContent =
      `Sample ${st.sample + 1} computed. Step through the passes.`;
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
  const res = await fetch(`${API}/api/update`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ lr: getLR() }),
  });
  const d = await res.json();
  st.lossHistory = d.loss_history;
  updateEpoch(d.epoch);
  renderLossChart();

  const pct = ((d.loss_before - d.loss_after) / d.loss_before * 100).toFixed(1);
  const improved = d.loss_after < d.loss_before;

  document.getElementById("loss-now").textContent =
    `Loss: ${d.loss_before.toFixed(4)} → ${d.loss_after.toFixed(4)}  (${improved ? "▼ −" : "▲ +"}${Math.abs(pct)}%)`;

  // Update the last step info to show the weight deltas
  const last = st.steps[st.steps.length - 1];
  last.updated = true;
  last.updateData = d;
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
  const already = st.steps[total - 1].updated;
  document.getElementById("btn-update").classList.toggle("hidden", !isLast || already);

  drawNetwork(step, st.data);
  renderInfo(step, st.data);
}

/* ════════════════════════════════════════════════════════
   Draw network SVG
   ════════════════════════════════════════════════════════ */
function drawNetwork(step, d) {
  const arch = d.architecture;
  const cont = document.getElementById("net-container");
  cont.innerHTML = "";

  const W  = Math.max(cont.clientWidth || 650, 400);
  const H  = 420;
  const ML = 100, MR = 60, MT = 44, MB = 24;
  const nL = arch.length;

  const svg = d3.select("#net-container").append("svg").attr("width", W).attr("height", H);

  // Glow filter
  const defs = svg.append("defs");
  const f = defs.append("filter").attr("id", "glow").attr("x","-50%").attr("y","-50%").attr("width","200%").attr("height","200%");
  f.append("feGaussianBlur").attr("stdDeviation", "5").attr("result", "blur");
  const m = f.append("feMerge");
  m.append("feMergeNode").attr("in","blur");
  m.append("feMergeNode").attr("in","SourceGraphic");

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

  // ── Gradient value at weight index wIdx, output neuron, input neuron si ──
  const gradWOf = (wIdx, di, si) => {
    if (!d.grad_w || !d.grad_w[wIdx]) return null;
    const g = d.grad_w[wIdx];
    return di < g.length && si < g[di].length ? g[di][si] : null;
  };

  // ── Draw edges ──────────────────────────────────────────
  for (let l = 0; l < nL - 1; l++) {
    const srcYs = yOf(l);
    const dstYs = yOf(l + 1);

    const isFwdActive = step.phase === "forward"  && step.layer === l + 1;
    const isBwdActive = step.phase === "backward" && step.wIdx  === l;
    const isUpdate    = step.phase === "update";

    srcYs.forEach((sy, si) => {
      dstYs.forEach((dy, di) => {
        let color   = "#1e3a5f";
        let width   = 0.5;
        let opacity = 0.18;
        let extraClass = "";

        if (isUpdate && step.updateData && l < step.updateData.deltas_w.length) {
          const dw = step.updateData.deltas_w[l];
          if (di < dw.length && si < dw[di].length) {
            const delta = dw[di][si];
            const mag   = Math.min(Math.abs(delta) * 8, 1);
            color   = delta > 0 ? "#3b82f6" : "#ef4444";
            width   = 0.5 + mag * 3;
            opacity = 0.15 + mag * 0.7;
          }
        } else if (d.weights && l < d.weights.length) {
          const w = d.weights[l];
          if (di < w.length && si < w[di].length) {
            const wv  = w[di][si];
            const mag = Math.min(Math.abs(wv) / 2, 1);
            color   = wv > 0 ? "#1d4ed8" : "#7f1d1d";
            width   = 0.3 + mag * 2;
            opacity = 0.08 + mag * 0.45;
          }
        }

        if (isFwdActive) { color = "#22c55e"; width = Math.max(width, 1.2); opacity = Math.max(opacity, 0.55); extraClass = "edge-pulse"; }
        if (isBwdActive) { color = "#f97316"; width = Math.max(width, 1.2); opacity = Math.max(opacity, 0.55); extraClass = "edge-pulse"; }

        const line = svg.append("line")
          .attr("x1", xOf(l)).attr("y1", sy)
          .attr("x2", xOf(l+1)).attr("y2", dy)
          .attr("stroke", color)
          .attr("stroke-width", width)
          .attr("opacity", opacity);
        if (extraClass) line.classed(extraClass, true);
      });
    });

    // Backward direction arrow (chevron label on active edge set)
    if (isBwdActive) {
      const midX = (xOf(l) + xOf(l+1)) / 2;
      const midY = H / 2;
      svg.append("text")
        .attr("x", midX).attr("y", midY - 18)
        .attr("text-anchor", "middle")
        .attr("fill", "#f97316").attr("font-size", "14px")
        .text("← ∇");
    }
    if (isFwdActive) {
      const midX = (xOf(l) + xOf(l+1)) / 2;
      const midY = H / 2;
      svg.append("text")
        .attr("x", midX).attr("y", midY - 18)
        .attr("text-anchor", "middle")
        .attr("fill", "#22c55e").attr("font-size", "14px")
        .text("→");
    }
  }

  // ── Draw neurons ─────────────────────────────────────────
  arch.forEach((nN, l) => {
    const xs  = xOf(l);
    const ys  = yOf(l);
    const isHL = step.layer === l;
    const isBwd = step.phase === "backward" && step.layer === l;
    const isUpd = step.phase === "update";

    // Layer label
    const lname = l === 0 ? "Input" : (l === nL - 1 ? "Output" : `Hidden ${l}`);
    const fnname = l > 0 ? (l === nL - 1 ? "σ(z)" : "ReLU(z)") : "";
    svg.append("text")
      .attr("x", xs).attr("y", MT - 20)
      .attr("text-anchor", "middle")
      .attr("fill", isHL ? "#fbbf24" : "#475569")
      .attr("font-size", "11px").attr("font-weight", isHL ? "700" : "400")
      .text(lname);
    if (fnname) {
      svg.append("text")
        .attr("x", xs).attr("y", MT - 8)
        .attr("text-anchor", "middle")
        .attr("fill", isHL ? "#fbbf24" : "#334155")
        .attr("font-size", "9px")
        .text(fnname);
    }
    if (nN > MAX_N) {
      svg.append("text")
        .attr("x", xs).attr("y", H - MB + 14)
        .attr("text-anchor", "middle")
        .attr("fill", "#475569").attr("font-size", "9px")
        .text(`+ ${nN - MAX_N} more`);
    }

    ys.forEach((y, ni) => {
      const act  = actOf(l, ni);
      const g    = svg.append("g").attr("transform", `translate(${xs},${y})`);

      // Determine fill / stroke based on phase + layer
      let fill   = "#1e293b";
      let stroke = "#334155";
      let showVal = null;

      // Forward: show activation once computed
      const fwdComputed = (step.phase === "forward"  && l <= step.layer)
                       || step.phase === "loss"
                       || step.phase === "backward"
                       || step.phase === "update";

      if (fwdComputed && act !== null) {
        if (l === nL - 1) {
          fill = d3.interpolateRgb("#818cf8", "#fb923c")(act);
        } else if (l === 0) {
          fill = d3.interpolateRgb("#1e3a5f", "#22c55e")(Math.min(Math.abs(act) / 2, 1));
        } else {
          fill = d3.interpolateRgb("#1e293b", "#22c55e")(Math.min(act, 1));
        }
        stroke = "#475569";
        showVal = act.toFixed(2);
      }

      // Backward: gradient highlighted on current backward layer
      if (isBwd && d.grad_z && step.wIdx < d.grad_z.length) {
        const gz = d.grad_z[step.wIdx];
        if (gz && ni < gz.length) {
          fill   = d3.interpolateRgb("#7c2d12", "#f97316")(Math.min(Math.abs(gz[ni]) * 3, 1));
          stroke = "#f97316";
          showVal = gz[ni].toFixed(3);
        }
      }

      // Glow ring for current layer
      if (isHL) {
        g.append("circle")
          .attr("r", R + 5).attr("fill", "none")
          .attr("stroke", "#fbbf24").attr("stroke-width", 1.5).attr("opacity", 0.5)
          .attr("filter", "url(#glow)");
      }

      g.append("circle").attr("r", R)
        .attr("fill", fill).attr("stroke", stroke)
        .attr("stroke-width", isHL ? 2 : 1);

      if (showVal != null) {
        g.append("text")
          .attr("text-anchor", "middle").attr("dy", "0.35em")
          .attr("fill", "white").attr("font-size", "9px").attr("font-weight", "bold")
          .text(showVal);
      }

      // Input layer: feature names on the left
      if (l === 0 && FEAT[ni]) {
        svg.append("text")
          .attr("x", xs - R - 6).attr("y", y).attr("dy", "0.35em")
          .attr("text-anchor", "end")
          .attr("fill", isHL ? "#e2e8f0" : "#475569").attr("font-size", "10px")
          .text(FEAT[ni]);
      }

      // Output layer: class labels on the right
      if (l === nL - 1 && ni === 0) {
        svg.append("text").attr("x", xs + R + 8).attr("y", y - 8)
          .attr("fill", "#818cf8").attr("font-size", "10px").attr("font-weight", "600")
          .text("Cat 🐱");
        svg.append("text").attr("x", xs + R + 8).attr("y", y + 8)
          .attr("fill", "#fb923c").attr("font-size", "10px").attr("font-weight", "600")
          .text("Dog 🐶");
        if (act !== null) {
          svg.append("text").attr("x", xs + R + 8).attr("y", y + 24)
            .attr("fill", "#64748b").attr("font-size", "9px")
            .text(`p=${act.toFixed(3)}`);
        }
      }
    });
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
            <tr><td>${f}</td>
                <td>${raw[i].toFixed(3)}</td>
                <td class="${norm[i] >= 0 ? "vp" : "vn"}">${norm[i].toFixed(3)}</td></tr>
          `).join("")}
        </table>
        <p style="margin-top:10px;font-size:12px;color:#64748b">
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
      const W    = d.weights[step.wIdx];
      const b    = d.biases[step.wIdx];
      const prev = d.activations[l - 1];
      title.textContent = `Forward — Hidden Layer ${l}  (ReLU)`;

      const rows = acts.slice(0, MAX_N).map((a, ni) => {
        const z   = pres[ni];
        const cls = a > 0.01 ? "vp" : "v0";
        return `<tr><td>n${ni+1}</td>
          <td class="${z >= 0 ? "vp" : "vn"}">${z.toFixed(3)}</td>
          <td class="${cls}">${a.toFixed(3)}</td>
          ${a < 0.001 && z < 0 ? '<td class="vn" style="font-size:10px">⚡ dead</td>' : '<td></td>'}</tr>`;
      }).join("") + (acts.length > MAX_N ? `<tr><td colspan="4" class="muted">… ${acts.length - MAX_N} more</td></tr>` : "");

      body.innerHTML = `
        <table><tr><th>Neuron</th><th>z (pre-act.)</th><th>a = ReLU(z)</th><th></th></tr>${rows}</table>`;

      // Formula for neuron 0
      const z0   = pres[0];
      const a0   = acts[0];
      const w0   = W[0];
      const b0   = b[0];
      const terms = w0.slice(0, 4).map((w, i) => {
        const sgn = (w >= 0 && i > 0) ? "+ " : (w < 0 ? "− " : "");
        return `${i === 0 ? "" : "\n       "}${sgn}${Math.abs(w).toFixed(3)} × ${prev[i].toFixed(3)}`;
      }).join("");
      fbox.textContent =
        `Neuron 1 (example):\nz₁ = ${terms}\n     + ${b0.toFixed(3)}  ← bias\n   = ${z0.toFixed(3)}\n\na₁ = ReLU(${z0.toFixed(3)}) = max(0, ${z0.toFixed(3)}) = ${a0.toFixed(3)}`;
      fbox.style.display = "block";
      break;
    }

    // ── FORWARD OUTPUT ─────────────────────────────────────
    case "fwd_output": {
      const prob = d.probability;
      const pred = d.prediction;
      const z_out = d.pre_activations[d.pre_activations.length - 1][0];
      const W = d.weights[step.wIdx];
      const b = d.biases[step.wIdx];
      const prev = d.activations[step.layer - 1];
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

      const terms = W[0].slice(0, MAX_N).map((w, i) =>
        `${i === 0 ? "" : (w >= 0 ? "+ " : "− ")}${Math.abs(w).toFixed(3)} × ${(prev[i] || 0).toFixed(3)}`
      ).join("\n       ");
      fbox.textContent =
        `z = ${terms}${W[0].length > MAX_N ? "\n    + …" : ""}\n  + ${b[0].toFixed(3)}  ← bias\n  = ${z_out.toFixed(4)}\n\nŷ = σ(${z_out.toFixed(4)}) = 1 / (1 + e^${(- z_out).toFixed(4)})\n  = ${prob.toFixed(4)}`;
      fbox.style.display = "block";
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
        <p style="margin-top:10px;font-size:12px;color:#64748b">
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
        <p style="font-size:12px;color:#94a3b8;margin-bottom:8px">
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
        <p style="margin-top:10px;font-size:12px;color:#64748b">
          ${dz > 0 ? "dz > 0: output was too high (reduce it)" : "dz < 0: output was too low (increase it)"}<br>
          This gradient now propagates backward to the hidden layer.
        </p>`;
      fbox.textContent =
        `dL/dz_out = ŷ − y\n           = ${prob.toFixed(4)} − ${lbl}\n           = ${dz.toFixed(4)}\n\ndL/dW_out = dL/dz · a_hidden^T\n\ndL/da_hidden = W_out^T · dL/dz\n              (propagated to hidden layer)`;
      fbox.style.display = "block";
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
        return `<tr>
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
        <p style="margin-top:8px;font-size:12px;color:#64748b">
          Neurons with z≤0 had ReLU output = 0.<br>
          Their gradient is <strong>blocked</strong> (dead neurons).
        </p>`;
      fbox.textContent =
        `da_hidden  = W_next^T · dz_next    (backprop)\ndz_hidden  = da · ReLU'(z)\nReLU'(z)   = 1 if z > 0, else 0\n\ndL/dW_hidden = dz · x^T\ndL/db_hidden = dz`;
      fbox.style.display = "block";
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
          <p style="margin-top:10px;font-size:13px;color:#86efac">
            ✓ Weights updated! Loss ${ud.loss_after < ud.loss_before ? "decreased" : "changed"} by ${Math.abs(pct)}%.<br>
            Run <strong>▶ Compute</strong> again to do the next training step.
          </p>`;
      } else {
        const lr = getLR();
        body.innerHTML = `
          <p style="font-size:13px;margin-bottom:10px">
            All gradients computed. Gradient descent will update every weight:
          </p>
          <div style="background:#0f172a;border-radius:8px;padding:12px;font-family:monospace;font-size:12px;color:#94a3b8">
W  ← W  − α · dL/dW\nb  ← b  − α · dL/db\nα  = ${lr}
          </div>
          <p style="margin-top:10px;font-size:12px;color:#64748b">
            • Connections lit in <span style="color:#3b82f6">blue</span>: weight increases (dw &lt; 0)<br>
            • Connections lit in <span style="color:#ef4444">red</span>: weight decreases (dw &gt; 0)
          </p>`;
      }
      fbox.style.display = "none";
      break;
    }
  }
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
  st.lossChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: st.lossHistory.map((_, i) => i + 1),
      datasets: [{
        label: "Loss per epoch",
        data: st.lossHistory,
        borderColor: "#6366f1",
        backgroundColor: "rgba(99,102,241,.08)",
        borderWidth: 2,
        pointRadius: st.lossHistory.length < 30 ? 3 : 0,
        fill: true, tension: 0.3,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 11 } } },
        tooltip: { callbacks: { label: c => `Loss: ${c.raw.toFixed(4)}` } },
      },
      scales: {
        x: { ticks: { color: "#475569", maxTicksLimit: 8 }, grid: { color: "#1e293b" } },
        y: { ticks: { color: "#475569" }, grid: { color: "#1e293b" }, beginAtZero: false },
      },
    },
  });
}

function updateEpoch(n) {
  document.getElementById("epoch-badge").textContent = `Epoch ${n}`;
}
