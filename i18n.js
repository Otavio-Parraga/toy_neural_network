/* ════════════════════════════════════════════════════════
   Internationalization (English / Brazilian Portuguese)
   - I18N[lang][key] holds every user-visible string; "{name}" is interpolated by t().
   - Static HTML: data-i18n="key" (text), data-i18n-html="key" (markup),
     data-i18n-attr="attr:key;attr2:key2", optional data-i18n-params='{"n":1}'.
   - Dynamic strings in app.js call t(key, params) at render time.
   Numbers keep "." as the decimal separator so they match the formulas.
   Loaded before app.js.
   ════════════════════════════════════════════════════════ */
const I18N = {
  en: {
    "doc.title": "MLP Training — Forward & Backward Pass",

    // Header
    "header.eyebrow": "Neural networks, one step at a time",
    "header.title": "MLP Training",
    "header.titleDim": "Forward & Backward Pass",
    "header.subtitle": "Step through one complete training iteration, layer by layer.",
    "epoch": "Epoch {n}",
    "theme.toDark": "Switch to dark theme",
    "theme.toLight": "Switch to light theme",
    "lang.label": "Language: English. Mudar para português",

    // Controls
    "ctrl.aria": "Network controls",
    "ctrl.hiddenLayers": "Hidden Layers",
    "ctrl.addLayer": "+ Layer",
    "ctrl.rmLayer": "− Layer",
    "ctrl.reset": "↺ Reset weights",
    "ctrl.lr": "Learning Rate",
    "ctrl.sample": "Sample",
    "ctrl.sampleNote": "(first of batch)",
    "ctrl.batchSize": "Batch size (B)",
    "ctrl.fullBatch": "Full batch",
    "ctrl.compute": "▶ Compute Forward + Backward",

    // Hints / status
    "hint.start": "Select a sample and click to begin",
    "hint.computing": "Computing…",
    "hint.batchDone": "Batch of {B} computed (starting at sample {s}). Step through the passes.",
    "hint.sampleDone": "Sample {s} computed. Step through the passes.",
    "hint.error": "Error: {msg}",
    "hint.updateFailed": "Update failed: {msg}",
    "hint.updated": "Weights updated! Run ▶ Compute again to see the next iteration.",
    "err.server": "⚠ Cannot reach server.<br>Start it on the server with <code>./start.sh --detach</code> and open the trycloudflare.com URL it prints.",
    "err.runComputeFirst": "Run compute first",
    "err.nonFinite": "Update would make weights non-finite; try a smaller learning rate",

    // Architecture row
    "layer.input": "Input",
    "layer.hidden": "Hidden {n}",
    "layer.output": "Output",

    // Step navigator
    "nav.aria": "Training step navigator",
    "nav.prev": "← Prev",
    "nav.next": "Next →",
    "nav.counter": "Step {n} / {total}",
    "phase.forward": "FORWARD",
    "phase.loss": "LOSS",
    "phase.backward": "BACKWARD",
    "phase.update": "UPDATE",
    "step.input": "Input Layer",
    "step.fwdHidden": "Hidden Layer {l}",
    "step.fwdOutput": "Output Layer",
    "step.loss": "Loss Computation",
    "step.bwdOutput": "Output Gradients",
    "step.bwdHidden": "Hidden Layer {l} Gradients",
    "step.update": "Weight Update",

    // Network card
    "net.hintEdge": "Hover a connection to see its weight",
    "net.hintNode": "Click a neuron to see its calculation",
    "net.showWeightsTitle": "Label connections with their weights (in larger layers, only those of the selected neuron)",
    "net.showWeights": "Show weights",
    "net.placeholder": "Click <strong>▶ Compute</strong> to start the training step visualization",
    "net.more": "+ {n} more",
    "net.neuronAria": "{name}: show its calculation",

    // Features and classes
    "feat.0": "Weight (kg)", "feat.1": "Ear Pointiness", "feat.2": "Meow/Bark Ratio", "feat.3": "Agility Score",
    "featShort.0": "Weight (kg)", "featShort.1": "Ear Point.", "featShort.2": "Meow/Bark", "featShort.3": "Agility",
    "cls.0": "Cat 🐱", "cls.1": "Dog 🐶",
    "clsPlain.0": "Cat", "clsPlain.1": "Dog",

    // Neuron names
    "neuron.input": "Input · x{i}",
    "neuron.output": "Output · ŷ",

    // Tooltips
    "tip.source": "(source)",
    "tip.target": "(target)",
    "tip.contrib": "This connection's contribution to z{j} of {name}.",
    "tip.srcPending": "Source activation not computed yet — step forward to see w · a.",
    "tip.batchAvg": "batch avg ∂L/∂w",
    "tip.focusNote": "δ · a is for the focus sample; the update uses the mean over B = {B}.",
    "tip.gradPending": "Gradient not computed yet — backprop reaches this layer in a later step.",
    "tip.oldW": "old w",
    "tip.newW": "new w",
    "tip.avg": " (avg)",
    "tip.applied": "Applied with α = {lr}.",
    "tip.preview": "Preview with α = {lr}.",
    "tip.weightSub": "Weight · {from} → {to} layer",
    "tip.layerSub": "{layer} layer",
    "tip.sigmoid": "sigmoid",
    "tip.feature": "feature",
    "tip.rawX": "raw x",
    "tip.bias": "bias b{j}",
    "tip.trueY": "true y",
    "tip.zaPending": "z and a are computed in a later forward step.",
    "tip.biasOldNew": "bias old → new",

    // Formula box
    "fx.selected": "selected",
    "fx.example": "example · click a neuron",
    "fx.meanStd": "μ, σ: mean and std of \"{feat}\"",
    "fx.overN": "over all {n} samples.",
    "fx.overAll": "over all the samples.",
    "fx.fedTo": "x̂{i} is fed to every neuron of {layer}.",
    "fx.termPer": "(one w · {src} term per neuron of {layer})",
    "fx.biasLbl": "+ b{j} (bias)",
    "fx.inactive": "     z ≤ 0 → this neuron outputs 0 (inactive)",
    "fx.onePerIn": "(one per incoming weight)",
    "fx.batchAvgUsed": "Batch average over B = {B} (used by the update):",
    "fx.batchAvg": "Batch average over B = {B}:",
    "fx.bceNote": "(sigmoid + BCE: the derivative simplifies to ŷ − y)",
    "fx.sumOver": "Σ w · δ over {layer} ({n} term)",
    "fx.sumOverPl": "Σ w · δ over {layer} ({n} terms)",
    "fx.blocked": "z ≤ 0 → gradient blocked (ReLU' = 0)",
    "fx.inputsNoDelta": "Inputs are data: no δ to backpropagate.",
    "fx.weightsLeaving": "Weights leaving x̂{i} (into {layer}):",
    "fx.onePerFed": "(one per neuron it feeds)",
    "fx.gradBatch": "∂L/∂w = batch average over B = {B}",
    "fx.gradFromBwd": "∂L/∂w from the backward pass",
    "fx.leavingInput": "(weights leaving this input)",
    "fx.incomingBias": "(incoming weights and bias)",
    "fx.applied": "✓ Applied: these are the new values.",
    "fx.preview": "Preview: click “Apply Weight Update” to commit.",

    // Info panel
    "info.initial": "Step through the forward and backward pass to see the math at each layer.",
    "info.reset": "Architecture reset. Click ▶ Compute to run a training step.",
    "info.inputTitle": "Input — Sample #{n}  (True: {cls})",
    "th.feature": "Feature",
    "th.raw": "Raw value",
    "th.norm": "Normalized",
    "th.neuron": "Neuron",
    "th.preAct": "z (pre-act.)",
    "th.value": "Value",
    "th.quantity": "Quantity",
    "th.meaning": "Meaning",
    "th.before": "Before",
    "th.after": "After",
    "th.received": "da (received)",
    "info.inputNote": "Features are <strong>standardized</strong>: x̂ = (x − μ) / σ<br>These 4 values enter the network as input activations.",
    "info.fwdHiddenTitle": "Forward — Hidden Layer {l}  (ReLU)",
    "info.dead": "⚡ dead",
    "info.more": "… {n} more",
    "info.fwdOutputTitle": "Forward — Output Layer  (Sigmoid)",
    "info.zPre": "z (pre-activation)",
    "info.pDog": "ŷ = σ(z)  = P(Dog)",
    "info.prediction": "Prediction",
    "info.trueLabel": "True label",
    "info.correct": "Correct?",
    "info.yes": "✓ Yes",
    "info.no": "✗ No",
    "info.lossTitle": "Loss — Binary Cross-Entropy",
    "info.trueY": "True label  y",
    "info.predY": "Prediction  ŷ",
    "info.lossL": "Loss  L",
    "info.lossNote": "High loss → the network is wrong (or uncertain).<br>The backward pass will compute how to reduce this loss.",
    "info.bwdOutputTitle": "Backward — Output Layer Gradients",
    "info.bceIntro": "For <strong>BCE + Sigmoid</strong>, the gradient simplifies elegantly:",
    "info.matrix": "matrix ({r}×{c})",
    "info.tooHigh": "dz > 0: output was too high (reduce it)",
    "info.tooLow": "dz < 0: output was too low (increase it)",
    "info.propagates": "This gradient now propagates backward to the hidden layer.",
    "info.bwdHiddenTitle": "Backward — Hidden Layer {l} Gradients",
    "info.blocked": "✗ blocked",
    "info.pass": "✓ pass",
    "info.deadNote": "Neurons with z≤0 had ReLU output = 0.<br>Their gradient is <strong>blocked</strong> (dead neurons).",
    "info.updateTitle": "Weight Update — Gradient Descent",
    "info.loss": "Loss",
    "info.doneDecreased": "✓ Weights updated! Loss decreased by {pct}%.",
    "info.doneChanged": "✓ Weights updated! Loss changed by {pct}%.",
    "info.runAgain": "Run <strong>▶ Compute</strong> again to do the next training step.",
    "info.allGrads": "All gradients computed. Gradient descent will update every weight:",
    "info.legendInc": "• Connections lit in <span style=\"color:var(--w-inc)\">blue</span>: weight increases (dw &lt; 0)",
    "info.legendDec": "• Connections lit in <span style=\"color:var(--w-dec)\">rose</span>: weight decreases (dw &gt; 0)",
    "update.apply": "Apply Weight Update",

    // Loss chart
    "loss.card": "Loss (batch mean) over updates",
    "loss.empty": "No updates yet — apply a weight update to see the loss curve",
    "loss.dataset": "Mean batch loss per update",
    "loss.tooltip": "Loss: {v}",
    "loss.nowBatch": "Mean batch loss",
    "loss.nowSingle": "Loss",

    // Legend
    "legend.title": "Legend",
    "legend.fwd": "Forward signal / activation",
    "legend.bwd": "Backward gradient signal",
    "legend.winc": "Weight update (Δw > 0)",
    "legend.wdec": "Weight update (Δw < 0)",
    "legend.hl": "Current layer (highlighted)",
    "legend.cat": "Cat class",
    "legend.dog": "Dog class",

    // Mini-batch panel
    "batch.avgNote": "Gradients are <strong>batch-averaged</strong>: ∂L/∂W = (1/B) Σᵢ ∂Lᵢ/∂W, with B = {B}.",
    "batch.head": "Mini-batch · B = {B}",
    "batch.focus": "focus: <strong>#{n}</strong> ({k}/{B}) · mean loss",
    "batch.bwdNote": "The values above are for the focus sample only (per-sample ∂Lᵢ). {avg} The update applies the average, not the focus sample's gradient.",
    "batch.updNote": "{avg} Edge colors show Δw = −α · (1/B) Σᵢ ∂Lᵢ/∂W.",
    "batch.updAfter": "Loss before/after above is the mean over the batch.",
    "batch.lossNote": "Batch loss L = (1/B) Σᵢ Lᵢ = <strong>{v}</strong>. The panel above shows Lᵢ for the focus sample.",
    "batch.thSample": "Sample",
    "batch.thTrue": "True",
    "batch.thBefore": "Lᵢ before",
    "batch.thAfter": "Lᵢ after",
    "batch.mean": "Mean (1/B) Σ Lᵢ",
    "batch.chipTitle": "Sample #{n} · {cls} · ŷ={p} · L={l}",
    "batch.hint": "Click a sample to show it in the network.",
  },

  pt: {
    "doc.title": "Treinamento de MLP — Passagens Forward e Backward",

    "header.eyebrow": "Redes neurais, um passo de cada vez",
    "header.title": "Treinamento de MLP",
    "header.titleDim": "Passagens Forward e Backward",
    "header.subtitle": "Percorra uma iteração completa de treinamento, camada por camada.",
    "epoch": "Época {n}",
    "theme.toDark": "Mudar para o tema escuro",
    "theme.toLight": "Mudar para o tema claro",
    "lang.label": "Idioma: português. Switch to English",

    "ctrl.aria": "Controles da rede",
    "ctrl.hiddenLayers": "Camadas ocultas",
    "ctrl.addLayer": "+ Camada",
    "ctrl.rmLayer": "− Camada",
    "ctrl.reset": "↺ Reiniciar pesos",
    "ctrl.lr": "Taxa de aprendizado",
    "ctrl.sample": "Amostra",
    "ctrl.sampleNote": "(primeira do lote)",
    "ctrl.batchSize": "Tamanho do lote (B)",
    "ctrl.fullBatch": "Lote completo",
    "ctrl.compute": "▶ Calcular Forward + Backward",

    "hint.start": "Escolha uma amostra e clique para começar",
    "hint.computing": "Calculando…",
    "hint.batchDone": "Lote de {B} amostras calculado (a partir da amostra {s}). Avance pelas passagens.",
    "hint.sampleDone": "Amostra {s} calculada. Avance pelas passagens.",
    "hint.error": "Erro: {msg}",
    "hint.updateFailed": "Falha na atualização: {msg}",
    "hint.updated": "Pesos atualizados! Clique em ▶ Calcular novamente para ver a próxima iteração.",
    "err.server": "⚠ Não foi possível conectar ao servidor.<br>Inicie-o no servidor com <code>./start.sh --detach</code> e abra a URL trycloudflare.com que ele exibe.",
    "err.runComputeFirst": "Execute o cálculo primeiro",
    "err.nonFinite": "A atualização tornaria os pesos não finitos; tente uma taxa de aprendizado menor",

    "layer.input": "Entrada",
    "layer.hidden": "Oculta {n}",
    "layer.output": "Saída",

    "nav.aria": "Navegação pelas etapas do treinamento",
    "nav.prev": "← Anterior",
    "nav.next": "Próxima →",
    "nav.counter": "Etapa {n} / {total}",
    "phase.forward": "FORWARD",
    "phase.loss": "PERDA",
    "phase.backward": "BACKWARD",
    "phase.update": "ATUALIZAÇÃO",
    "step.input": "Camada de entrada",
    "step.fwdHidden": "Camada oculta {l}",
    "step.fwdOutput": "Camada de saída",
    "step.loss": "Cálculo da perda",
    "step.bwdOutput": "Gradientes da saída",
    "step.bwdHidden": "Gradientes da camada oculta {l}",
    "step.update": "Atualização dos pesos",

    "net.hintEdge": "Passe o mouse sobre uma conexão para ver seu peso",
    "net.hintNode": "Clique em um neurônio para ver seu cálculo",
    "net.showWeightsTitle": "Rotular as conexões com seus pesos (em camadas maiores, apenas as do neurônio selecionado)",
    "net.showWeights": "Mostrar pesos",
    "net.placeholder": "Clique em <strong>▶ Calcular</strong> para iniciar a visualização da etapa de treinamento",
    "net.more": "+ {n} a mais",
    "net.neuronAria": "{name}: mostrar seu cálculo",

    "feat.0": "Peso (kg)", "feat.1": "Orelhas Pontudas", "feat.2": "Razão Miado/Latido", "feat.3": "Índice de Agilidade",
    "featShort.0": "Peso (kg)", "featShort.1": "Orelhas", "featShort.2": "Miado/Latido", "featShort.3": "Agilidade",
    "cls.0": "Gato 🐱", "cls.1": "Cachorro 🐶",
    "clsPlain.0": "Gato", "clsPlain.1": "Cachorro",

    "neuron.input": "Entrada · x{i}",
    "neuron.output": "Saída · ŷ",

    "tip.source": "(origem)",
    "tip.target": "(destino)",
    "tip.contrib": "Contribuição desta conexão para z{j} de {name}.",
    "tip.srcPending": "Ativação de origem ainda não calculada — avance para ver w · a.",
    "tip.batchAvg": "média do lote ∂L/∂w",
    "tip.focusNote": "δ · a é da amostra em foco; a atualização usa a média sobre B = {B}.",
    "tip.gradPending": "Gradiente ainda não calculado — a retropropagação chega a esta camada em uma etapa posterior.",
    "tip.oldW": "w antigo",
    "tip.newW": "w novo",
    "tip.avg": " (média)",
    "tip.applied": "Aplicado com α = {lr}.",
    "tip.preview": "Prévia com α = {lr}.",
    "tip.weightSub": "Peso · camada {from} → {to}",
    "tip.layerSub": "Camada {layer}",
    "tip.sigmoid": "sigmoide",
    "tip.feature": "atributo",
    "tip.rawX": "x bruto",
    "tip.bias": "viés b{j}",
    "tip.trueY": "y real",
    "tip.zaPending": "z e a são calculados em uma etapa forward posterior.",
    "tip.biasOldNew": "viés antigo → novo",

    "fx.selected": "selecionado",
    "fx.example": "exemplo · clique em um neurônio",
    "fx.meanStd": "μ, σ: média e desvio de \"{feat}\"",
    "fx.overN": "sobre todas as {n} amostras.",
    "fx.overAll": "sobre todas as amostras.",
    "fx.fedTo": "x̂{i} alimenta cada neurônio de {layer}.",
    "fx.termPer": "(um termo w · {src} por neurônio de {layer})",
    "fx.biasLbl": "+ b{j} (viés)",
    "fx.inactive": "     z ≤ 0 → este neurônio produz 0 (inativo)",
    "fx.onePerIn": "(um por peso de entrada)",
    "fx.batchAvgUsed": "Média no lote (B = {B}), usada na atualização:",
    "fx.batchAvg": "Média do lote sobre B = {B}:",
    "fx.bceNote": "(sigmoide + BCE: a derivada vira ŷ − y)",
    "fx.sumOver": "Σ w · δ sobre {layer} ({n} termo)",
    "fx.sumOverPl": "Σ w · δ sobre {layer} ({n} termos)",
    "fx.blocked": "z ≤ 0 → gradiente bloqueado (ReLU' = 0)",
    "fx.inputsNoDelta": "Entradas são dados: não há δ a propagar.",
    "fx.weightsLeaving": "Pesos que saem de x̂{i} (para {layer}):",
    "fx.onePerFed": "(um por neurônio destino)",
    "fx.gradBatch": "∂L/∂w = média do lote sobre B = {B}",
    "fx.gradFromBwd": "∂L/∂w vindo da passagem backward",
    "fx.leavingInput": "(pesos que saem desta entrada)",
    "fx.incomingBias": "(pesos de entrada e viés)",
    "fx.applied": "✓ Aplicado: estes são os novos valores.",
    "fx.preview": "Prévia: confirme em “Aplicar atualização”.",

    "info.initial": "Avance pelas passagens forward e backward para ver a matemática de cada camada.",
    "info.reset": "Arquitetura reiniciada. Clique em ▶ Calcular para executar uma etapa de treinamento.",
    "info.inputTitle": "Entrada — Amostra #{n}  (Real: {cls})",
    "th.feature": "Atributo",
    "th.raw": "Valor bruto",
    "th.norm": "Normalizado",
    "th.neuron": "Neurônio",
    "th.preAct": "z (pré-ativ.)",
    "th.value": "Valor",
    "th.quantity": "Grandeza",
    "th.meaning": "Significado",
    "th.before": "Antes",
    "th.after": "Depois",
    "th.received": "da (recebido)",
    "info.inputNote": "Os atributos são <strong>padronizados</strong>: x̂ = (x − μ) / σ<br>Esses 4 valores entram na rede como ativações de entrada.",
    "info.fwdHiddenTitle": "Forward — Camada oculta {l}  (ReLU)",
    "info.dead": "⚡ morto",
    "info.more": "… mais {n}",
    "info.fwdOutputTitle": "Forward — Camada de saída  (Sigmoide)",
    "info.zPre": "z (pré-ativação)",
    "info.pDog": "ŷ = σ(z)  = P(Cachorro)",
    "info.prediction": "Predição",
    "info.trueLabel": "Rótulo real",
    "info.correct": "Correto?",
    "info.yes": "✓ Sim",
    "info.no": "✗ Não",
    "info.lossTitle": "Perda — Entropia Cruzada Binária",
    "info.trueY": "Rótulo real  y",
    "info.predY": "Predição  ŷ",
    "info.lossL": "Perda  L",
    "info.lossNote": "Perda alta → a rede está errada (ou incerta).<br>A passagem backward vai calcular como reduzir essa perda.",
    "info.bwdOutputTitle": "Backward — Gradientes da camada de saída",
    "info.bceIntro": "Com <strong>BCE + Sigmoide</strong>, o gradiente se simplifica de forma elegante:",
    "info.matrix": "matriz ({r}×{c})",
    "info.tooHigh": "dz > 0: a saída ficou alta demais (reduzi-la)",
    "info.tooLow": "dz < 0: a saída ficou baixa demais (aumentá-la)",
    "info.propagates": "Esse gradiente agora se propaga para trás até a camada oculta.",
    "info.bwdHiddenTitle": "Backward — Gradientes da camada oculta {l}",
    "info.blocked": "✗ bloqueado",
    "info.pass": "✓ passa",
    "info.deadNote": "Neurônios com z≤0 tiveram saída ReLU = 0.<br>O gradiente deles é <strong>bloqueado</strong> (neurônios mortos).",
    "info.updateTitle": "Atualização dos pesos — Gradiente Descendente",
    "info.loss": "Perda",
    "info.doneDecreased": "✓ Pesos atualizados! A perda diminuiu {pct}%.",
    "info.doneChanged": "✓ Pesos atualizados! A perda mudou {pct}%.",
    "info.runAgain": "Clique em <strong>▶ Calcular</strong> novamente para fazer a próxima etapa de treinamento.",
    "info.allGrads": "Todos os gradientes foram calculados. O gradiente descendente vai atualizar cada peso:",
    "info.legendInc": "• Conexões em <span style=\"color:var(--w-inc)\">azul</span>: o peso aumenta (dw &lt; 0)",
    "info.legendDec": "• Conexões em <span style=\"color:var(--w-dec)\">rosa</span>: o peso diminui (dw &gt; 0)",
    "update.apply": "Aplicar atualização",

    "loss.card": "Perda (média do lote) por atualização",
    "loss.empty": "Nenhuma atualização ainda — aplique uma atualização dos pesos para ver a curva de perda",
    "loss.dataset": "Perda média do lote por atualização",
    "loss.tooltip": "Perda: {v}",
    "loss.nowBatch": "Perda média do lote",
    "loss.nowSingle": "Perda",

    "legend.title": "Legenda",
    "legend.fwd": "Sinal forward / ativação",
    "legend.bwd": "Sinal do gradiente (backward)",
    "legend.winc": "Atualização de peso (Δw > 0)",
    "legend.wdec": "Atualização de peso (Δw < 0)",
    "legend.hl": "Camada atual (destacada)",
    "legend.cat": "Classe Gato",
    "legend.dog": "Classe Cachorro",

    "batch.avgNote": "Os gradientes são <strong>médias do lote</strong>: ∂L/∂W = (1/B) Σᵢ ∂Lᵢ/∂W, com B = {B}.",
    "batch.head": "Minilote · B = {B}",
    "batch.focus": "foco: <strong>#{n}</strong> ({k}/{B}) · perda média",
    "batch.bwdNote": "Os valores acima são apenas da amostra em foco (∂Lᵢ por amostra). {avg} A atualização aplica a média, não o gradiente da amostra em foco.",
    "batch.updNote": "{avg} As cores das conexões mostram Δw = −α · (1/B) Σᵢ ∂Lᵢ/∂W.",
    "batch.updAfter": "A perda antes/depois acima é a média sobre o lote.",
    "batch.lossNote": "Perda do lote L = (1/B) Σᵢ Lᵢ = <strong>{v}</strong>. O painel acima mostra Lᵢ da amostra em foco.",
    "batch.thSample": "Amostra",
    "batch.thTrue": "Real",
    "batch.thBefore": "Lᵢ antes",
    "batch.thAfter": "Lᵢ depois",
    "batch.mean": "Média (1/B) Σ Lᵢ",
    "batch.chipTitle": "Amostra #{n} · {cls} · ŷ={p} · L={l}",
    "batch.hint": "Clique em uma amostra para exibi-la na rede.",
  },
};

const LANG_KEY = "mlp-lang";

// Saved choice, else the browser language (pt* → pt), else English.
function detectLang() {
  let s = null;
  try { s = localStorage.getItem(LANG_KEY); } catch (e) {}
  if (s === "en" || s === "pt") return s;
  const nav = (navigator.languages && navigator.languages[0]) || navigator.language || "";
  return /^pt\b/i.test(nav) ? "pt" : "en";
}

let LANG = detectLang();

// Translate `key`, interpolating {name} placeholders from `params`. Falls back to English, then the key.
function t(key, params) {
  const dict = I18N[LANG] || I18N.en;
  let s = key in dict ? dict[key] : (key in I18N.en ? I18N.en[key] : key);
  if (params) s = s.replace(/\{(\w+)\}/g, (m, k) => (k in params ? String(params[k]) : m));
  return s;
}

// Localized feature / class names by index (the backend always sends English names).
const featName  = i => t(`feat.${i}`);
const featShort = i => t(`featShort.${i}`);
const clsName   = i => t(`cls.${i}`);
const clsPlain  = i => t(`clsPlain.${i}`);

// Known backend error messages → translation keys (the API stays English).
function trError(msg) {
  const m = String(msg);
  if (m === "Run compute first") return t("err.runComputeFirst");
  if (m.startsWith("Update would make weights non-finite")) return t("err.nonFinite");
  return m;
}

function i18nParams(el) {
  try { return el.dataset.i18nParams ? JSON.parse(el.dataset.i18nParams) : null; } catch (e) { return null; }
}

// Apply translations to static markup (and to dynamic elements tagged via setI18n).
function applyI18n(root = document) {
  document.documentElement.lang = LANG === "pt" ? "pt-BR" : "en";
  document.title = t("doc.title");
  root.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n, i18nParams(el)); });
  root.querySelectorAll("[data-i18n-html]").forEach(el => { el.innerHTML = t(el.dataset.i18nHtml, i18nParams(el)); });
  root.querySelectorAll("[data-i18n-attr]").forEach(el => {
    el.dataset.i18nAttr.split(";").forEach(pair => {
      const [attr, key] = pair.split(":").map(s => s && s.trim());
      if (attr && key) el.setAttribute(attr, t(key, i18nParams(el)));
    });
  });
  const btn = document.getElementById("lang-toggle");
  if (btn) btn.dataset.active = LANG;
}

// Set a translated message on an element and remember the key, so a language
// switch re-renders it in place. `html` = the string contains markup.
function setI18n(el, key, params = null, html = false) {
  if (typeof el === "string") el = document.getElementById(el);
  if (!el) return;
  el.removeAttribute(html ? "data-i18n" : "data-i18n-html");
  el.setAttribute(html ? "data-i18n-html" : "data-i18n", key);
  if (params) el.dataset.i18nParams = JSON.stringify(params);
  else el.removeAttribute("data-i18n-params");
  if (html) el.innerHTML = t(key, params); else el.textContent = t(key, params);
}

function setLang(lang) {
  if (lang !== "en" && lang !== "pt") return;
  LANG = lang;
  try { localStorage.setItem(LANG_KEY, lang); } catch (e) {}
  applyI18n();
  if (typeof refreshLangViews === "function") refreshLangViews();   // app.js: JS-rendered views
}

// Static text is translated as soon as this script runs (it is loaded at the end of <body>).
applyI18n();

window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("lang-toggle");
  if (btn) btn.addEventListener("click", () => setLang(LANG === "pt" ? "en" : "pt"));
});
