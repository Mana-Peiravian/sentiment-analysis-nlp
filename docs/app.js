let MODEL = null;

const $ = (sel) => document.querySelector(sel);
const textEl = $("#text");
const statusEl = $("#status");
const resultEl = $("#result");
const predEl = $("#pred");
const probsEl = $("#probs");
const btnAnalyze = $("#analyze");
const btnClear = $("#clear");

// Simple tokenizer that mirrors scikit's default token_pattern: (?u)\b\w\w+\b
function tokenize(s) {
  return (s || "")
    .toLowerCase()
    .normalize("NFKC")
    .match(/\b\w\w+\b/gu) || [];
}

// Build a term -> count map
function termCounts(tokens) {
  const m = Object.create(null);
  for (const t of tokens) m[t] = (m[t] || 0) + 1;
  return m;
}

// Compute TF-IDF with scikit defaults: raw tf * idf then L2 normalize
function vectorize(text, vocab, idf) {
  const tokens = tokenize(text);
  const counts = termCounts(tokens);

  const n = vocab.length;
  const x = new Float32Array(n);

  // Build a quick lookup: token -> index
  const indexOf = Object.create(null);
  for (let i = 0; i < n; i++) indexOf[vocab[i]] = i;

  // raw tf * idf
  for (const t in counts) {
    const j = indexOf[t];
    if (j !== undefined) x[j] = counts[t] * idf[j];
  }

  // L2 normalize
  let norm2 = 0;
  for (let j = 0; j < n; j++) norm2 += x[j] * x[j];
  norm2 = Math.sqrt(norm2) || 1.0;
  for (let j = 0; j < n; j++) x[j] /= norm2;

  return x;
}

// logits = W x + b, softmax -> probs
function softmax(logits) {
  const m = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function loadModel() {
  try {
    statusEl.textContent = "Loading model…";
    const resp = await fetch("./model/model.json", { cache: "no-cache" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    MODEL = await resp.json();
    statusEl.textContent = "Model ready ✅";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Failed to load model ❌";
  }
}

function infer() {
  if (!MODEL) return;
  const text = textEl.value.trim();
  if (!text) return;

  const x = vectorize(text, MODEL.vocabulary, MODEL.idf);

  const C = MODEL.classes.length;
  const F = MODEL.meta.n_features;
  const logits = new Array(C).fill(0);

  // logits[c] = dot(W[c], x) + b[c]
  for (let c = 0; c < C; c++) {
    let sum = 0;
    const row = MODEL.coef[c];
    for (let j = 0; j < F; j++) sum += row[j] * x[j];
    logits[c] = sum + MODEL.intercept[c];
  }

  const probs = softmax(logits);
  const topIdx = probs
    .map((p, i) => [p, i])
    .sort((a, b) => b[0] - a[0])[0][1];

  // UI
  predEl.textContent = `${MODEL.classes[topIdx]} (${(probs[topIdx] * 100).toFixed(1)}%)`;
  probsEl.innerHTML = "";
  for (let i = 0; i < C; i++) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div style="width:120px">${MODEL.classes[i]}</div>
      <div class="bar" style="flex:1"><div class="fill" style="width:${(probs[i]*100).toFixed(1)}%"></div></div>
      <div style="width:64px;text-align:right">${(probs[i]*100).toFixed(1)}%</div>
    `;
    probsEl.appendChild(row);
  }
  resultEl.classList.remove("hidden");
}

btnAnalyze.addEventListener("click", infer);
btnClear.addEventListener("click", () => {
  textEl.value = "";
  resultEl.classList.add("hidden");
  textEl.focus();
});
textEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) infer();
});

loadModel();
