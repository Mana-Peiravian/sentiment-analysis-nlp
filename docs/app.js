// ─────────────────────────────────────────────────────────────────────────────
// Config
// If you use a CDN, set MODEL_URL to that absolute URL. Otherwise keep relative.
const MODEL_URL = "./model/model.json";

// ─────────────────────────────────────────────────────────────────────────────
// DOM helpers
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const statusChip = $("#model-status");
const textEl     = $("#text");
const analyzeBtn = $("#analyze");
const clearBtn   = $("#clear");
const resultCard = $("#result-card");
const predPill   = $("#pred-pill");
const confEl     = $("#confidence");
const probsEl    = $("#probs");
const cuesEl     = $("#cues");
const metaVocab  = $("#meta-vocab");
const metaClasses= $("#meta-classes");
const toastEl    = $("#toast");

let MODEL = null;

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
function setStatus(state, msg){
  const dot = statusChip.querySelector(".dot");
  const txt = statusChip.querySelector(".txt");
  dot.className = "dot " + (state==="ok" ? "dot--ok" : state==="err" ? "dot--err" : "dot--loading");
  txt.textContent = msg;
}

function toast(msg, ms=1800){
  toastEl.textContent = msg;
  toastEl.hidden = false;
  setTimeout(()=> toastEl.hidden = true, ms);
}

// Tokenizer mirroring scikit's default: (?u)\b\w\w+\b
function tokenize(s){
  return (s || "").toLowerCase().normalize("NFKC").match(/\b\w\w+\b/gu) || [];
}
function termCounts(tokens){
  const m = Object.create(null);
  for (const t of tokens) m[t] = (m[t] || 0) + 1;
  return m;
}
// TF-IDF with L2 norm
function vectorize(text, vocab, idf){
  const tokens = tokenize(text);
  const counts = termCounts(tokens);
  const n = vocab.length, x = new Float32Array(n);

  // token -> index lookup
  const indexOf = Object.create(null);
  for (let i=0;i<n;i++) indexOf[vocab[i]] = i;

  for (const t in counts){
    const j = indexOf[t];
    if (j !== undefined) x[j] = counts[t] * idf[j];
  }

  // L2
  let norm2 = 0;
  for (let j=0;j<n;j++) norm2 += x[j]*x[j];
  norm2 = Math.sqrt(norm2) || 1;
  for (let j=0;j<n;j++) x[j] /= norm2;
  return {x, tokens};
}
// softmax
function softmax(arr){
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const s = exps.reduce((a,b)=>a+b,0);
  return exps.map(v => v/s);
}

// Explain: top token contributions for the predicted class
function topCues(x, tokens, vocab, coefRow, k=6){
  // Build index lookup
  const idx = Object.create(null);
  for (let i=0;i<vocab.length;i++) idx[vocab[i]] = i;

  // score = x[j] * w[j]
  const scores = [];
  const unique = new Set(tokens); // avoid repeating same token
  for (const t of unique){
    const j = idx[t];
    if (j === undefined) continue;
    const s = x[j] * coefRow[j];
    if (s !== 0) scores.push([t, s]);
  }
  scores.sort((a,b)=>Math.abs(b[1]) - Math.abs(a[1]));
  return scores.slice(0, k);
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
function renderProbs(classes, probs){
  probsEl.innerHTML = "";
  classes.forEach((name, i) => {
    const row = document.createElement("div");
    row.className = "row";
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = name;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.className = "fill";
    bar.appendChild(fill);
    const pct = document.createElement("div");
    pct.className = "percent";
    const p = probs[i] * 100;
    pct.textContent = `${p.toFixed(1)}%`;
    row.append(label, bar, pct);
    probsEl.appendChild(row);

    // animate width
    requestAnimationFrame(()=> fill.style.width = `${p.toFixed(1)}%`);
  });
}

function renderCues(cues){
  cuesEl.innerHTML = "";
  if (cues.length === 0){
    const none = document.createElement("div");
    none.className = "subtle";
    none.textContent = "No strong token cues detected (likely OOV or very short text).";
    cuesEl.appendChild(none);
    return;
  }
  for (const [t, s] of cues){
    const chip = document.createElement("div");
    chip.className = "cue";
    chip.innerHTML = `${t} <small>${s>=0?"+":""}${s.toFixed(3)}</small>`;
    cuesEl.appendChild(chip);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference
function infer(){
  if (!MODEL) return;
  const raw = textEl.value.trim();
  if (!raw){
    toast("Type something first");
    return;
  }

  const {x, tokens} = vectorize(raw, MODEL.vocabulary, MODEL.idf);

  const C = MODEL.classes.length;
  const F = MODEL.meta.n_features;
  const logits = new Array(C).fill(0);

  for (let c=0;c<C;c++){
    const row = MODEL.coef[c];
    let sum = 0;
    for (let j=0;j<F;j++) sum += row[j] * x[j];
    logits[c] = sum + MODEL.intercept[c];
  }
  const probs = softmax(logits);
  const topIdx = probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0])[0][1];

  // UI
  const topP = probs[topIdx]*100;
  predPill.textContent = `Prediction: ${MODEL.classes[topIdx]}`;
  confEl.textContent = `Confidence: ${topP.toFixed(1)}%`;
  predPill.style.borderColor = topP >= 66 ? "#1d3b23" : topP >= 40 ? "#3b2f1d" : "#3b1d1d";
  predPill.style.background = topP >= 66 ? "rgba(16,48,28,.7)" : topP >= 40 ? "rgba(48,36,16,.7)" : "rgba(48,16,16,.7)";
  renderProbs(MODEL.classes, probs);

  const cues = topCues(x, tokens, MODEL.vocabulary, MODEL.coef[topIdx], 6);
  renderCues(cues);

  resultCard.hidden = false;
  localStorage.setItem("sentiment:last", raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
async function loadModel(){
  try{
    setStatus("loading","Loading model…");
    const resp = await fetch(MODEL_URL, {cache:"no-cache"});
    if(!resp.ok) throw new Error(`HTTP ${resp.status}`);
    MODEL = await resp.json();

    // metadata
    metaVocab.textContent  = MODEL.vocabulary.length.toLocaleString();
    metaClasses.textContent= MODEL.classes.join(" · ");

    setStatus("ok","Model ready");
    analyzeBtn.disabled = false;

    // restore last text
    const last = localStorage.getItem("sentiment:last");
    if (last) textEl.value = last;

  }catch(err){
    console.error(err);
    setStatus("err","Load failed");
    toast("Failed to load model.json");
  }
}

// Sample chips
$("#samples").addEventListener("click", (e)=>{
  const b = e.target.closest(".chip");
  if(!b) return;
  textEl.value = b.dataset.text || "";
  textEl.focus();
});

// Actions
analyzeBtn.addEventListener("click", infer);
clearBtn.addEventListener("click", ()=>{
  textEl.value = "";
  resultCard.hidden = true;
  textEl.focus();
});
textEl.addEventListener("keydown", (e)=>{
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) infer();
});

// Go
loadModel();
