// ========= CONFIG =========
const LR_MODEL_URL = "./model/model.json"; // keep relative for GitHub Pages
const BERT_MODEL_ID = "Xenova/twitter-roberta-base-sentiment-latest"; // 3-class: neg/neu/pos

// ========= DOM =========
const $  = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const statusChip  = $("#model-status");
const textEl      = $("#text");
const analyzeBtn  = $("#analyze");
const clearBtn    = $("#clear");
const resultCard  = $("#result-card");
const predPill    = $("#pred-pill");
const confEl      = $("#confidence");
const probsEl     = $("#probs");
const cuesEl      = $("#cues");
const metaMode    = $("#meta-mode");
const metaVocab   = $("#meta-vocab");
const metaClasses = $("#meta-classes");
const toastEl     = $("#toast");

// ========= STATE =========
let ACTIVE_MODE = "lr";
let LR = null;          // {classes, coef, intercept, vocabulary, idf, meta}
let BERT = null;        // transformers pipeline
let transformers = null;// module cache

// ========= UI helpers =========
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
function setMeta(mode, vocabInfo, classesInfo){
  metaMode.textContent = mode.toUpperCase();
  metaVocab.textContent = vocabInfo;
  metaClasses.textContent = classesInfo;
}

// ========= Shared rendering =========
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
    requestAnimationFrame(()=> fill.style.width = `${p.toFixed(1)}%`);
  });
}
function renderCues(cues){
  cuesEl.innerHTML = "";
  if (!cues || cues.length === 0){
    const none = document.createElement("div");
    none.className = "subtle";
    none.textContent = "No strong token cues detected.";
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
function updateHeader(predLabel, topP){
  predPill.textContent = `Prediction: ${predLabel}`;
  confEl.textContent = `Confidence: ${topP.toFixed(1)}%`;
  predPill.style.borderColor = topP >= 66 ? "#1d3b23" : topP >= 40 ? "#3b2f1d" : "#3b1d1d";
  predPill.style.background = topP >= 66 ? "rgba(16,48,28,.7)" : topP >= 40 ? "rgba(48,36,16,.7)" : "rgba(48,16,16,.7)";
  resultCard.hidden = false;
}

// ========= LR (TF-IDF + Logistic Regression) =========
function tokenize(s){ return (s||"").toLowerCase().normalize("NFKC").match(/\b\w\w+\b/gu) || []; }
function termCounts(tokens){ const m=Object.create(null); for(const t of tokens) m[t]=(m[t]||0)+1; return m; }
function vectorize(text, vocab, idf){
  const tokens = tokenize(text);
  const counts = termCounts(tokens);
  const n = vocab.length, x = new Float32Array(n);
  const indexOf = Object.create(null);
  for (let i=0;i<n;i++) indexOf[vocab[i]] = i;
  for (const t in counts){ const j = indexOf[t]; if (j !== undefined) x[j] = counts[t] * idf[j]; }
  let norm2 = 0; for (let j=0;j<n;j++) norm2 += x[j]*x[j]; norm2 = Math.sqrt(norm2)||1;
  for (let j=0;j<n;j++) x[j] /= norm2;
  return { x, tokens };
}
function softmax(arr){
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const s = exps.reduce((a,b)=>a+b,0);
  return exps.map(v => v/s);
}
function topCuesLR(x, tokens, vocab, coefRow, k=6){
  const idx = Object.create(null);
  for (let i=0;i<vocab.length;i++) idx[vocab[i]] = i;
  const scores = [];
  const unique = new Set(tokens);
  for (const t of unique){
    const j = idx[t];
    if (j === undefined) continue;
    const s = x[j]*coefRow[j];
    if (s !== 0) scores.push([t, s]);
  }
  scores.sort((a,b)=>Math.abs(b[1]) - Math.abs(a[1]));
  return scores.slice(0, k);
}
async function loadLR(){
  setStatus("loading","Loading LR…");
  const resp = await fetch(LR_MODEL_URL, { cache: "no-cache" });
  if (!resp.ok) throw new Error(`LR HTTP ${resp.status}`);
  LR = await resp.json();
  setStatus("ok","LR ready");
  analyzeBtn.disabled = false;
  setMeta("LR", `${LR.vocabulary.length.toLocaleString()} terms`, LR.classes.join(" · "));
}
function inferLR(){
  const raw = textEl.value.trim();
  if (!raw){ toast("Type something first"); return; }
  const {x, tokens} = vectorize(raw, LR.vocabulary, LR.idf);
  const C = LR.classes.length, F = LR.meta.n_features;
  const logits = new Array(C).fill(0);
  for (let c=0;c<C;c++){
    const row = LR.coef[c];
    let sum = 0; for (let j=0;j<F;j++) sum += row[j]*x[j];
    logits[c] = sum + LR.intercept[c];
  }
  const probs = softmax(logits);
  const topIdx = probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0])[0][1];
  updateHeader(LR.classes[topIdx], probs[topIdx]*100);
  renderProbs(LR.classes, probs);
  renderCues(topCuesLR(x, tokenize(raw), LR.vocabulary, LR.coef[topIdx], 6));
}

// ========= BERT (in-browser via @xenova/transformers) =========
// We lazy-load on first switch to keep initial page small.
async function ensureBERT(){
  if (BERT) return;
  try{
    setStatus("loading","Loading BERT…");
    transformers = await import("https://cdn.jsdelivr.net/npm/@xenova/transformers");
    const { env, pipeline } = transformers;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;       // cache model files in the browser
    // Optional: env.backends.onnx.wasm.numThreads = 1; // tweak if needed
    BERT = await pipeline("text-classification", BERT_MODEL_ID);
    setStatus("ok","BERT ready");
    setMeta("BERT", "—", "negative · neutral · positive");
    analyzeBtn.disabled = false;
  } catch (e){
    console.error(e);
    setStatus("err","BERT load failed");
    toast("Failed to load BERT");
  }
}
function remapBertLabels(arr){
  // twitter-roberta-base-sentiment labels: LABEL_0 neg, LABEL_1 neu, LABEL_2 pos
  const map = { LABEL_0: "negative", LABEL_1: "neutral", LABEL_2: "positive",
                negative:"negative", neutral:"neutral", positive:"positive" };
  return arr.map(x => ({ label: map[x.label] || x.label, score: x.score }));
}
async function inferBERT(){
  const raw = textEl.value.trim();
  if (!raw){ toast("Type something first"); return; }
  await ensureBERT();
  // Request top-3 so we can draw the bars
  const out = await BERT(raw, { topk: 3 });
  const results = remapBertLabels(Array.isArray(out) ? out : [out]);
  // Normalize order to [neg, neu, pos]
  const clsOrder = ["negative","neutral","positive"];
  const probs = clsOrder.map(c => {
    const found = results.find(r => r.label === c);
    return found ? found.score : 0;
  });
  const topIdx = probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0])[0][1];
  updateHeader(clsOrder[topIdx], probs[topIdx]*100);
  renderProbs(clsOrder, probs);

  // Simple saliency: highlight tokens that appear in the predicted polarity list
  // (Placeholder lightweight cue; true attribution would require gradients.)
  const tokens = tokenize(raw);
  const freq = Object.create(null);
  for (const t of tokens) freq[t]=(freq[t]||0)+1;
  const cues = Object.entries(freq)
    .map(([t,c]) => [t, c * ((probs[topIdx]||0) - 0.5)]) // crude weighting by confidence
    .sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]))
    .slice(0,6);
  renderCues(cues);
}

// ========= Mode switching & events =========
async function switchMode(mode){
  if (mode === ACTIVE_MODE) return;
  ACTIVE_MODE = mode;
  $$(".tab").forEach(b=>{
    const active = b.dataset.mode === mode;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  });
  resultCard.hidden = true;
  if (mode === "lr"){
    analyzeBtn.disabled = !LR;
    setStatus(LR ? "ok" : "loading", LR ? "LR ready" : "Loading LR…");
    setMeta("LR", LR ? `${LR.vocabulary.length.toLocaleString()} terms` : "—",
                 LR ? LR.classes.join(" · ") : "—");
  } else {
    analyzeBtn.disabled = !BERT;
    setStatus(BERT ? "ok" : "loading", BERT ? "BERT ready" : "Loading BERT…");
    setMeta("BERT", "—", "negative · neutral · positive");
    if (!BERT) await ensureBERT();
  }
}
$(".tabs").addEventListener("click", async (e)=>{
  const b = e.target.closest(".tab"); if (!b) return;
  await switchMode(b.dataset.mode);
});

// Actions
$("#samples").addEventListener("click", (e)=>{
  const b = e.target.closest(".chip"); if(!b) return;
  textEl.value = b.dataset.text || "";
  textEl.focus();
});
analyzeBtn.addEventListener("click", ()=> ACTIVE_MODE === "lr" ? inferLR() : inferBERT());
clearBtn.addEventListener("click", ()=>{
  textEl.value = "";
  resultCard.hidden = true;
  textEl.focus();
});
textEl.addEventListener("keydown", (e)=>{
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) (ACTIVE_MODE === "lr" ? inferLR() : inferBERT());
});

// ========= Bootstrap =========
(async function init(){
  try{
    await loadLR();             // always load LR first (tiny, instant)
    analyzeBtn.disabled = false;
  }catch(err){
    console.error(err);
    setStatus("err","LR load failed");
    toast("Failed to load LR model.json");
  }
  // If you want BERT to prefetch automatically, uncomment:
  // ensureBERT();
})();
