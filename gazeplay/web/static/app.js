"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const SVGNS = "http://www.w3.org/2000/svg";

const VIDEO_RE = /\.(mp4|mov|avi|mkv|m4v|wmv|webm|mts|mpg|mpeg)$/i;
const GAZE_RE = /\.(tsv|csv|txt|xlsx|xls|xlsm)$/i;
const EGO_HINT = /(tobii|scene|ego|glass|eye|first|fpv|pov|gaze|child)/i;
const ROOM_HINT = /(room|ext|camera|cam\b|side|wall|overview|external|lab)/i;

const NEUTRAL = "#e6e4de";
const NON_OBJECT_FILL = {
  "No object in hand": NEUTRAL,
  "No object": NEUTRAL,
  "Hands not visible": "url(#hatch)",
  "Child not visible": "url(#hatch)",
  "No gaze data": "url(#hatch)",
};
const MATCH_FILL = { Yes: "#0ca30c", No: "#d03b3b" };
const LABEL_TEXT = { ego: "eye-tracking video", room: "room video", gaze: "gaze data file" };

const state = {
  config: null,
  files: { ego: null, room: null, gaze: null },
  uploading: {},
  errors: {},
  preset: "balanced",
  objects: [],
  jobId: null,
  poll: null,
  result: null,
  zoom: 1,
  autoDownloaded: false,
};

/* ------------------------------------------------------------------ helpers */
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v);
  }
  for (const c of children) if (c !== null && c !== undefined) e.append(c);
  return e;
}
function svgEl(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
function fmtBytes(n) {
  if (n > 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n > 1e6) return (n / 1e6).toFixed(1) + " MB";
  return Math.max(1, Math.round(n / 1e3)) + " KB";
}
function fmtSec(s) {
  if (s === null || s === undefined) return "-";
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return `${m} min ${r} s`;
}
function fmtClock(t, decimals = 1) {
  t = Math.max(0, t);
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(decimals).padStart(decimals ? 3 + decimals : 2, "0")}`;
}
function fmtEta(s) {
  if (s === null || s === undefined || !isFinite(s)) return "";
  if (s < 50) return "Less than a minute left";
  const m = Math.round(s / 60);
  return m < 60 ? `About ${m} min left` : `About ${Math.floor(m / 60)} h ${m % 60} min left`;
}
function pct(v) { return v === null || v === undefined ? "-" : `${Math.round(v)}%`; }
function showView(name) {
  for (const v of ["setup", "progress", "results", "error"]) $(`#${v}View`).hidden = v !== name;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ------------------------------------------------------------------ init */
async function init() {
  try {
    state.config = await (await fetch("/api/config")).json();
  } catch (e) {
    state.config = { presets: [], stages: [] };
  }
  if (state.config.detector_ready === false) {
    const b = $("#engineBadge");
    b.hidden = false;
    b.className = "badge badge-warn";
    b.textContent = "Detector not installed - run the setup script";
  }
  initPresets();
  initObjects();
  initAdvanced();
  initDropzones();
  $("#runBtn").addEventListener("click", runAnalysis);
  $("#cancelBtn").addEventListener("click", cancelRun);
  $("#newRun").addEventListener("click", () => { showView("setup"); updateRunButton(); });
  $("#errBack").addEventListener("click", () => { showView("setup"); updateRunButton(); });
  $("#openFolder").addEventListener("click", openFolder);
  $$(".zoom button").forEach(b => b.addEventListener("click", () => {
    $$(".zoom button").forEach(x => x.classList.toggle("active", x === b));
    state.zoom = Number(b.dataset.zoom);
    renderTimeline();
  }));
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => state.result && renderTimeline(), 150); });
  window.addEventListener("beforeunload", e => {
    if (state.poll) { e.preventDefault(); e.returnValue = ""; }
  });
}

function initPresets() {
  const group = $("#presetGroup");
  const presets = state.config.presets.length ? state.config.presets : [
    { id: "fast", label: "Fast" }, { id: "balanced", label: "Balanced" }, { id: "accurate", label: "Accurate" }];
  const saved = localStorage.getItem("gp.preset");
  if (saved && presets.some(p => p.id === saved)) state.preset = saved;
  for (const p of presets) {
    const b = el("button", { type: "button", role: "radio", text: p.label });
    b.addEventListener("click", () => { state.preset = p.id; localStorage.setItem("gp.preset", p.id); sync(); });
    b.dataset.id = p.id;
    group.append(b);
  }
  function sync() {
    $$("button", group).forEach(b => {
      b.classList.toggle("active", b.dataset.id === state.preset);
      b.setAttribute("aria-checked", b.dataset.id === state.preset);
    });
    const p = presets.find(x => x.id === state.preset);
    $("#presetHelp").textContent = p && p.description ? p.description.replace(" - ", " — ") : "";
  }
  sync();
}

/* ------------------------------------------------------------------ objects (chips) */
function initObjects() {
  try { state.objects = JSON.parse(localStorage.getItem("gp.objects") || "[]"); } catch { state.objects = []; }
  const input = $("#objectsInput");
  $("#chipsBox").addEventListener("click", () => input.focus());
  input.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === ",") { e.preventDefault(); commitObjects(); }
    else if (e.key === "Backspace" && !input.value && state.objects.length) {
      state.objects.pop(); saveObjects(); renderChips();
    }
  });
  input.addEventListener("input", () => { if (/[,;\n]/.test(input.value)) commitObjects(); });
  input.addEventListener("blur", commitObjects);
  renderChips();
}
function commitObjects() {
  const input = $("#objectsInput");
  const parts = input.value.split(/[,;\n]/).map(s => s.trim().toLowerCase().replace(/\s+/g, " ")).filter(Boolean);
  for (const p of parts) if (!state.objects.includes(p)) state.objects.push(p);
  input.value = "";
  saveObjects();
  renderChips();
}
function saveObjects() { localStorage.setItem("gp.objects", JSON.stringify(state.objects)); }
function renderChips() {
  const box = $("#chips");
  box.replaceChildren(...state.objects.map((o, i) => el("span", { class: "chip" },
    el("span", { text: o }),
    el("button", { type: "button", title: `Remove ${o}`, "aria-label": `Remove ${o}`, text: "×",
      onclick: e => { e.stopPropagation(); state.objects.splice(i, 1); saveObjects(); renderChips(); } }))));
  $("#objectsInput").placeholder = state.objects.length ? "Add another object..."
    : "e.g. red ball, teddy bear, blocks, toy car - press Enter or comma after each";
}

/* ------------------------------------------------------------------ advanced */
function initAdvanced() {
  $$("input[name=sync]").forEach(r => r.addEventListener("change", () => {
    $("#manualOffset").disabled = $("input[name=sync]:checked").value !== "manual";
  }));
  const tol = $("#gazeTol");
  tol.addEventListener("input", () => { $("#gazeTolVal").textContent = Number(tol.value).toFixed(1); });
}
function collectAdvanced() {
  return {
    sync_mode: $("input[name=sync]:checked").value,
    manual_offset_s: Number($("#manualOffset").value || 0),
    gaze_offset_s: Number($("#gazeOffset").value || 0),
    child_selection: $("#childSel").value,
    contact: $("#contactSel").value,
    gaze_tolerance_pct: Number($("#gazeTol").value),
    include_people: $("#includePeople").checked,
    fixations_only: $("#fixOnly").checked,
    review_video: $("#optReviewVideo").checked,
    max_duration_s: $("#maxDur").value ? Number($("#maxDur").value) : null,
  };
}

/* ------------------------------------------------------------------ drop zones & uploads */
function initDropzones() {
  for (const zone of $$(".drop")) {
    const kind = zone.dataset.kind;
    const input = $("input[type=file]", zone);
    zone.addEventListener("click", e => {
      if (e.target.closest(".icon-btn") || state.files[kind] || state.uploading[kind]) return;
      input.click();
    });
    zone.addEventListener("keydown", e => {
      if ((e.key === "Enter" || e.key === " ") && !state.files[kind] && !state.uploading[kind]) { e.preventDefault(); input.click(); }
    });
    input.addEventListener("change", () => { if (input.files[0]) uploadFile(kind, input.files[0]); input.value = ""; });
    zone.addEventListener("dragover", e => { e.preventDefault(); e.stopPropagation(); zone.classList.add("dragover"); overlay(false); });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", e => {
      e.preventDefault(); e.stopPropagation();
      zone.classList.remove("dragover"); overlay(false);
      const files = [...e.dataTransfer.files];
      if (files.length === 1) uploadFile(kind, files[0]);
      else if (files.length > 1) autoAssign(files);
    });
    renderZone(kind);
  }
  const hasFiles = e => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragover", e => {
    e.preventDefault();
    if (!$("#setupView").hidden && hasFiles(e) && !e.target.closest?.(".drop")) overlay(true);
  });
  window.addEventListener("dragleave", e => { if (!e.relatedTarget) overlay(false); });
  window.addEventListener("drop", e => {
    e.preventDefault(); overlay(false);
    if ($("#setupView").hidden) return;
    autoAssign([...e.dataTransfer.files]);
  });
}
function overlay(show) { $("#dropOverlay").hidden = !show; }

function autoAssign(files) {
  const free = k => !state.files[k] && !state.uploading[k];
  const gazeFiles = files.filter(f => GAZE_RE.test(f.name));
  const videos = files.filter(f => VIDEO_RE.test(f.name));
  if (gazeFiles.length) uploadFile("gaze", gazeFiles[0]);
  const pending = [];
  for (const v of videos) {
    const ego = EGO_HINT.test(v.name), room = ROOM_HINT.test(v.name);
    if (ego && !room && free("ego") && !pending.some(p => p.kind === "ego")) pending.push({ kind: "ego", file: v });
    else if (room && !ego && free("room") && !pending.some(p => p.kind === "room")) pending.push({ kind: "room", file: v });
    else pending.push({ kind: null, file: v });
  }
  for (const p of pending) {
    if (!p.kind) p.kind = ["ego", "room"].find(k => free(k) && !pending.some(q => q.kind === k));
    if (p.kind) uploadFile(p.kind, p.file);
  }
  const unknown = files.filter(f => !GAZE_RE.test(f.name) && !VIDEO_RE.test(f.name));
  if (unknown.length) {
    const kind = ["ego", "room", "gaze"].find(free) || "gaze";
    state.errors[kind] = `"${unknown[0].name}" is not a video or gaze data file.`;
    renderZone(kind);
  }
}

function uploadFile(kind, file) {
  const ok = kind === "gaze" ? GAZE_RE.test(file.name) : VIDEO_RE.test(file.name);
  if (!ok) {
    state.errors[kind] = `"${file.name}" doesn't look like a ${kind === "gaze" ? "gaze data file (.tsv, .csv, .xlsx)" : "video file"}.`;
    renderZone(kind);
    return;
  }
  if (state.files[kind]) removeFile(kind, false);
  if (state.uploading[kind]) state.uploading[kind].xhr.abort();
  const xhr = new XMLHttpRequest();
  state.uploading[kind] = { name: file.name, size: file.size, pct: 0, xhr };
  state.errors[kind] = null;
  renderZone(kind);
  xhr.upload.onprogress = e => {
    if (!e.lengthComputable || !state.uploading[kind]) return;
    state.uploading[kind].pct = e.loaded / e.total;
    const zone = $(`.drop[data-kind=${kind}]`);
    const bar = $(".upbar > div", zone), info = $(".file-info", zone);
    if (bar) bar.style.width = `${Math.round(100 * e.loaded / e.total)}%`;
    if (info) info.textContent = e.loaded >= e.total ? "Checking file..." : `Uploading... ${Math.round(100 * e.loaded / e.total)}%`;
  };
  xhr.onload = () => {
    delete state.uploading[kind];
    let data;
    try { data = JSON.parse(xhr.responseText); } catch { data = { error: "Upload failed." }; }
    if (xhr.status === 200) state.files[kind] = data;
    else state.errors[kind] = data.error || "Upload failed.";
    renderZone(kind);
    updateRunButton();
  };
  xhr.onerror = () => {
    delete state.uploading[kind];
    state.errors[kind] = "Upload failed - is the GazePlay window still running?";
    renderZone(kind);
    updateRunButton();
  };
  xhr.onabort = () => { delete state.uploading[kind]; renderZone(kind); updateRunButton(); };
  xhr.open("POST", `/api/upload?kind=${kind}`);
  xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.send(file);
  updateRunButton();
}

function removeFile(kind, render = true) {
  const f = state.files[kind];
  if (f) fetch(`/api/upload/${f.id}`, { method: "DELETE" }).catch(() => {});
  state.files[kind] = null;
  if (render) { renderZone(kind); updateRunButton(); }
}

function renderZone(kind) {
  const zone = $(`.drop[data-kind=${kind}]`);
  const body = $(".drop-body", zone);
  zone.classList.remove("ready", "error");
  const up = state.uploading[kind], f = state.files[kind], err = state.errors[kind];
  const xBtn = (title, fn) => el("button", { class: "icon-btn", type: "button", title, "aria-label": title, text: "×",
    onclick: e => { e.stopPropagation(); fn(); } });
  if (up) {
    body.replaceChildren(el("div", { class: "file-row" },
      el("div", { class: "file-meta" },
        el("div", { class: "file-name", text: up.name, title: up.name }),
        el("div", { class: "file-info", text: `Uploading... ${Math.round(up.pct * 100)}%` }),
        el("div", { class: "upbar" }, el("div", { style: `width:${Math.round(up.pct * 100)}%` }))),
      xBtn("Cancel upload", () => up.xhr.abort())));
  } else if (f) {
    zone.classList.add("ready");
    const check = el("span", { class: "check" });
    check.innerHTML = '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
    body.replaceChildren(el("div", { class: "file-row" }, check,
      el("div", { class: "file-meta" },
        el("div", { class: "file-name", text: f.name, title: f.name }),
        el("div", { class: "file-info", text: `${f.summary} | ${fmtBytes(f.size)}` })),
      xBtn("Remove file", () => removeFile(kind))));
  } else {
    const nodes = [];
    if (err) { zone.classList.add("error"); nodes.push(el("p", { class: "err-text", text: err })); }
    const b = el("p", { class: "browse" });
    b.append("Drop file here or ", el("b", { text: "browse" }));
    nodes.push(b);
    body.replaceChildren(...nodes);
  }
}

function updateRunButton() {
  const ready = ["ego", "room", "gaze"].every(k => state.files[k]);
  const busy = Object.keys(state.uploading).length > 0;
  $("#runBtn").disabled = !ready || busy || !!state.poll;
  const missing = ["ego", "room", "gaze"].filter(k => !state.files[k]).map(k => LABEL_TEXT[k]);
  $("#runHint").textContent = busy ? "Waiting for uploads to finish..."
    : missing.length ? `Still needed: ${missing.join(", ")}.`
    : `Ready. Processing a 10-minute session takes roughly 10-25 minutes on a regular computer.`;
}

/* ------------------------------------------------------------------ run & progress */
async function runAnalysis() {
  commitObjects();
  const body = {
    ego: state.files.ego.id, room: state.files.room.id, gaze: state.files.gaze.id,
    objects: state.objects.join(", "), preset: state.preset, advanced: collectAdvanced(),
  };
  $("#runBtn").disabled = true;
  let data;
  try {
    const r = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    data = await r.json();
    if (!r.ok) throw new Error(data.error || "Could not start the analysis.");
  } catch (e) {
    $("#runHint").textContent = e.message;
    updateRunButton();
    return;
  }
  state.jobId = data.job_id;
  state.autoDownloaded = false;
  buildStageList();
  $("#progTitle").textContent = "Analysing your session...";
  $("#progBar").style.width = "0%";
  $("#progPct").textContent = "0%";
  $("#logBox").textContent = "";
  showView("progress");
  state.poll = setInterval(pollStatus, 800);
  pollStatus();
}

function buildStageList() {
  const list = $("#stageList");
  list.replaceChildren(...state.config.stages.map(s =>
    el("li", { "data-id": s.id }, el("span", { class: "dot" }), el("span", { text: s.label }), el("span", { class: "sub" }))));
}

async function pollStatus() {
  let s;
  try { s = await (await fetch(`/api/status/${state.jobId}`)).json(); } catch { return; }
  renderProgress(s);
  if (["done", "error", "cancelled"].includes(s.state)) {
    clearInterval(state.poll);
    state.poll = null;
    document.title = "GazePlay Analyzer";
    if (s.state === "done") showResults(s.result);
    else if (s.state === "error") showError(s);
    else { showView("setup"); updateRunButton(); }
  }
}

function renderProgress(s) {
  const p = Math.round(100 * (s.overall || 0));
  $("#progBar").style.width = `${p}%`;
  $("#progPct").textContent = `${p}%`;
  document.title = `${p}% - GazePlay Analyzer`;
  $("#progMessage").textContent = s.state === "queued" ? "Waiting for the previous analysis to finish..." : (s.message || "");
  $("#progEta").textContent = fmtEta(s.eta_s);
  const ids = state.config.stages.map(x => x.id);
  const cur = ids.indexOf(s.stage);
  $$("#stageList li").forEach((li, i) => {
    li.classList.toggle("done", s.state === "done" || i < cur);
    li.classList.toggle("active", s.state === "running" && i === cur);
    const sub = $(".sub", li);
    sub.textContent = (i === cur && ["room", "gaze"].includes(s.stage)) ? `${Math.round(100 * (s.stage_frac || 0))}%` : "";
  });
  const box = $("#logBox");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
  box.textContent = (s.logs || []).join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

async function cancelRun() {
  if (!state.jobId) return;
  $("#progTitle").textContent = "Cancelling...";
  await fetch(`/api/cancel/${state.jobId}`, { method: "POST" }).catch(() => {});
}

function showError(s) {
  $("#errMessage").textContent = s.error || "Unknown error.";
  $("#errLog").textContent = (s.logs || []).join("\n");
  showView("error");
}

async function openFolder() {
  const r = await (await fetch(`/api/open-folder/${state.jobId}`, { method: "POST" })).json().catch(() => ({}));
  if (!r.ok && r.path) $("#savedTo").textContent = `Results folder: ${r.path}`;
}

/* ------------------------------------------------------------------ results */
function showResults(r) {
  state.result = r;
  showView("results");
  const m = r.metrics;
  $("#resultsSub").textContent =
    `${fmtSec(m.duration_s)} analysed (${m.analysed_from.slice(0, 8)} to ${m.analysed_to.slice(0, 8)}) | objects ${r.object_source}: ${r.objects.join(", ")}`;
  $("#savedTo").textContent = `Saved automatically to: ${r.out_dir}`;
  $("#dlExcel").href = r.downloads.excel;
  $("#dlCsv").href = r.downloads.csv;
  $("#dlVideo").hidden = !r.downloads.video;
  if (r.downloads.video) $("#dlVideo").href = r.downloads.video;

  // warnings
  const w = $("#warnings");
  w.replaceChildren();
  if (r.warnings && r.warnings.length) {
    w.append(el("div", { class: "callout" }, el("b", { text: "Please check" }),
      el("ul", {}, ...r.warnings.map(x => el("li", { text: x })))));
  }

  // stat tiles
  const sync = r.sync;
  const syncTxt = sync.method === "audio" ? `Videos synced from sound: ${sync.offset_s >= 0 ? "+" : ""}${sync.offset_s.toFixed(2)} s (${sync.confidence})`
    : sync.method === "manual" ? `Manual offset ${sync.offset_s.toFixed(2)} s` : "No sync offset applied";
  const hero = el("div", { class: "stat hero" },
    el("div", { class: "label", text: "Looking at the object being played with" }),
    el("div", { class: "value", text: pct(m.looking_at_played_pct) }),
    el("div", { class: "detail", text: m.looking_at_played_pct === null ? "No play with gaze data was found"
      : `of play time with gaze data (${fmtSec(m.looking_at_played_time_s)})` }),
    el("div", { class: "meter" }, el("div", { style: `width:${m.looking_at_played_pct || 0}%` })));
  const tile = (label, value, detail) => el("div", { class: "stat" },
    el("div", { class: "label", text: label }), el("div", { class: "value", text: value }), el("div", { class: "detail", text: detail }));
  $("#stats").replaceChildren(hero,
    tile("Time playing with objects", fmtSec(m.playing_time_s), `${pct(m.playing_pct)} of the session`),
    tile("Play bouts", String(m.play_bouts), m.play_bouts ? `mean ${fmtSec(m.mean_bout_s)} | ${pct(m.bouts_with_look_pct)} include a look` : "no play detected"),
    tile("Valid gaze data", pct(m.gaze_valid_pct), syncTxt));

  // video
  const video = $("#reviewVideo");
  if (r.downloads.video) {
    $("#videoCard").hidden = false;
    video.src = `${r.downloads.video}?inline=1`;
    video.ontimeupdate = () => movePlayhead(r.t_start + video.currentTime);
  } else {
    $("#videoCard").hidden = true;
    video.removeAttribute("src");
  }

  renderTimeline();
  renderObjectTable();
  renderBouts();

  if (!state.autoDownloaded && r.downloads.excel) {
    state.autoDownloaded = true;
    const a = el("a", { href: r.downloads.excel, download: r.file_names.excel || "" });
    document.body.append(a);
    a.click();
    a.remove();
  }
}

function colorFor(label) {
  const r = state.result;
  if (r.colors[label]) return r.colors[label];
  if (NON_OBJECT_FILL[label]) return NON_OBJECT_FILL[label];
  if (label && label.includes(" + ")) return r.colors[label.split(" + ")[0]] || "#8a8984";
  return "#8a8984";
}

function findSeg(segs, t) {
  let lo = 0, hi = segs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segs[mid].end <= t) lo = mid + 1;
    else if (segs[mid].start > t) hi = mid - 1;
    else return segs[mid];
  }
  return null;
}

const TL = { laneH: 34, gap: 8, top: 6, axisH: 24 };

function renderTimeline() {
  const r = state.result;
  if (!r) return;
  const scroll = $("#tlScroll");
  const svg = $("#timelineSvg");
  const W = Math.max(300, scroll.clientWidth) * state.zoom;
  const H = TL.top + 3 * (TL.laneH + TL.gap) + TL.axisH;
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const t0 = r.t_start, t1 = r.t_end;
  const x = t => 1 + (t - t0) / (t1 - t0) * (W - 2);
  state.tl = { W, x, t0, t1 };

  const defs = svgEl("defs");
  const pat = svgEl("pattern", { id: "hatch", width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
  pat.append(svgEl("rect", { width: 5, height: 5, fill: "#f1f0ec" }), svgEl("rect", { width: 2, height: 5, fill: "#cfccc4" }));
  defs.append(pat);
  svg.append(defs);

  const lanes = [
    { segs: r.segments.interaction, fill: s => colorFor(s.label) },
    { segs: r.segments.gaze, fill: s => colorFor(s.label) },
    { segs: r.segments.match, fill: s => MATCH_FILL[s.label] || null },
  ];
  lanes.forEach((lane, i) => {
    const y = TL.top + i * (TL.laneH + TL.gap);
    svg.append(svgEl("rect", { x: 0, y, width: W, height: TL.laneH, rx: 6, fill: "#f6f5f2" }));
    const g = svgEl("g");
    for (const s of lane.segs) {
      const f = lane.fill(s);
      if (!f) continue;
      const x1 = x(s.start), x2 = x(s.end), w = x2 - x1;
      const inset = w > 4 ? 1 : 0;  // 2px surface gap between neighbouring segments
      g.append(svgEl("rect", { x: x1 + inset, y, width: Math.max(0.8, w - 2 * inset), height: TL.laneH,
        rx: w > 8 ? 3 : 0, fill: f }));
    }
    svg.append(g);
  });

  // time axis
  const axisY = TL.top + 3 * (TL.laneH + TL.gap) - 2;
  const span = t1 - t0;
  const steps = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];
  const step = steps.find(s => (s / span) * W >= 80) || 1800;
  const first = Math.ceil(t0 / step) * step;
  const axis = svgEl("g", { "font-size": 11, fill: "#85837d", "font-family": "inherit" });
  for (let t = first; t <= t1 + 1e-6; t += step) {
    const xx = x(t);
    axis.append(svgEl("line", { x1: xx, x2: xx, y1: axisY, y2: axisY + 5, stroke: "#cfccc4" }));
    const label = svgEl("text", { x: xx, y: axisY + 17, "text-anchor": xx < 20 ? "start" : xx > W - 20 ? "end" : "middle" });
    label.textContent = fmtClock(t, 0);
    axis.append(label);
  }
  svg.append(axis);

  // crosshair + playhead
  const ch = svgEl("line", { id: "crosshair", y1: 0, y2: axisY, stroke: "#17181a", "stroke-width": 1, opacity: 0 });
  const ph = svgEl("line", { id: "playhead", y1: 0, y2: axisY, stroke: "#1d4f91", "stroke-width": 2, opacity: 0 });
  svg.append(ph, ch);

  svg.onpointermove = e => hoverTimeline(e);
  svg.onpointerleave = () => { $("#tooltip").hidden = true; ch.setAttribute("opacity", 0); };
  svg.onclick = e => {
    const t = timeAt(e);
    seekVideo(t);
  };
  renderLegend();
  const video = $("#reviewVideo");
  if (video.src) movePlayhead(r.t_start + (video.currentTime || 0));
}

function timeAt(e) {
  const { W, t0, t1 } = state.tl;
  const rect = $("#timelineSvg").getBoundingClientRect();
  const px = Math.min(Math.max(0, e.clientX - rect.left), W);
  return t0 + (px / W) * (t1 - t0);
}

function hoverTimeline(e) {
  const r = state.result;
  const t = timeAt(e);
  const ch = $("#crosshair");
  ch.setAttribute("x1", state.tl.x(t));
  ch.setAttribute("x2", state.tl.x(t));
  ch.setAttribute("opacity", 0.5);
  const si = findSeg(r.segments.interaction, t), sg = findSeg(r.segments.gaze, t), sm = findSeg(r.segments.match, t);
  const tip = $("#tooltip");
  const row = (name, seg) => {
    const col = seg ? colorFor(seg.label) : "#ccc";
    return el("div", { class: "row" },
      el("span", { class: "key", style: `background:${col.startsWith("url") ? "#cfccc4" : col}` }),
      el("span", { class: "v", text: seg ? seg.label : "-" }),
      el("span", { class: "n", text: name }));
  };
  const match = sm ? sm.label : "n/a";
  const pill = el("span", { class: `pill ${match === "Yes" ? "yes" : match === "No" ? "no" : "na"}`,
    text: match === "Yes" ? "Looking at the played object" : match === "No" ? "Looking elsewhere" : "Not applicable" });
  tip.replaceChildren(el("div", { class: "t", text: fmtClock(t) }), row("playing with", si), row("looking at", sg),
    el("div", { style: "margin-top:8px" }, pill));
  tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = e.clientX + 14, top = e.clientY + 14;
  if (left + tw > window.innerWidth - 8) left = e.clientX - tw - 14;
  if (top + th > window.innerHeight - 8) top = e.clientY - th - 14;
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}

function movePlayhead(t) {
  const ph = $("#playhead");
  if (!ph || !state.tl) return;
  const xx = state.tl.x(t);
  ph.setAttribute("x1", xx);
  ph.setAttribute("x2", xx);
  ph.setAttribute("opacity", 0.9);
  const scroll = $("#tlScroll");
  if (state.zoom > 1 && (xx < scroll.scrollLeft || xx > scroll.scrollLeft + scroll.clientWidth)) {
    scroll.scrollLeft = Math.max(0, xx - scroll.clientWidth / 3);
  }
}

function seekVideo(t) {
  const r = state.result, video = $("#reviewVideo");
  movePlayhead(t);
  if (!r.downloads.video) return;
  const target = Math.max(0, t - r.t_start);
  const go = () => { video.currentTime = target; };
  if (video.readyState >= 1) go(); else video.addEventListener("loadedmetadata", go, { once: true });
}

function renderLegend() {
  const r = state.result;
  const used = new Set();
  for (const s of [...r.segments.interaction, ...r.segments.gaze]) for (const p of s.label.split(" + ")) used.add(p);
  const item = (label, fill, hatch = false) => el("span", { class: "item" },
    el("span", { class: `sw${hatch ? " hatch" : ""}`, style: hatch ? "" : `background:${fill}` }), el("span", { text: label }));
  const items = Object.keys(r.colors).filter(k => used.has(k)).map(k => item(k, r.colors[k]));
  $("#legend").replaceChildren(...items, el("span", { class: "sep" }),
    item("No object (hands free / gaze elsewhere)", NEUTRAL), item("No data (hands hidden / no gaze)", "", true),
    el("span", { class: "sep" }), item("Looking at the played object", MATCH_FILL.Yes), item("Looking elsewhere", MATCH_FILL.No));
}

function renderObjectTable() {
  const r = state.result;
  const rows = r.per_object.filter(o => o["Play Time (s)"] > 0 || o["Total Gaze Time (s)"] > 0);
  const max = Math.max(1, ...rows.map(o => o["Play Time (s)"]));
  const thead = el("thead", {}, el("tr", {},
    el("th", { text: "Object" }), el("th", { text: "Play time" }), el("th", { class: "num", text: "Played" }),
    el("th", { class: "num", text: "Looked while playing" }), el("th", { class: "num", text: "Total gaze" })));
  const tbody = el("tbody");
  for (const o of rows) {
    const play = o["Play Time (s)"], look = o["Looking-While-Playing Time (s)"];
    const lookPct = o["% of Play Time Looking at It"];
    tbody.append(el("tr", {},
      el("td", {}, el("span", { class: "objname" },
        el("span", { class: "sw", style: `background:${r.colors[o.Object] || "#8a8984"}` }), el("span", { text: o.Object }))),
      el("td", {}, el("div", { class: "minibar", title: `${play.toFixed(1)} s played, ${look.toFixed(1)} s of it looking` },
        el("div", { class: "play", style: `width:${100 * play / max}%` }),
        el("div", { class: "look", style: `width:${100 * look / max}%` }))),
      el("td", { class: "num", text: fmtSec(play) }),
      el("td", { class: "num", text: lookPct === null ? "-" : `${Math.round(lookPct)}%` }),
      el("td", { class: "num", text: fmtSec(o["Total Gaze Time (s)"]) })));
  }
  if (!rows.length) tbody.append(el("tr", {}, el("td", { colspan: 5, text: "No objects were played with or looked at." })));
  $("#objTable").replaceChildren(thead, tbody);
  const key = (label, color) => el("span", { class: "item" }, el("span", { class: "sw", style: `background:${color}` }), el("span", { text: label }));
  $("#objLegend").replaceChildren(key("Play time", "#b7d3f6"), key("Looking at it while playing", "#1c5cab"));
}

function renderBouts() {
  const r = state.result;
  const thead = el("thead", {}, el("tr", {},
    ...["#", "Start", "Duration", "Played with", "Looking at it", "Looked?", "First look after", "Also looked at"]
      .map((h, i) => el("th", { class: [0, 2, 4, 6].includes(i) ? "num" : "", text: h }))));
  const tbody = el("tbody");
  for (const b of r.bouts) {
    const looked = b["Looked at It?"];
    const tr = el("tr", { title: "Click to jump the review video here" },
      el("td", { class: "num", text: b.Bout }),
      el("td", { text: b.Start.slice(0, 10) }),
      el("td", { class: "num", text: fmtSec(b["Duration (s)"]) }),
      el("td", {}, el("span", { class: "objname" },
        el("span", { class: "sw", style: `background:${colorFor(b["Object(s) Played With"])}` }),
        el("span", { text: b["Object(s) Played With"] }))),
      el("td", { class: "num", text: b["% Time Looking at It"] === null ? "-" : `${Math.round(b["% Time Looking at It"])}%` }),
      el("td", {}, el("span", { class: `pill ${looked === "Yes" ? "yes" : looked === "No" ? "no" : "na"}`, text: looked })),
      el("td", { class: "num", text: b["Time to First Look (s)"] === null ? "-" : fmtSec(b["Time to First Look (s)"]) }),
      el("td", { text: b["Other Things Looked At"] || "" }));
    tr.addEventListener("click", () => {
      seekVideo(b["Start (s)"]);
      $("#videoCard").scrollIntoView({ behavior: "smooth", block: "center" });
    });
    tbody.append(tr);
  }
  if (!r.bouts.length) tbody.append(el("tr", {}, el("td", { colspan: 8, text: "No play bouts detected." })));
  $("#boutTable").replaceChildren(thead, tbody);
}

init();
