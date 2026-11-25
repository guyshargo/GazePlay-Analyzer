"use strict";
/* Video boxes, linked timelines and synchronisation.

   Single source of truth (so the two timelines can never disagree):
       offset                  room time = eye-tracking time + offset
       egoStart, egoEnd        the analysed part on the eye-tracking clock
   The room markers are always derived: roomStart = egoStart + offset, roomEnd = egoEnd + offset.
   Moving a marker on either timeline therefore moves its partner on the other timeline.

   Workflow: both videos uploaded -> automatic audio sync starts in the background (once per video
   pair) -> timelines become active -> user fine-tunes markers / offset -> Confirm locks the settings
   (Edit unlocks them). Manual changes never restart the audio sync, and the analysis receives only
   the confirmed values. */
(function () {
  const KINDS = ["ego", "room"];
  const NAMES = { ego: "eye-tracking video", room: "room video" };
  const COLORS = { ego: "#2a78d6", room: "#1baf7a" };
  const MIN_LEN = 1.0;     // shortest analysable part (s)
  const N_THUMBS = 14;

  const VS = {
    pairKey: null,          // "egoId|roomId" of the pair the state below belongs to
    status: "waiting",      // waiting (videos missing) | running (audio sync) | ready
    step: "", progress: 0,
    audio: null,            // {available, offset, confidence, message}
    offset: 0, egoStart: 0, egoEnd: 0,
    locked: false,          // true once confirmed (until Edit)
    confirmed: null,        // last confirmed settings, sent to the analysis
    separate: false,        // "move the videos separately" (align by eye)
    dur: { ego: 0, room: 0 }, fps: { ego: 25, room: 25 },
    vids: {}, stages: {}, tls: {}, thumbs: {}, waves: {}, srcId: {},
    raf: null, poll: null, errorMsg: "",
  };

  /* ---------------------------------------------------------------- helpers */
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const r3 = v => Math.round(v * 1000) / 1000;
  const roomStart = () => VS.egoStart + VS.offset;
  const roomEnd = () => VS.egoEnd + VS.offset;
  const segOf = v => (v === "ego" ? [VS.egoStart, VS.egoEnd] : [roomStart(), roomEnd()]);
  const toEgo = (v, t) => (v === "ego" ? t : t - VS.offset);
  const active = () => VS.status === "ready" && !VS.locked;
  function fmtTC(t) {
    const neg = t < 0;
    t = Math.round(Math.abs(t) * 100) / 100;
    const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t - h * 3600 - m * 60;
    const ss = s.toFixed(2).padStart(5, "0");
    return (neg ? "-" : "") + (h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${String(m).padStart(2, "0")}:${ss}`);
  }
  function parseTC(str) {
    str = String(str).trim().replace(",", ".");
    if (!str) return NaN;
    const parts = str.split(":").map(Number);
    if (parts.some(p => isNaN(p))) return NaN;
    return parts.reduce((acc, p) => acc * 60 + p, 0);
  }
  const fmtOff = o => `${o >= 0 ? "+" : "−"}${Math.abs(o).toFixed(3)}`;
  function fmtLen(s) {
    if (s < 60) return `${s.toFixed(1)} s`;
    const m = Math.floor(s / 60);
    return `${m} min ${(s - m * 60).toFixed(1)} s`;
  }
  function bounds(o = VS.offset) {   // valid eye-tracking range for a given offset
    return { lo: Math.max(0, -o), hi: Math.min(VS.dur.ego, VS.dur.room - o) };
  }
  const offsetIsValid = o => { const b = bounds(o); return b.hi - b.lo >= MIN_LEN; };

  /* ---------------------------------------------------------------- public API */
  window.VideoSync = {
    init, renderBox, onVideosChanged,
    isConfirmed: () => !!VS.confirmed && VS.locked && VS.status === "ready",
    isSyncing: () => VS.status === "running",
    getConfirmed: () => (VS.confirmed && VS.locked ? { ...VS.confirmed } : null),
    _state: VS,   // for tests / debugging
  };

  function init() {
    for (const v of KINDS) {
      buildStage(v);
      buildTimeline(v);
    }
    buildBar();
    let rt;
    window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(drawAll, 120); });
    document.addEventListener("keydown", onKey);
    render();
  }

  /* ---------------------------------------------------------------- video boxes */
  function buildStage(v) {
    const zone = $(`.drop[data-kind=${v}]`);
    const video = el("video", { preload: "auto", playsinline: "" });
    video.muted = v === "room";            // avoid an echo when both play
    const stage = el("div", { class: "vstage", hidden: "" },
      video,
      el("span", { class: `vtag ${v}`, text: v === "ego" ? "POV" : "ROOM" }),
      el("span", { class: "vbadge" }),
      el("div", { class: "vmsg", hidden: "" }),
      el("div", { class: "vbar" },
        el("span", { class: "vname" }), el("span", { class: "vinfo" }),
        el("button", { type: "button", class: "vbtn vmute", title: "Sound on / off", "aria-label": "Sound on / off",
          onclick: e => { e.stopPropagation(); video.muted = !video.muted; renderStage(v); } }),
        el("button", { type: "button", class: "vbtn vremove", title: `Remove the ${NAMES[v]}`, "aria-label": `Remove the ${NAMES[v]}`,
          text: "×", onclick: e => { e.stopPropagation(); pauseAll(); removeFile(v); } })));
    zone.append(stage);
    VS.vids[v] = video;
    VS.stages[v] = stage;
    video.addEventListener("error", () => unsupported(v));
    video.addEventListener("loadedmetadata", () => { if (!video.videoWidth) unsupported(v); renderStage(v); });
    for (const ev of ["seeked", "timeupdate", "play", "pause"]) video.addEventListener(ev, () => { renderStage(v); renderTimeline(v); renderBar(); });
  }

  function renderBox(v) {
    if (!VS.stages[v]) return;
    const zone = $(`.drop[data-kind=${v}]`);
    const f = state.files[v];
    const stage = VS.stages[v];
    const video = VS.vids[v];
    zone.classList.toggle("has-video", !!f);
    $(".vdrop-empty", zone).hidden = !!f;
    stage.hidden = !f;
    if (!f) {
      if (VS.srcId[v]) { video.pause(); video.removeAttribute("src"); video.load(); VS.srcId[v] = null; }
      return;
    }
    if (VS.srcId[v] !== f.id) {
      VS.srcId[v] = f.id;
      video.dataset.proxy = "";
      showMsg(v, null);
      video.src = `/api/media/${f.id}`;
      video.load();
      $(".vname", stage).textContent = f.name;
      $(".vname", stage).title = f.name;
      $(".vinfo", stage).textContent = f.summary;
      VS.thumbs[v] = null;
      VS.waves[v] = null;
      const img = new Image();
      img.onload = () => { if (VS.srcId[v] === f.id) { VS.thumbs[v] = img; drawStrip(v); } };
      img.src = `/api/thumbs/${f.id}?n=${N_THUMBS}`;
    }
    renderStage(v);
  }

  function unsupported(v) {
    const video = VS.vids[v], f = state.files[v];
    if (!f) return;
    if (video.dataset.proxy === "1") {
      showMsg(v, "This video can't be shown in the browser, even after conversion. Synchronisation still works from the sound.");
      return;
    }
    video.dataset.proxy = "1";
    const id = f.id;
    showMsg(v, "This video format can't be played in the browser. Preparing a preview copy...");
    fetch(`/api/proxy/${id}`, { method: "POST" }).then(() => {
      const timer = setInterval(async () => {
        if (!state.files[v] || state.files[v].id !== id) { clearInterval(timer); return; }
        const st = await (await fetch(`/api/proxy/${id}`)).json();
        if (st.state === "running") showMsg(v, `Preparing a preview copy for the browser... ${Math.round(100 * (st.progress || 0))}%`);
        else {
          clearInterval(timer);
          if (st.state === "done") { showMsg(v, null); video.src = `/api/media/${id}?proxy=1`; video.load(); }
          else showMsg(v, "Could not prepare a browser preview of this video. Synchronisation still works from the sound.");
        }
      }, 1000);
    });
  }
  function showMsg(v, text) {
    const m = $(".vmsg", VS.stages[v]);
    m.hidden = !text;
    m.textContent = text || "";
  }

  function renderStage(v) {
    const stage = VS.stages[v], video = VS.vids[v];
    if (!stage || stage.hidden) return;
    const mute = $(".vmute", stage);
    if (mute.dataset.state !== String(video.muted)) {
      mute.innerHTML = video.muted
        ? '<svg viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16.5 9.5l5 5m0-5l-5 5" stroke="currentColor" stroke-width="1.8"/></svg>'
        : '<svg viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16 8.5a5 5 0 010 7M18.5 6a8.5 8.5 0 010 12" stroke="currentColor" stroke-width="1.8" fill="none"/></svg>';
      mute.dataset.state = String(video.muted);
    }
    const badge = $(".vbadge", stage);
    const t = video.currentTime || 0;
    if (VS.status !== "ready") { badge.textContent = fmtTC(t); badge.className = "vbadge"; return; }
    const [s, e] = segOf(v);
    const fr = 1 / VS.fps[v];
    // where the linked position wants this video to be (may be outside this video)
    const want = v === "room" && !VS.separate ? VS.vids.ego.currentTime + VS.offset : t;
    if (v === "room" && !VS.separate && (want < -fr || want > VS.dur.room + fr)) {
      badge.textContent = want < 0 ? "ROOM VIDEO NOT STARTED YET" : "ROOM VIDEO ENDED";
      badge.className = "vbadge out";
    } else if (Math.abs(t - s) < fr * 0.6) { badge.textContent = `START  ${fmtTC(t)}`; badge.className = "vbadge mark"; }
    else if (Math.abs(t - e) < fr * 0.6) { badge.textContent = `END  ${fmtTC(t)}`; badge.className = "vbadge mark"; }
    else { badge.textContent = fmtTC(t); badge.className = t < s || t > e ? "vbadge dim" : "vbadge"; }
  }

  /* ---------------------------------------------------------------- timelines */
  function buildTimeline(v) {
    const wrap = $(`.tlwrap[data-v=${v}]`);
    const tl = el("div", { class: "tl is-disabled", "data-v": v, style: `--vc:${COLORS[v]}` },
      el("div", { class: "tl-track" },
        el("canvas", { class: "tl-film" }), el("canvas", { class: "tl-wave" }),
        el("div", { class: "tl-shade l" }), el("div", { class: "tl-shade r" }), el("div", { class: "tl-range" }),
        el("div", { class: "tl-mark s", tabindex: "0", role: "slider", "aria-label": `Analysis start in the ${NAMES[v]}`,
          title: "Analysis start - drag to move (the other video's marker follows)" }, el("span", { class: "tl-flag", text: "START" })),
        el("div", { class: "tl-mark e", tabindex: "0", role: "slider", "aria-label": `Analysis end in the ${NAMES[v]}`,
          title: "Analysis end - drag to move (the other video's marker follows)" }, el("span", { class: "tl-flag", text: "END" })),
        el("div", { class: "tl-head" }),
        el("div", { class: "tl-ph" }, el("span", { class: "spin" }), el("span", { class: "tl-ph-text" }))),
      el("div", { class: "tl-ticks" }),
      el("div", { class: "tl-read" },
        el("label", {}, el("span", { class: "rl", text: "Start" }), el("input", { class: "tc", "data-pt": "start", spellcheck: "false", "aria-label": `Start time in the ${NAMES[v]}` })),
        el("label", {}, el("span", { class: "rl", text: "End" }), el("input", { class: "tc", "data-pt": "end", spellcheck: "false", "aria-label": `End time in the ${NAMES[v]}` })),
        el("span", { class: "tl-now" }, "Position ", el("b", { text: "-" }))));
    wrap.replaceChildren(tl);
    VS.tls[v] = tl;

    const track = $(".tl-track", tl);
    const timeAtX = x => { const r = track.getBoundingClientRect(); return clamp((x - r.left) / r.width, 0, 1) * VS.dur[v]; };
    track.addEventListener("pointerdown", e => {
      if (VS.status !== "ready") return;
      const mark = e.target.closest(".tl-mark");
      const which = mark ? (mark.classList.contains("s") ? "start" : "end") : null;
      if (which && VS.locked) return;          // markers are locked after Confirm
      e.preventDefault();
      track.setPointerCapture(e.pointerId);
      pauseAll();
      const move = ev => {
        const t = timeAtX(ev.clientX);
        if (which) { setMarker(which, toEgo(v, t)); showMoment(which === "start" ? VS.egoStart : VS.egoEnd); }
        else seekFrom(v, t);
        render();
      };
      move(e);
      const up = () => {
        track.removeEventListener("pointermove", move);
        track.removeEventListener("pointerup", up);
        track.removeEventListener("pointercancel", up);
        render();
      };
      track.addEventListener("pointermove", move);
      track.addEventListener("pointerup", up);
      track.addEventListener("pointercancel", up);
    });
    $$(".tl-mark", tl).forEach(m => m.addEventListener("keydown", e => {
      if (!active() || (e.key !== "ArrowLeft" && e.key !== "ArrowRight")) return;
      e.preventDefault(); e.stopPropagation();
      const which = m.classList.contains("s") ? "start" : "end";
      const step = (e.shiftKey ? 1 : 1 / VS.fps.ego) * (e.key === "ArrowLeft" ? -1 : 1);
      setMarker(which, (which === "start" ? VS.egoStart : VS.egoEnd) + step);
      showMoment(which === "start" ? VS.egoStart : VS.egoEnd);
      render();
    }));
    $$("input.tc", tl).forEach(inp => {
      const commit = () => {
        const t = parseTC(inp.value);
        if (!isNaN(t) && active()) {
          setMarker(inp.dataset.pt, toEgo(v, t));
          showMoment(inp.dataset.pt === "start" ? VS.egoStart : VS.egoEnd);
        }
        render();
      };
      inp.addEventListener("change", commit);
      inp.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); inp.blur(); } e.stopPropagation(); });
    });
  }

  function renderTimeline(v) {
    const tl = VS.tls[v];
    if (!tl) return;
    const ready = VS.status === "ready";
    tl.classList.toggle("is-disabled", !ready);
    tl.classList.toggle("is-locked", ready && VS.locked);
    const ph = $(".tl-ph", tl);
    ph.hidden = ready;
    ph.classList.toggle("busy", VS.status === "running");
    $(".tl-ph-text", ph).textContent = VS.status === "running" ? "Synchronising from the sound tracks..."
      : !state.files[v] ? `Add the ${NAMES[v]}` : `Waiting for the ${NAMES[v === "ego" ? "room" : "ego"]}`;
    const inputs = $$("input.tc", tl);
    inputs.forEach(i => { i.disabled = !active(); });
    if (!ready) {
      inputs.forEach(i => { i.value = ""; });
      $(".tl-now b", tl).textContent = "-";
      return;
    }
    const dur = VS.dur[v] || 1;
    const [s, e] = segOf(v);
    const pc = t => (100 * clamp(t, 0, dur)) / dur;
    $(".tl-shade.l", tl).style.width = `${pc(s)}%`;
    $(".tl-shade.r", tl).style.width = `${100 - pc(e)}%`;
    const range = $(".tl-range", tl);
    range.style.left = `${pc(s)}%`;
    range.style.width = `${pc(e) - pc(s)}%`;
    $(".tl-mark.s", tl).style.left = `${pc(s)}%`;
    $(".tl-mark.e", tl).style.left = `${pc(e)}%`;
    $$(".tl-mark", tl).forEach(m => { m.tabIndex = active() ? 0 : -1; m.setAttribute("aria-disabled", String(!active())); });
    const t = VS.vids[v].currentTime || 0;
    $(".tl-head", tl).style.left = `${pc(t)}%`;
    $(".tl-now b", tl).textContent = fmtTC(t);
    for (const inp of inputs) if (document.activeElement !== inp) inp.value = fmtTC(inp.dataset.pt === "start" ? s : e);
  }

  function drawStrip(v) {
    const tl = VS.tls[v];
    if (!tl) return;
    const dpr = window.devicePixelRatio || 1;
    const film = $(".tl-film", tl), wave = $(".tl-wave", tl);
    for (const c of [film, wave]) {
      c.width = Math.max(1, Math.round(c.clientWidth * dpr));
      c.height = Math.max(1, Math.round(c.clientHeight * dpr));
    }
    const fc = film.getContext("2d");
    fc.clearRect(0, 0, film.width, film.height);
    const img = VS.thumbs[v];
    if (img) {
      const n = N_THUMBS, tw = img.width / n, th = img.height, cw = film.width / n, ch = film.height;
      for (let i = 0; i < n; i++) {
        const scale = Math.max(ch / th, cw / tw);
        const sw = cw / scale, sh = ch / scale;
        fc.drawImage(img, i * tw + (tw - sw) / 2, (th - sh) / 2, sw, sh, i * cw, 0, cw - dpr, ch);
      }
    }
    const wc = wave.getContext("2d");
    wc.clearRect(0, 0, wave.width, wave.height);
    const w = VS.waves[v];
    if (w && w.has_audio && w.peaks.length) {
      const mid = wave.height / 2, total = VS.dur[v] || w.duration, per = w.duration / w.peaks.length;
      wc.fillStyle = COLORS[v];
      for (let x = 0; x < wave.width; x++) {
        const t0 = (x / wave.width) * total, t1 = ((x + 1) / wave.width) * total;
        let pk = 0;
        for (let i = Math.floor(t0 / per); i <= Math.min(w.peaks.length - 1, Math.floor(t1 / per)); i++) pk = Math.max(pk, w.peaks[i] || 0);
        const h = Math.max(1, pk * (mid - 2));
        wc.fillRect(x, mid - h, 1, 2 * h);
      }
    } else if (w) {
      wc.fillStyle = "#85837d";
      wc.font = `${11 * dpr}px system-ui, sans-serif`;
      wc.fillText("no sound track", 8 * dpr, wave.height / 2 + 4 * dpr);
    }
    drawTicks(v);
  }
  function drawTicks(v) {
    const box = $(".tl-ticks", VS.tls[v]);
    const W = box.clientWidth, dur = VS.dur[v];
    if (!W || !dur || VS.status !== "ready") { box.replaceChildren(); return; }
    const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];
    const step = steps.find(s => (s / dur) * W >= 64) || 3600;
    const out = [];
    for (let t = 0; t <= dur + 1e-6; t += step) {
      const x = (t / dur) * W;
      if (x < 16 || x > W - 16) continue;
      out.push(el("span", { style: `left:${x}px`, text: fmtTC(t).replace(/\.00$/, "") }));
    }
    box.replaceChildren(...out);
  }
  function drawAll() { KINDS.forEach(drawStrip); }

  /* ---------------------------------------------------------------- the sync model */
  function setMarker(which, egoT) {
    const b = bounds();
    egoT = r3(egoT);   // exact (ms) - typed times stay exactly as typed
    VS.errorMsg = "";
    if (which === "start") VS.egoStart = clamp(egoT, b.lo, VS.egoEnd - MIN_LEN);
    else VS.egoEnd = clamp(egoT, VS.egoStart + MIN_LEN, b.hi);
  }

  function setOffset(o) {
    o = r3(o);
    if (!isFinite(o) || !offsetIsValid(o)) {
      VS.errorMsg = `An offset of ${isFinite(o) ? fmtOff(o) + " s" : "that value"} leaves less than ${MIN_LEN} s of overlap between the videos.`;
      renderBar();
      return false;
    }
    VS.errorMsg = "";
    const ob = bounds(), nb = bounds(o);
    // markers sitting at the very start / end of the usable range stay there ("analyse everything")
    const pinStart = VS.egoStart <= ob.lo + 1e-3, pinEnd = VS.egoEnd >= ob.hi - 1e-3;
    let s = pinStart ? nb.lo : clamp(VS.egoStart, nb.lo, nb.hi - MIN_LEN);
    let e = pinEnd ? nb.hi : clamp(VS.egoEnd, s + MIN_LEN, nb.hi);
    if (e - s < MIN_LEN) { s = nb.lo; e = nb.hi; }
    VS.offset = o;
    VS.egoStart = s;
    VS.egoEnd = e;
    followRoom();
    render();
    return true;
  }

  /* ---------------------------------------------------------------- automatic audio sync */
  function onVideosChanged() {
    const e = state.files.ego, r = state.files.room;
    const key = e && r ? `${e.id}|${r.id}` : null;
    if (key !== VS.pairKey) {
      // a different pair of videos: forget everything that belonged to the old pair
      pauseAll();
      clearInterval(VS.poll);
      VS.pairKey = key;
      VS.audio = null;
      VS.confirmed = null;
      VS.locked = false;
      VS.separate = false;
      VS.errorMsg = "";
      VS.status = key ? "running" : "waiting";
      if (key) {
        VS.dur = { ego: e.info.duration_s, room: r.info.duration_s };
        VS.fps = { ego: e.info.fps || 25, room: r.info.fps || 25 };
        startAudioSync(key);   // the ONLY place the automatic sync is started
      }
    }
    KINDS.forEach(renderBox);
    render();
  }

  async function startAudioSync(key) {
    VS.step = "Starting...";
    VS.progress = 0.02;
    try {
      const r = await fetch("/api/sync/start", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ego: state.files.ego.id, room: state.files.room.id }) });
      if (!r.ok) throw new Error((await r.json()).error || "could not start");
    } catch (err) {
      if (key === VS.pairKey) finishAudioSync(key, { available: false, offset_s: 0, confidence: "none",
        message: `Automatic synchronisation could not start (${err.message}). The offset starts at 0 s - set it by hand.` });
      return;
    }
    clearInterval(VS.poll);
    VS.poll = setInterval(async () => {
      if (key !== VS.pairKey) { clearInterval(VS.poll); return; }
      let j;
      try { j = await (await fetch(`/api/sync/status?key=${encodeURIComponent(key)}`)).json(); } catch { return; }
      if (key !== VS.pairKey) return;
      VS.step = j.step || "";
      VS.progress = j.progress || 0;
      if (j.state === "done") { clearInterval(VS.poll); finishAudioSync(key, j); }
      else renderBar();
    }, 400);
    render();
  }

  function finishAudioSync(key, j) {
    // only the job of the current pair, and only while it is still running, may set the offset:
    // a late or repeated result can never overwrite the user's adjustments
    if (key !== VS.pairKey || VS.status !== "running") return;
    VS.audio = { available: !!j.available, offset: r3(j.offset_s || 0), confidence: j.confidence || "none",
      message: j.message || "" };
    let o = VS.audio.available ? VS.audio.offset : 0;
    if (!offsetIsValid(o)) {
      VS.audio.message = `The sound tracks suggested ${fmtOff(o)} s, which leaves no overlap between the videos; the offset starts at 0 s instead.`;
      VS.audio.available = false;
      o = 0;
    }
    VS.status = "ready";
    VS.offset = o;
    const b = bounds(o);
    VS.egoStart = b.lo;
    VS.egoEnd = b.hi;
    KINDS.forEach(v => {
      const f = state.files[v];
      fetch(`/api/waveform/${f.id}`).then(r => r.json()).then(d => {
        if (VS.pairKey === key) { VS.waves[v] = d; drawStrip(v); }
      }).catch(() => {});
    });
    showMoment(VS.egoStart);
    render();
    requestAnimationFrame(drawAll);
  }

  /* ---------------------------------------------------------------- playback (linked) */
  function setT(v, t) {
    const video = VS.vids[v];
    if (!video || !state.files[v]) return;
    video.currentTime = clamp(t, 0, Math.max(0, VS.dur[v] - 0.01));
  }
  function showMoment(egoT) { setT("ego", egoT); setT("room", egoT + VS.offset); }   // the same moment in both
  function seekFrom(v, t) { if (VS.separate) setT(v, t); else showMoment(toEgo(v, t)); }
  function followRoom() { if (!VS.separate) setT("room", (VS.vids.ego.currentTime || 0) + VS.offset); }
  function pauseAll() {
    KINDS.forEach(v => { const x = VS.vids[v]; if (x) { x.pause(); x.playbackRate = 1; } });
  }
  function isPlaying() { return KINDS.some(v => VS.vids[v] && !VS.vids[v].paused); }
  function playBoth() {
    if (isPlaying()) { pauseAll(); render(); return; }
    followRoom();
    KINDS.forEach(v => { const p = VS.vids[v].play(); if (p && p.catch) p.catch(() => {}); });
    loop();
  }
  function stepFrames(n) {
    pauseAll();
    if (VS.separate) KINDS.forEach(v => setT(v, VS.vids[v].currentTime + n / VS.fps[v]));
    else showMoment(VS.vids.ego.currentTime + n / VS.fps.ego);
  }
  function loop() {
    if (VS.raf) return;
    const tick = () => {
      const ego = VS.vids.ego, room = VS.vids.room;
      if (!VS.separate && !ego.paused && !room.paused) {
        // keep the room video locked to the eye-tracking video + offset
        const drift = room.currentTime - (ego.currentTime + VS.offset);
        if (Math.abs(drift) > 0.3) { room.currentTime = ego.currentTime + VS.offset; room.playbackRate = 1; }
        else if (Math.abs(drift) > 0.015) room.playbackRate = clamp(1 - drift * 2.5, 0.85, 1.15);
        else if (room.playbackRate !== 1) room.playbackRate = 1;
      }
      KINDS.forEach(v => { renderTimeline(v); renderStage(v); });
      if (isPlaying()) VS.raf = requestAnimationFrame(tick);
      else { VS.raf = null; render(); }
    };
    VS.raf = requestAnimationFrame(tick);
  }

  /* ---------------------------------------------------------------- sync bar */
  function buildBar() {
    const bar = $("#syncBar");
    const btn = (id, text, cls, title, onclick) => el("button", { id, type: "button", class: cls, title, text, onclick });
    const step = (label, delta, title) => el("button", { type: "button", class: "step-btn", text: label, title,
      onclick: () => { if (active()) setOffset(VS.offset + delta()); } });
    bar.replaceChildren(
      el("div", { class: "sb-status", id: "sbStatus" }),
      el("div", { class: "sb-grid" },
        el("div", { class: "sb-cell" },
          el("label", { class: "sb-title", for: "offsetInput", text: "Synchronisation offset" }),
          el("div", { class: "offset-ctl" },
            step("−1 s", () => -1, "Room video 1 s earlier"),
            step("−0.1", () => -0.1, "Room video 0.1 s earlier"),
            step("−1 fr", () => -1 / VS.fps.room, "Room video one frame earlier"),
            el("div", { class: "offset-input" }, el("input", { id: "offsetInput", type: "text", inputmode: "decimal", spellcheck: "false" }), el("span", { text: "s" })),
            step("+1 fr", () => 1 / VS.fps.room, "Room video one frame later"),
            step("+0.1", () => 0.1, "Room video 0.1 s later"),
            step("+1 s", () => 1, "Room video 1 s later")),
          el("div", { class: "sb-help", id: "offsetHelp" }),
          btn("returnAudioBtn", "↺ Return to original audio sync", "btn-secondary small-btn", "Restore the offset found from the sound tracks",
            () => { if (active() && VS.audio && VS.audio.available) setOffset(VS.audio.offset); })),
        el("div", { class: "sb-cell" },
          el("div", { class: "sb-title", text: "Check the alignment" }),
          el("div", { class: "transport" },
            btn("tpStart", "⏮", "tp", "Go to the start marker (both videos)", () => { pauseAll(); showMoment(VS.egoStart); render(); }),
            btn("tpBack", "◂", "tp", "One frame back", () => { stepFrames(-1); render(); }),
            btn("tpPlay", "▶ Play both", "tp play", "Play both videos together", () => { playBoth(); render(); }),
            btn("tpFwd", "▸", "tp", "One frame forward", () => { stepFrames(1); render(); }),
            btn("tpEnd", "⏭", "tp", "Go to the end marker (both videos)", () => { pauseAll(); showMoment(VS.egoEnd - 1 / VS.fps.ego); render(); })),
          el("label", { class: "check sep-row" }, el("input", { type: "checkbox", id: "separateChk" }),
            " Move the two videos separately (align by eye)"),
          btn("matchBtn", "Use the current frames as the same moment", "btn-secondary small-btn", "Set the offset so that the two frames now shown are the same moment",
            () => {
              if (!active()) return;
              const o = VS.vids.room.currentTime - VS.vids.ego.currentTime;
              if (setOffset(o)) { VS.separate = false; render(); }
            })),
        el("div", { class: "sb-cell sb-confirm" },
          el("div", { class: "sb-title", text: "Synchronisation settings" }),
          el("div", { class: "sb-summary", id: "sbSummary" }),
          el("div", { class: "confirm-row" },
            btn("confirmSyncBtn", "✓ Confirm synchronisation", "btn-primary", "Lock these settings for the analysis", confirmSync),
            btn("editSyncBtn", "✎ Edit", "btn-secondary", "Unlock the synchronisation to change it", editSync),
            btn("cancelEditBtn", "Cancel changes", "btn-ghost", "Go back to the last confirmed settings", cancelEdit)))));
    const inp = $("#offsetInput");
    const commit = () => {
      const v = parseFloat(String(inp.value).replace(",", ".").replace("−", "-"));
      if (active() && !isNaN(v)) setOffset(v);
      render();
    };
    inp.addEventListener("change", commit);
    inp.addEventListener("keydown", e => {
      e.stopPropagation();
      if (e.key === "Enter") { e.preventDefault(); commit(); inp.select(); }
      else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        if (active()) setOffset(VS.offset + (e.key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 0.1 : 0.01));
        inp.select();
      }
    });
    $("#separateChk").addEventListener("change", e => {
      VS.separate = e.target.checked && active();
      if (!VS.separate) followRoom();
      render();
    });
  }

  function confirmSync() {
    if (!active() || VS.egoEnd - VS.egoStart < MIN_LEN) return;
    pauseAll();
    VS.separate = false;
    VS.confirmed = {
      confirmed: true,
      offset: r3(VS.offset),
      ego_start: r3(VS.egoStart), ego_end: r3(VS.egoEnd),
      room_start: r3(roomStart()), room_end: r3(roomEnd()),
      audio_offset: VS.audio && VS.audio.available ? VS.audio.offset : null,
      audio_confidence: VS.audio ? VS.audio.confidence : null,
    };
    VS.locked = true;
    render();
  }
  function editSync() {
    if (VS.status !== "ready") return;
    VS.locked = false;   // the last confirmed values stay as the starting point (and for "Cancel changes")
    render();
  }
  function cancelEdit() {
    if (!VS.confirmed) return;
    VS.offset = VS.confirmed.offset;
    VS.egoStart = VS.confirmed.ego_start;
    VS.egoEnd = VS.confirmed.ego_end;
    VS.separate = false;
    VS.locked = true;
    VS.errorMsg = "";
    showMoment(VS.egoStart);
    render();
  }

  function renderBar() {
    const bar = $("#syncBar");
    if (!bar || !$("#sbStatus")) return;
    const st = $("#sbStatus");
    bar.dataset.state = VS.status === "ready" ? (VS.locked ? "confirmed" : "ready") : VS.status;
    const kids = [];
    if (VS.status === "waiting") {
      kids.push(el("span", { class: "sb-ico idle", text: "1" }),
        el("div", {}, el("b", { text: "Add both videos to start the automatic synchronisation." }),
          el("span", { text: "The timelines become active once the videos are synchronised." })));
    } else if (VS.status === "running") {
      kids.push(el("span", { class: "spin big" }),
        el("div", { class: "grow" }, el("b", { text: "Synchronising the videos from their sound tracks..." }),
          el("span", { text: VS.step }),
          el("div", { class: "sb-progress" }, el("div", { style: `width:${Math.round(100 * Math.max(0.05, VS.progress))}%` }))));
    } else if (VS.locked) {
      kids.push(el("span", { class: "sb-ico ok", text: "✓" }),
        el("div", {}, el("b", { text: `Synchronisation confirmed: offset ${fmtOff(VS.confirmed.offset)} s` }),
          el("span", { text: "These settings will be used for the analysis. Click Edit to change them." })));
    } else {
      const a = VS.audio || {};
      if (a.available) {
        const o = a.offset;
        const what = Math.abs(o) < 0.0005 ? "the videos start at the same moment"
          : `the room camera started ${Math.abs(o).toFixed(3)} s ${o > 0 ? "before" : "after"} the glasses`;
        const warn = a.confidence === "low";
        kids.push(el("span", { class: `sb-ico ${warn ? "warn" : "ok"}`, text: warn ? "!" : "✓" }),
          el("div", {}, el("b", { text: `Audio sync complete: ${what}.` }),
            el("span", {}, el("span", { class: `conf ${a.confidence}`, text: `${a.confidence} confidence` }),
              warn ? " The sound tracks did not match clearly - check carefully with the markers and Play both."
                : " Check it with the markers and Play both, fine-tune if needed, then confirm.")));
      } else {
        kids.push(el("span", { class: "sb-ico warn", text: "!" }),
          el("div", {}, el("b", { text: "Automatic synchronisation was not possible." }), el("span", { text: a.message || "" })));
      }
    }
    st.replaceChildren(...kids);

    const ready = VS.status === "ready";
    const can = active();
    const inp = $("#offsetInput");
    if (document.activeElement !== inp) inp.value = ready ? fmtOff(VS.offset) : "";
    inp.disabled = !can;
    $$(".offset-ctl .step-btn").forEach(b => { b.disabled = !can; });
    const help = $("#offsetHelp");
    help.replaceChildren();
    if (VS.errorMsg && can) help.append(el("span", { class: "err-inline", text: VS.errorMsg }));
    else if (ready) {
      help.append("Room video time = eye-tracking time + offset.");
      if (VS.audio && VS.audio.available) {
        const d = VS.offset - VS.audio.offset;
        help.append(" ", el("span", { class: Math.abs(d) < 0.0005 ? "same" : "changed",
          text: Math.abs(d) < 0.0005 ? "Same as the audio sync." : `${fmtOff(d)} s from the audio sync (${fmtOff(VS.audio.offset)} s).` }));
      }
    }
    $("#returnAudioBtn").disabled = !can || !(VS.audio && VS.audio.available) || Math.abs(VS.offset - VS.audio.offset) < 0.0005;
    for (const id of ["tpStart", "tpBack", "tpPlay", "tpFwd", "tpEnd"]) $(`#${id}`).disabled = !ready;
    $("#tpPlay").textContent = isPlaying() ? "❚❚ Pause" : "▶ Play both";
    const sep = $("#separateChk");
    sep.checked = VS.separate;
    sep.disabled = !can;
    $("#matchBtn").hidden = !VS.separate;
    $("#matchBtn").disabled = !can;

    const sum = $("#sbSummary");
    if (ready) {
      const len = VS.egoEnd - VS.egoStart;
      sum.replaceChildren(
        el("div", {}, "Analysed part: ", el("b", { text: fmtLen(len) })),
        el("div", { class: "seg-line" }, el("span", { class: "dot ego" }), `POV ${fmtTC(VS.egoStart)} → ${fmtTC(VS.egoEnd)}`),
        el("div", { class: "seg-line" }, el("span", { class: "dot room" }), `Room ${fmtTC(roomStart())} → ${fmtTC(roomEnd())}`));
    } else sum.replaceChildren(el("span", { class: "muted", text: "Available after the automatic synchronisation." }));
    $("#confirmSyncBtn").hidden = VS.locked;
    $("#confirmSyncBtn").disabled = !can;
    $("#editSyncBtn").hidden = !VS.locked;
    $("#cancelEditBtn").hidden = VS.locked || !VS.confirmed || !ready;
  }

  function render() {
    KINDS.forEach(v => { renderTimeline(v); renderStage(v); });
    renderBar();
    if (typeof updateRunButton === "function") updateRunButton();
  }

  function onKey(e) {
    if (VS.status !== "ready" || $("#setupView").hidden) return;
    if (e.target.matches && e.target.matches("input, textarea, select, button, [role=slider]")) return;
    const inSync = e.target.closest && (e.target.closest(".vgrid") || e.target.closest("#syncBar"));
    if (!inSync && document.activeElement !== document.body) return;
    if (e.key === " " && inSync) { e.preventDefault(); playBoth(); render(); }
    else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && inSync) {
      e.preventDefault();
      if (e.shiftKey) { pauseAll(); showMoment(VS.vids.ego.currentTime + (e.key === "ArrowLeft" ? -1 : 1)); }
      else stepFrames(e.key === "ArrowLeft" ? -1 : 1);
      render();
    }
  }
})();
