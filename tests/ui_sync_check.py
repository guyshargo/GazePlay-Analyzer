"""Browser check of the synchronisation workflow (needs: pip install playwright && playwright install chromium).

    python tests/make_synthetic.py                      # once, creates tests/synthetic/
    GAZEPLAY_BACKEND=mock python app.py --no-browser    # in another terminal
    python tests/ui_sync_check.py
"""
import json
import os
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
D = str(ROOT / "synthetic") + "/"
OUT = str(ROOT / "screenshots") + "/"
os.makedirs(OUT, exist_ok=True)
TRUE_OFFSET = 7.3
fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def parse_tc(s):
    parts = [float(p) for p in s.split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get("CHROME") or None,
                          args=["--autoplay-policy=no-user-gesture-required"])
    pg = b.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
    errors, sync_starts, run_bodies = [], [], []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
    pg.on("request", lambda r: sync_starts.append(r.url) if r.url.endswith("/api/sync/start") else
          (run_bodies.append(r.post_data) if r.url.endswith("/api/run") else None))
    # hold the first status answers at "running" so the loading state can be seen and photographed
    held = {"n": 0}

    def hold(route):
        if held["n"] < 4:
            held["n"] += 1
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps({"state": "running", "step": "Reading the sound tracks of both videos...", "progress": 0.45}))
        else:
            route.continue_()
    pg.route("**/api/sync/status*", hold)

    S = lambda: pg.evaluate("""(() => { const s = VideoSync._state; return {status: s.status, offset: s.offset,
        egoStart: s.egoStart, egoEnd: s.egoEnd, locked: s.locked, separate: s.separate, confirmed: s.confirmed,
        egoT: s.vids.ego.currentTime, roomT: s.vids.room.currentTime, audio: s.audio}; })()""")
    shown = lambda v, pt: parse_tc(pg.input_value(f".tl[data-v={v}] input[data-pt={pt}]"))
    mark_time = lambda v, cls: pg.evaluate(f"""(() => {{ const s = VideoSync._state;
        return parseFloat(document.querySelector('.tl[data-v={v}] .tl-mark.{cls}').style.left) / 100 * s.dur['{v}']; }})()""")

    def invariant(label):
        s = S()
        rs, re_ = shown("room", "start"), shown("room", "end")
        es, ee = shown("ego", "start"), shown("ego", "end")
        ok = (abs(rs - (es + s["offset"])) < 0.011 and abs(re_ - (ee + s["offset"])) < 0.011 and
              abs(mark_time("room", "s") - (s["egoStart"] + s["offset"])) < 0.01 and
              abs(mark_time("ego", "s") - s["egoStart"]) < 0.01 and
              abs(mark_time("room", "e") - (s["egoEnd"] + s["offset"])) < 0.01)
        check(ok, f"{label}: room markers = POV markers + offset (POV {es:.2f}-{ee:.2f}, room {rs:.2f}-{re_:.2f}, offset {s['offset']:+.3f})")
        return s

    def drag(v, cls, frac):
        h = pg.locator(f".tl[data-v={v}] .tl-mark.{cls}").bounding_box()
        tr = pg.locator(f".tl[data-v={v}] .tl-track").bounding_box()
        pg.mouse.move(h["x"] + h["width"] / 2, h["y"] + h["height"] / 2)
        pg.mouse.down()
        pg.mouse.move(tr["x"] + tr["width"] * frac, h["y"] + h["height"] / 2, steps=6)
        pg.mouse.up()
        pg.wait_for_timeout(350)

    pg.goto(os.environ.get("GAZEPLAY_URL", "http://127.0.0.1:8765/"))
    pg.evaluate("localStorage.clear()")
    pg.reload()
    print("\n1. Layout and defaults")
    check(pg.locator(".vgrid .vcol").count() == 2, "two video boxes side by side")
    check(pg.locator(".tl.is-disabled").count() == 2, "both timelines start as grey placeholders")
    check(pg.locator(".gdrop").count() == 1 and pg.locator(".gdrop").bounding_box()["y"] > pg.locator(".vgrid").bounding_box()["y"] + 200,
          "gaze data box sits below the video section")
    check(not pg.is_checked("#includePeople"), "'Count people as gaze targets' unchecked by default")
    check(pg.input_value("#childSel") == "only", "child default = 'There is only one person'")
    check(pg.locator("text=Output").count() == 0 and pg.locator("#optReviewVideo").count() == 0, "Output section removed")
    pg.screenshot(path=OUT + "v0_empty.png", full_page=True)

    print("\n2. Upload -> automatic audio sync in the background")
    pg.set_input_files('.drop[data-kind=ego] input[type=file]', D + "tobii_scene.mp4")
    pg.wait_for_selector(".drop[data-kind=ego].has-video")
    check(pg.locator(".tl.is-disabled").count() == 2 and len(sync_starts) == 0, "after one video: timelines still grey, no sync yet")
    pg.set_input_files('.drop[data-kind=room] input[type=file]', D + "room_camera.mp4")
    pg.wait_for_selector("#syncBar[data-state=running]", timeout=10000)
    check(pg.locator("#syncBar .spin.big").is_visible() and pg.locator(".tl-ph.busy").count() == 2,
          "loading indicator visible while synchronising (spinner + timelines)")
    check(pg.is_disabled("#confirmSyncBtn") and pg.is_disabled("#offsetInput"), "controls disabled while synchronising")
    pg.screenshot(path=OUT + "v1_syncing.png", full_page=True)
    pg.wait_for_selector("#syncBar[data-state=ready]", timeout=20000)
    pg.wait_for_timeout(800)
    s = S()
    check(len(sync_starts) == 1, f"audio sync started exactly once (calls: {len(sync_starts)})")
    check(abs(s["audio"]["offset"] - TRUE_OFFSET) < 0.02 and abs(s["offset"] - s["audio"]["offset"]) < 1e-9,
          f"audio sync result applied: {s['offset']:+.3f} s (true {TRUE_OFFSET:+.3f})")
    check(pg.locator(".tl.is-disabled").count() == 0, "timelines enabled and coloured after sync")
    check(abs(s["roomT"] - (s["egoT"] + s["offset"])) < 0.02, f"videos show the same moment (POV {s['egoT']:.3f}, room {s['roomT']:.3f})")
    invariant("initial")
    pg.screenshot(path=OUT + "v2_ready.png", full_page=True)

    print("\n3. Linked markers")
    drag("ego", "s", 0.20)
    s = invariant("drag POV start")
    check(abs(s["roomT"] - (s["egoT"] + s["offset"])) < 0.02 and abs(s["egoT"] - s["egoStart"]) < 0.02,
          "dragging a marker shows that moment in both videos")
    drag("room", "e", 0.70)
    invariant("drag room end")
    inp = pg.locator(".tl[data-v=room] input[data-pt=start]")
    inp.fill("00:20.00"); inp.press("Enter"); pg.wait_for_timeout(300)
    s = invariant("type room start 00:20.00")
    check(abs(s["egoStart"] - (20.0 - s["offset"])) < 0.0015, f"typed room start kept exactly (POV start {s['egoStart']:.3f} = 20.000 - offset)")
    pg.locator(".tl[data-v=ego] .tl-mark.e").focus(); pg.keyboard.press("ArrowLeft"); pg.wait_for_timeout(200)
    invariant("keyboard: POV end one frame back")

    print("\n4. Offset control")
    seg_before = (S()["egoStart"], S()["egoEnd"])
    pg.fill("#offsetInput", "7.34"); pg.press("#offsetInput", "Enter"); pg.wait_for_timeout(300)
    s = invariant("offset typed 7.34")
    check(abs(s["offset"] - 7.34) < 1e-9 and (s["egoStart"], s["egoEnd"]) == seg_before,
          "typed offset applied; POV markers unchanged, room markers moved")
    check(abs(s["roomT"] - (s["egoT"] + 7.34)) < 0.02, "room video re-positioned to POV time + new offset")
    check(len(sync_starts) == 1, "manual change did not restart the audio sync")
    pg.click(".step-btn:has-text('+0.1')"); pg.wait_for_timeout(200)
    check(abs(S()["offset"] - 7.44) < 1e-9, "+0.1 step -> 7.440")
    pg.click(".step-btn:has-text('-1 fr'), .step-btn:has-text('−1 fr')"); pg.wait_for_timeout(200)
    check(abs(S()["offset"] - 7.40) < 1e-9, "-1 frame step (25 fps) -> 7.400")
    invariant("after steps")
    check("from the audio sync" in pg.inner_text("#offsetHelp"), f"difference to audio sync shown: '{pg.inner_text('#offsetHelp')}'")
    pg.fill("#offsetInput", "500"); pg.press("#offsetInput", "Enter"); pg.wait_for_timeout(200)
    check(abs(S()["offset"] - 7.40) < 1e-9 and pg.locator(".err-inline").is_visible(), "impossible offset (500 s) refused with a message")
    pg.click("#returnAudioBtn"); pg.wait_for_timeout(200)
    s = invariant("Return to original audio sync")
    check(abs(s["offset"] - s["audio"]["offset"]) < 1e-9 and pg.is_disabled("#returnAudioBtn"), "offset restored to the audio result")

    print("\n5. Align by eye (videos moved separately)")
    pg.check("#separateChk")
    s0 = S()
    tr = pg.locator(".tl[data-v=room] .tl-track").bounding_box()
    pg.mouse.click(tr["x"] + tr["width"] * 0.5, tr["y"] + 15); pg.wait_for_timeout(300)
    s1 = S()
    check(abs(s1["egoT"] - s0["egoT"]) < 0.01 and abs(s1["roomT"] - s0["roomT"]) > 1, "only the room video moved")
    pg.click("#matchBtn"); pg.wait_for_timeout(300)
    s2 = invariant("use current frames as the same moment")
    check(abs(s2["offset"] - round(s1["roomT"] - s1["egoT"], 3)) < 0.0015 and not s2["separate"], "offset = room frame - POV frame; linking restored")
    pg.click("#returnAudioBtn"); pg.wait_for_timeout(200)

    print("\n6. Confirm / Edit")
    pg.fill("#offsetInput", "7.320"); pg.press("#offsetInput", "Enter"); pg.wait_for_timeout(200)
    expect = S()
    pg.set_input_files('.drop[data-kind=gaze] input[type=file]', D + "gaze_export.tsv")
    pg.wait_for_selector(".drop[data-kind=gaze].ready")
    check(pg.is_disabled("#runBtn"), f"Run disabled until the sync is confirmed ('{pg.inner_text('#runHint')}')")
    pg.click("#confirmSyncBtn"); pg.wait_for_timeout(300)
    s = S()
    check(s["locked"] and pg.is_visible("#editSyncBtn") and not pg.is_visible("#confirmSyncBtn"), "Confirm replaced by Edit")
    check(s["confirmed"]["offset"] == 7.32 and abs(s["confirmed"]["ego_start"] - expect["egoStart"]) < 0.0011, "confirmed values stored")
    check(pg.is_disabled("#offsetInput") and pg.locator(".tl.is-locked").count() == 2, "controls locked after confirming")
    before = (s["egoStart"], s["egoEnd"], s["offset"])
    drag("ego", "s", 0.05)
    s = S()
    check((s["egoStart"], s["egoEnd"], s["offset"]) == before, "markers cannot be moved while confirmed")
    check(not pg.is_disabled("#runBtn"), "Run enabled once confirmed")
    pg.screenshot(path=OUT + "v3_confirmed.png", full_page=True)
    pg.click("#editSyncBtn"); pg.wait_for_timeout(200)
    check(not S()["locked"] and pg.is_disabled("#runBtn") and not pg.is_disabled("#offsetInput"), "Edit unlocks; Run waits for a new confirmation")
    pg.fill("#offsetInput", "7.9"); pg.press("#offsetInput", "Enter"); pg.wait_for_timeout(200)
    pg.click("#cancelEditBtn"); pg.wait_for_timeout(200)
    s = S()
    check(s["locked"] and abs(s["offset"] - 7.32) < 1e-9, "Cancel changes restores the confirmed settings")
    pg.click("#editSyncBtn"); pg.wait_for_timeout(200)
    drag("room", "s", 0.30)
    pg.fill("#offsetInput", "7.310"); pg.press("#offsetInput", "Enter"); pg.wait_for_timeout(200)
    pg.click("#confirmSyncBtn"); pg.wait_for_timeout(300)
    final = S()["confirmed"]
    print("   confirmed:", final)
    check(len(sync_starts) == 1, "audio sync never re-ran during manual work")

    print("\n7. The analysis uses exactly the confirmed settings")
    while pg.locator(".chip button").count():
        pg.locator(".chip button").first.click()
    pg.fill("#objectsInput", "ball, block, toy car"); pg.keyboard.press("Enter")
    pg.click("#runBtn")
    pg.wait_for_selector("#resultsView:not([hidden])", timeout=240000)
    sent = json.loads(run_bodies[-1])["sync"]
    check(sent == final, "request carried the confirmed settings unchanged")
    used = pg.evaluate("state.result.sync_used")
    check(abs(used["offset"] - final["offset"]) < 1e-9 and abs(used["ego_start"] - final["ego_start"]) < 1e-9
          and abs(used["ego_end"] - final["ego_end"]) < 1e-9, f"server used them: {used}")
    csv = pg.evaluate("state.result.files.csv")
    df = pd.read_csv(csv, keep_default_na=False)
    diffs = (df["Room Video Time (s)"] - df["Time (s)"]).round(6).unique()
    check(len(diffs) == 1 and abs(diffs[0] - final["offset"]) < 1e-6, f"every row: room time = POV time + {final['offset']} (found {diffs})")
    check(df["Time (s)"].min() >= final["ego_start"] - 1e-6 and df["Time (s)"].max() <= final["ego_end"],
          f"rows only inside the confirmed part ({df['Time (s)'].min():.2f}-{df['Time (s)'].max():.2f})")
    print("   results tile:", pg.inner_text(".stat:nth-child(4) .detail"))
    check(not errors, f"no browser errors {errors}")
    b.close()

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
