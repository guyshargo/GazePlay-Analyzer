# GazePlay Analyzer

**Was the child looking at the object they were playing with?**

GazePlay Analyzer takes three inputs from one play session:

1. the **first-person video** from Tobii Pro Glasses,
2. a **room video** from an external camera, and
3. the **gaze data**: either the `gazedata.gz` file from the Glasses 3 recording folder, or a Tobii Pro Lab data export (TSV, CSV or Excel).

It then works out, moment by moment, **which object the child is holding** (from the room video) and **which object the child is looking at** (from the eye-tracking video and the gaze points). Everything runs on an ordinary computer; no GPU is needed.

---

## Quick start

| | Windows | macOS / Linux |
|---|---|---|
| **Install and start** | Double-click `setup_and_run.bat` | Run `./setup_and_run.sh` in a terminal |

* The **first start** installs everything into a private folder (`.venv`) and downloads the detection models, about 600–800 MB in total. It takes 5–15 minutes and needs internet. Later starts open in seconds and work offline.
* Python 3.10–3.12 is needed ([python.org](https://www.python.org/downloads/); on Windows tick *"Add python.exe to PATH"*).
* The app opens in your web browser at `http://127.0.0.1:8765`. It runs only on your own computer, and **no data leaves the machine**. Keep the black console window open while you work.

## Using it

1. **Add the two videos** to the two boxes side by side: *POV / Eye-tracking video* and *Room video*. Each file is checked right away (duration, resolution, frame rate).
2. **Synchronisation starts by itself.** As soon as both videos are in, the app synchronises them from their sound in the background, with a loading indicator. The timelines under the videos stay grey until it finishes, then turn blue and green and become active.
3. **Check, fine-tune and confirm** with the timelines, the offset control and *Play both* (see below), then click **Confirm synchronisation**.
4. **Add the gaze data** in the box below the videos.
5. *(Optional, recommended)* **Type the toys used in the session**, e.g. `red ball, teddy bear, blocks, toy car`. Colours help tell similar toys apart. The list is remembered for the next session. If you leave it empty, the objects are found automatically.
6. Choose a **quality preset** (Fast / Balanced / Accurate) and click **Run analysis**. The button is enabled once all files are added and the synchronisation is confirmed.

You can also drop all three files anywhere on the page; they are sorted into the right boxes automatically.

Progress, the current step and the time remaining are shown live. When the analysis finishes:

* the **Excel report downloads automatically**, and every output is also saved in `results/<video name>_<date>/`;
* the results page shows the key numbers, a **session timeline** (hover for details, click to jump the review video to that moment), a per-object table and a list of play bouts.

### Synchronising the videos

Everything is based on one number, the **synchronisation offset**: *room video time = eye-tracking time + offset*. A positive offset means the room camera started first. Each video has a timeline under it, showing thumbnails, the sound waveform (claps show up as spikes) and two markers, **START** and **END**, for the part to analyse.

* **Automatic audio sync.** This runs once, as soon as both videos are uploaded. It finds the offset from the sound tracks and places the markers to cover the whole overlapping part. The result and its confidence are shown above the controls.
* **Linked markers.** The markers on the two timelines are always linked through the offset. Moving START or END on either video, by dragging, typing a time, or using the arrow keys on a focused marker, moves the matching marker on the other video. Both videos then show that moment, so you can see at once whether they match.
* **Synchronisation offset control.** Type an offset, or nudge it with −1 s / −0.1 s / −1 frame / +1 frame / +0.1 s / +1 s (or ↑/↓ in the box: ±0.01 s, Shift ±0.1 s). The room markers and the room picture follow immediately. Below the box, the difference from the audio result is shown. Offsets that would leave the videos without at least 1 s of overlap are refused.
* **Return to original audio sync** restores the offset found from the sound.
* **Check the alignment.** *Play both* plays the two videos locked together at the current offset. You can also step frame by frame or jump to the START/END markers. *Move the two videos separately* lets you find the same event (e.g. a clap) in each video on its own. *Use the current frames as the same moment* then sets the offset from the two frames shown, which is useful when a video has no sound.
* **Confirm synchronisation** locks the offset and markers, and the button becomes **Edit**. Edit unlocks them for further changes; *Cancel changes* returns to the last confirmed settings.

Manual changes never restart the automatic audio sync and never overwrite what you set. It runs again only if you replace one of the videos. The analysis only starts with a confirmed synchronisation, and it uses exactly the confirmed offset and part. The Summary sheet lists both the audio result and the confirmed values, and in the Timeline sheet *Room Video Time* = *Time* + offset on every row.

If a video has no usable sound, the offset starts at 0 s and the message says so; set it by hand as above. Video formats the browser can't play (e.g. some `.avi`, `.mts` or HEVC files) are converted automatically into a small preview copy with the same timing.

### Preparing the files

**Straight from a Glasses 3 recording folder (simplest).** Use `scenevideo.mp4` as the eye-tracking video and `gazedata.gz` as the gaze data, exactly as they come off the glasses. The gaze file is read directly: timestamps in seconds share a clock with the scene video, and the normalised `gaze2d` coordinates are scaled to the video size. This raw format has no fixation classification, so *Use fixation samples only* does not apply.

**Or from Tobii Pro Lab:**

* **Gaze data:** *Analyze → Data Export*. Include at least **Recording timestamp** and **Gaze point X / Y**. **Eye movement type** is optional but useful. Export one recording per file. The default `.tsv` format works as is, including European decimal commas.
* **Eye-tracking video:** export the scene-camera video *without* the gaze overlay if you can. The overlay circle can hide small objects.
* Both videos should have their **sound track**, which is used to synchronise them automatically.

---

## Output

### Excel workbook (`*_gaze_play_results.xlsx`)

| Sheet | Contents |
|---|---|
| **Summary** | Key results in plain language, synchronisation details, input files, settings, warnings and a key to the labels |
| **Timeline** | One row per time point (default 5 per second). The first three columns are **Timestamp**, **Interacted Object** and **Gazed-at Object**. Then come **Looking at Played Object** (Yes / No / n/a, colour-coded), the room-video time, the gaze coordinates, the eye-movement type, confidence values, and the raw labels before smoothing |
| **Play Bouts** | Each continuous period of play with one object: start, duration, % of the bout spent looking at the object, whether the child looked at it at all, **time to first look**, and what else the child looked at |
| **Per Object** | Play time, looking-while-playing time, % of play time looking at the object, total gaze time, detection rates, and a chart |
| **Objects Found** | *(automatic mode only)* what the discovery step saw and why each tag was kept or ignored |

A `*_timeline.csv` holds the same data as the Timeline sheet, `*_review.mp4` (always produced) is a side-by-side video (room camera | eye-tracking camera) with the detections, hands, gaze point and final labels drawn on it, and `*_summary.json` is a machine-readable summary.

### What the labels mean

| Column | Label | Meaning |
|---|---|---|
| Interacted Object | *object name* | A hand is on or holding this object |
| | `ball + block` | Two objects in the hands at once |
| | `No object in hand` | Child and hands visible, nothing held |
| | `Hands not visible` | Child visible, both hands hidden |
| | `Child not visible` | No child in the room video |
| Gazed-at Object | *object name* | The gaze point is on the object, or within the accuracy margin |
| | `No object` | Valid gaze, but not on a listed object |
| | `Outside camera view` | Valid gaze outside the scene camera's picture, typically looking down at the lap or hands. What was looked at can't be seen |
| | `No gaze data` | Blink, looking away or tracking lost |
| Looking at Played Object | `Yes` / `No` | Whether the gazed-at object is the one in hand |
| | `n/a` | Not playing, no gaze data, or gaze outside the camera view (unknown rather than "No") |

---

## How it works

```
 Room video ──► people + 17 body keypoints (YOLO26 pose) ──► pick the child ──► hand positions
            └─► objects in a zoomed crop around the child (YOLOE) ───────────┐
                                                                             ├─► "Interacted object"
 Eye-tracking video + gaze points ──► objects in a crop around the gaze ─────┼─► "Gazed-at object"
                                      point (YOLOE, outlines/masks)          │
 Sound tracks of both videos ──► cross-correlation ──► time offset ──────────┘
                                                                             ▼
             one common time axis ─► smoothing ─► Yes/No match ─► Excel / CSV / review video
```

1. **Synchronisation.** The automatic step works from the sound, then you check and confirm the result in the app, and the analysis uses the confirmed offset and part. Both cameras hear the same room (voices, toys, a clap), so the app turns each sound track into a "loudness onset" signal and cross-correlates the two. The best match gives the offset between the recordings, usually to within a few hundredths of a second, with a confidence rating. Gaze timestamps are already aligned with the scene video.
2. **Objects in the room.** If you name the toys, those names are used as YOLOE *text prompts*. Otherwise a prompt-free YOLOE model scans 32 frames from both videos. It then drops people, furniture, room surfaces and very generic tags, merges synonyms (e.g. *toy block* → *block*), and keeps objects seen in at least 8% of frames. **The same name list is used for both videos**, so the labels can be compared directly.
3. **Interaction (room video).** A pose model finds each person's shoulders, hips, elbows and wrists. By default the child is assumed to be the only person in view, so the most confidently detected person is used. When an adult is also present, choose *the smallest person in view*: the person with the smallest typical torso length, which still works when people sit or stand. Hand centres are placed slightly beyond the wrists along the forearm. An object counts as *played with* when a hand is within 0.3 torso lengths of its segmentation outline. That measure scales with the child's size and distance from the camera.
4. **Gaze (eye-tracking video).** The gaze point is the median of the samples within ±50 ms of each time point. Objects are detected in a crop around the gaze point, which gives small objects more resolution. The gazed-at object is the one whose **outline** contains the gaze point (the smallest one if several do, e.g. a toy in front of a person). Otherwise it is the nearest object within an accuracy margin of 2% of the frame width, about 1.5° of visual angle.
5. **Combining.** Both streams are put on the eye-tracking time axis. Short flickers are removed with a majority vote over 1 s (interaction) and 0.4 s (gaze), and the Yes / No match, play bouts and statistics are computed.

### Efficiency choices (CPU only)

* **Frame sampling** at 5 time points per second, instead of all 25 frames per second, gives about 5× less work. That is still well below typical fixation durations (200–400 ms).
* **Focused crops:** objects are detected only around the child (room video) and around the gaze point (eye-tracking video). That is faster and more accurate for small toys than processing the full frame.
* **Skipping:** frames with no gaze sample (blinks) skip detection, and so does the room video when no child is visible.
* **Sequential decoding**, without random seeking. Small, recent models: YOLOE-26s and YOLO26n-pose, which Ultralytics designed for fast CPU inference.

| Preset | Time points / s | Models | Rough time for a 10-min session* |
|---|---|---|---|
| Fast | 3 | YOLOE-26s, YOLO26n-pose, 512 px | ~8–12 min |
| Balanced (default) | 5 | YOLOE-26s, YOLO26n-pose, 640 px | ~15–25 min |
| Accurate | 10 | YOLOE-26m, YOLO26s-pose, 800 px | ~60+ min |

\*Estimates for a typical 4–8-core laptop; the app shows the real remaining time while it runs. If an NVIDIA GPU with CUDA PyTorch is present, it is used automatically and runs much faster.

---

## Getting the most reliable results

* **Name the toys.** Automatic discovery is convenient, but a list you type gives consistent labels across both videos. Labs usually reuse a fixed toy set, and the list is remembered.
* **Two identical toys** (two red balls) cannot be told apart by name. Add distinguishing words if possible (`red ball`, `blue ball`).
* **Camera placement:** the room camera should see the child's hands and the toys, without the child's back to the camera.
* **Check the review video** for a few minutes of each session, especially the first time you use a new room or camera setup. Also check it whenever the Summary reports low sync confidence.
* If an adult is also in the room, set *Which person is the child?* to *the smallest person in view* under Advanced settings. The default assumes only the child is visible.

### Known limitations

* The *interaction* label is based on hand–object proximity. Holding or touching is detected well, while "playing without touching" (e.g. watching a rolling ball) is not counted as interaction.
* The tracker can confuse the child and an adult when they overlap closely for a long time.
* An object held up to the face may be partly hidden in the room video, so detection relies on smoothing over short gaps.
* With *Count people as gaze targets* turned on, gaze on the child's own hands appears as `person`.

---

## Batch / command line

```bash
# one session
.venv/bin/python analyze.py --ego scene.mp4 --room room.mp4 --gaze gaze.tsv --objects "ball, teddy bear"

# with a confirmed synchronisation, as in the app (the two starts are the same moment)
.venv/bin/python analyze.py --ego scene.mp4 --room room.mp4 --gaze gazedata.gz --ego-start 12.4 --room-start 19.7 --ego-end 604.8

# many sessions: CSV/Excel with columns ego, room, gaze (optional: objects, offset, out)
.venv/bin/python analyze.py --manifest sessions.csv --preset fast
```

On Windows use `.venv\Scripts\python` instead of `.venv/bin/python`. Run `python analyze.py --help` for all options.

## Advanced settings (in the app)

| Setting | Default | When to change it |
|---|---|---|
| Which person is the child? | There is only one person | Choose *the smallest person in view* when an adult is also in the room |
| Hand–object contact | Normal (0.3 torso lengths) | *Strict* (0.2) or *Lenient* (0.45) |
| Gaze accuracy margin | 2% of frame width | Increase if calibration was poor; decrease for tightly packed objects |
| Gaze time offset | 0 s | Only if the gaze export does not start with the scene video |
| Count people as gaze targets | Off | Turn on so that "looking at the parent" shows up as `person` |
| Use fixation samples only | Off | Uses only Pro Lab "Fixation" samples (needs *Eye movement type*; not available in `gazedata.gz`) |

## Project structure

```
app.py                 start the web interface
analyze.py             command line / batch processing
setup_and_run.bat/.sh  one-click install + start
setup_models.py        downloads the models (run by the setup scripts)
gazeplay/
  config.py            settings, presets, label names, vocabularies
  gaze_data.py         gaze loader: Glasses 3 gazedata.gz and Pro Lab TSV/CSV/Excel
  sync.py              audio-based synchronisation
  video_io.py          fast sampled frame reading
  detectors.py         YOLOE + YOLO pose back-end
  room_analysis.py     child selection, hands, hand-object contact
  gaze_analysis.py     gaze-point-to-object mapping
  timeline.py          smoothing, matching, bouts, statistics
  report.py            Excel / CSV / JSON writers
  review_video.py      side-by-side annotated video
  pipeline.py          runs all steps with progress reporting
  web/                 local web interface (Flask + plain HTML/CSS/JS; videosync.js = video boxes, timelines, sync)
tests/                 synthetic end-to-end tests (mock detector), sync-settings tests, browser sync check
```

## Testing

`./run_tests.sh` (or `python tests/test_synthetic.py`) generates a 70-second synthetic session: two videos with a shared sound track and a 7.3 s offset, plus the gaze data in both formats: a Pro Lab-style TSV (decimal commas, blinks, 1920×1080 coordinates) and a Glasses 3 `gazedata.gz` (including a stretch of gaze below the camera view). It then runs the full pipeline with a colour-based stand-in for YOLO and checks the synchronisation and the labels against the ground truth. It needs no models or internet. `tests/test_confirmed_sync.py` checks that the confirmed synchronisation is validated and used exactly. To try the interface with that stand-in: `GAZEPLAY_BACKEND=mock python app.py`. With the stand-in server running, `tests/ui_sync_check.py` runs about 50 browser checks of the synchronisation workflow (requires Playwright).
