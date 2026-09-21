# Video Analysis Server

Web front end for the analysis pipelines that already live in this project.
Upload a video from a phone or a PC browser, pick **Person Analysis** or
**Vehicle Analysis**, and get structured results back. The server listens on
**port 7802**.

```
http://SERVER_IP:7802
```

Two processes, on purpose:

```
 phone / PC browser
        |  resumable 32 MB chunks
        v
 FastAPI :7802  ------------------ SQLite (data/analysis.db) ------------------+
   upload, validation, job creation, status APIs, ROI post-processing          |
   never touches the GPU                                                       |
                                                                               |
 worker.py (separate process) <------ FIFO queue -----------------------------+
   claims ONE job at a time, runs it in a child process
        |
        +-- person : YOLO11 -> ByteTrack -> SOLIDER ReID -> MiVOLO
        |            -> tracks.jsonl.gz (every bbox, ~5 Hz, final identities)
        +-- vehicle: YOLO11 -> ByteTrack -> plate YOLO -> PP-OCRv5 -> plate FSM
        |
        v
   SQLite rows + JSON files (+ an annotated video if asked), verified,
   then the source video is deleted
```

**Person analysis needs no ROI up front.** GPU analysis stores every visible
person's bounding box at ~5 samples per second; the Entry/Exit polygons are drawn
*afterwards*, on the result page, and are applied by replaying that stored
history. Redrawing the ROI recomputes entry/exit in a couple of seconds and never
re-runs the GPU - which also means the ROI can still be changed long after the
source video has been deleted.

**The annotated result video is opt-in and off by default.** Tick
**결과 영상 생성** before starting a job and it renders one with the persistent
IDs, the ROIs and a running entry/exit tally burnt in; leave it unticked (the
default) and nothing but structured data is written.

---

## 1. Installation

Python 3.10 or 3.11. From the **project root** (`ai-counter/`):

```bash
# 1. PyTorch with CUDA (RTX 30xx / Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. the analysis stack (ultralytics, opencv, onnxruntime-gpu, ...)
pip install -r requirements.txt

# 3. MiVOLO v2 - must be installed without build isolation and without deps
pip install wheel
pip install --no-deps --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git

# 4. the web server itself
pip install -r video_analysis_server/requirements.txt
```

`ffprobe` is optional. When it is on `PATH` the upload validator records the
codec and container as well; OpenCV is what actually gates validation, because
OpenCV is what the analysis decodes with.

### Model weights

Weights are **shared with the CLI tools** — the server reads them from the
project-root `weights/` directory (`storage.weights_dir` in
`video_analysis_server/config.yaml`, default `../weights`).

```
weights/
├── yolo11x.pt                       person + vehicle detector
├── yolov8x_person_face.pt           face detector feeding MiVOLO
├── solider_swin_base_msmt17.pth     SOLIDER ReID  (matches reid.model)
├── license_plate_yolo11x.pt         license plate detector
└── ocr/
    ├── det.onnx / det.yml           PP-OCRv5 mobile text detection
    └── korean_rec.onnx / .yml       PP-OCRv5 korean recogniser
```

Fetch them with the existing downloaders, from the project root:

```bash
python main.py --download-weights      # person stack (+ SOLIDER for reid.model)
python detect_car.py --download-weights  # plate detector + PP-OCRv5
```

MiVOLO v2 itself is pulled into the HuggingFace cache on first run.

---

## 2. Running

Two commands, two processes. Start them in either order.

```bash
cd video_analysis_server

# web server (port 7802) - no GPU
python server.py

# GPU worker - in a second terminal
python worker.py
```

Useful worker flags:

```bash
python worker.py --once            # take at most one queued job, then exit
python worker.py --poll 1.0        # queue poll interval, seconds
python worker.py --run-job job_x   # run exactly this job in the foreground
```

### Accessing it from a smartphone

The phone and the server must be on the same network.

1. Find the server's LAN address (`ipconfig` on Windows, `ip addr` on Linux).
2. Open `http://<that address>:7802` in the phone's browser.
3. Allow port 7802 through the firewall, once:

   ```powershell
   New-NetFirewallRule -DisplayName "Video Analysis 7802" -Direction Inbound `
       -Protocol TCP -LocalPort 7802 -Action Allow
   ```

`server.host` is `0.0.0.0`, so no other change is needed.

### Directories

| What | Where |
|---|---|
| upload chunks (`upload.part`) | `F:/videos/uploads/<upload_id>/` |
| assembled source video | `F:/videos/uploads/<upload_id>/source.<ext>` |
| JSON results, `tracks.jsonl.gz`, `frame.jpg` | `F:/videos/results/<job_id>/` |
| annotated video (opt-in) | `F:/videos/results/<job_id>/result.mp4` |
| SQLite database | `video_analysis_server/data/analysis.db` |

The root is `storage.video_root` in `config.yaml` (default `F:/videos`). Only the
database stays beside the server: it is small and must be backed up, while
everything under `video_root` is large and per-job.
| model weights | `weights/` (project root) |
| server config | `video_analysis_server/config.yaml` |
| analysis config | `config.yaml` (project root, merged in first) |

---

## 3. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | web interface |
| `GET` | `/result?job=<job_id>` | **result dashboard** (heatmap + charts) |
| `POST` | `/api/uploads/init` | start or **resume** an upload |
| `PUT` | `/api/uploads/{upload_id}/chunk?index=N` | upload one chunk |
| `GET` | `/api/uploads/{upload_id}/status` | upload progress + missing chunks |
| `POST` | `/api/uploads/{upload_id}/complete` | finish and validate the video |
| `GET` | `/api/uploads` | every registered video and its job |
| `DELETE` | `/api/uploads/{upload_id}` | drop a video that is not queued/running |
| `GET` | `/api/uploads/{upload_id}/frame?at=1` | representative JPEG frame (ROI editor) |
| `POST` | `/api/jobs` | create an analysis job |
| `GET` | `/api/jobs` | all jobs + which one holds the GPU |
| `GET` | `/api/jobs/{job_id}` | job status, progress, queue position |
| `POST` | `/api/jobs/{job_id}/person-roi` | **apply** Entry/Exit ROIs to a finished job (recomputes from `tracks.jsonl.gz`; repeatable) |
| `POST` | `/api/jobs/{job_id}/start` | queue a configured job (also: retry) |
| `GET` | `/api/jobs/{job_id}/result` | structured result (summary + rows + events + heatmap) |
| `GET` | `/api/jobs/{job_id}/frame` | the stored first frame (JPEG) |
| `GET` | `/api/jobs/{job_id}/video` | the annotated result video, if one was made |
| `GET` | `/api/jobs/{job_id}/result/{name}.json` | one raw result file |
| `GET` | `/api/health` | port, queue depth, running job, render switch |

Minimal upload flow:

```bash
# 1. init  (returns upload_id, chunk_size, and the chunks still missing)
curl -X POST localhost:7802/api/uploads/init -H 'Content-Type: application/json' \
     -d '{"filename":"festival_01.mp4","file_size":37580963840}'

# 2. one chunk, raw body, streamed to its offset on the server
curl -X PUT "localhost:7802/api/uploads/$ID/chunk?index=0" --data-binary @chunk_0000

# 3. finish + validate
curl -X POST localhost:7802/api/uploads/$ID/complete

# 4. job  (create_video defaults to false - omit it for structured output only)
curl -X POST localhost:7802/api/jobs -H 'Content-Type: application/json' \
     -d '{"upload_id":"'$ID'","analysis_type":"vehicle","create_video":false}'
curl -X POST localhost:7802/api/jobs/$JOB/start -H 'Content-Type: application/json' \
     -d '{"create_video":true}'          # or opt in at start time
```

---

## 4. Job state lifecycle

Upload progress and analysis progress are stored separately, in two tables.

```
uploads.status
  UPLOADING ──► UPLOAD_VERIFYING ──► UPLOADED ──► WAITING_CONFIG
                                                       │
                                                  (job runs)
                                                       ▼
                                                 SOURCE_DELETED
  any step ──► FAILED

jobs.status
  WAITING_CONFIG ──► QUEUED ──► PROCESSING ──► RESULT_VERIFYING ──► COMPLETED
  any step ──► FAILED
```

* `WAITING_CONFIG` — the video is valid; person jobs are waiting for their ROIs.
* `QUEUED` — in the FIFO queue. `queue_position` is 1-based.
* `PROCESSING` — this job owns the GPU. Exactly one job is ever in this state.
* `RESULT_VERIFYING` — analysis done, results being checked before deletion.
* `COMPLETED` → the source video is deleted → the upload becomes `SOURCE_DELETED`.
* `FAILED` — the source video is **kept** so the job can be inspected and retried.

The `jobs` table carries a `priority` column (FIFO orders by
`priority DESC, queue_seq ASC`), so priorities can be added later without a
migration. The first version always writes `priority = 0`.

---

## 5. Resumable upload

**Browser side** (`static/upload.js`): `File.slice(start, end)` produces one
chunk at a time and hands the `Blob` to `fetch()`. The 35 GB file is never read
into memory; peak browser RAM is one chunk.

**Server side** (`server.py`): `/api/uploads/init` creates a **sparse `.part`
file of the full final size** in `data/temp/`. Each `PUT .../chunk?index=N`
seeks to `N * chunk_size` and streams the request body there in ≤ 8 MB blocks —
the request body is never buffered whole either. Only when the whole chunk has
arrived is its bit set in the upload's `chunk_map` (a `'0'`/`'1'` string, one
character per chunk) inside one SQLite transaction.

That map is what makes resume exact:

* A chunk that dies mid-flight leaves its bit `0`, so it is simply re-sent.
* `POST /api/uploads/init` with the same `filename` + `file_size` returns the
  **existing** `upload_id` plus `missing_chunks` and `next_chunk`.
* The browser skips every chunk already marked received.

```
35 GB file, connection dies at 24 GB

  wrong:  restart from 0 GB
  right:  init returns next_chunk = 768  ->  resume at ~24 GB
```

Failed chunks are retried in the browser with exponential backoff (8 attempts).
Beyond that the card shows the error and picking the same file again resumes.

Chunk size is `upload.chunk_size_mb` (default 32 MB); the client may propose its
own in `/api/uploads/init`.

### Upload validation

`POST /api/uploads/{id}/complete` refuses (409) while any chunk is missing.
Otherwise the `.part` file is moved into `data/uploads/` and checked:

* the file exists and its size matches the declared `file_size` exactly
* OpenCV can open the container
* width, height and FPS are sane, and **one real frame decodes**
* frame count and duration are recorded
* `ffprobe`, when installed, adds codec and container name

Pass → `UPLOADED`. Fail → `FAILED`, with the reason, and no job can be created.

---

## 6. Person analysis

The work is split in two, and only the first half needs a GPU:

```
GPU ANALYSIS                            ROI POST-PROCESSING
  source video                            tracks.jsonl.gz
  -> YOLO11 detection                     -> bbox -> ground point
  -> ByteTrack                            -> ROIMonitor state machine
  -> SOLIDER ReID (+ FaceNet fusion)      -> ENTRY / EXIT events
  -> MiVOLO age / gender                  -> SQLite + JSON + dashboard
  -> tracks.jsonl.gz  (~5 Hz bboxes)
  -> verify -> delete the source video    (no decoder, no GPU, repeatable)
```

### Track history (`tracks.jsonl.gz`)

One JSON object per sampled instant, gzipped JSON Lines:

```json
{"frame":15228,"time":507.6,"people":[
  {"person_name":"BlueFox","local_track_id":17,"bbox":[292.3,248.2,426.0,660.1]},
  {"person_name":"SilverBear","local_track_id":21,"bbox":[514.1,231.4,632.2,671.0]}]}
```

* **~5 samples per second of VIDEO time.** The slot index is derived from the
  timestamp (`int(t / 0.2)`), never accumulated, so 25 / 29.97 / 30 / 59.94 fps
  all give exactly 5.00 samples/s with no cumulative drift. Measured: 5.00/s at
  25 fps, 5.02/s at 29.97 fps, 5.00/s at 30 fps.
* **Storage rate only.** Detection, tracking, ReID and MiVOLO keep running at
  their own cadence - a 187-frame clip is still analysed 187 times and merely
  *stored* 32 times.
* **Final ReID identities.** A track's name at sample time is provisional (it can
  be null, or a name later folded into somebody else), so the raw file is
  rewritten against the final `IdentityGallery` before it is compressed - the
  same `gallery.resolve()` rule the ROI events use. `local_track_id` is kept
  alongside for debugging the local tracker.
* **Bounded memory.** Written line-by-line to `tracks.raw.jsonl` during analysis,
  then streamed raw -> gzip and the raw file deleted. Measured on 200 000
  samples: 88 MB raw -> 1.7 MB gzip at a peak Python heap of **0.38 MB**.
* The whole **bbox** is stored, not a ground point, so `ground_band_ratio` can be
  changed later and applied retroactively.

### Applying an ROI

`POST /api/jobs/{job_id}/person-roi` with `{entry_roi, exit_roi}` (polygons in
original video pixels) replays the stored boxes and rewrites the ROI-derived
results only:

| Rewritten | Preserved |
|---|---|
| `person_events` (replaced, never appended) | `estimated_age`, `age_group` |
| `person_results.entry_time` / `exit_time` | `estimated_gender`, `gender_confidence` |
| `summary.json`, `person_events.json` | `first_seen`, `last_seen`, `visible_seconds` |
| `jobs.summary`, `jobs.entry_roi/exit_roi` | `persons.json` demographics, heatmap, frame.jpg |

`database.save_person_roi_results()` exists precisely so that
`save_person_results()` - which rebuilds the person rows from scratch - is never
used for an ROI-only change.

Before any ROI is configured the summary says so explicitly rather than
pretending nobody crossed:

```json
{"roi_configured": false, "entry_count": null, "exit_count": null}
```

### Debounce timing at 5 Hz

The live monitor debounces in frames (`enter_frames: 3` at 30 fps = 0.1 s).
Reusing 3 against 5 Hz samples would mean 0.6 s - six times stricter. The replay
therefore converts the frame thresholds to seconds using the analysed fps, then
back to samples at the storage rate, keeping the behaviour equivalent.



### ROI coordinate format

Both ROIs are polygons in **original video pixels**:

```json
{
  "entry_roi": [[120, 540], [900, 540], [980, 1070], [60, 1070]],
  "exit_roi":  [[1000, 540], [1800, 540], [1860, 1070], [960, 1070]],
  "start": true
}
```

At least 3 points each, any number of vertices, any shape. The ROI editor sizes
its `<canvas>` to the frame's natural resolution and lets CSS scale it for
display, so what the browser posts is already in source-video pixels — nothing
is rescaled on the server.

### The bottom-15 % ground point

The camera sits on a ~2 m tripod and the ROIs are **floor regions**, so the
question is where a person is *standing*, not where their head is. The exact
bottom line `y2` jitters by several pixels per frame, so the bottom band is used
instead of that single line:

```
    x1, y1, x2, y2   = bounding box
    bbox_height      = y2 - y1
    band_start_y     = y1 + bbox_height * (1 - ground_band_ratio)

    ┌───────────────┐
    │               │
    │    person     │
    │               │
    ├───────────────┤  <- band_start_y  (85 % of the box height)
    │  bottom 15 %  │
    │       ●       │  <- the ground point
    └───────────────┘  <- y2

    ground_x = (x1 + x2) / 2
    ground_y = band_start_y + (y2 - band_start_y) / 2
             = y1 + (y2 - y1) * 0.925        (with the default 0.15)
```

This is implemented **once**, in `pipeline.person_ground_point()` at the project
root, and every ROI test in the pipeline calls it. The band width is
configurable:

```yaml
person:
  ground_band_ratio: 0.15
```

### Entry / Exit events and debouncing

A person is counted when their ground point **transitions into** an ROI — not on
every frame they spend inside it. `pipeline.ROIMonitor` runs the state machine:

```
    OUTSIDE ──(inside for enter_frames in a row)──► INSIDE   ← ENTRY / EXIT event
    INSIDE  ──(outside for exit_frames in a row)──► OUTSIDE  ← lane re-arms
```

```yaml
roi:
  enter_frames: 3               # consecutive inside frames before an event
  exit_frames: 5                # consecutive outside frames before the lane re-arms
  min_repeat_seconds: 2.0       # same identity + same ROI may not re-fire faster
  count_first_seen_inside: false
  require_entry_before_exit: true
```

* One or two jittery frames on the polygon edge produce nothing.
* Standing inside the ROI produces nothing after the first event.
* A tracker lane that is *born* inside an ROI starts in `INSIDE` and is not
  counted — that case is almost always an ID switch of somebody already standing
  there. Set `count_first_seen_inside: true` to count it anyway.

### Exit ordering — an exit needs a matching entry

The two ROIs are two lanes of the same state machine, so on their own they fire
independently: anyone who appears in the middle of the scene and then walks into
the exit region would be counted as leaving, even though they never crossed the
entry region. `require_entry_before_exit` (default **true**) closes that:

```
counted as an exit        entry ROI ──► … ──► exit ROI     (same identity)
NOT counted as an exit    (appears anywhere) ──► exit ROI
```

Matching is decided **after** the identities are final, not when the transition
fires — an entry is often recorded before its track has a ReID name, or under a
name that is later merged away, and only the final gallery can say whether an
exit follows an entry by the *same* person.

An exit with no matching entry is **not discarded**. It is written to
`person_events.json` and to the `person_events` table with `matched = false`, and
reported as `summary.unmatched_exit_count`, so the number is auditable instead of
silently vanishing. The usual cause is somebody who was already inside when the
recording started.

Set `require_entry_before_exit: false` to count every exit-region crossing.
That is the right choice only when the entry gate is off-camera — otherwise the
exit total will exceed the entry total for no physical reason.

Note a tracker lane born *inside* the exit ROI is already ignored by
`count_first_seen_inside`, so someone who simply appears there never counted
even before this rule; the rule is about walking in from elsewhere in frame.

Events are attributed to the **persistent SOLIDER ReID identity**, not to the
tracker id. A person who is occluded and comes back with a new `local_track_id`
is still the same `person_name`, so they are not counted twice. Because identity
is assigned a few frames after a track appears (and can later be merged into
another identity), every ROI event is re-attributed to the identity the track
ended up as, and the per-identity de-duplication is applied again on those final
names.

### Age and gender

The existing MiVOLO v2 pipeline, unchanged: it samples a person every
`demographics.interval` frames, slows to `relaxed_interval` once
`enough_samples` are in, and aggregates per persistent identity — a winsorised
confidence-weighted mean for age, a confidence-weighted vote for gender that
returns `unknown` when it is not decisive. MiVOLO never runs on every frame.

Age groups are configurable:

```yaml
person:
  age_groups:
    under_20: [0, 20]
    20s:      [20, 30]
    30s:      [30, 40]
    40s:      [40, 50]
    50s:      [50, 60]
    60_plus:  [60, 200]
```

Gender and age statistics are computed over the people who were actually
**counted in** through the Entry ROI.

---

## 7. Vehicle analysis

No ROI configuration: the **entire frame** is the ROI. The existing vehicle
detector, plate detector and PP-OCRv5 korean recogniser are reused as-is.

### Plate stabilisation

A single OCR frame is never trusted. Every reading votes for its tracked
vehicle, weighted by CTC confidence × plate grammar score × crop quality, and
readings that differ only in an unresolved Hangul syllable are folded into the
resolved one. A plate is only used as an identity once:

```yaml
vehicle:
  min_plate_score: 0.8   # the plate is well-formed
  min_reads: 2           # at least this many OCR votes agree
```

This is why a one-character OCR error does not create a second vehicle.

### ENTRY / LEFT_FRAME / EXIT state machine

Keyed on the **stabilised plate text**, not the tracker id, so one car that
picks up three track ids is still one vehicle:

```
   UNKNOWN
      │ first stable appearance of a new plate
      ▼
   ENTERED  ──── still visible ────► (no event)
      │ absent for lost_seconds
      ▼
   LEFT_FRAME
      │ the same plate appears again
      ▼
   EXITED
      │ absent for lost_seconds
      ▼
   LEFT_FRAME   (the next sighting counts as an ENTRY again)
```

```yaml
vehicle:
  lost_seconds: 3.0      # undetected for this long before LEFT_FRAME
```

A vehicle counts as visible on **every** frame its tracked box is present, even
on frames where plate detection or OCR was not attempted, so a failed detection
frame can never produce `LEFT_FRAME`. Only a genuine `lost_seconds` absence can.

---

## 8. Results

Nothing but structured data is kept.

```
data/results/job_000123/          (person)        data/results/job_000124/   (vehicle)
├── summary.json                                  ├── summary.json
├── persons.json                                  ├── vehicles.json
├── person_events.json                            ├── vehicle_events.json
├── heatmap.json                                  ├── job_metadata.json
├── job_metadata.json                             └── frame.jpg
├── frame.jpg      the video's FIRST frame
└── raw/           pipeline-native persons.json + reid_events.json + roi_events.json
```

`frame.jpg` is the only image the server keeps: one still, taken from the video's
first frame **before** the analysis starts, so the result dashboard still has a
backdrop long after the source video has been deleted. It is a kept result, not a
temporary artifact, and the cleanup step leaves it alone.

SQLite tables: `uploads`, `jobs`, `person_results`, `person_events`,
`vehicle_results`, `vehicle_events`.

`summary.json` (person):

```json
{
  "job_id": "job_000123",
  "analysis_type": "person",
  "summary": {
    "entry_count": 842,
    "exit_count": 795,
    "unmatched_exit_count": 34,
    "require_entry_before_exit": true,
    "gender":     { "male": 412, "female": 401, "unknown": 29 },
    "age_groups": { "under_20": 104, "20s": 235, "30s": 211,
                    "40s": 146, "50s": 91, "60_plus": 55 },
    "entries_by_time": [ { "start_seconds": 0, "end_seconds": 300, "count": 37 } ],
    "exits_by_time":   [ { "start_seconds": 0, "end_seconds": 300, "count": 31 } ]
  },
  "persons": [
    { "person_name": "BlueFox", "entry_time": 142.6, "exit_time": 526.2,
      "estimated_age": 31.7, "age_group": "30s", "estimated_gender": "female" }
  ]
}
```

`summary.json` (vehicle):

```json
{
  "job_id": "job_000124",
  "analysis_type": "vehicle",
  "summary": {
    "entry_count": 124, "exit_count": 117,
    "total_vehicle_entries": 124, "total_vehicle_exits": 117,
    "current_entered_vehicle_count": 7
  },
  "vehicles": [
    { "plate_number": "12가3456", "entry_time": 73.2, "exit_time": 981.4,
      "first_seen": 73.2, "last_seen": 981.4, "status": "EXITED" }
  ]
}
```

This is a temporary festival field survey, so there is deliberately **no**
week-over-week growth and **no** revisit rate. Time-window histograms use
`person.time_bucket_seconds` / `vehicle.time_bucket_seconds` (default 300 s).

---

## 9. Result dashboard (`/result`)

Every completed job gets a dashboard at

```
http://SERVER_IP:7802/result?job=job_000123
```

reachable from the **결과 대시보드** button on the job card. It is drawn entirely
with inline SVG and canvas - **no chart library and no CDN** - because the
festival site may have no internet at all and the page still has to work.

### What it shows (person jobs)

| Block | Form | Why that form |
|---|---|---|
| 총 입장 / 총 퇴장 / 고유 방문객 / 영상 길이 | stat tiles | headline numbers are not a chart |
| unmatched-exit notice | callout | only shown when `unmatched_exit_count > 0` |
| **체류 히트맵** | first frame + sequential heat overlay | continuous magnitude on a grid |
| 시간대별 방문객 | 2 lines, or grouped columns when ≤ 8 buckets | trend over time |
| 성별 분포 | one horizontal stacked bar | part-to-whole |
| 연령대 분포 | columns, one hue | ordered bands; bar length is the magnitude |
| 방문객 목록 | table | more classes than colours can carry |

Vehicle jobs get the same shell: stat tiles (입차 / 출차 / 현재 내부 / 고유 번호판),
the stored frame, the time chart, and the plate table.

### The heatmap

The grid is accumulated during analysis from the **same bottom-band ground point
the Entry/Exit ROIs test** (`pipeline.person_ground_point`), so the picture and
the counts can never disagree. One unit is *one person visible in that cell for
one frame*, i.e. it measures dwell time, not head-count; the dashboard divides by
FPS to label the scale in seconds.

* grid size: `person.heatmap.cols` (default 64); rows follow the video's aspect
* stored as raw integer counts in `heatmap.json` - the viewer scales them, so two
  jobs stay comparable
* the scale runs to the **98th percentile** of non-empty cells, so one very busy
  cell cannot flatten everything else; cells above it clamp to the top step
* empty cells stay fully transparent - where nobody ever stood is information too
* toggles for the heat layer and the ROI outlines, and a table twin that splits
  total dwell across Entry ROI / Exit ROI / elsewhere

Turn it off with `person.heatmap.enabled: false`; the rest of the dashboard is
unaffected.

### Colour

The palette is **computed, not chosen by eye** - it was run through the data-viz
validator against this page's actual surface (`#1a212b`, dark):

| Role | Hex | Used for |
|---|---|---|
| series 1 | `#3987e5` | 입장 / 남성 / 연령대 막대 / Entry ROI |
| series 2 | `#d95926` | 퇴장 / 여성 / Exit ROI |
| no data | `#898781` | 미상 (not measured, so deliberately not a hue) |
| heat ramp | `#184f95 → #9ec5f4` | one blue hue, monotone lightness |

Worst adjacent pair: CVD ΔE 11.3 (target ≥ 8), normal-vision ΔE 16.7 (floor
≥ 15), every mark ≥ 4:1 against the surface. The ROI editor on the upload page
uses the same two hues, so Entry is blue and Exit is orange everywhere.

Nothing is encoded by colour alone: every chart carries a legend and a
**표로 보기** table twin, and the age/gender charts are directly labelled.

---

## 10. Annotated result video (opt-in)

Off by default. The **결과 영상 생성** checkbox sits next to the *Start Analysis*
button on both configuration screens and is **unchecked** every time — it is
never remembered, because rendering costs a second full decode of the video and
that should always be a deliberate choice.

```
POST /api/jobs                     {"create_video": true}
POST /api/jobs/{id}/person-roi     {"create_video": true, ...}
POST /api/jobs/{id}/start          {"create_video": true}
```

The flag lives on the job row (`jobs.create_video`), so two jobs in the same
queue can differ.

### What is drawn

Person jobs render with the project's existing two-pass renderer — pass 1
analyses, pass 2 draws — so every frame shows the **final** identity, not the
one that was guessed in the first second and corrected later. On screen:

* the **persistent anonymous ID** (`BlueFox`), never the tracker id
* estimated age / gender, and the `RE-ID MATCH` banner when someone is recovered
* the **Entry / Exit ROI polygons** that actually did the counting
* the tally obeys `require_entry_before_exit`, so an unmatched exit never moves
  the on-screen `EXIT` number either
* the **bottom-band ground point** — the white dot, plus the 85 % band line, so
  you can see exactly which position was tested against the polygon
* a running **`ENTRY n  EXIT n`** tally, bottom-left, drawn last so no bounding
  box can cover it
* `COUNTED: ENTRY` / `COUNTED: EXIT` on the person's label for ~1.2 s at the
  moment they are counted

Vehicle jobs reuse `detect_car`'s plate renderer (boxes, the settled plate
number, and the plate-crop panel down the right side) and get the same tally.

The counters come from the **resolved** event list — the same one the JSON
reports — so the number burnt into the last frame always equals
`summary.entry_count`. Resolution deliberately runs *before* rendering for that
reason.

### Cost

One extra decode of the whole video, plus encoding. On the test clips that is
roughly **+30-60 %** wall-clock on top of the analysis, and the file is large
(a 6 s 1080p clip renders to ~21 MB; a 106 s clip with the plate panel to
~100 MB). The GPU queue is unaffected — it is still one job at a time.

### Turning the option off entirely

```yaml
storage:
  allow_result_video: false   # the checkbox stops having any effect
```

That key is only the *permission*. A job renders solely because its own
`create_video` flag was set. With the permission off, a job that asks for a
video logs a warning and completes as a structured-output-only job.

Files: `data/results/job_xxx/result.mp4`. It is a kept result —
`keep_result_video: true` — and the cleanup step leaves it alone.

---

## 11. Automatic source deletion

The source video is temporary, but it is never removed before the results are
proven good. `worker.verify_results()` must pass **all** of these:

1. the analysis finished without a fatal error
2. the SQLite transaction committed (`person_results` + `person_events`, or
   `vehicle_results` + `vehicle_events`)
3. all four required JSON files exist and are non-empty
4. every one of them parses as JSON
5. `summary.json` carries the right `job_id` and the required summary fields
   (`entry_count`, `exit_count`, `frames_processed`, plus `gender` and
   `age_groups` for person jobs)
6. the committed row counts are consistent with the summary, and at least one
   frame was processed
7. the result video matches what was asked for: **present and decodable**
   when `create_video` was set, and **absent** when it was not

Only then does `cleanup_job()` delete:

* the assembled source video in `data/uploads/`
* that upload's chunks and `.part` file in `data/temp/`
* `raw/reid_debug/` and any temporary frame/crop directories

and the upload becomes `SOURCE_DELETED`. Kept forever: the SQLite database, the
JSON result files, `frame.jpg`, `result.mp4` when one was requested, the job
metadata and the statistics.

Rendering happens *inside* the analysis run, before verification, so the source
video is always still on disk when the renderer needs it.

To keep sources for debugging, set either switch in `config.yaml`:

```yaml
storage:
  delete_source_after_success: false
  keep_source_video: true
```

---

## 12. Failure recovery

* Analysis runs in a **child process**. A CUDA OOM, a decoding crash or a
  segfault kills that child only; the supervisor records `FAILED` with the
  message and moves on to the next queued job. The GPU memory goes back to the
  OS with the process.
* FastAPI is a **separate process** and never calls an Analyzer, so no analysis
  failure can take the web server down. A job failing does not interrupt
  uploads in flight.
* A `FAILED` job **keeps its source video**. Press *Retry* in the UI (or
  `POST /api/jobs/{job_id}/start`) to put it back in the queue.
* If the worker is killed while a job is running, the next `python worker.py`
  re-queues every job left in `PROCESSING` / `RESULT_VERIFYING`.
* An upload interrupted at any point stays `UPLOADING` with its chunk map
  intact and resumes on the next `/api/uploads/init`.
* Unhandled request errors return a JSON 500 rather than killing the server.

---

## 13. Verifying the result-video switch

The default is off, and that is checkable:

```bash
# 1. no job renders unless its own flag is set
sqlite3 video_analysis_server/data/analysis.db \
        'SELECT job_id, create_video FROM jobs;'

# 2. so only those jobs have a video
ls video_analysis_server/data/results/*/result.mp4

# 3. and every job says which it was
grep -h result_video video_analysis_server/data/results/*/job_metadata.json
#    -> "result_video": null        (box left unchecked)
#    -> "result_video": "result.mp4" (box ticked)

# 4. the verifier enforces both directions
grep -n "did not ask for one\|was requested but none" video_analysis_server/worker.py
```

On Windows PowerShell, (2) is:

```powershell
Get-ChildItem -Recurse video_analysis_server\data\results -Filter result.mp4
```

Mechanically: `analyzer.load_config()` always sets `video.render = false`;
`PersonAnalysis._analysis_config()` turns it back on for the single job whose
`create_video` flag is set, and `VehicleAnalysis` builds its `Renderer` only
then (`no_render=True` otherwise). A job that did not ask for a video and
somehow produced one **fails verification and keeps its source video**, so the
mistake is visible rather than silent.

---

## 14. Configuration

`video_analysis_server/config.yaml` is layered on top of the project-root
`config.yaml` (root first), so every detector / ReID / MiVOLO threshold already
tuned for the CLI tools stays in force.

```yaml
server:   { host: 0.0.0.0, port: 7802 }
upload:   { chunk_size_mb: 32, resumable: true, max_file_gb: 200 }
video:    { max_frames: 0, start_frame: 0 }      # 0 = whole video
analysis: { max_gpu_jobs: 1, base_config: ../config.yaml, progress_interval: 50 }
person:   { ground_band_ratio: 0.15, time_bucket_seconds: 300, age_groups: {...},
            heatmap: { enabled: true, cols: 64 } }
roi:      { enter_frames: 3, exit_frames: 5, min_repeat_seconds: 2.0 }
vehicle:  { lost_seconds: 3.0, min_plate_score: 0.8, min_reads: 2, ... }
storage:
  keep_first_frame: true          # one still per job, the dashboard backdrop
  delete_source_after_success: true
  keep_source_video: false
  allow_result_video: true        # PERMISSION only; each job opts in itself
  keep_result_video: true         # a rendered video survives cleanup
  keep_json: true
  keep_database_results: true
  weights_dir: ../weights
```

`video.max_frames` is a debugging knob — set it to e.g. `300` for a fast
end-to-end smoke test of the whole workflow, then put it back to `0`.

---

## 15. Where the code lives

```
video_analysis_server/
├── server.py                   FastAPI :7802  (uploads, jobs, status; no GPU)
├── worker.py                   single-GPU FIFO worker + per-job child process
├── database.py                 SQLite schema and queries
├── analyzer/
│   ├── __init__.py             sys.path bootstrap + merged config loading
│   ├── models.py               re-export of the root model wrappers
│   ├── solider_swin.py         re-export of the root SOLIDER backbone
│   ├── pipeline.py             person job: ROI counting, age groups, JSON output
│   └── vehicle_pipeline.py     vehicle job: plate ENTRY/LEFT_FRAME/EXIT, JSON
├── static/
│   ├── index.html, upload.js       upload + ROI editor
│   ├── result.html, result.js      the /result dashboard (no chart library)
│   └── style.css
├── data/{temp,uploads,results}/ + analysis.db
├── config.yaml
└── requirements.txt
```

Nothing here re-implements a model. The detector, tracker, SOLIDER ReID, MiVOLO,
vehicle detector and plate OCR all stay in the project-root modules
(`models.py`, `pipeline.py`, `solider_swin.py`, `detect_car.py`), which the CLI
tools (`main.py`, `detect_car.py`) still use unchanged. The web workflow only
added to them:

* `pipeline.person_ground_point()` / `person_ground_band()` — the bottom-band rule
* `pipeline.ROIMonitor` — debounced Entry/Exit transitions
* `Analyzer._update_ground()` / `heatmap_json()` — the dwell heatmap, off the same
  ground point (`person.heatmap.enabled` turns it off)
* `video.render: false` — the switch that skips all rendering
* `Analyzer.progress_cb` — job progress reporting
* `detect_car.PlateReader(no_render=True)` — build no `Renderer`
* `pipeline.draw_roi_polygons()` / `draw_count_bar()` and the ground-point
  marker in `draw_overlay()` — what the opt-in video draws on top of the
  overlay the CLI already produced
