# Local person analysis: detection → tracking → long-term ReID → age/gender

Fully local, GPU-only pipeline over `test.mp4`. No cloud APIs, no network calls at
inference time (weights are downloaded once, up front).

```
test.mp4
   ↓  YOLO11-X                person detection
   ↓  ByteTrack               short-term local_track_id  (temporary!)
   ↓  SOLIDER Swin-Base       appearance embedding per crop
   ↓  Active + Lost gallery   persistent anonymous person_name  ("BlueFox")
   ↓  MiVOLO v2               age + gender from face AND body
   ↓  track-level aggregation winsorised age, weighted gender vote
output/result.mp4   output/persons.json   output/reid_events.json   output/reid_debug/
```

The point of the whole design is that **`local_track_id` and `person_name` are
separate things**. ByteTrack's ids are throwaway; identity lives in the ReID
gallery. When someone walks out and comes back three tracks later, the gallery
recognises them from appearance and hands the old name back.

---

## 1. Model choices

| Stage | Model | Why |
|---|---|---|
| Detection | **YOLO11-X** (`weights/yolo11x.pt`) | Current best accuracy in the ultralytics family; ~20 ms/frame at 960 px on an RTX 3080, so the extra accuracy over `l`/`m` is free here. Person class only. |
| Local tracking | **ByteTrack** | Motion/IoU only — deliberately *no* appearance model, so long-term identity cannot leak in through the tracker. `botsort` is selectable in `config.yaml`. |
| ReID | **SOLIDER Swin-Base**, MSMT17 fine-tune (mAP 77.1 / R1 90.7) | MSMT17 (not Market1501): multi-camera, multi-scene, generalises better to arbitrary footage. Base over Small because it measured better *on this clip* (+3 pts recall) and costs nothing — at our batch sizes ReID is kernel-launch bound, not compute bound, so Base runs at the same ~34 ms/crop. `solider_swin_small` / `_tiny` are one config line away. |
| Age + gender | **MiVOLO v2** (`iitolstykh/mivolo_v2`) | Checked for something clearly better; there isn't one that is both open and locally deployable. MiVOLO v2 remains SOTA-class on APPA-REAL/LAGENDA, and crucially it is a *dual-input* model: it takes the face crop **and** the body crop in one forward pass, so it still produces an estimate when the face is turned away — which is most of the time in overhead surveillance. A face-only model would simply abstain on most frames here. |

Everything runs in `torch.inference_mode()` with FP16 autocast.

## 2. Install (Windows, Python 3.10/3.11, CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# MiVOLO's setup.py needs pkg_resources, so it must skip build isolation,
# and --no-deps stops it downgrading ultralytics to 8.1:
pip install wheel
pip install --no-deps --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git

python main.py --download-weights      # yolo11x, yolov8x_person_face, SOLIDER Swin-Base
```

To use a different SOLIDER backbone, change **one** line in `config.yaml`
(`reid.model`) and download the matching checkpoint:

```bash
python main.py --download-weights --reid-model solider_swin_small
```

`reid.weights: auto` resolves to `weights/<reid.model>_msmt17.pth`. If you do set
an explicit path, the loader checks the checkpoint really belongs to the named
architecture and refuses with a clear message rather than a wall of
`size mismatch` errors.

`--download-weights` fills `weights/`. MiVOLO v2 itself is pulled into the
HuggingFace cache the first time you run with demographics enabled.

> `timm==0.8.13.dev0` is pinned because MiVOLO subclasses timm's `VOLO` and
> relies on its exact constructor signature (timm ≥ 0.9 inserted `pos_drop_rate`
> and the positional args shift). `models.py` also shims the two `timm.models`
> symbols MiVOLO imports, so a newer timm *imports* — but the pin is the safe path.

## 3. Run

```bash
python main.py                     # whole video, config.yaml
python main.py --max-frames 300    # quick check
python main.py --no-demographics   # detection + tracking + ReID only
python main.py --input other.mp4 --output output/other.mp4
```

## 4. Files

```
main.py           entry point, CLI, weight downloads
detect_car.py     separate pipeline: vehicle -> license plate -> Korean OCR (section 9)
models.py         model wrappers (detector, face detector, ReID, MiVOLO) + timing
pipeline.py       identity logic: gallery, matching, aggregation, drawing, JSON
solider_swin.py   third-party SOLIDER Swin backbone, vendored verbatim
config.yaml       every threshold
weights/          checkpoints
output/           result.mp4, persons.json, reid_events.json, reid_debug/
```

`solider_swin.py` is the one file that is not project logic — it is the SOLIDER
model definition copied from `tinyvision/SOLIDER-REID` (minus its unused
`mmcv`/`cv2` imports, plus a device fix). It is kept separate so the 1400 lines
of upstream code do not bury the ~700 lines that actually implement the system.

## 5. How re-entry recognition works

1. A new `local_track_id` appears. It has **no identity yet** (drawn as `identifying...`).
2. Each frame it gets a crop-quality score (confidence × resolution × aspect ×
   border-cut × sharpness × occlusion). Bad crops are never used.
3. Once `min_embeddings_to_assign` good embeddings exist they are averaged into
   one query vector.
4. The query is scored against every gallery identity **that is not currently
   owned by another live track** — a person cannot be in two places at once.
   Both ACTIVE and LOST identities are searched.
   * appearance = mean of the top-`match_topk` cosine similarities against that
     person's embedding bank
   * temporal = `exp(-gap / tau)`, spatial = `exp(-(dist/diag) / sigma)`
   * `final = 0.85·appearance + 0.10·temporal + 0.05·spatial` ranks the candidates
5. **Appearance alone gates the decision**: `best ≥ similarity_threshold` *and*
   `best − second_best ≥ second_best_margin`. Time and position only break ties,
   so a match can never be carried by timing or position.
6. Accept → the old `person_name` comes back, `reentry_count` increments, a new
   segment opens, `RE-ID MATCH  Similarity: 0.92` shows for ~1.5 s, and a
   side-by-side crop lands in `output/reid_debug/`.
   Reject → a brand new name. When uncertain the system prefers a false new
   person over merging two different people.
7. **Second look (`reassess_embeddings`).** Step 3 has to decide from two crops
   so the video can show a name immediately — and that is exactly when a
   returning person gets mistaken for somebody new. Once the track has collected
   8 embeddings the query is far more reliable (measured: 0.67 → 0.74 recall at
   the same false-merge rate), so the gallery is asked once more. If the track is
   sitting on an identity it invented for itself and now clearly matches a known
   lost person, the duplicate is **merged** into it — embeddings, age/gender
   samples and segments are transferred and the throwaway name disappears.
   Identities that were already matched, or shared with another track, are never
   touched. On `test.mp4` this recovers 13 of the 18 re-entries.

8. **Duplicate healing.** The ambiguity rule in step 5 has a failure mode that
   compounds: once a person is accidentally stored under two names, every later
   re-entry matches *both* copies about equally, the margin check calls it
   ambiguous, refuses, and creates a **third** copy. Measured on this clip, 85 %
   of ambiguous refusals were of exactly this kind - the two "rival" identities
   were 0.82+ similar *to each other* (the worst pair scored 0.95). So a rival is
   only treated as competing evidence if it is plausibly a different person; if
   its own embedding bank matches the leader's, it is folded into it
   (`duplicate_merged`) and the gallery heals itself.

   Deciding two banks are the same person needs a stricter test than matching a
   single query, because the obvious statistic - the best of up to 400 cross
   pairs - is max-like and easily satisfied by chance. Reusing
   `similarity_threshold` (0.82) here false-merged **12.9 %** of provably
   different people and chained four identities into one. The shipped rule
   requires a high peak **and** broad agreement (`duplicate_peak_similarity`
   0.90 with `duplicate_mean_similarity` 0.76): 83 % of true duplicates caught
   at a 0.7 % false rate.

Each person keeps up to `max_embeddings_per_person` embeddings, not one. When
the bank is full the sample dropped is the one with the worst
`quality × (1 − redundancy)`, so the bank retains genuinely *different* views
(front/side/back, near/far) instead of 20 near-duplicates of one pose.

Lost people are kept for the whole video (`keep_lost_until_video_end`).

## 6. Tuning

The two knobs that matter are in `reid:`:

| | effect |
|---|---|
| `similarity_threshold` ↑ | fewer wrong merges, more duplicate identities |
| `similarity_threshold` ↓ | more re-entries recovered, more wrong merges |
| `second_best_margin` ↑ | refuses ambiguous cases (two similar-looking people) |
| `duplicate_peak/mean_similarity` ↓ | merges duplicates harder — risks collapsing two real people |

When somebody who *should* be recognised keeps getting a new name, turn on
`debug.log_rejections` and `debug.save_reject_crops`. Every refusal is then
written to `reid_events.json` as a `reid_reject` with its top-3 candidates
(including how similar those candidates are to each other), and near-miss pairs
are dumped to `output/reid_debug/rejected/`. The `reason` field separates the two
causes that need opposite fixes: `below_threshold` (lower the threshold / improve
crop quality) vs `ambiguous` (a rival was equally similar — check whether it is a
duplicate of the same person).

The default `0.82 / 0.05` was calibrated on this footage, not guessed. Using
co-occurring tracks as known-different people (they cannot be the same person),
gallery matching at a fixed **2 % wrong-merge rate** gives this recall:

| query embeddings | swin_small | swin_base |
|---|---|---|
| 1 | 0.636 | 0.660 |
| 2 | 0.652 | 0.678 |
| 4 | 0.675 | 0.698 |
| **8** | 0.713 | **0.743** |

Two things fall out of this. The **number of embeddings averaged into the query
matters more than the backbone** (+0.08 from 1→8, vs +0.03 from Small→Base) —
which is what `reassess_embeddings` exploits. And a Small+Base ensemble scored
0.748, statistically indistinguishable from Base alone at twice the ReID cost,
so it was dropped.

Note also `neck_feat: after`. SOLIDER's own eval config uses `before`, but on
this footage the pre-BNNeck features are badly compressed (different people sit
at cosine 0.93, same person at 0.97 — unusable), while the post-BNNeck features
separate cleanly (0.42 vs 0.81). Measured, not assumed.

`tracking.detect_low_conf` matters more than it looks: ByteTrack's whole trick is
its second association pass over *low*-scoring boxes, so the detector runs at
0.10 and `detection.confidence` is applied as ByteTrack's high threshold. Feeding
the tracker only pre-filtered 0.4+ boxes fragments tracks badly.

## 7. Results on `test.mp4` (RTX 3080, 1918×1078, 1086 frames, 36 s)

| | |
|---|---|
| processing | ~83 s end-to-end, **~13.1 fps** |
| detector | 20.0 ms/frame |
| ReID | 34.0 ms/crop (933 crops; small batches, flip-augmented) |
| face detector | 19.1 ms/call (279 calls, not every frame) |
| MiVOLO | 20.7 ms/crop (329 crops) |
| people found | 65 persistent identities, 117 local tracks |
| re-entries recovered | **21 ReID matches**, gaps of 3–26 s |

All 21 side-by-side crops in `output/reid_debug/` were inspected by eye: **19 are
clearly the same person** (`BronzeHarbor` recovered 3 times across tracks
1/11/36/91; `GreenMeadow` 0.92, blue shirt + khaki trousers, gone 15 s;
`JadeStone` 3 times on the grey AERO tee), one is ambiguous (two near-black leg
crops, 0.3 s gap — ID recovery rather than a real re-entry) and one is a genuine
false merge at 0.892.

That false merge survives because it is a real appearance confusion, not a thin
gallery — the identity involved holds 17 embeddings. Pushing
`similarity_threshold` to 0.90 would kill it along with 12 of the 16 correct
matches, so 0.82 is the right operating point for this clip.

Measured back-to-back on the same machine, `merge_duplicates` on vs off:

| | duplicates kept | duplicates healed |
|---|---|---|
| identities | 70 | **65** |
| re-entries recovered | 18 | **21** |
| throughput | 10.57 fps | 10.73 fps |

(The absolute fps here is lower than the 13.1 quoted above only because the
machine was busier during this pair of runs — an isolated detector benchmark was
unchanged at 18.9 ms/frame. The comparison above is apples-to-apples.)

For reference, the original Swin-Small + 2-embedding-query configuration produced
13 matches (~11 correct) and 75 identities.

This is a hard clip: crowded office corridor, overhead angle, heavy mutual
occlusion, many people in similar white/dark office clothing. The remaining
identities still include ~15 visible for under 0.6 s — partially visible people
at the corridor entrance that never produce a usable crop. Raise
`detection.min_box_height` or `reid.min_quality` to suppress them.

## 8. Outputs

`persons.json` — one entry per persistent person: `person_name`, `first_seen`,
`last_seen`, `estimated_age`, `estimated_gender`, `reentry_count`, `segments`,
plus sample counts and the local track ids it used.

`reid_events.json` — `new_person` / `reid_match` / `lost` / `track_end`, each with
time, `person_name`, `local_track_id`, and for matches the `similarity`,
`second_best_similarity` and `gap_seconds`.

`reid_debug/` — `NAME_previous.jpg`, `NAME_reentry_00N.jpg` and a labelled
`NAME_match_00N.jpg` side-by-side, so a wrong match takes one glance to spot.

Per-frame JSON is off by default (`debug.save_frame_results`).

Aggregation, as required, is never single-frame: age is a winsorised
confidence-weighted mean (samples with a visible face count 1.5×), gender is a
confidence-weighted vote that returns `unknown` below
`gender_min_confidence`. Both are stored on the *person*, so they survive
re-entries. The video shows `(?)` until `minimum_samples` is reached.

## 9. License plates — `detect_car.py`

A second, self-contained pipeline in the same repo. Same rules: everything local,
weights downloaded once up front.

```
car_test.mp4
   ↓  YOLO11-X                vehicle detection + ByteTrack  (one id per car)
   ↓  YOLO11-X (plate)        plate detection, run on each *upscaled* vehicle crop
   ↓  PP-OCRv5 korean         text recognition on the plate crop
   ↓  per-track voting        one number per car, not one per frame
output_car/result.mp4   output_car/plates.json
```

```bash
python detect_car.py --download-weights
python detect_car.py                       # car_test.mp4 -> output_car/result.mp4
python detect_car.py --max-frames 300      # quick debug run
```

### Model choices

| Stage | Model | Why |
|---|---|---|
| Vehicles | **YOLO11-X** | Already in `weights/`, and its COCO classes cover car / truck / bus / motorcycle. Only used to produce crops and a track id. |
| Plates | **YOLO11-X fine-tuned on plates** (`morsetechlab/yolov11-license-plate-detection`, `-v1x`) | The strongest openly available plate detector: same YOLO11-X backbone, so it needs no new runtime, and the `x` variant is the most accurate of the six it ships. |
| OCR | **PP-OCRv5, Korean recogniser** (`PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx` + `PP-OCRv5_mobile_det_onnx`) | The plates are Korean, which rules out the plate-specific OCR models — they are trained on Latin charsets. PP-OCRv5 is the current best open Korean text recogniser. It runs as **ONNX through onnxruntime**, not through `paddleocr`: no `paddlepaddle` install, no dependency conflicts, and it shares the GPU with YOLO. |

`onnxruntime-gpu` needs CUDA/cuDNN DLLs that a torch+cu121 install already ships
in `torch/lib`, so `detect_car.py` puts that directory on the DLL search path
rather than asking you to install the CUDA toolkit again. Without it the
recogniser silently falls back to CPU.

### Why two-stage detection

Plates in this footage are **20–60 px wide**. Running the plate detector on the
full 1104×604 frame finds almost nothing, because the letterbox to 640 px shrinks
the plate further. So each vehicle box is cropped and upscaled to ≥640 px wide
first — a 40 px plate becomes ~250 px — and the plate box is then mapped back to
frame coordinates. The OCR crop is taken from the **original** frame, never from
the already-resampled vehicle crop.

The other thing that matters: PP-OCR squashes whatever you hand it to 48 px tall.
A plate box includes the plate's frame and mounting, so the *text* ends up far
smaller than 48 px. `_crop_variants` therefore tries several vertical trims and
lets the plate grammar pick the winner — worth roughly 2× on confidence
(0.42 → 0.86 on the clip's clearest plate).

### The unresolved-syllable rule

A Korean plate is `12가3456`. On a 40 px plate the syllable is an ~8 px glyph, and
the recogniser answers with a Latin look-alike — `85아0527` comes back as
`85a0527` — with the correct syllable nowhere in its top candidates (probability
`<0.001`). Two things follow:

* **Constraining the decoder to the 40 legal plate syllables does not help.** It
  just picks the least-bad of forty near-zero probabilities, and the digits get
  damaged too (`85아0527` → `8580527`). The mask is still used as *one* candidate,
  because it does cleanly strip the junk letters the recogniser bolts onto the
  ends of a reading, but it never gets to invent a syllable.
* **So the slot is marked unknown**: `85?0527`. The digits are read reliably and
  are kept; the syllable is reported as unread rather than guessed. Guessing it
  would be fabricating a number plate.

If some frame *does* resolve the syllable, `merge_votes` folds the `?` readings
into it — but only if the resolved reading earned at least 25% of the unknown
reading's own weight. Without that floor, a single hallucinated syllable captures
every vote that had only ever agreed on the digits, which is exactly what a
1-frame `85러0527` did before the threshold was added.

### Per-track voting, and why the render is a second pass

Nothing is decided from one frame. Every OCR hit votes for its track, weighted by
`OCR confidence × plate grammar × crop quality` (crop quality being Laplacian
sharpness × size, so motion-blurred frames count for little, and frames below
`--min-quality` are not read at all). The winner is drawn in green once it is
well-formed, amber otherwise.

That means the number is only settled once the car has *left*. A single-pass
render would therefore spend the first half of every vehicle showing a number
that is still wrong — the BMW at 77 s reads `52607883` for seventeen frames
before it becomes `26다7883`. So `detect_car.py` analyses the whole clip first
and draws it afterwards, and every car carries the number the whole clip agreed
on from its first frame. The second pass is one more decode of the video plus
drawing — around 100 fps, seconds against minutes of inference.

The panel thumbnail is likewise the *sharpest* crop seen for that car, not the
current frame's.

### Results on `car_test.mp4` (RTX 3080, 1104×604, 3175 frames, 106 s)

Analysis 448 s (7.1 fps); the render pass adds ~40 s at 80 fps. 299 vehicle
tracks → 62 plates read → **20 well-formed → 16 distinct numbers**.

| | plate | first seen | reads |
|---|---|---|---|
| | `07오0437` | 1.6 s | 15 |
| | `191우5787` | 23.5 s | 4 |
| | `84오8311` | 34.3 s | 22 |
| | `55노0734` | 37.8 s | 14 |
| | `80부4349` | 41.2 s | 11 |
| ? | `85?0527` | 49.1 s | 12 |
| | `45호7684` | 51.7 s | 20 |
| ? | `68?4699` | 57.4 s | 13 |
| | `62고7084` | 62.2 s | 24 |
| | `66누8682` | 68.0 s | 27 |
| | `26다7883` | 74.6 s | 21 |
| | `18부6819` | 77.1 s | 30 |
| | `45부9111` | 89.7 s | 12 |
| | `64도3714` | 102.0 s | 15 |

(plus `84호8311`, a 1-read duplicate of `84오8311`, and `수은55181`, a misread bus.)

Two things this shows. The syllable **is** recoverable once a car gets close
enough — 12 of the 14 above resolved it, and the two that did not (`85?0527`,
`68?4699`) are the vehicles that never came nearer than ~45 px of plate. And the
count that matters is the well-formed one: 299 tracks includes every parked car
across the intersection whose plate is a dozen pixels of grey, and the pipeline
correctly declines to read those rather than inventing numbers for them.

### Output

`result.mp4` is the frame plus a right-hand panel listing each vehicle's
**cropped plate image** next to its recognised text, track id, vehicle class and
vote share — the sharpest crop seen for that car, not the current frame's.

`plates.json` — `plates_deduplicated` is the answer to "which cars went past":
one entry per distinct well-formed number, since a car that leaves frame behind a
pillar comes back as a new track id and would otherwise be counted twice.
`plates` keeps the per-track detail: `plate`, `valid_format`,
`syllable_unresolved`, `vote_share`, `reads`, `first_seen_s`, `last_seen_s` and
the runner-up readings.

Plate numbers identify vehicles, not people, and nothing leaves the machine — but
a `?` in a result means *not read*, and an entry with `valid_format: false` is a
partial reading, not a plate number. Neither should be treated as an
identification.

## 10. Privacy

`BlueFox`, `SilverPine` and the rest are randomly generated aliases from two word
lists. They exist only so you can see at a glance whether ReID kept the same
person on the same identity. No facial identification, no attempt to determine
anyone's real name, and nothing leaves the machine. Age and gender are
AI-estimated *appearance* attributes, labelled `estimated_*` for that reason, and
should not be treated as facts about a person.
