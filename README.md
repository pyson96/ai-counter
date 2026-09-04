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
| people found | 70 persistent identities, 117 local tracks |
| re-entries recovered | **18 ReID matches**, gaps of 3–26 s |

All 18 side-by-side crops in `output/reid_debug/` were inspected by eye: **16 are
clearly the same person** (`BronzeHarbor` recovered 3 times across tracks
1/11/36/91; `GreenMeadow` 0.92, blue shirt + khaki trousers, gone 15 s;
`JadeStone` 3 times on the grey AERO tee), one is ambiguous (two near-black leg
crops, 0.3 s gap — ID recovery rather than a real re-entry) and one is a genuine
false merge at 0.892.

That false merge survives because it is a real appearance confusion, not a thin
gallery — the identity involved holds 17 embeddings. Pushing
`similarity_threshold` to 0.90 would kill it along with 12 of the 16 correct
matches, so 0.82 is the right operating point for this clip.

For reference, the earlier Swin-Small + 2-embedding-query configuration produced
13 matches (~11 correct) and 75 identities. The change is +5 correct recoveries
and 5 fewer duplicate identities at identical throughput.

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

## 9. Privacy

`BlueFox`, `SilverPine` and the rest are randomly generated aliases from two word
lists. They exist only so you can see at a glance whether ReID kept the same
person on the same identity. No facial identification, no attempt to determine
anyone's real name, and nothing leaves the machine. Age and gender are
AI-estimated *appearance* attributes, labelled `estimated_*` for that reason, and
should not be treated as facts about a person.
