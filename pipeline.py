"""Analysis pipeline: local tracking -> long-term ReID identity -> demographics.

The single most important idea in this file is the separation between

    local_track_id      temporary, produced by ByteTrack/BoT-SORT, may change
                        every time a person leaves and re-enters the frame

    person_name         persistent anonymous identity ("BlueFox"), owned by the
                        ReID gallery and recovered by appearance matching

Everything a persistent person needs - the anonymous name, the embedding bank,
age/gender samples, the ACTIVE/LOST flag, visit segments - lives in PersonState.
Gallery matching, aggregation and drawing all live here too, on purpose: the
whole identity logic is meant to be readable in one pass.

The names are randomly generated aliases. They carry no real-world identity and
no attempt is made to recognise who anybody actually is.
"""

from __future__ import annotations

import json
import os
import random
import time

import cv2
import numpy as np

import models

# ---------------------------------------------------------------------------
# anonymous names
# ---------------------------------------------------------------------------
ADJECTIVES = [
    "Blue", "Silver", "Amber", "Green", "Red", "Golden", "Arctic", "Bright",
    "Crimson", "Violet", "Copper", "Ivory", "Jade", "Coral", "Indigo", "Onyx",
    "Scarlet", "Teal", "Bronze", "Cobalt", "Sunny", "Misty", "Rusty", "Velvet",
]
NOUNS = [
    "Fox", "Pine", "Bird", "Wolf", "Maple", "Bear", "Moon", "River",
    "Hawk", "Cedar", "Otter", "Falcon", "Willow", "Lynx", "Heron", "Aspen",
    "Raven", "Badger", "Comet", "Harbor", "Meadow", "Stone", "Ember", "Lark",
]

# Distinct, high-contrast BGR box colours, cycled by internal id.
PALETTE = [
    (255, 128, 0), (0, 200, 255), (0, 255, 128), (255, 0, 200), (60, 60, 255),
    (255, 255, 0), (128, 0, 255), (0, 165, 255), (180, 255, 0), (255, 0, 90),
    (0, 255, 255), (200, 120, 255), (90, 255, 180), (255, 190, 100), (120, 200, 0),
]


class NameGenerator:
    """Adjective+Noun aliases, guaranteed unique for one analysis run."""

    def __init__(self, seed=1234):
        rng = random.Random(seed)
        self.pool = [a + n for a in ADJECTIVES for n in NOUNS]
        rng.shuffle(self.pool)
        self.used = set()
        self._n = 0

    def next(self):
        while self.pool:
            name = self.pool.pop()
            if name not in self.used:
                self.used.add(name)
                self._n += 1
                return name
        self._n += 1  # pathological case: more people than the word lists allow
        name = f"Person{self._n:03d}"
        self.used.add(name)
        return name


# ---------------------------------------------------------------------------
# crop helpers
# ---------------------------------------------------------------------------
def safe_crop(frame, box, pad=0.0):
    """Crop a bbox from a frame, clipped to the image. Returns None if degenerate."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    if pad:
        dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh
    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(w, round(x2)))
    y2 = int(min(h, round(y2)))
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    return frame[y1:y2, x1:x2]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


def crop_quality(frame, box, conf, others=()):
    """Score in [0, 1] telling how useful this crop is as an appearance sample.

    Combines detection confidence, resolution, body aspect ratio, whether the
    box is cut off by the frame border, image sharpness and occlusion by other
    people. A low score means: do not trust this crop for ReID or age/gender.
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 4 or bh <= 8:
        return 0.0
    fh, fw = frame.shape[:2]

    size = min(1.0, bh / 128.0)

    ratio = bh / max(1.0, bw)
    aspect = 1.0 if 1.5 <= ratio <= 4.5 else max(0.2, 1.0 - abs(ratio - 3.0) / 4.0)

    margin = 3
    cut = (x1 < margin) + (y1 < margin) + (x2 > fw - margin) + (y2 > fh - margin)
    border = (1.0, 0.75, 0.5, 0.4, 0.3)[min(cut, 4)]

    crop = safe_crop(frame, box)
    if crop is None:
        return 0.0
    small = cv2.cvtColor(cv2.resize(crop, (64, 128)), cv2.COLOR_BGR2GRAY)
    sharp = float(np.clip(cv2.Laplacian(small, cv2.CV_32F).var() / 150.0, 0.35, 1.0))

    occl = 1.0
    for other in others:
        if _iou(box, other) > 0.30:
            occl = 0.6
            break

    return float(np.clip(conf * size * aspect * border * sharp * occl, 0.0, 1.0))


def face_for_box(person_box, faces):
    """Pick the face box belonging to a person box (largest face in its upper half)."""
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    best, best_area = None, 0.0
    for fx1, fy1, fx2, fy2, _conf in faces:
        cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        if not (px1 <= cx <= px2 and py1 <= cy <= py1 + 0.55 * ph):
            continue
        area = (fx2 - fx1) * (fy2 - fy1)
        if area > best_area:
            best, best_area = (fx1, fy1, fx2, fy2), area
    return best


# ---------------------------------------------------------------------------
# persistent person identity
# ---------------------------------------------------------------------------
class PersonState:
    """Everything we persistently know about one anonymous person."""

    def __init__(self, name, internal_id, cfg, t_now):
        self.name = name
        self.internal_id = internal_id
        self.cfg = cfg

        self.active = True
        self.current_local_track_id = None
        self.local_track_ids = []

        self.first_seen = t_now
        self.last_seen = t_now
        self.reentry_count = 0
        self.segments = []

        # appearance bank: several embeddings per person (pose / lighting / scale)
        self.embeddings = np.zeros((0, 0), np.float32)
        self.emb_quality = []
        self.last_store_frame = -10 ** 9

        self.age_samples = []       # dicts: age, weight, has_face, time
        self.gender_samples = []    # dicts: gender, conf, weight, time
        self.last_demo_frame = -10 ** 9

        self.last_box = None
        self.last_center = None
        self.best_crop = None
        self.best_crop_quality = 0.0

    # -- lifecycle ---------------------------------------------------------
    def bind(self, local_track_id, t_now):
        self.active = True
        self.current_local_track_id = local_track_id
        self.local_track_ids.append(local_track_id)
        self.segments.append({"start": round(t_now, 3), "end": round(t_now, 3)})
        self.last_seen = t_now

    def touch(self, t_now, box):
        self.last_seen = t_now
        self.last_box = box
        self.last_center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        if self.segments:
            self.segments[-1]["end"] = round(t_now, 3)

    def unbind(self):
        self.active = False
        self.current_local_track_id = None
        if self.segments:
            self.segments[-1]["end"] = round(self.last_seen, 3)

    # -- appearance bank ---------------------------------------------------
    def store_embedding(self, emb, quality, frame_idx, crop=None):
        """Gated insert: only good crops, and not too close together in time."""
        rc = self.cfg["reid"]
        if quality < rc["gallery_min_quality"]:
            return False
        if frame_idx - self.last_store_frame < rc["gallery_min_interval"]:
            return False
        if crop is not None and quality > self.best_crop_quality:
            self.best_crop = crop.copy()
            self.best_crop_quality = quality
        if not self.insert_embedding(emb, quality):
            return False
        self.last_store_frame = frame_idx
        return True

    def insert_embedding(self, emb, quality):
        """Ungated insert into the appearance bank (used when merging people)."""
        rc = self.cfg["reid"]
        cap = int(rc["max_embeddings_per_person"])
        if self.embeddings.shape[0] == 0:
            self.embeddings = emb[None, :].astype(np.float32)
            self.emb_quality = [quality]
        elif self.embeddings.shape[0] < cap:
            self.embeddings = np.vstack([self.embeddings, emb[None, :]])
            self.emb_quality.append(quality)
        else:
            # Bank is full: drop whichever sample is worst once redundancy is
            # taken into account, so the bank keeps DIFFERENT views of the person.
            sims = self.embeddings @ self.embeddings.T
            np.fill_diagonal(sims, -1.0)
            redundancy = sims.max(1)
            value = np.asarray(self.emb_quality) * (1.0 - np.clip(redundancy, 0.0, 1.0))
            worst = int(value.argmin())
            new_red = float(np.max(self.embeddings @ emb))
            if quality * (1.0 - np.clip(new_red, 0.0, 1.0)) <= value[worst]:
                return False
            self.embeddings[worst] = emb
            self.emb_quality[worst] = quality
        return True

    def appearance_score(self, query, topk):
        """Cosine similarity of a query embedding against this person's bank."""
        if self.embeddings.shape[0] == 0:
            return 0.0
        sims = self.embeddings @ query
        k = min(int(topk), sims.shape[0])
        return float(np.sort(sims)[-k:].mean())

    # -- demographics ------------------------------------------------------
    def add_demographics(self, sample, quality, has_face):
        weight = quality * (1.5 if has_face else 1.0)
        self.age_samples.append(
            {"age": sample["age"], "weight": weight, "has_face": bool(has_face), "time": self.last_seen}
        )
        self.gender_samples.append(
            {
                "gender": sample["gender"],
                "conf": sample["gender_conf"],
                "weight": weight * sample["gender_conf"],
                "time": self.last_seen,
            }
        )

    def estimated_age(self):
        """Winsorised, confidence-weighted mean - one bad frame cannot move it."""
        if not self.age_samples:
            return None
        ages = np.array([s["age"] for s in self.age_samples], np.float64)
        weights = np.array([s["weight"] for s in self.age_samples], np.float64)
        trim = float(self.cfg["demographics"].get("age_trim", 0.1))
        if len(ages) >= 5 and trim > 0:
            lo, hi = np.quantile(ages, [trim, 1.0 - trim])
            ages = np.clip(ages, lo, hi)
        if weights.sum() <= 0:
            return float(np.median(ages))
        return float(np.average(ages, weights=weights))

    def estimated_gender(self):
        """Confidence-weighted vote; 'unknown' when the vote is not decisive."""
        if not self.gender_samples:
            return "unknown", 0.0
        votes = {}
        for s in self.gender_samples:
            votes[s["gender"]] = votes.get(s["gender"], 0.0) + s["weight"]
        total = sum(votes.values())
        if total <= 0:
            return "unknown", 0.0
        label, weight = max(votes.items(), key=lambda kv: kv[1])
        ratio = weight / total
        if ratio < float(self.cfg["demographics"].get("gender_min_confidence", 0.6)):
            return "unknown", ratio
        return label, ratio

    def demographics_confident(self):
        return len(self.age_samples) >= int(self.cfg["demographics"].get("minimum_samples", 5))

    def color(self):
        return PALETTE[self.internal_id % len(PALETTE)]

    def to_json(self):
        age = self.estimated_age()
        gender, gconf = self.estimated_gender()
        return {
            "person_name": self.name,
            "internal_id": self.internal_id,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "estimated_age": None if age is None else round(age, 1),
            "estimated_gender": gender,
            "gender_confidence": round(gconf, 3),
            "demographics_confident": self.demographics_confident(),
            "age_sample_count": len(self.age_samples),
            "gender_sample_count": len(self.gender_samples),
            "reentry_count": self.reentry_count,
            "embedding_count": int(self.embeddings.shape[0]),
            "local_track_ids": self.local_track_ids,
            "total_visible_seconds": round(sum(s["end"] - s["start"] for s in self.segments), 3),
            "segments": [{"start": s["start"], "end": s["end"]} for s in self.segments],
        }


class IdentityGallery:
    """Active + lost persistent identities and the appearance matching logic."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.persons = {}
        self.names = NameGenerator(seed=cfg.get("seed", 1234))
        self._next_id = 0

    def create(self, t_now):
        person = PersonState(self.names.next(), self._next_id, self.cfg, t_now)
        self._next_id += 1
        self.persons[person.name] = person
        return person

    def candidates(self, exclude_names):
        return [
            p for p in self.persons.values()
            if p.name not in exclude_names and p.embeddings.shape[0] > 0
        ]

    def match(self, query, t_now, center, frame_diag, exclude_names):
        """Score every candidate identity against one query embedding.

        Appearance is what gates the decision (`similarity_threshold` and
        `second_best_margin` both apply to the cosine similarity). Time since
        disappearance and last known position only re-rank candidates that are
        already appearance-plausible, so a match can never be carried by
        timing or position alone.
        """
        rc = self.cfg["reid"]
        w = rc["score_weights"]
        tau = float(rc["temporal_tau_seconds"])
        sigma = float(rc["spatial_sigma"])
        max_gap = float(rc["spatial_max_gap_seconds"])
        topk = int(rc.get("match_topk", 2))

        scored = []
        for person in self.candidates(exclude_names):
            appearance = person.appearance_score(query, topk)
            gap = max(0.0, t_now - person.last_seen)
            temporal = float(np.exp(-gap / tau))
            if person.last_center is not None and gap <= max_gap:
                dist = float(np.hypot(center[0] - person.last_center[0], center[1] - person.last_center[1]))
                spatial = float(np.exp(-(dist / frame_diag) / sigma))
            else:
                spatial = 0.5  # no usable spatial evidence -> neutral
            final = w["reid"] * appearance + w["temporal"] * temporal + w["spatial"] * spatial
            scored.append(
                {
                    "person": person,
                    "appearance": appearance,
                    "temporal": temporal,
                    "spatial": spatial,
                    "final": final,
                    "gap": gap,
                }
            )
        scored.sort(key=lambda s: s["final"], reverse=True)
        return scored

    def merge(self, source, target, t_now, local_track_id):
        """Fold a duplicate identity into the one it turned out to be.

        Used when a track was named too early from a couple of weak crops and a
        richer query later shows it is somebody the gallery already knows.
        """
        for emb, quality in zip(source.embeddings, source.emb_quality):
            target.insert_embedding(emb, quality)
        target.age_samples.extend(source.age_samples)
        target.gender_samples.extend(source.gender_samples)
        target.segments.extend(source.segments)
        target.segments.sort(key=lambda seg: seg["start"])
        target.first_seen = min(target.first_seen, source.first_seen)
        target.last_seen = max(target.last_seen, source.last_seen)
        target.last_demo_frame = max(target.last_demo_frame, source.last_demo_frame)
        target.reentry_count += 1
        target.active = True
        target.current_local_track_id = local_track_id
        target.local_track_ids.append(local_track_id)
        target.last_box, target.last_center = source.last_box, source.last_center
        self.persons.pop(source.name, None)
        return target

    @staticmethod
    def accept(scored, threshold, margin):
        """Threshold + ambiguity check. When in doubt, refuse (a new person is
        cheaper than merging two different people into one identity)."""
        if not scored:
            return False
        best = scored[0]["appearance"]
        second = scored[1]["appearance"] if len(scored) > 1 else 0.0
        return best >= threshold and (best - second) >= margin


class TrackState:
    """A local tracker lane. Owns no identity of its own - it borrows one."""

    def __init__(self, local_track_id, frame_idx, t_now):
        self.id = local_track_id
        self.person_name = None
        self.first_frame = frame_idx
        self.first_time = t_now
        self.last_frame = frame_idx
        self.box = None
        self.conf = 0.0
        self.quality = 0.0
        self.last_emb_frame = -10 ** 9
        self.pending = []            # (embedding, quality) collected before assignment
        self.history = []            # first N embeddings, for the later re-check
        self.reassessed = False
        self.banner_until = -1.0     # RE-ID MATCH overlay end time
        self.banner_similarity = 0.0


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_DUPLEX


HUD_HEIGHT = 34


def _overlaps(a, b):
    return not (a[0] >= b[2] or b[0] >= a[2] or a[1] >= b[3] or b[1] >= a[3])


def _label_block(frame, box, lines, color, font_scale, top_limit=0, occupied=None):
    """Draw a translucent label panel for `box`.

    Sits above the box when there is room, otherwise below it or inside it, and
    slides down past labels that are already on screen - so in a crowded frame
    the names stay readable instead of stacking on top of each other.
    `lines` = [(text, relative_scale, colour_or_None)].
    """
    pad = 7
    sizes = [cv2.getTextSize(t, FONT, s * font_scale, 1)[0] for t, s, _ in lines]
    width = max(w for w, _ in sizes) + 2 * pad
    height = sum(h + 7 for _, h in sizes) + 2 * pad - 7

    fh, fw = frame.shape[:2]
    x = int(np.clip(box[0], 0, max(0, fw - width - 1)))
    lo, hi = top_limit, max(top_limit, fh - height - 1)

    candidates = [int(box[1]) - height - 4, int(box[3]) + 4, int(box[1]) + 4]
    y = int(np.clip(candidates[0], lo, hi))
    if occupied is not None:
        for base in candidates:
            for step in range(5):
                cand = int(np.clip(base + step * (height + 3), lo, hi))
                rect = (x, cand, x + width, cand + height)
                if not any(_overlaps(rect, o) for o in occupied):
                    y = cand
                    break
            else:
                continue
            break
        occupied.append((x, y, x + width, y + height))

    x2, y2 = min(fw, x + width), min(fh, y + height)
    if x2 <= x or y2 <= y:
        return
    region = frame[y:y2, x:x2]
    shade = np.full_like(region, 22)
    cv2.addWeighted(shade, 0.72, region, 0.28, 0, region)
    cv2.rectangle(frame, (x, y), (x2 - 1, y2 - 1), color, 1)
    cv2.rectangle(frame, (x, y), (x + 4, y2 - 1), color, -1)

    cy = y + pad
    for (text, scale, tcolor), (_, th) in zip(lines, sizes):
        cy += th
        cv2.putText(frame, text, (x + pad + 4, cy), FONT, scale * font_scale,
                    tcolor or (245, 245, 245), 1, cv2.LINE_AA)
        cy += 7


def draw_overlay(frame, tracks, gallery, live_ids, t_now, cfg, frame_idx, fps):
    vis = cfg["visualization"]
    fs = float(vis.get("font_scale", 0.75))
    th = int(vis.get("thickness", 2))

    top = 0
    if vis.get("show_hud", True):
        top = HUD_HEIGHT
        n_active = sum(1 for p in gallery.persons.values() if p.active)
        hud = (f"frame {frame_idx}   t={t_now:6.2f}s   people seen: {len(gallery.persons)}"
               f"   active: {n_active}   lost: {len(gallery.persons) - n_active}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], HUD_HEIGHT), (20, 20, 20), -1)
        cv2.putText(frame, hud, (12, 23), FONT, 0.62, (240, 240, 240), 1, cv2.LINE_AA)

    occupied = []
    order = sorted(
        (t for t in live_ids if tracks[t].box is not None),
        key=lambda t: (tracks[t].box[3] - tracks[t].box[1]),
    )
    for tid in order:
        tr = tracks[tid]
        if tr.box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in tr.box]

        if tr.person_name is None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 140, 140), 1)
            _label_block(frame, (x1, y1, x2, y2), [("identifying...", 0.85, (200, 200, 200))],
                         (140, 140, 140), fs, top, occupied)
            continue

        person = gallery.persons[tr.person_name]
        color = person.color()
        banner = vis.get("show_reid_match", True) and t_now <= tr.banner_until
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, th + (1 if banner else 0))

        lines = []
        if vis.get("show_person_name", True):
            lines.append((person.name, 1.25, (255, 255, 255)))
        if banner:
            lines.append(("RE-ID MATCH", 1.0, (0, 235, 255)))
            lines.append((f"Similarity: {tr.banner_similarity:.2f}", 0.85, (0, 235, 255)))
        if vis.get("show_age", True):
            age = person.estimated_age()
            if age is None:
                lines.append(("Age: --", 0.9, None))
            else:
                mark = "" if person.demographics_confident() else " (?)"
                lines.append((f"Age: ~{age:.0f}{mark}", 0.9, None))
        if vis.get("show_gender", True):
            gender, _ = person.estimated_gender()
            lines.append((f"Gender: {gender.capitalize()}", 0.9, None))
        if person.reentry_count:
            lines.append((f"re-entries: {person.reentry_count}", 0.8, (190, 190, 190)))
        if vis.get("show_local_track_id", False):
            lines.append((f"local id {tr.id}", 0.75, (170, 170, 170)))

        _label_block(frame, (x1, y1, x2, y2), lines, color, fs, top, occupied)

    return frame


def _save_match_debug(debug_dir, person, prev_crop, new_crop, similarity, index):
    """Previous crop | new crop, side by side, so a wrong match is obvious."""
    os.makedirs(debug_dir, exist_ok=True)
    if new_crop is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{person.name}_reentry_{index:03d}.jpg"), new_crop)
    if prev_crop is None or new_crop is None:
        return
    cv2.imwrite(os.path.join(debug_dir, f"{person.name}_previous.jpg"), prev_crop)

    height = 320
    def _fit(img):
        scale = height / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * scale)), height))

    left, right = _fit(prev_crop), _fit(new_crop)
    gap = np.full((height, 8, 3), 40, np.uint8)
    body = np.hstack([left, gap, right])
    header = np.full((56, body.shape[1], 3), 30, np.uint8)
    cv2.putText(header, f"{person.name}   similarity {similarity:.3f}", (10, 24),
                FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "previous  |  re-entry", (10, 46), FONT, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(debug_dir, f"{person.name}_match_{index:03d}.jpg"),
                np.vstack([header, body]))


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------
class Analyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = models.Stats()
        self.device = models.resolve_device(cfg)

        out_dir = cfg["video"]["output_dir"]
        os.makedirs(out_dir, exist_ok=True)

        print(f"[init] device: {self.device}")
        self.detector = models.PersonDetector(cfg, self.device, self.stats)
        self.reid = models.ReIDExtractor(cfg, self.device, self.stats)
        self.demographics = None
        self.face_detector = None
        if cfg["demographics"].get("enabled", True):
            self.face_detector = models.FaceDetector(cfg, self.device, self.stats)
            self.demographics = models.DemographicsEstimator(cfg, self.device, self.stats)

        self.gallery = IdentityGallery(cfg)
        self.tracks = {}
        self.events = []
        self.frame_records = []
        self.match_counter = {}

    # -- events ------------------------------------------------------------
    def _event(self, t_now, kind, person=None, local_track_id=None, **extra):
        ev = {"time": round(t_now, 3), "event": kind}
        if person is not None:
            ev["person_name"] = person.name
        if local_track_id is not None:
            ev["local_track_id"] = int(local_track_id)
        ev.update(extra)
        self.events.append(ev)
        if self.cfg["debug"].get("verbose", True):
            detail = " ".join(f"{k}={v}" for k, v in extra.items())
            print(f"[{kind.upper():<11}] t={t_now:7.2f}s  "
                  f"{(person.name if person else '-'):<12} local_track_id={local_track_id} {detail}")
        return ev

    # -- identity assignment ----------------------------------------------
    def _busy_names(self, live_ids, own_track_id):
        """Identities another live track already owns - a person cannot be in
        two places at once, so these are never match candidates."""
        return {
            p.name for p in self.gallery.persons.values()
            if p.active and p.current_local_track_id in live_ids
            and p.current_local_track_id != own_track_id
        }

    def _reassess_identity(self, tr, frame, t_now, live_ids, frame_diag):
        """Second look once the track has collected enough embeddings.

        The first assignment has to be made from one or two crops so the video
        can show a name straight away, and that is exactly when a returning
        person gets mistaken for somebody new. Averaging ~8 embeddings makes the
        query far more reliable (measured: 0.67 -> 0.74 recall at the same
        false-merge rate), so we ask the gallery once more. Only an identity
        THIS track invented for itself may be corrected - an identity that was
        already matched to somebody, or is shared with another track, is left alone.
        """
        tr.reassessed = True
        person = self.gallery.persons.get(tr.person_name)
        if person is None or person.local_track_ids != [tr.id]:
            return

        rc = self.cfg["reid"]
        query = np.mean([e for e, _ in tr.history], axis=0)
        query /= max(1e-9, np.linalg.norm(query))
        center = ((tr.box[0] + tr.box[2]) / 2.0, (tr.box[1] + tr.box[3]) / 2.0)
        busy = self._busy_names(live_ids, tr.id) | {person.name}

        scored = self.gallery.match(query, t_now, center, frame_diag, busy)
        if not self.gallery.accept(scored, rc["similarity_threshold"], rc["second_best_margin"]):
            return

        best = scored[0]
        target, second = best["person"], (scored[1]["appearance"] if len(scored) > 1 else 0.0)
        prev_crop, new_crop = target.best_crop, safe_crop(frame, tr.box)
        merged = self.gallery.merge(person, target, t_now, tr.id)

        tr.person_name = merged.name
        tr.banner_until = t_now + float(self.cfg["visualization"].get("reid_match_seconds", 1.5))
        tr.banner_similarity = best["appearance"]
        self._event(
            t_now, "reid_match", merged, tr.id,
            similarity=round(best["appearance"], 4),
            second_best_similarity=round(second, 4),
            final_score=round(best["final"], 4),
            gap_seconds=round(best["gap"], 2),
            was_lost=True,
            reentry_count=merged.reentry_count,
            corrected_from=person.name,
            embeddings_used=len(tr.history),
        )
        if self.cfg["debug"].get("save_reid_crops", True):
            idx = self.match_counter.get(merged.name, 0) + 1
            self.match_counter[merged.name] = idx
            _save_match_debug(self.cfg["debug"]["reid_debug_dir"], merged,
                              prev_crop, new_crop, best["appearance"], idx)

    def _assign_identity(self, tr, frame, t_now, live_ids, frame_diag):
        """Give an unassigned local track a persistent identity (old or new)."""
        rc = self.cfg["reid"]
        query = np.mean([e for e, _ in tr.pending], axis=0)
        query /= max(1e-9, np.linalg.norm(query))
        best_quality = max(q for _, q in tr.pending)
        center = ((tr.box[0] + tr.box[2]) / 2.0, (tr.box[1] + tr.box[3]) / 2.0)

        busy = self._busy_names(live_ids, tr.id)
        scored = self.gallery.match(query, t_now, center, frame_diag, busy)
        best = scored[0] if scored else None
        second = scored[1]["appearance"] if len(scored) > 1 else 0.0
        accepted = self.gallery.accept(scored, rc["similarity_threshold"], rc["second_best_margin"])

        crop = safe_crop(frame, tr.box)
        if accepted:
            person = best["person"]
            was_lost = not person.active
            gap = best["gap"]
            prev_crop = person.best_crop
            person.bind(tr.id, t_now)
            if gap >= float(rc["min_reentry_gap_seconds"]):
                person.reentry_count += 1
            tr.banner_until = t_now + float(self.cfg["visualization"].get("reid_match_seconds", 1.5))
            tr.banner_similarity = best["appearance"]
            self._event(
                t_now, "reid_match", person, tr.id,
                similarity=round(best["appearance"], 4),
                second_best_similarity=round(second, 4),
                final_score=round(best["final"], 4),
                gap_seconds=round(gap, 2),
                was_lost=bool(was_lost),
                reentry_count=person.reentry_count,
            )
            if self.cfg["debug"].get("save_reid_crops", True):
                idx = self.match_counter.get(person.name, 0) + 1
                self.match_counter[person.name] = idx
                _save_match_debug(self.cfg["debug"]["reid_debug_dir"], person,
                                  prev_crop, crop, best["appearance"], idx)
        else:
            person = self.gallery.create(t_now)
            person.bind(tr.id, t_now)
            self._event(
                t_now, "new_person", person, tr.id,
                best_similarity=round(best["appearance"], 4) if best else 0.0,
                best_candidate=best["person"].name if best else None,
                reason="below_threshold" if best else "empty_gallery",
            )

        tr.person_name = person.name
        for emb, quality in tr.pending:
            person.store_embedding(emb, quality, tr.last_frame, crop if quality == best_quality else None)
        tr.history = list(tr.pending)
        tr.pending.clear()
        return person

    # -- demographics ------------------------------------------------------
    def _run_demographics(self, frame, live_ids, frame_idx, faces_cache):
        dc = self.cfg["demographics"]
        due = []
        for tid in live_ids:
            tr = self.tracks[tid]
            if tr.person_name is None or tr.quality < float(dc["min_quality"]):
                continue
            person = self.gallery.persons[tr.person_name]
            if len(person.age_samples) >= int(dc["max_samples"]):
                continue
            interval = int(dc["interval"])
            if len(person.age_samples) >= int(dc["enough_samples"]):
                interval = int(dc["relaxed_interval"])
            if frame_idx - person.last_demo_frame < interval:
                continue
            due.append(tr)
        if not due:
            return

        if faces_cache[0] is None:
            faces_cache[0] = self.face_detector.detect(frame)
        faces = faces_cache[0]

        body_crops, face_crops, owners, has_face = [], [], [], []
        for tr in due:
            body = safe_crop(frame, tr.box)
            if body is None:
                continue
            fbox = face_for_box(tr.box, faces)
            face = safe_crop(frame, fbox, pad=0.1) if fbox is not None else None
            body_crops.append(body)
            face_crops.append(face)
            owners.append(tr)
            has_face.append(face is not None)

        for tr, sample, hf in zip(owners, self.demographics.predict(face_crops, body_crops), has_face):
            person = self.gallery.persons[tr.person_name]
            person.add_demographics(sample, tr.quality, hf)
            person.last_demo_frame = frame_idx

    # -- main loop ---------------------------------------------------------
    def run(self):
        cfg = self.cfg
        vcfg = cfg["video"]
        cap = cv2.VideoCapture(vcfg["input"])
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {vcfg['input']}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_diag = float(np.hypot(width, height))
        start_frame = int(vcfg.get("start_frame", 0))
        max_frames = int(vcfg.get("max_frames", 0))
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer = cv2.VideoWriter(vcfg["output"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {vcfg['output']}")

        print(f"[video] {vcfg['input']}  {width}x{height} @ {fps:.2f}fps  {total} frames")
        print(f"[video] writing {vcfg['output']}")

        rc = cfg["reid"]
        reassess_n = int(rc.get("reassess_embeddings", 8))
        max_age = int(cfg["tracking"]["max_age_frames"])
        frame_idx = start_frame
        processed = 0
        wall0 = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and processed >= max_frames:
                break
            t_now = frame_idx / fps

            detections = self.detector.track(frame)
            boxes = [d["xyxy"] for d in detections]
            live_ids = []

            # 1) refresh local tracks
            for det in detections:
                tid = det["local_track_id"]
                tr = self.tracks.get(tid)
                if tr is None:
                    tr = TrackState(tid, frame_idx, t_now)
                    self.tracks[tid] = tr
                tr.box = det["xyxy"]
                tr.conf = det["conf"]
                tr.last_frame = frame_idx
                others = [b for b in boxes if b is not det["xyxy"]]
                tr.quality = crop_quality(frame, tr.box, tr.conf, others)
                live_ids.append(tid)
                if tr.person_name is not None:
                    self.gallery.persons[tr.person_name].touch(t_now, tr.box)

            # 2) which tracks want a ReID feature this frame?
            need = []
            for tid in live_ids:
                tr = self.tracks[tid]
                if tr.quality < float(rc["min_quality"]):
                    continue
                if tr.person_name is None:
                    need.append(tr)                       # unidentified: try every frame
                elif frame_idx - tr.last_emb_frame >= int(rc["extraction_interval"]):
                    need.append(tr)

            if need:
                crops = [safe_crop(frame, tr.box) for tr in need]
                keep = [(tr, c) for tr, c in zip(need, crops) if c is not None]
                if keep:
                    embeddings = self.reid.extract([c for _, c in keep])
                    for (tr, crop), emb in zip(keep, embeddings):
                        tr.last_emb_frame = frame_idx
                        if tr.person_name is None:
                            tr.pending.append((emb, tr.quality))
                            if len(tr.pending) >= int(rc["min_embeddings_to_assign"]):
                                self._assign_identity(tr, frame, t_now, live_ids, frame_diag)
                        else:
                            self.gallery.persons[tr.person_name].store_embedding(
                                emb, tr.quality, frame_idx, crop
                            )
                            if len(tr.history) < reassess_n:
                                tr.history.append((emb, tr.quality))
                            if not tr.reassessed and len(tr.history) >= reassess_n:
                                self._reassess_identity(tr, frame, t_now, live_ids, frame_diag)

            # 3) age / gender (at most one face-detector pass per frame)
            if self.demographics is not None:
                self._run_demographics(frame, live_ids, frame_idx, [None])

            # 4) retire local tracks; their identity survives in the gallery
            for tid in list(self.tracks):
                tr = self.tracks[tid]
                if tid in live_ids or frame_idx - tr.last_frame <= max_age:
                    continue
                gone_at = tr.last_frame / fps
                self._event(gone_at, "track_end", None, tid,
                            person_name=tr.person_name,
                            frames=tr.last_frame - tr.first_frame + 1)
                if tr.person_name is not None:
                    person = self.gallery.persons[tr.person_name]
                    if person.current_local_track_id == tid:
                        person.unbind()
                        self._event(gone_at, "lost", person, tid,
                                    visible_seconds=round(person.segments[-1]["end"]
                                                          - person.segments[-1]["start"], 2))
                del self.tracks[tid]

            # 5) draw + write
            draw_overlay(frame, self.tracks, self.gallery, live_ids, t_now, cfg, frame_idx, fps)
            writer.write(frame)

            if cfg["debug"].get("save_frame_results", False):
                self.frame_records.append({
                    "frame": frame_idx,
                    "time": round(t_now, 3),
                    "people": [
                        {
                            "person_name": self.tracks[t].person_name,
                            "local_track_id": t,
                            "bbox": [round(float(v), 1) for v in self.tracks[t].box],
                            "confidence": round(self.tracks[t].conf, 3),
                            "crop_quality": round(self.tracks[t].quality, 3),
                        }
                        for t in live_ids
                    ],
                })

            frame_idx += 1
            processed += 1
            if processed % 50 == 0:
                elapsed = time.perf_counter() - wall0
                print(f"  ... {processed} frames  ({processed / elapsed:.1f} fps)  "
                      f"people={len(self.gallery.persons)}")

        cap.release()
        writer.release()

        # close whatever is still open at the end of the video
        end_t = (frame_idx - 1) / fps
        for tr in self.tracks.values():
            if tr.person_name is not None:
                person = self.gallery.persons[tr.person_name]
                if person.current_local_track_id == tr.id:
                    person.unbind()
                    self._event(end_t, "track_end", person, tr.id, reason="video_end")

        wall = time.perf_counter() - wall0
        return self._finish(processed, wall, fps)

    # -- outputs -----------------------------------------------------------
    def _finish(self, processed, wall, fps):
        cfg = self.cfg
        out_dir = cfg["video"]["output_dir"]
        persons = sorted(self.gallery.persons.values(), key=lambda p: p.first_seen)

        persons_path = os.path.join(out_dir, "persons.json")
        with open(persons_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "video": os.path.basename(cfg["video"]["input"]),
                    "frames_processed": processed,
                    "fps": round(fps, 3),
                    "duration_seconds": round(processed / fps, 3),
                    "person_count": len(persons),
                    "models": {
                        "detector": cfg["detection"]["model"],
                        "tracker": cfg["tracking"]["tracker"],
                        "reid": cfg["reid"]["model"],
                        "demographics": cfg["demographics"]["model"] if self.demographics else None,
                    },
                    "note": "person_name values are randomly generated anonymous aliases; "
                            "age and gender are AI-estimated appearance attributes, not identity claims.",
                    "persons": [p.to_json() for p in persons],
                },
                fh, indent=2,
            )

        events_path = os.path.join(out_dir, "reid_events.json")
        with open(events_path, "w", encoding="utf-8") as fh:
            json.dump(self.events, fh, indent=2)

        if cfg["debug"].get("save_frame_results", False):
            with open(os.path.join(out_dir, "frames.json"), "w", encoding="utf-8") as fh:
                json.dump(self.frame_records, fh)

        report = self.stats.report()
        n_reentry = sum(1 for e in self.events if e["event"] == "reid_match")

        print("\n" + "=" * 74)
        print(f"  input fps                : {fps:.2f}")
        print(f"  frames processed         : {processed}")
        print(f"  total processing time    : {wall:.1f} s")
        print(f"  processing fps           : {processed / max(1e-9, wall):.2f}")
        for name in ("detector", "reid", "face_detector", "demographics"):
            if name in report:
                r = report[name]
                print(f"  {name:<24} : {r['total_s']:7.1f} s   {r['ms_per_call']:6.1f} ms/call   "
                      f"{r['ms_per_item']:6.1f} ms/item   ({r['items']} items)")
        print("-" * 74)
        print(f"  persistent people        : {len(persons)}")
        print(f"  re-id matches (re-entry) : {n_reentry}")
        for p in persons:
            age = p.estimated_age()
            gender, _ = p.estimated_gender()
            print(f"    {p.name:<14} {p.first_seen:6.2f}s - {p.last_seen:6.2f}s  "
                  f"segments={len(p.segments)} reentries={p.reentry_count}  "
                  f"age={'--' if age is None else f'{age:.1f}'} gender={gender} "
                  f"(emb={p.embeddings.shape[0]}, demo_samples={len(p.age_samples)})")
        print("=" * 74)
        print(f"  {cfg['video']['output']}\n  {persons_path}\n  {events_path}")

        return {
            "persons": [p.to_json() for p in persons],
            "events": self.events,
            "performance": {
                "input_fps": round(fps, 3),
                "frames_processed": processed,
                "processing_seconds": round(wall, 3),
                "processing_fps": round(processed / max(1e-9, wall), 3),
                "stages": report,
            },
        }
