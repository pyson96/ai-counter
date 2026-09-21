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

import gzip
import json
import math
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
# ground position + Entry/Exit ROI state machine
# ---------------------------------------------------------------------------

DEFAULT_GROUND_BAND_RATIO = 0.15


def person_ground_point(box, band_ratio=DEFAULT_GROUND_BAND_RATIO):
    """Where the person is STANDING, from their bounding box.

    The camera sits on a ~2 m tripod and the Entry/Exit ROIs are floor regions,
    so the head is the wrong thing to test against them - the feet are what is
    inside the polygon. The exact bottom line y2 jitters by several pixels every
    frame, so instead of that single line we take the bottom `band_ratio` slice
    of the box and use its centre:

        +---------------+
        |               |
        |    person     |
        +---------------+  <- y1 + h * (1 - band_ratio)
        | bottom band  o|  <- the returned point sits in the middle of it
        +---------------+  <- y2

    With the default 0.15 that is (cx, y1 + h * 0.925). One helper, used by the
    whole pipeline - do not recompute this rule anywhere else.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    height = y2 - y1
    band_ratio = min(max(float(band_ratio), 0.0), 1.0)
    band_start = y1 + height * (1.0 - band_ratio)
    return ((x1 + x2) / 2.0, band_start + (y2 - band_start) / 2.0)


def person_ground_band(box, band_ratio=DEFAULT_GROUND_BAND_RATIO):
    """The bottom-band rectangle itself (x1, band_start_y, x2, y2)."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    band_ratio = min(max(float(band_ratio), 0.0), 1.0)
    return (x1, y1 + (y2 - y1) * (1.0 - band_ratio), x2, y2)


def point_in_polygon(point, polygon):
    """Even-odd ray casting. `polygon` is [[x, y], ...] in video pixels."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if (yi > y) != (yj > y):
            cross = xi + (y - yi) * (xj - xi) / ((yj - yi) or 1e-9)
            if x < cross:
                inside = not inside
        j = i
    return inside


DEFAULT_LINE_HYSTERESIS_PX = 8.0
DEFAULT_LINE_CONFIRM = 3


def line_segments(points):
    """[[x,y], ...] -> [((ax,ay),(bx,by)), ...]. Fewer than 2 points = no line."""
    pts = [(float(p[0]), float(p[1])) for p in (points or []) if len(p) >= 2]
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def signed_side_of_line(point, points):
    """Which side of the line a point falls on: +1, -1, or 0 when undecidable.

    0 means "do not judge": the point is past the end of the line (they walked
    around it, they did not cross it) or exactly on it. Distance to the nearest
    segment decides which segment's orientation applies, so a bent line works
    the same as a straight one.

    The sign is arbitrary but CONSISTENT - which of the two is "inside" is the
    user's choice, stored alongside the line.
    """
    best = None
    px, py = float(point[0]), float(point[1])
    segments = line_segments(points)
    last = len(segments) - 1
    for idx, ((ax, ay), (bx, by)) in enumerate(segments):
        vx, vy = bx - ax, by - ay
        length2 = vx * vx + vy * vy
        if length2 <= 0.0:
            continue
        t = ((px - ax) * vx + (py - ay) * vy) / length2
        clamped = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        dx, dy = px - (ax + vx * clamped), py - (ay + vy * clamped)
        distance = math.hypot(dx, dy)
        if best is None or distance < best[0]:
            best = (distance, vx * (py - ay) - vy * (px - ax), t, idx)
    if best is None:
        return 0, 0.0
    distance, cross, t, idx = best
    # Only the two ENDS of the whole line are blind: past them somebody walked
    # around the line rather than through it. An interior corner is not a gap -
    # on the outside of a bend no segment has the point within its span, and
    # refusing to judge there left a wedge where an approach went unseen, so
    # the crossing itself was never the first thing observed.
    if (idx == 0 and t < 0.0) or (idx == last and t > 1.0):
        return 0, distance
    if cross == 0.0:
        return 0, distance
    return (1 if cross > 0 else -1), distance


class CrossingLine:
    """One line with an inside and an outside; crossing it is the event.

        outside -> inside   ENTRY
        inside  -> outside  EXIT

    This replaces the two-polygon scheme. A doorway is a threshold, not two
    areas: with polygons the counts depended on whether somebody happened to
    stand inside a region long enough, and a person walking straight through
    could register both, either or neither. A line has no interior to linger
    in - you are on one side or the other, and only changing sides counts.

    Two kinds of jitter are handled:

    * the box wobbles on the line itself. `hysteresis_px` is a dead band around
      the line where no judgement is made at all, so a foot placed on the
      threshold does not produce a burst of crossings;
    * one bad frame puts the box on the far side. `confirm_frames` consecutive
      observations on the new side are required before the side actually flips.

    Walking around the END of the line is not a crossing, so observations past
    either endpoint are ignored (see signed_side_of_line).

    State is per local track, because that is what is observable frame to
    frame; the caller re-attributes the event to the persistent ReID identity,
    which may only be assigned seconds later.
    """

    def __init__(self, points, inside=1, confirm_frames=DEFAULT_LINE_CONFIRM,
                 hysteresis_px=DEFAULT_LINE_HYSTERESIS_PX):
        self.points = [[float(p[0]), float(p[1])] for p in (points or []) if len(p) >= 2]
        self.inside = 1 if int(inside or 1) >= 0 else -1
        self.confirm_frames = max(1, int(confirm_frames))
        self.hysteresis_px = max(0.0, float(hysteresis_px))
        self.states = {}                 # local_track_id -> lane

    @property
    def active(self):
        return len(self.points) >= 2

    def side_of(self, point):
        """+1 / -1 / 0, with the dead band applied."""
        side, distance = signed_side_of_line(point, self.points)
        if side == 0 or distance < self.hysteresis_px:
            return 0
        return side

    def label(self, side):
        return "entry" if side == self.inside else "exit"

    def update(self, track_id, point, t_now=None):
        """Feed one observation. Returns "entry", "exit", or None.

        The first decisive observation of a track only establishes which side it
        started on - somebody already inside when the recording starts has not
        entered, and counting them would inflate every total.
        """
        if not self.active:
            return None
        side = self.side_of(point)
        if side == 0:
            return None                  # on the line, or past its end
        lane = self.states.get(track_id)
        if lane is None:
            self.states[track_id] = {"side": side, "candidate": 0, "streak": 0}
            return None                  # where they began, not a crossing
        if side == lane["side"]:
            lane["candidate"] = 0
            lane["streak"] = 0
            return None
        if lane["candidate"] == side:
            lane["streak"] += 1
        else:
            lane["candidate"] = side
            lane["streak"] = 1
        if lane["streak"] < self.confirm_frames:
            return None
        lane["side"] = side
        lane["candidate"] = 0
        lane["streak"] = 0
        return self.label(side)

    def drop(self, track_id):
        self.states.pop(track_id, None)


def build_crossing_line(roi_cfg, analysis_fps=None, sample_fps=None):
    """A CrossingLine from config, or None when no line is configured.

    `confirm_frames` is tuned at the analysis frame rate. When the same rule is
    replayed against the 5 Hz track history the threshold is restated in
    samples, so the two paths behave the same in SECONDS rather than in frames.
    """
    cfg = roi_cfg or {}
    line = cfg.get("line") or cfg.get("crossing_line") or []
    if len(line) < 2:
        return None
    confirm = int(cfg.get("confirm_frames", DEFAULT_LINE_CONFIRM))
    if analysis_fps and sample_fps:
        seconds = confirm / (float(analysis_fps) or 30.0)
        confirm = max(1, int(math.ceil(seconds * float(sample_fps))))
    return CrossingLine(
        line,
        inside=cfg.get("inside", 1),
        confirm_frames=confirm,
        hysteresis_px=float(cfg.get("hysteresis_px", DEFAULT_LINE_HYSTERESIS_PX)),
    )


class ROIMonitor:
    """Debounced OUTSIDE -> ENTERING -> INSIDE transitions for one polygon.

    A detection box wobbles around the polygon edge, so a single frame that
    happens to overlap is not an event: `enter_frames` consecutive inside
    frames are required before one is emitted, and `exit_frames` consecutive
    outside frames before the lane can fire again. That is the whole
    anti-double-counting rule - while somebody stands inside the ROI their
    state stays INSIDE and nothing more is counted.

    State is kept per local track, because that is what is observable frame to
    frame; the emitted event carries the track id and the caller re-attributes
    it to the persistent ReID identity (which may only be assigned, or merged
    into somebody else, several seconds later).
    """

    OUTSIDE, ENTERING, INSIDE, LEAVING = "OUTSIDE", "ENTERING", "INSIDE", "LEAVING"

    def __init__(self, kind, polygon, enter_frames=3, exit_frames=5,
                 count_first_seen_inside=False):
        self.kind = kind                       # "entry" | "exit"
        self.polygon = [[float(p[0]), float(p[1])] for p in (polygon or [])]
        self.enter_frames = max(1, int(enter_frames))
        self.exit_frames = max(1, int(exit_frames))
        self.count_first_seen_inside = bool(count_first_seen_inside)
        self.states = {}                       # local_track_id -> dict

    @property
    def active(self):
        return len(self.polygon) >= 3

    def _lane(self, track_id, inside_now):
        lane = self.states.get(track_id)
        if lane is None:
            # A track born INSIDE the ROI is almost always an id switch of
            # somebody already standing there, so it starts as INSIDE and is
            # not counted unless the caller explicitly asks for it.
            start = self.INSIDE if (inside_now and not self.count_first_seen_inside) else self.OUTSIDE
            lane = self.states[track_id] = {"state": start, "streak": 0}
        return lane

    def update(self, track_id, point, t_now):
        """Feed one observation. Returns True exactly on a counted transition."""
        if not self.active:
            return False
        inside = point_in_polygon(point, self.polygon)
        lane = self._lane(track_id, inside)
        state = lane["state"]

        if inside:
            if state in (self.OUTSIDE, self.ENTERING):
                lane["streak"] = lane["streak"] + 1 if state == self.ENTERING else 1
                lane["state"] = self.ENTERING
                if lane["streak"] >= self.enter_frames:
                    lane["state"] = self.INSIDE
                    lane["streak"] = 0
                    lane["entered_at"] = t_now
                    return True
            else:                                   # INSIDE / LEAVING -> still inside
                lane["state"] = self.INSIDE
                lane["streak"] = 0
        else:
            if state in (self.INSIDE, self.LEAVING):
                lane["streak"] = lane["streak"] + 1 if state == self.LEAVING else 1
                lane["state"] = self.LEAVING
                if lane["streak"] >= self.exit_frames:
                    lane["state"] = self.OUTSIDE
                    lane["streak"] = 0
            else:
                lane["state"] = self.OUTSIDE
                lane["streak"] = 0
        return False

    def drop(self, track_id):
        self.states.pop(track_id, None)


ROI_COUNT_MODES = ("transition", "independent")


def roi_owner_key(ev):
    """Who a ROI visit belongs to: the ReID identity, or the raw track."""
    return ev.get("person_name") or f"track:{ev.get('local_track_id')}"


def classify_roi_events(events, mode="transition", require_entry_before_exit=False):
    """Decide which ROI visits count toward the ENTRY / EXIT totals.

    `events` are debounced ROI ARRIVALS - "this person is now inside the entry
    polygon" - one per ROIMonitor firing. Which of those arrivals is worth
    counting is a separate question, answered here so the live pipeline and the
    after-the-fact ROI replay can never drift apart.

    transition (default)
        Only a DIRECTIONAL crossing between the two regions counts: somebody
        last seen in the Exit ROI who is now in the Entry ROI has entered, and
        the reverse is an exit. This is what a doorway actually looks like -
        the two polygons are the outside and the inside, and only movement
        BETWEEN them means anything. Arriving in the Entry ROI from anywhere
        else (off-camera, the middle of the room) is recorded but not counted.

        The person may pass through un-covered floor on the way; only the last
        region they were in matters. Requiring the polygons to touch would mean
        almost nothing ever counted.

    independent
        Every arrival counts on its own - the older behaviour, kept because it
        is the right rule when the two regions are unrelated areas rather than
        two sides of a threshold. `require_entry_before_exit` additionally
        drops exits by people who were never counted in.

    Events are annotated in place and returned in time order. Each gains
    `counted`; transition mode also records `from_roi`, the region the person
    came from (None if they had not been in either).
    """
    ordered = sorted(events, key=lambda e: (float(e.get("time") or 0.0),
                                            e.get("event") or ""))
    if mode == "independent":
        entered = set()
        for ev in ordered:
            owner = roi_owner_key(ev)
            if ev.get("event") == "entry":
                entered.add(owner)
                ev["matched"] = True
                ev["counted"] = True
            else:
                ev["matched"] = owner in entered
                ev["counted"] = ev["matched"] or not require_entry_before_exit
        return ordered

    last_roi = {}                       # owner -> the region they were last in
    for ev in ordered:
        owner = roi_owner_key(ev)
        kind = ev.get("event")
        previous = last_roi.get(owner)
        ev["from_roi"] = previous
        # crossing over from the other region is the event; re-arriving in the
        # same one (stepped out, stepped back) is not
        ev["counted"] = previous is not None and previous != kind
        ev["matched"] = ev["counted"]   # older readers only know this name
        last_roi[owner] = kind
    return ordered


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
        self.emb_frame = []          # which frame each stored embedding came from
        self.last_store_frame = -10 ** 9

        # Secondary face bank. Faces are tiny in surveillance footage, so this
        # only ever refines the body decision - it never decides on its own.
        self.face_embeddings = np.zeros((0, 0), np.float32)
        self.face_quality = []
        self.last_face_frame = -10 ** 9

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
        if not self.insert_embedding(emb, quality, frame_idx):
            return False
        self.last_store_frame = frame_idx
        return True

    def insert_embedding(self, emb, quality, frame_idx=-1):
        """Ungated insert into the appearance bank (used when merging people)."""
        rc = self.cfg["reid"]
        cap = int(rc["max_embeddings_per_person"])
        if self.embeddings.shape[0] == 0:
            self.embeddings = emb[None, :].astype(np.float32)
            self.emb_quality = [quality]
            self.emb_frame = [frame_idx]
        elif self.embeddings.shape[0] < cap:
            self.embeddings = np.vstack([self.embeddings, emb[None, :]])
            self.emb_quality.append(quality)
            self.emb_frame.append(frame_idx)
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
            self.emb_frame[worst] = frame_idx
        return True

    def store_face(self, emb, quality, frame_idx):
        rc = self.cfg["reid"]
        cap = int(rc.get("face_fusion", {}).get("max_faces_per_person", 12))
        if frame_idx - self.last_face_frame < int(rc["gallery_min_interval"]):
            return False
        if self.face_embeddings.shape[0] == 0:
            self.face_embeddings = emb[None, :].astype(np.float32)
            self.face_quality = [quality]
        elif self.face_embeddings.shape[0] < cap:
            self.face_embeddings = np.vstack([self.face_embeddings, emb[None, :]])
            self.face_quality.append(quality)
        else:
            worst = int(np.argmin(self.face_quality))
            if quality <= self.face_quality[worst]:
                return False
            self.face_embeddings[worst] = emb
            self.face_quality[worst] = quality
        self.last_face_frame = frame_idx
        return True

    def face_score(self, query, topk):
        if self.face_embeddings.shape[0] == 0 or query is None:
            return None
        sims = self.face_embeddings @ query
        k = min(int(topk), sims.shape[0])
        return float(np.sort(sims)[-k:].mean())

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
            "face_embedding_count": int(self.face_embeddings.shape[0]),
            "embedding_frames": sorted(int(f) for f in self.emb_frame),
            "local_track_ids": self.local_track_ids,
            "total_visible_seconds": round(sum(s["end"] - s["start"] for s in self.segments), 3),
            "segments": [{"start": s["start"], "end": s["end"]} for s in self.segments],
        }


class IdentityGallery:
    """Active + lost persistent identities and the appearance matching logic."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.persons = {}
        self.aliases = {}   # retired name -> the identity it was folded into
        self.names = NameGenerator(seed=cfg.get("seed", 1234))
        self._next_id = 0

    def resolve(self, name):
        """Follow merges to the identity a name ended up as."""
        seen = set()
        while name in self.aliases and name not in seen:
            seen.add(name)
            name = self.aliases[name]
        return name

    def live(self, tr):
        """The identity this track currently belongs to, following merges.

        A track keeps whatever name it was given, but that identity can later be
        folded into another one by absorb(), which REMOVES it from `persons`.
        Looking the old name up directly then raises KeyError and takes the
        whole analysis down with it - which is what happened to a job that died
        on KeyError: 'CoralBadger' after ~1 s. Several live tracks can point at
        the same identity, and merge() only re-points the one that triggered it,
        so the others are left holding a name that no longer exists.

        Returns None only if the name is unknown entirely (it never existed, or
        the gallery was rebuilt), which callers treat as "nothing to update".
        """
        name = getattr(tr, "person_name", None)
        if name is None:
            return None
        resolved = self.resolve(name)
        person = self.persons.get(resolved)
        if person is None:
            return None
        if resolved != name:
            tr.person_name = resolved      # keep the track honest from here on
        return person

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

    def match(self, query, t_now, center, frame_diag, exclude_names, face_query=None):
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
        fc = rc.get("face_fusion", {}) or {}
        face_on = bool(fc.get("enabled", False)) and face_query is not None
        face_w = float(fc.get("weight", 0.2))
        base_thr = float(rc["similarity_threshold"])
        fused_thr = base_thr + float(fc.get("threshold_shift", -0.06))

        scored = []
        for person in self.candidates(exclude_names):
            body = person.appearance_score(query, topk)
            face = person.face_score(face_query, topk) if face_on else None
            if face is None:
                appearance, threshold = body, base_thr
            else:
                # Fusion measured on this footage: 0.8*body + 0.2*face lifts recall
                # 0.72 -> 0.77 at the same false-merge rate. The fused score sits
                # lower than a body-only score at equal confidence, hence its own
                # (also measured) threshold rather than reusing the body one.
                appearance = (1.0 - face_w) * body + face_w * face
                threshold = fused_thr
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
                    "body": body,
                    "face": face,
                    "threshold": threshold,
                    "temporal": temporal,
                    "spatial": spatial,
                    "final": final,
                    "gap": gap,
                }
            )
        scored.sort(key=lambda s: s["final"], reverse=True)
        return scored

    def absorb(self, source, target):
        """Fold one stored identity into another and delete the duplicate."""
        for emb, quality, fr in zip(source.embeddings, source.emb_quality, source.emb_frame):
            target.insert_embedding(emb, quality, fr)
        for emb, quality in zip(source.face_embeddings, source.face_quality):
            target.store_face(emb, quality, target.last_face_frame + 10 ** 9)
        target.age_samples.extend(source.age_samples)
        target.gender_samples.extend(source.gender_samples)
        target.segments.extend(source.segments)
        target.segments.sort(key=lambda seg: seg["start"])
        target.first_seen = min(target.first_seen, source.first_seen)
        target.last_seen = max(target.last_seen, source.last_seen)
        target.last_demo_frame = max(target.last_demo_frame, source.last_demo_frame)
        target.reentry_count += source.reentry_count + 1
        target.local_track_ids.extend(
            t for t in source.local_track_ids if t not in target.local_track_ids
        )
        if source.best_crop is not None and source.best_crop_quality > target.best_crop_quality:
            target.best_crop, target.best_crop_quality = source.best_crop, source.best_crop_quality
        self.persons.pop(source.name, None)
        self.aliases[source.name] = target.name
        return target

    def merge(self, source, target, t_now, local_track_id):
        """Fold a duplicate identity into the one it turned out to be, and hand
        the live track over to it.

        Used when a track was named too early from a couple of weak crops and a
        richer query later shows it is somebody the gallery already knows.
        """
        self.absorb(source, target)
        target.active = True
        target.current_local_track_id = local_track_id
        if local_track_id not in target.local_track_ids:
            target.local_track_ids.append(local_track_id)
        target.last_box, target.last_center = source.last_box, source.last_center
        return target

    @staticmethod
    def accept(scored, threshold, margin, dup_thresholds=None):
        """Threshold + ambiguity check.

        Returns (accepted, duplicates). The ambiguity rule exists to stop two
        DIFFERENT people who look alike being merged. It must not fire when the
        rival is the same person already stored under a second name - otherwise
        one duplicate breeds more duplicates: the returning person matches both
        copies equally, gets refused, and a third copy is created. So a rival
        whose own embedding bank matches the leader's is set aside as a
        duplicate rather than treated as competing evidence.

        Note the rivals are scanned by appearance, not by list order, because
        `scored` is ranked by the combined score.
        """
        if not scored:
            return False, []
        best = scored[0]
        if best["appearance"] < best.get("threshold", threshold):
            return False, []
        duplicates = []
        for rival in scored[1:]:
            if best["appearance"] - rival["appearance"] >= margin:
                continue                                    # not close enough to matter
            if dup_thresholds and is_duplicate(best["person"], rival["person"], *dup_thresholds):
                duplicates.append(rival["person"])          # same person, second name
                continue
            return False, []                                # a real rival - refuse
        return True, duplicates


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
        self.faces = []              # face embeddings seen so far (query side)
        self.reassessed = False
        self.banner_until = -1.0     # RE-ID MATCH overlay end time
        self.banner_similarity = 0.0


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_DUPLEX

# Same two hues the ROI editor and the result dashboard use, in BGR:
# Entry = series-1 blue #3987e5, Exit = series-2 orange #d95926.
ROI_COLORS = {"entry": (229, 135, 57), "exit": (38, 89, 217)}

# Track history is written raw during analysis and canonicalised afterwards.
TRACKS_RAW_NAME = "tracks.raw.jsonl"
TRACKS_NAME = "tracks.jsonl.gz"
COUNT_FLASH_SECONDS = 1.2


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


def draw_roi_polygons(frame, cfg):
    """Outline the Entry / Exit floor regions the counting actually used."""
    roi = cfg.get("roi") or {}
    for kind in ("entry", "exit"):
        polygon = roi.get(f"{kind}_polygon") or []
        if len(polygon) < 3:
            continue
        pts = np.array([[int(p[0]), int(p[1])] for p in polygon], np.int32).reshape(-1, 1, 2)
        color = ROI_COLORS[kind]
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        label = f"{kind.upper()} ROI"
        ox, oy = int(polygon[0][0]) + 6, max(18, int(polygon[0][1]) - 8)
        cv2.putText(frame, label, (ox, oy), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (ox, oy), FONT, 0.6, color, 1, cv2.LINE_AA)


def draw_count_bar(frame, entries, exits, t_now, frame_idx):
    """The running ENTRY / EXIT tally, bottom-left, over a dark plate."""
    h, w = frame.shape[:2]
    pad, bar_h = 14, 54
    y0 = h - bar_h - pad
    x0 = pad
    width = 330
    region = frame[y0:y0 + bar_h, x0:x0 + min(width, w - 2 * pad)]
    if region.size:
        shade = np.full_like(region, 16)
        cv2.addWeighted(shade, 0.94, region, 0.06, 0, region)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + bar_h), (70, 70, 70), 1)
    cv2.putText(frame, f"ENTRY {entries}", (x0 + 14, y0 + 36), FONT, 0.95,
                ROI_COLORS["entry"], 2, cv2.LINE_AA)
    cv2.putText(frame, f"EXIT {exits}", (x0 + 180, y0 + 36), FONT, 0.95,
                ROI_COLORS["exit"], 2, cv2.LINE_AA)


def draw_overlay(frame, tracks, gallery, live_ids, t_now, cfg, frame_idx, fps,
                 counts=None, flashes=()):
    vis = cfg["visualization"]
    fs = float(vis.get("font_scale", 0.75))
    th = int(vis.get("thickness", 2))
    band = float((cfg.get("person") or {}).get("ground_band_ratio", DEFAULT_GROUND_BAND_RATIO))

    if vis.get("show_roi", True):
        draw_roi_polygons(frame, cfg)

    top = 0
    if vis.get("show_hud", True):
        top = HUD_HEIGHT
        visible = sum(1 for t in live_ids if tracks[t].person_name is not None)
        hud = (f"frame {frame_idx}   t={t_now:6.2f}s   people seen: {len(gallery.persons)}"
               f"   on screen: {visible}")
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

        person = gallery.live(tr)
        if person is None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 140, 140), 1)
            continue
        color = person.color()
        banner = vis.get("show_reid_match", True) and t_now <= tr.banner_until
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, th + (1 if banner else 0))

        # the bottom-band ground point - the single position the ROI test uses
        if vis.get("show_ground_point", True):
            gx, gy = person_ground_point(tr.box, band)
            by1 = y1 + (y2 - y1) * (1.0 - band)
            cv2.line(frame, (x1, int(by1)), (x2, int(by1)), color, 1, cv2.LINE_AA)
            cv2.circle(frame, (int(gx), int(gy)), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(gx), int(gy)), 5, color, 2, cv2.LINE_AA)

        lines = []
        if vis.get("show_person_name", True):
            lines.append((person.name, 1.25, (255, 255, 255)))
        flash = next((f for f in flashes if f[0] == person.name), None)
        if flash is not None:
            kind = flash[1]
            lines.append((f"COUNTED: {kind.upper()}", 1.05, ROI_COLORS[kind]))
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

    # drawn last so no bounding box or label can cover the running tally
    if counts is not None and vis.get("show_counts", True):
        draw_count_bar(frame, counts[0], counts[1], t_now, frame_idx)

    return frame


def _mean_unit(vectors):
    """Average a list of unit vectors back onto the unit sphere (None if empty)."""
    if not vectors:
        return None
    v = np.mean(np.stack(vectors), axis=0)
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else (v / n)


def _reject_reason(scored, threshold, margin, duplicate_threshold=None):
    """Say precisely WHY a candidate was refused - the two failure modes need
    different fixes, so they must not share one label."""
    if not scored:
        return "empty_gallery"
    best = scored[0]["appearance"]
    second = max((c["appearance"] for c in scored[1:]), default=0.0)
    if best < scored[0].get("threshold", threshold):
        return "below_threshold"
    if best - second < margin:
        return "ambiguous"          # a genuinely different person looked just as similar
    return "rejected"


def bank_similarity(a, b, topk=2):
    """Two ways of asking how alike two stored identities are.

    peak  - mean of the top-K of ALL cross pairs. Answers "do these two banks
            contain a pair of near-identical views?". Being a max-like statistic
            over up to 400 pairs it is easily satisfied by chance, so it can
            never be used on its own.
    mean  - symmetric mean of each embedding's best match in the other bank.
            Answers "do the banks agree BROADLY?", which a lucky pair cannot fake.

    Calibrated on this footage against co-occurring tracks (people who are
    provably different): peak>=0.90 alone false-merges 2.5% of pairs, and
    peak>=0.82 - the value naively borrowed from similarity_threshold - fires on
    12.9%. Requiring peak>=0.90 AND mean>=0.76 detects 83% of true duplicates at
    a 0.7% false rate.
    """
    if a.embeddings.shape[0] == 0 or b.embeddings.shape[0] == 0:
        return 0.0, 0.0
    sims = a.embeddings @ b.embeddings.T
    flat = sims.ravel()
    k = min(topk, flat.size)
    peak = float(np.sort(flat)[-k:].mean())
    mean = float((sims.max(1).mean() + sims.max(0).mean()) / 2.0)
    return peak, mean


def is_duplicate(a, b, peak_thr, mean_thr):
    """True when two stored identities are almost certainly the same person."""
    peak, mean = bank_similarity(a, b)
    return peak >= peak_thr and mean >= mean_thr


def _top_candidates(scored, n=3):
    top = scored[0]["person"] if scored else None
    out = []
    for c in scored[:n]:
        row = {"person_name": c["person"].name,
               "appearance": round(c["appearance"], 4),
               "body": round(c["body"], 4),
               "face": None if c.get("face") is None else round(c["face"], 4),
               "gap_seconds": round(c["gap"], 2),
               "embeddings": int(c["person"].embeddings.shape[0])}
        if top is not None and c["person"] is not top:
            peak, mean = bank_similarity(top, c["person"])
            row["sim_to_top1"] = round(peak, 4)
            row["mean_sim_to_top1"] = round(mean, 4)
        out.append(row)
    return out


def _save_match_debug(debug_dir, person, prev_crop, new_crop, similarity, index, prefix=""):
    """Previous crop | new crop, side by side, so a wrong match is obvious."""
    os.makedirs(debug_dir, exist_ok=True)
    if new_crop is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}{person.name}_reentry_{index:03d}.jpg"), new_crop)
    if prev_crop is None or new_crop is None:
        return
    cv2.imwrite(os.path.join(debug_dir, f"{prefix}{person.name}_previous.jpg"), prev_crop)

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
    cv2.imwrite(os.path.join(debug_dir, f"{prefix}{person.name}_match_{index:03d}.jpg"),
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
        self.face_embedder = None
        face_fusion = (cfg["reid"].get("face_fusion") or {}).get("enabled", False)
        if cfg["demographics"].get("enabled", True) or face_fusion:
            self.face_detector = models.FaceDetector(cfg, self.device, self.stats)
        if cfg["demographics"].get("enabled", True):
            self.demographics = models.DemographicsEstimator(cfg, self.device, self.stats)
        if face_fusion:
            self.face_embedder = models.FaceEmbedder(cfg, self.device, self.stats)

        self.gallery = IdentityGallery(cfg)
        self.tracks = {}
        self.events = []
        self.frame_records = []
        self.match_counter = {}
        self.reject_counter = 0
        self.render_log = []

        # The web server never wants an annotated video - only the numbers -
        # so rendering is a switch rather than an assumption.
        self.render_enabled = bool(cfg["video"].get("render", True))
        # Called as progress(frame_index, total_frames) every progress_interval
        # frames; the worker uses it to update the job row.
        self.progress_cb = None
        self.progress_interval = int(cfg["video"].get("progress_interval", 50))
        self.cancelled = None            # optional callable -> stop early

        # Entry / Exit floor ROIs (person analysis). Absent -> no ROI counting.
        pcfg = cfg.get("person") or {}
        self.ground_band_ratio = float(pcfg.get("ground_band_ratio", DEFAULT_GROUND_BAND_RATIO))
        roi_cfg = cfg.get("roi") or {}
        self.roi_monitors = []
        for kind in ("entry", "exit"):
            polygon = roi_cfg.get(f"{kind}_polygon") or []
            if len(polygon) >= 3:
                self.roi_monitors.append(ROIMonitor(
                    kind, polygon,
                    enter_frames=int(roi_cfg.get("enter_frames", 3)),
                    exit_frames=int(roi_cfg.get("exit_frames", 5)),
                    count_first_seen_inside=bool(roi_cfg.get("count_first_seen_inside", False)),
                ))
        self.roi_min_repeat_seconds = float(roi_cfg.get("min_repeat_seconds", 2.0))
        # How a ROI arrival becomes a count - see classify_roi_events().
        # "transition": only Exit->Entry counts as an entry and Entry->Exit as
        # an exit. "independent": every arrival counts by itself.
        self.roi_count_mode = str(roi_cfg.get("count_mode", "transition")).lower()
        if self.roi_count_mode not in ROI_COUNT_MODES:
            self.roi_count_mode = "transition"
        # Only meaningful in "independent" mode: drop exits by people who were
        # never counted in. In "transition" mode the ordering is inherent.
        self.require_entry_before_exit = bool(
            roi_cfg.get("require_entry_before_exit", False))
        self.roi_events = []             # raw events, re-attributed in _finish
        self._roi_last = {}              # (owner_key, kind) -> t of last counted event
        self._roi_resolved = False

        # Dwell heatmap over the SAME bottom-band ground point the ROIs use, so
        # the picture and the counts can never disagree. One cell per person per
        # frame, i.e. it accumulates time-spent, not head-count.
        # ---- track-position history ------------------------------------
        # Sampled on VIDEO time, so 25 / 29.97 / 30 / 59.94 fps all yield the
        # same ~5 samples per second. This is storage only: the detector, the
        # tracker, ReID and MiVOLO all keep their own cadence.
        tcfg = cfg.get("tracks") or {}
        self.tracks_enabled = bool(tcfg.get("enabled", True))
        self.track_sample_fps = float(tcfg.get("sample_fps", 5.0)) or 5.0
        self.track_interval = 1.0 / self.track_sample_fps
        self.track_decimals = int(tcfg.get("bbox_decimals", 1))
        self._track_slot = -1            # last emitted sample slot, drift-free
        self._track_fh = None
        self.track_samples = 0
        self.tracks_raw_path = None
        self.tracks_path = None

        hm_cfg = pcfg.get("heatmap") or {}
        self.heatmap_enabled = bool(hm_cfg.get("enabled", True))
        self.heatmap_cols = max(8, int(hm_cfg.get("cols", 64)))
        self.heatmap = None              # allocated in run(), once the size is known

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

    def _dup_threshold(self):
        rc = self.cfg["reid"]
        if not rc.get("merge_duplicates", True):
            return None
        return (float(rc.get("duplicate_peak_similarity", 0.90)),
                float(rc.get("duplicate_mean_similarity", 0.76)))

    def _fold_duplicates(self, duplicates, target, t_now, local_track_id):
        """Heal the gallery: the same person stored under several names becomes one."""
        for dup in duplicates:
            if dup.name not in self.gallery.persons or dup is target:
                continue
            peak, mean = bank_similarity(target, dup)
            self.gallery.absorb(dup, target)
            self._event(t_now, "duplicate_merged", target, local_track_id,
                        absorbed=dup.name, peak_similarity=round(peak, 4),
                        mean_similarity=round(mean, 4))

    def _save_reject_debug(self, scored, crop, t_now, local_track_id, stage):
        """Dump near-miss comparisons so it is visible whether a refused match
        was actually the same person (threshold too strict) or not (model limit)."""
        dbg = self.cfg["debug"]
        if not dbg.get("save_reject_crops", False) or crop is None or not scored:
            return
        best = scored[0]
        if best["appearance"] < float(dbg.get("reject_crop_min_similarity", 0.6)):
            return
        out = os.path.join(dbg["reid_debug_dir"], "rejected")
        self.reject_counter += 1
        _save_match_debug(out, best["person"], best["person"].best_crop, crop,
                          best["appearance"], self.reject_counter,
                          prefix=f"REJECT_{stage}_t{t_now:.1f}s_track{local_track_id}_vs_")

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

        scored = self.gallery.match(query, t_now, center, frame_diag, busy,
                                    face_query=_mean_unit(tr.faces))
        ok, duplicates = self.gallery.accept(
            scored, rc["similarity_threshold"], rc["second_best_margin"], self._dup_threshold()
        )
        if not ok:
            if scored and self.cfg["debug"].get("log_rejections", False):
                self._event(
                    t_now, "reid_reject", person, tr.id,
                    reason=_reject_reason(scored, rc["similarity_threshold"], rc["second_best_margin"]),
                    best_similarity=round(scored[0]["appearance"], 4),
                    embeddings_used=len(tr.history),
                    candidates=_top_candidates(scored),
                )
            self._save_reject_debug(scored, safe_crop(frame, tr.box), t_now, tr.id, "reassess")
            return

        best = scored[0]
        target = best["person"]
        second = max((c["appearance"] for c in scored[1:]), default=0.0)
        prev_crop, new_crop = target.best_crop, safe_crop(frame, tr.box)
        self._fold_duplicates(duplicates, target, t_now, tr.id)
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
        face_query = _mean_unit(tr.faces)
        scored = self.gallery.match(query, t_now, center, frame_diag, busy, face_query=face_query)
        best = scored[0] if scored else None
        second = max((c["appearance"] for c in scored[1:]), default=0.0)
        accepted, duplicates = self.gallery.accept(
            scored, rc["similarity_threshold"], rc["second_best_margin"], self._dup_threshold()
        )

        crop = safe_crop(frame, tr.box)
        if accepted:
            person = best["person"]
            was_lost = not person.active
            gap = best["gap"]
            prev_crop = person.best_crop
            self._fold_duplicates(duplicates, person, t_now, tr.id)
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
                reason=_reject_reason(scored, rc["similarity_threshold"], rc["second_best_margin"]),
                best_similarity=round(best["appearance"], 4) if best else 0.0,
                second_best_similarity=round(second, 4),
                best_candidate=best["person"].name if best else None,
                embeddings_used=len(tr.pending),
                candidates=_top_candidates(scored),
            )
            self._save_reject_debug(scored, crop, t_now, tr.id, "assign")

        tr.person_name = person.name
        for emb, quality in tr.pending:
            person.store_embedding(emb, quality, tr.last_frame, crop if quality == best_quality else None)
        tr.history = list(tr.pending)
        tr.pending.clear()
        return person

    # -- ground point: heatmap + Entry / Exit ROIs -------------------------
    def _update_ground(self, live_ids, t_now, frame_idx, width, height):
        """The ground point is computed ONCE per visible track per frame, and
        both the dwell heatmap and the ROI state machines read that same value."""
        for tid in live_ids:
            tr = self.tracks[tid]
            if tr.box is None:
                continue
            point = person_ground_point(tr.box, self.ground_band_ratio)
            if self.heatmap is not None:
                rows, cols = self.heatmap.shape
                col = int(point[0] / max(width, 1) * cols)
                row = int(point[1] / max(height, 1) * rows)
                if 0 <= col < cols and 0 <= row < rows:
                    self.heatmap[row, col] += 1.0
            for monitor in self.roi_monitors:
                if not monitor.update(tid, point, t_now):
                    continue
                # Prefer the persistent ReID identity: tracker ids change every
                # time somebody is briefly occluded, the identity does not.
                owner = (self.gallery.resolve(tr.person_name)
                         if tr.person_name else f"track:{tid}")
                key = (owner, monitor.kind)
                last = self._roi_last.get(key)
                if last is not None and t_now - last < self.roi_min_repeat_seconds:
                    continue                      # same identity, same ROI, moments ago
                self._roi_last[key] = t_now
                self.roi_events.append({
                    "time": round(float(t_now), 3),
                    "frame": int(frame_idx),
                    "event": monitor.kind,        # "entry" | "exit"
                    "local_track_id": int(tid),
                    "person_name": tr.person_name,
                    "point": [round(point[0], 1), round(point[1], 1)],
                })
                self._event(t_now, f"roi_{monitor.kind}",
                            self.gallery.persons.get(owner), tid,
                            x=round(point[0], 1), y=round(point[1], 1))

    # -- track history -----------------------------------------------------
    def _open_tracks(self):
        """Raw JSONL, appended as the video is decoded and never held in RAM."""
        if not self.tracks_enabled:
            return
        out_dir = self.cfg["video"]["output_dir"]
        tracks_dir = self.cfg.get("tracks", {}).get("dir") or out_dir
        os.makedirs(tracks_dir, exist_ok=True)
        self.tracks_raw_path = os.path.join(tracks_dir, TRACKS_RAW_NAME)
        self.tracks_path = os.path.join(tracks_dir, TRACKS_NAME)
        self._track_fh = open(self.tracks_raw_path, "w", encoding="utf-8",
                              newline="\n")

    def _sample_tracks(self, live_ids, t_now, frame_idx):
        """Emit at most one line per 1/sample_fps slice of VIDEO time.

        The slot index is computed from the timestamp rather than accumulated,
        so a dropped or duplicated frame cannot make the sample times drift.
        """
        if self._track_fh is None:
            return
        slot = int(t_now / self.track_interval + 1e-9)
        if slot <= self._track_slot:
            return
        self._track_slot = slot
        people = []
        for tid in live_ids:
            tr = self.tracks.get(tid)
            if tr is None or tr.box is None:
                continue
            people.append({
                "person_name": tr.person_name,      # provisional; resolved later
                "local_track_id": int(tid),
                "bbox": [round(float(v), self.track_decimals) for v in tr.box[:4]],
            })
        if not people:
            return
        self._track_fh.write(json.dumps(
            {"frame": int(frame_idx), "time": round(float(t_now), 3),
             "people": people}, separators=(",", ":")) + "\n")
        self.track_samples += 1

    def finalize_tracks(self):
        """Raw -> canonical identities -> gzip, streaming, then drop the raw file.

        A track's name at sample time is provisional: it may have been null, or
        a name ReID later folded into somebody else. The final gallery is the
        only authority, so every line is rewritten against it here - the same
        rule `_resolve_roi_events` applies to ROI events.
        """
        if self._track_fh is not None:
            self._track_fh.close()
            self._track_fh = None
        if not self.tracks_raw_path or not os.path.exists(self.tracks_raw_path):
            return None

        by_track = {}
        for person in self.gallery.persons.values():
            for tid in person.local_track_ids:
                by_track[int(tid)] = person.name

        def canonical(entry):
            name = entry.get("person_name")
            name = self.gallery.resolve(name) if name else None
            if name is None or name not in self.gallery.persons:
                name = by_track.get(int(entry["local_track_id"]))
            return name

        t0 = time.perf_counter()
        lines = 0
        # line by line, in and out - a 10-hour video never lands in memory
        with open(self.tracks_raw_path, "r", encoding="utf-8") as src, \
                gzip.open(self.tracks_path, "wt", encoding="utf-8",
                          compresslevel=6, newline="\n") as dst:
            for raw in src:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                for entry in rec["people"]:
                    entry["person_name"] = canonical(entry)
                dst.write(json.dumps(rec, separators=(",", ":"),
                                     ensure_ascii=False) + "\n")
                lines += 1

        os.remove(self.tracks_raw_path)
        self.tracks_raw_path = None
        print(f"[tracks] {lines} samples -> {self.tracks_path} "
              f"({os.path.getsize(self.tracks_path) / 1e6:.1f} MB, "
              f"{time.perf_counter() - t0:.1f}s)")
        return {"file": TRACKS_NAME, "path": self.tracks_path,
                "samples": lines, "sample_fps": self.track_sample_fps}

    def counted_roi_events(self):
        """The events that make up the headline counts.

        Visits that do not count are kept in the record (flagged counted=False)
        rather than dropped, so the summary, the JSON and the burnt-in video
        counter are all derived from one list and cannot disagree.
        """
        self._resolve_roi_events()
        return [e for e in self.roi_events if e.get("counted", True)]

    def heatmap_json(self):
        """The dwell grid as plain JSON: raw counts plus what they mean.

        Raw counts, not normalised - the viewer decides how to scale them, and
        two jobs stay comparable. `frames_per_unit` is 1, i.e. one unit is one
        person visible in that cell for one frame.
        """
        if self.heatmap is None:
            return None
        grid = self.heatmap
        rows, cols = grid.shape
        return {
            "cols": int(cols),
            "rows": int(rows),
            "max": float(grid.max()),
            "total": float(grid.sum()),
            "unit": "person-frames",
            "cells": [[int(v) for v in row] for row in grid.astype(np.int64)],
        }

    def _resolve_roi_events(self):
        """Attribute every ROI event to the identity the track ENDED UP as.

        An event may have fired before the track was named, or under a name that
        was later folded into somebody else; both are fixed here, and the
        per-identity de-duplication is applied a second time on the final names.
        """
        if self._roi_resolved:
            return self.roi_events
        # final name per local track id, from the identities that survived
        by_track = {}
        for person in self.gallery.persons.values():
            for tid in person.local_track_ids:
                by_track[int(tid)] = person.name

        resolved, seen = [], {}
        for ev in sorted(self.roi_events, key=lambda e: e["time"]):
            name = ev.get("person_name")
            name = self.gallery.resolve(name) if name else None
            if name is None or name not in self.gallery.persons:
                name = by_track.get(int(ev["local_track_id"]))
            owner = name or f"track:{ev['local_track_id']}"
            key = (owner, ev["event"])
            last = seen.get(key)
            if last is not None and ev["time"] - last < self.roi_min_repeat_seconds:
                continue
            seen[key] = ev["time"]
            out = dict(ev)
            out["person_name"] = name
            resolved.append(out)
        # Counting comes last, on the FINAL names: a visit may have fired before
        # its track had a name, or under one later merged away, so only now can
        # we say which region this person was really in beforehand.
        self.roi_events = classify_roi_events(
            resolved, self.roi_count_mode, self.require_entry_before_exit)
        self._roi_resolved = True
        return self.roi_events

    def _frame_faces(self, frame, cache):
        """Face detection is run at most once per frame and shared by the face
        ReID signal and the age/gender model."""
        if cache[0] is None:
            cache[0] = self.face_detector.detect(frame)
        return cache[0]

    # -- demographics ------------------------------------------------------
    def _run_demographics(self, frame, live_ids, frame_idx, faces_cache):
        dc = self.cfg["demographics"]
        due = []
        for tid in live_ids:
            tr = self.tracks[tid]
            if tr.person_name is None or tr.quality < float(dc["min_quality"]):
                continue
            person = self.gallery.live(tr)
            if person is None:
                continue
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

        faces = self._frame_faces(frame, faces_cache)

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
            person = self.gallery.live(tr)
            if person is None:
                continue
            person.add_demographics(sample, tr.quality, hf)
            person.last_demo_frame = frame_idx

    def _collect_faces(self, frame, tracks, faces_cache, frame_idx):
        """Face embedding for each track that has a visible, large-enough face."""
        faces = self._frame_faces(frame, faces_cache)
        if len(faces) == 0:
            return
        owners, crops = [], []
        for tr in tracks:
            fbox = face_for_box(tr.box, faces)
            crop = safe_crop(frame, fbox, pad=0.15) if fbox is not None else None
            if self.face_embedder.usable(crop):
                owners.append(tr)
                crops.append(crop)
        if not crops:
            return
        cap = int((self.cfg["reid"].get("face_fusion") or {}).get("max_query_faces", 8))
        for tr, emb in zip(owners, self.face_embedder.extract(crops)):
            if len(tr.faces) < cap:
                tr.faces.append(emb)
            owner = self.gallery.live(tr)
            if owner is not None:
                owner.store_face(emb, tr.quality, frame_idx)

    # -- pass 2: draw the video from the FINAL identities -------------------
    def _render_video(self, fps, width, height):
        cfg = self.cfg
        out = cfg["video"]["output"]
        cap = cv2.VideoCapture(cfg["video"]["input"])
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {out}")

        t0 = time.perf_counter()
        # cumulative ENTRY / EXIT at each frame, from the resolved events
        events = sorted(self.counted_roi_events(), key=lambda e: int(e.get("frame", 0)))
        flash_frames = max(1, int(fps * COUNT_FLASH_SECONDS))
        first = self.render_log[0][0] if self.render_log else 0
        if first:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)

        written = 0
        cursor = 0
        n_entry = n_exit = 0
        for frame_idx, entries in self.render_log:
            ok, frame = cap.read()
            if not ok:
                break
            while cursor < len(events) and int(events[cursor].get("frame", 0)) <= frame_idx:
                if events[cursor]["event"] == "entry":
                    n_entry += 1
                else:
                    n_exit += 1
                cursor += 1
            flashes = [(e.get("person_name"), e["event"]) for e in events
                       if 0 <= frame_idx - int(e.get("frame", 0)) < flash_frames]
            live, tracks = [], {}
            for e in entries:
                name = self.gallery.resolve(e["name"]) if e["name"] else None
                if name is not None and name not in self.gallery.persons:
                    name = None
                shim = TrackState(e["tid"], frame_idx, frame_idx / fps)
                shim.box = np.array(e["box"], np.float32)
                shim.person_name = name
                shim.banner_until = float("inf") if e["banner"] else -1.0
                shim.banner_similarity = e["sim"]
                tracks[e["tid"]] = shim
                live.append(e["tid"])
            draw_overlay(frame, tracks, self.gallery, live, frame_idx / fps, cfg,
                         frame_idx, fps,
                         counts=(n_entry, n_exit) if self.roi_monitors else None,
                         flashes=flashes)
            writer.write(frame)
            written += 1
            if written % 200 == 0:
                print(f"  ... rendering {written}/{len(self.render_log)}")
        cap.release()
        writer.release()
        print(f"[video] pass 2 rendered {written} frames in {time.perf_counter() - t0:.1f}s")

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

        # Two-pass: analyse first, draw afterwards. A name assigned from two weak
        # crops is often corrected a second later, and a single-pass video keeps
        # the discarded name burnt into every frame already written. Pass 2
        # redraws from the FINAL identities, so what you watch matches the JSON.
        self.two_pass = bool(vcfg.get("two_pass", True))
        if not self.render_enabled:
            self.two_pass = False        # nothing to render in either pass
        writer = None
        if self.render_enabled and not self.two_pass:
            writer = cv2.VideoWriter(vcfg["output"], cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {vcfg['output']}")

        print(f"[video] {vcfg['input']}  {width}x{height} @ {fps:.2f}fps  {total} frames")
        if self.render_enabled:
            print(f"[video] {'two-pass render' if self.two_pass else 'streaming render'} "
                  f"-> {vcfg['output']}")
        else:
            print("[video] structured output only - no result video is written")

        rc = cfg["reid"]
        reassess_n = int(rc.get("reassess_embeddings", 8))
        max_age = int(cfg["tracking"]["max_age_frames"])
        self._open_tracks()

        if self.heatmap_enabled:
            cols = self.heatmap_cols
            rows = max(8, int(round(cols * height / max(width, 1))))
            self.heatmap = np.zeros((rows, cols), np.float32)

        frame_idx = start_frame
        processed = 0
        total_wanted = max(0, total - start_frame)
        if max_frames:
            total_wanted = min(total_wanted, max_frames) if total_wanted else max_frames
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
                owner = self.gallery.live(tr)
                if owner is not None:
                    owner.touch(t_now, tr.box)

            # 1b) everything that reads the bottom-band ground point: the dwell
            #     heatmap and the Entry / Exit ROI transitions
            if self.heatmap is not None or self.roi_monitors:
                self._update_ground(live_ids, t_now, frame_idx, width, height)

            # 1c) track-position history, sampled on video time
            self._sample_tracks(live_ids, t_now, frame_idx)

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

            faces_cache = [None]
            if need:
                crops = [safe_crop(frame, tr.box) for tr in need]
                keep = [(tr, c) for tr, c in zip(need, crops) if c is not None]
                if keep and self.face_embedder is not None:
                    self._collect_faces(frame, [tr for tr, _ in keep], faces_cache, frame_idx)
                if keep:
                    embeddings = self.reid.extract([c for _, c in keep])
                    for (tr, crop), emb in zip(keep, embeddings):
                        tr.last_emb_frame = frame_idx
                        if tr.person_name is None:
                            tr.pending.append((emb, tr.quality))
                            if len(tr.pending) >= int(rc["min_embeddings_to_assign"]):
                                self._assign_identity(tr, frame, t_now, live_ids, frame_diag)
                        else:
                            owner = self.gallery.live(tr)
                            if owner is not None:
                                owner.store_embedding(emb, tr.quality, frame_idx, crop)
                            if len(tr.history) < reassess_n:
                                tr.history.append((emb, tr.quality))
                            if not tr.reassessed and len(tr.history) >= reassess_n:
                                self._reassess_identity(tr, frame, t_now, live_ids, frame_diag)

            # 3) age / gender (reuses this frame's face detection)
            if self.demographics is not None:
                self._run_demographics(frame, live_ids, frame_idx, faces_cache)

            # 4) retire local tracks; their identity survives in the gallery
            for tid in list(self.tracks):
                tr = self.tracks[tid]
                if tid in live_ids or frame_idx - tr.last_frame <= max_age:
                    continue
                gone_at = tr.last_frame / fps
                self._event(gone_at, "track_end", None, tid,
                            person_name=tr.person_name,
                            frames=tr.last_frame - tr.first_frame + 1)
                person = self.gallery.live(tr)
                if person is not None:
                    if person.current_local_track_id == tid:
                        person.unbind()
                        self._event(gone_at, "lost", person, tid,
                                    visible_seconds=round(person.segments[-1]["end"]
                                                          - person.segments[-1]["start"], 2))
                for monitor in self.roi_monitors:
                    monitor.drop(tid)
                del self.tracks[tid]

            # 5) render now, or remember what to render in pass 2
            if not self.render_enabled:
                pass
            elif self.two_pass:
                self.render_log.append((frame_idx, [
                    {"tid": t,
                     "box": [float(v) for v in self.tracks[t].box],
                     "name": self.tracks[t].person_name,
                     "banner": bool(t_now <= self.tracks[t].banner_until),
                     "sim": float(self.tracks[t].banner_similarity)}
                    for t in live_ids
                ]))
            else:
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
            if processed % self.progress_interval == 0:
                elapsed = time.perf_counter() - wall0
                print(f"  ... {processed} frames  ({processed / elapsed:.1f} fps)  "
                      f"people={len(self.gallery.persons)}")
                if self.progress_cb is not None:
                    self.progress_cb(frame_idx - start_frame, total_wanted)
                if self.cancelled is not None and self.cancelled():
                    print("[run] cancellation requested - stopping early")
                    break

        cap.release()
        if writer is not None:
            writer.release()
        if self._track_fh is not None:
            self._track_fh.flush()

        # close whatever is still open at the end of the video
        end_t = (frame_idx - 1) / fps
        for tr in self.tracks.values():
            if tr.person_name is None:
                continue
            person = self.gallery.persons.get(self.gallery.resolve(tr.person_name))
            if person is not None and person.current_local_track_id == tr.id:
                person.unbind()
                self._event(end_t, "track_end", person, tr.id, reason="video_end")

        analysis_wall = time.perf_counter() - wall0
        # Resolve first: the counters burnt into the video must be the SAME
        # numbers the JSON reports, not the raw pre-dedup events.
        if self.roi_monitors:
            self._resolve_roi_events()
        # Canonicalise the track history against the FINAL gallery. Must happen
        # after the loop, because that is when identities stop changing.
        self.track_storage = self.finalize_tracks()
        if self.render_enabled and self.two_pass:
            self._render_video(fps, width, height)
        wall = time.perf_counter() - wall0
        return self._finish(processed, wall, fps, analysis_wall)

    # -- outputs -----------------------------------------------------------
    def _finish(self, processed, wall, fps, analysis_wall=None):
        cfg = self.cfg
        out_dir = cfg["video"]["output_dir"]
        persons = sorted(self.gallery.persons.values(), key=lambda p: p.first_seen)
        roi_events = self._resolve_roi_events() if self.roi_monitors else []
        video_path = cfg["video"]["output"] if self.render_enabled else None
        heatmap = self.heatmap_json()

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
        if analysis_wall is not None and analysis_wall < wall:
            print(f"    analysis pass          : {analysis_wall:.1f} s "
                  f"({processed / max(1e-9, analysis_wall):.2f} fps)")
            print(f"    render pass            : {wall - analysis_wall:.1f} s")
        print(f"  processing fps           : {processed / max(1e-9, wall):.2f}")
        for name in ("detector", "reid", "face_detector", "face_embed", "demographics"):
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
        if self.render_enabled:
            print(f"  {cfg['video']['output']}")
        print(f"  {persons_path}\n  {events_path}")

        if roi_events:
            with open(os.path.join(out_dir, "roi_events.json"), "w", encoding="utf-8") as fh:
                json.dump(roi_events, fh, indent=2)
            entries = sum(1 for e in roi_events if e["event"] == "entry")
            exits = sum(1 for e in roi_events if e["event"] == "exit")
            print(f"  ROI entries / exits      : {entries} / {exits}")

        return {
            "persons": [p.to_json() for p in persons],
            "events": self.events,
            "roi_events": roi_events,
            "heatmap": heatmap,
            "track_storage": getattr(self, "track_storage", None),
            "result_video": video_path,
            "fps": round(fps, 3),
            "frames_processed": processed,
            "performance": {
                "input_fps": round(fps, 3),
                "analysis_seconds": None if analysis_wall is None else round(analysis_wall, 3),
                "frames_processed": processed,
                "processing_seconds": round(wall, 3),
                "processing_fps": round(processed / max(1e-9, wall), 3),
                "stages": report,
            },
        }
