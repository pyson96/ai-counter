"""Person / visitor analysis for one uploaded video.

This is a thin driver around the project's existing pipeline. It does not add a
second tracker, a second ReID gallery or a second MiVOLO wrapper:

    root pipeline.Analyzer
        YOLO11 person detection  ->  ByteTrack local_track_id
        ->  SOLIDER ReID  ->  persistent anonymous person identity
        ->  MiVOLO v2 age / gender, aggregated per identity

What this module adds is the web workflow on top of it:

    * GPU analysis runs with NO ROI. Every visible person's bbox is stored at
      ~5 Hz of video time as tracks.jsonl.gz, keyed by the FINAL ReID identity.
    * Entry / Exit ROIs are applied afterwards by `recalculate_roi()`, which
      replays those stored boxes through the same ROIMonitor state machine -
      no decoder, no YOLO, no SOLIDER, no MiVOLO. The ROI can be changed as
      often as the user likes.
    * age groups, gender split and the time-window histograms
    * summary.json / persons.json / person_events.json / job_metadata.json

No result video is produced: `video.render` is forced to False, which is what
makes the root Analyzer skip both the writer and the second render pass.
"""

from __future__ import annotations

import json
import os
import time

import cv2

from . import PROJECT_ROOT, TRACKS_FILE  # noqa: F401  (sys.path)

import pipeline as core          # the project-root pipeline

person_ground_point = core.person_ground_point
person_ground_band = core.person_ground_band
point_in_polygon = core.point_in_polygon

DEFAULT_AGE_GROUPS = {
    "under_20": (0, 20),
    "20s": (20, 30),
    "30s": (30, 40),
    "40s": (40, 50),
    "50s": (50, 60),
    "60_plus": (60, 200),
}


def age_group_for(age, groups=None):
    """Map an estimated age onto a configured bucket label."""
    if age is None:
        return "unknown"
    groups = groups or DEFAULT_AGE_GROUPS
    for label, bounds in groups.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo <= age < hi:
            return label
    return "unknown"


FRAME_IMAGE = "frame.jpg"


def save_first_frame(video_path, result_dir, name=FRAME_IMAGE, quality=88):
    """Keep the video's FIRST frame as a still, before the source is deleted.

    The result dashboard draws the dwell heatmap on top of this image, so it has
    to outlive the source video. One JPEG per job - this is a kept result, not a
    temporary artifact, and it is the only image the server stores.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, name)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())
    return {"file": name, "width": int(frame.shape[1]), "height": int(frame.shape[0])}


def _bucket_histogram(times, bucket_seconds, duration):
    """Counts per fixed time window, as [{start, end, count}, ...]."""
    bucket_seconds = max(1.0, float(bucket_seconds))
    n = max(1, int((duration or 0) // bucket_seconds) + 1)
    counts = [0] * n
    for t in times:
        idx = min(n - 1, max(0, int(float(t) // bucket_seconds)))
        counts[idx] += 1
    return [
        {
            "start_seconds": round(i * bucket_seconds, 1),
            "end_seconds": round((i + 1) * bucket_seconds, 1),
            "count": c,
        }
        for i, c in enumerate(counts)
    ]


class PersonAnalysis:
    """Runs the person pipeline for one job and writes the structured output."""

    RESULT_VIDEO = "result.mp4"

    def __init__(self, cfg, video_path, result_dir, entry_roi=None, exit_roi=None,
                 progress_cb=None, job_id="job", create_video=False):
        # entry_roi / exit_roi are accepted for signature compatibility but are
        # NOT used by GPU analysis any more - ROIs are a post-processing step.
        self.cfg = cfg
        self.video_path = video_path
        self.result_dir = result_dir
        self.entry_roi = entry_roi or []
        self.exit_roi = exit_roi or []
        self.progress_cb = progress_cb
        self.job_id = job_id
        self.create_video = bool(create_video)
        self.frame_image = None
        self.video_path_out = None

    # -- configuration -----------------------------------------------------
    def _analysis_config(self):
        """The root analysis config, pointed at this job and this video."""
        cfg = json.loads(json.dumps(self.cfg))        # deep copy, plain types only
        raw_dir = os.path.join(self.result_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        video = cfg.setdefault("video", {})
        video["input"] = self.video_path
        video["output_dir"] = raw_dir
        # Rendering is opt-in per job. When it is off nothing is written at all;
        # when it is on the two-pass renderer draws the FINAL identities and the
        # settled entry/exit counters, so the video agrees with the JSON.
        if self.create_video:
            self.video_path_out = os.path.join(self.result_dir, self.RESULT_VIDEO)
            video["output"] = self.video_path_out
            video["render"] = True
            video["two_pass"] = True
        else:
            video["output"] = os.path.join(raw_dir, "unused.mp4")   # never written
            video["render"] = False
            video["two_pass"] = False
        video["progress_interval"] = int(
            (cfg.get("analysis") or {}).get("progress_interval", 50))

        debug = cfg.setdefault("debug", {})
        debug["save_reid_crops"] = False      # no image cache for a server run
        debug["save_reject_crops"] = False
        debug["save_frame_results"] = False
        debug["reid_debug_dir"] = os.path.join(raw_dir, "reid_debug")
        debug["verbose"] = False
        os.makedirs(debug["reid_debug_dir"], exist_ok=True)

        # No ROI during GPU analysis. The polygons are applied later, by
        # recalculate_roi(), against the stored track history.
        roi = dict(cfg.get("roi") or {})
        roi["entry_polygon"] = []
        roi["exit_polygon"] = []
        cfg["roi"] = roi
        cfg.setdefault("tracks", {})["dir"] = self.result_dir
        cfg.setdefault("person", {}).setdefault("ground_band_ratio", 0.15)

        vis = cfg.setdefault("visualization", {})
        vis.setdefault("show_roi", True)             # the polygons that did the counting
        vis.setdefault("show_counts", True)          # the running ENTRY / EXIT tally
        vis.setdefault("show_ground_point", True)    # the bottom-band point itself
        vis.setdefault("show_person_name", True)     # the persistent anonymous id
        return cfg

    # -- run ---------------------------------------------------------------
    def run(self):
        cfg = self._analysis_config()
        # taken up front: if the analysis dies later, the still is already saved
        self.frame_image = save_first_frame(self.video_path, self.result_dir)
        analyzer = core.Analyzer(cfg)
        if self.progress_cb is not None:
            analyzer.progress_cb = self.progress_cb
        started = time.time()
        result = analyzer.run()
        elapsed = time.time() - started
        return self._build_output(cfg, analyzer, result, elapsed)

    # -- structured output -------------------------------------------------
    def _build_output(self, cfg, analyzer, result, elapsed):
        pcfg = cfg.get("person") or {}
        groups = {k: tuple(v) for k, v in (pcfg.get("age_groups") or DEFAULT_AGE_GROUPS).items()}
        bucket = float(pcfg.get("time_bucket_seconds", 300))

        fps = float(result.get("fps") or 30.0)
        frames = int(result.get("frames_processed") or 0)
        duration = frames / fps if fps else 0.0
        roi_events = []                      # no ROI yet - see recalculate_roi()
        first_entry, last_exit = {}, {}

        persons = []
        for p in result["persons"]:
            age = p.get("estimated_age")
            name = p["person_name"]
            persons.append({
                "person_name": name,
                "entry_time": first_entry.get(name),
                "exit_time": last_exit.get(name),
                "first_seen": p.get("first_seen"),
                "last_seen": p.get("last_seen"),
                "estimated_age": age,
                "age_group": age_group_for(age, groups),
                "estimated_gender": p.get("estimated_gender", "unknown"),
                "gender_confidence": p.get("gender_confidence", 0.0),
                "demographics_confident": p.get("demographics_confident", False),
                "total_visible_seconds": p.get("total_visible_seconds"),
                "reentry_count": p.get("reentry_count", 0),
                "segments": p.get("segments", []),
            })

        # No ROI has been configured yet, so entry/exit are UNKNOWN rather than
        # zero - zero would claim an ROI was configured and nobody crossed it.
        summary = {
            "roi_configured": False,
            "entry_count": None,
            "exit_count": None,
            "unmatched_exit_count": None,
            "require_entry_before_exit": bool(
                (cfg.get("roi") or {}).get("require_entry_before_exit", False)),
            "unique_persons_detected": len(persons),
            "unique_persons_entered": None,
            "gender": _gender_stats(persons),
            "age_groups": _age_stats(persons, groups),
            "time_bucket_seconds": bucket,
            "entries_by_time": [],
            "exits_by_time": [],
            "video_duration_seconds": round(duration, 2),
            "frames_processed": frames,
            "fps": round(fps, 3),
        }

        # The heatmap is in GRID coordinates plus the video size it was built
        # from, so the dashboard can lay it over the saved first frame no matter
        # what resolution that frame is displayed at.
        heatmap = result.get("heatmap")
        if heatmap:
            heatmap = dict(heatmap)
            heatmap["video_width"] = int((self.frame_image or {}).get("width") or 0)
            heatmap["video_height"] = int((self.frame_image or {}).get("height") or 0)
            heatmap["entry_roi"] = self.entry_roi
            heatmap["exit_roi"] = self.exit_roi
            heatmap["ground_band_ratio"] = float(pcfg.get("ground_band_ratio", 0.15))

        metadata = {
            "job_id": self.job_id,
            "analysis_type": "person",
            "source_video": os.path.basename(self.video_path),
            "processing_seconds": round(elapsed, 1),
            "processing_fps": round(frames / max(elapsed, 1e-6), 2),
            "ground_band_ratio": float(pcfg.get("ground_band_ratio", 0.15)),
            "entry_roi": None,
            "exit_roi": None,
            "roi_configured": False,
            "track_storage": _track_meta(result.get("track_storage"),
                                         self.cfg, self.result_dir),
            "frame_image": self.frame_image,
            "models": {
                "detector": cfg["detection"]["model"],
                "tracker": cfg["tracking"]["tracker"],
                "reid": cfg["reid"]["model"],
                "demographics": (cfg["demographics"]["model"]
                                 if cfg["demographics"].get("enabled", True) else None),
            },
            "result_video": (self.RESULT_VIDEO if self.create_video else None),
            "note": "person_name values are randomly generated anonymous aliases; age and "
                    "gender are AI-estimated appearance attributes, not identity claims.",
            "performance": result.get("performance"),
        }

        payload = {
            "job_id": self.job_id,
            "analysis_type": "person",
            "summary": summary,
            "persons": persons,
            "frame_image": self.frame_image,
            "result_video": self.RESULT_VIDEO if self.create_video else None,
        }

        os.makedirs(self.result_dir, exist_ok=True)
        _write(os.path.join(self.result_dir, "summary.json"), payload)
        _write(os.path.join(self.result_dir, "persons.json"),
               {"job_id": self.job_id, "persons": persons})
        _write(os.path.join(self.result_dir, "person_events.json"),
               {"job_id": self.job_id, "roi_configured": False, "events": []})
        _write(os.path.join(self.result_dir, "job_metadata.json"), metadata)
        if heatmap:
            _write(os.path.join(self.result_dir, "heatmap.json"),
                   {"job_id": self.job_id, "heatmap": heatmap})

        return {
            "analysis_type": "person",
            "summary": summary,
            "persons": persons,
            "events": [],
            "track_storage": result.get("track_storage"),
            "heatmap": heatmap,
            "frame_image": self.frame_image,
            "result_video": self.video_path_out,
            "metadata": metadata,
        }


def _gender_stats(persons):
    out = {"male": 0, "female": 0, "unknown": 0}
    for p in persons:
        g = (p.get("estimated_gender") or "unknown").lower()
        out[g if g in out else "unknown"] += 1
    return out


def _age_stats(persons, groups):
    out = {label: 0 for label in groups}
    out["unknown"] = 0
    for p in persons:
        label = p.get("age_group")
        out[label if label in out else "unknown"] += 1
    return out


def _track_meta(storage, cfg, result_dir):
    """What job_metadata.json says about the stored track history."""
    tcfg = cfg.get("tracks") or {}
    meta = {
        "file": TRACKS_FILE,
        "sample_fps": float(tcfg.get("sample_fps", 5.0)),
        "bbox_format": "xyxy_pixels",
        "identity": "final_persistent_person_name",
        "samples": None,
    }
    if storage:
        meta["samples"] = storage.get("samples")
        meta["sample_fps"] = storage.get("sample_fps", meta["sample_fps"])
    return meta


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def run_person_analysis(cfg, video_path, result_dir, entry_roi=None, exit_roi=None,
                        progress_cb=None, job_id="job", create_video=False):
    """GPU analysis. The ROI arguments are accepted for backward compatibility
    and ignored - Entry/Exit are applied afterwards by recalculate_roi()."""
    return PersonAnalysis(cfg, video_path, result_dir, entry_roi, exit_roi,
                          progress_cb, job_id, create_video).run()


# ---------------------------------------------------------------------------
# ROI post-processing: apply (or re-apply) Entry/Exit polygons to a finished job
# ---------------------------------------------------------------------------

def recalculate_line(cfg, result_dir, line, inside, job_id="job"):
    """Recompute Entry/Exit for a COMPLETED job from ONE crossing line.

    Crossing the line inward is an entry, outward an exit. Like its polygon
    sibling it replays stored bounding boxes only - no detector, no decode - so
    the line can be redrawn as often as the user likes, long after the source
    video is gone.
    """
    from .roi_replay import replay_line

    def run(tracks_file, roi_cfg, pcfg, analysis_fps, sample_fps):
        return replay_line(tracks_file, line, inside, roi_cfg, pcfg,
                           analysis_fps=analysis_fps, sample_fps=sample_fps)

    rounded = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in line]
    return _recalculate(
        cfg, result_dir, job_id, run,
        summary_extra={
            "roi_mode": "line",
            "crossing_line": rounded,
            "inside_side": int(inside),
            # a crossing IS the count, so these polygon-era fields are moot
            "roi_count_mode": "line",
            "uncounted_visit_count": 0,
            "unmatched_exit_count": 0,
        },
        meta_extra={"roi_mode": "line", "crossing_line": rounded,
                    "inside_side": int(inside),
                    "entry_roi": None, "exit_roi": None},
    )


def recalculate_roi(cfg, result_dir, entry_roi, exit_roi, job_id="job"):
    """Recompute Entry/Exit results for a COMPLETED job from the older
    Entry/Exit polygon pair.

    Kept so jobs counted before the crossing line was introduced can still be
    recomputed; new jobs use recalculate_line().
    """
    from .roi_replay import replay

    def run(tracks_file, roi_cfg, pcfg, analysis_fps, sample_fps):
        return replay(tracks_file, entry_roi, exit_roi, roi_cfg, pcfg,
                      analysis_fps=analysis_fps, sample_fps=sample_fps)

    return _recalculate(
        cfg, result_dir, job_id, run,
        summary_extra={"roi_mode": "polygon"},
        meta_extra={"roi_mode": "polygon",
                    "entry_roi": entry_roi, "exit_roi": exit_roi},
    )


def _recalculate(cfg, result_dir, job_id, run_replay, summary_extra, meta_extra):
    """Shared body: load the stored analysis, replay it, rewrite every
    boundary-derived output. Only the replay differs between a line and a
    polygon pair, so everything else lives here once.
    """
    from .roi_replay import summarize, validate_tracks

    result_dir = str(result_dir)
    tracks_file = os.path.join(result_dir, TRACKS_FILE)
    ok, detail = validate_tracks(tracks_file)
    if not ok:
        raise FileNotFoundError(f"track history unusable: {detail}")

    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    persons = payload.get("persons") or []
    base = payload.get("summary") or {}

    pcfg = cfg.get("person") or {}
    groups = {k: tuple(v) for k, v in (pcfg.get("age_groups") or DEFAULT_AGE_GROUPS).items()}
    bucket = float(pcfg.get("time_bucket_seconds", 300))
    roi_cfg = cfg.get("roi") or {}
    sample_fps = float((cfg.get("tracks") or {}).get("sample_fps", 5.0))

    analysis_fps = float(base.get("fps") or 30.0)
    duration = float(base.get("video_duration_seconds") or 0.0)

    events, times = run_replay(tracks_file, roi_cfg, pcfg, analysis_fps, sample_fps)

    # per-person entry/exit times, rewritten from scratch every time
    for person in persons:
        slot = times.get(person["person_name"]) or {}
        person["entry_time"] = slot.get("entry_time")
        person["exit_time"] = slot.get("exit_time")

    summary = summarize(events, persons, roi_cfg, groups, bucket, duration,
                        base, _bucket_histogram)
    summary.update(summary_extra)

    payload["summary"] = summary
    payload["persons"] = persons
    _write(summary_path, payload)
    _write(os.path.join(result_dir, "persons.json"),
           {"job_id": job_id, "persons": persons})
    _write(os.path.join(result_dir, "person_events.json"),
           {"job_id": job_id, "roi_configured": True, "events": events})

    meta_path = os.path.join(result_dir, "job_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["roi_configured"] = True
        meta.update(meta_extra)
        _write(meta_path, meta)

    return {"summary": summary, "events": events, "persons": persons,
            "times": times}
