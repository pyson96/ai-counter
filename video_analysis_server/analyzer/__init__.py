"""Analysis package for the web server.

Nothing in here re-implements a detector, a tracker, SOLIDER or MiVOLO. The
project root already owns those and they stay the single implementation:

    <root>/models.py        YOLO11 person detector, ByteTrack, SOLIDER ReID
                            extractor, MiVOLO demographics, face detector
    <root>/solider_swin.py  the SOLIDER Swin backbone itself
    <root>/pipeline.py      tracking -> ReID identity -> demographics, plus the
                            bottom-band ground point and the Entry/Exit ROI
                            state machine
    <root>/detect_car.py    vehicle detector, plate detector, PP-OCRv5 Korean
                            recogniser and the per-track plate voting

This package adds the parts that only the web workflow needs: ROI-driven
entry/exit counting on top of the person pipeline, and the vehicle
ENTRY / LEFT_FRAME / EXIT state machine on top of the plate pipeline.

It also owns config loading, because the server, the worker and the analysis
code all need exactly the same merged view of the two config files. Importing
this module is cheap - it pulls in no torch, no OpenCV and no FastAPI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_ROOT = os.path.dirname(SERVER_ROOT)
PROJECT_ROOT = os.path.abspath(os.path.join(SERVER_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONFIG_PATH = os.path.join(SERVER_ROOT, "config.yaml")
DATA_DIR = os.path.join(SERVER_ROOT, "data")          # SQLite only
STATIC_DIR = os.path.join(SERVER_ROOT, "static")

# Re-pointed at `storage.video_root` once the config is read - see the bottom of
# this module. Uploads and results are large and per-job; the database is small
# and must stay next to the server, so only these three move.
VIDEO_ROOT = Path(DATA_DIR)
UPLOAD_ROOT = VIDEO_ROOT / "uploads"
RESULT_ROOT = VIDEO_ROOT / "results"

RESULT_VIDEO = "result.mp4"
FRAME_IMAGE = "frame.jpg"
TRACKS_FILE = "tracks.jsonl.gz"
TRACKS_RAW = "tracks.raw.jsonl"


def ensure_dirs():
    for path in (Path(DATA_DIR), VIDEO_ROOT, UPLOAD_ROOT, RESULT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def upload_dir(upload_id):
    """Everything belonging to one upload: the .part file and the assembled
    source video. Deleting the directory deletes the upload."""
    return UPLOAD_ROOT / str(upload_id)


def result_dir(job_id):
    """Everything belonging to one job: JSON, the still, the track history and
    the optional rendered video."""
    return RESULT_ROOT / str(job_id)


def result_video_path(job_id):
    return result_dir(job_id) / RESULT_VIDEO


def tracks_path(job_id):
    return result_dir(job_id) / TRACKS_FILE


def _deep_merge(base, extra):
    out = dict(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


_cache = {}


def load_config(path=None, reload=False):
    """Server config on top of the project-root analysis config.

    `analysis.base_config` in the server config points at the root config.yaml;
    the root is loaded first so every detector / ReID / MiVOLO threshold the
    project already tuned stays in force, and the server file only overrides
    what the web workflow changes.
    """
    import yaml

    path = os.path.abspath(path or CONFIG_PATH)
    if not reload and path in _cache:
        return _cache[path]

    with open(path, "r", encoding="utf-8") as fh:
        server_cfg = yaml.safe_load(fh) or {}

    base_ref = ((server_cfg.get("analysis") or {}).get("base_config")
                or os.path.join(PROJECT_ROOT, "config.yaml"))
    base_path = base_ref if os.path.isabs(base_ref) else os.path.join(
        os.path.dirname(path), base_ref)
    base_cfg = {}
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as fh:
            base_cfg = yaml.safe_load(fh) or {}

    cfg = _deep_merge(base_cfg, server_cfg)

    # Model weights are shared with the CLI tools, so the paths in the root
    # config are relative to the project root - make them absolute here.
    weights_dir = (cfg.get("storage") or {}).get("weights_dir", "weights")
    if not os.path.isabs(weights_dir):
        weights_dir = os.path.abspath(os.path.join(SERVER_ROOT, weights_dir))
    cfg.setdefault("storage", {})["weights_dir"] = weights_dir

    # Where uploads and per-job results live. Relative paths resolve against the
    # server folder; forward slashes are fine on Windows.
    root = (cfg.get("storage") or {}).get("video_root") or DATA_DIR
    root = Path(root)
    if not root.is_absolute():
        root = Path(SERVER_ROOT) / root
    cfg["storage"]["video_root"] = str(root)

    # Where the SQLite database lives. Unset -> beside the server under data/,
    # which is what every install had before this became configurable, so an
    # untouched config keeps its database exactly where it was.
    db = (cfg.get("storage") or {}).get("db_path")
    if db:
        db = Path(db)
        if not db.is_absolute():
            db = Path(SERVER_ROOT) / db
        cfg["storage"]["db_path"] = str(db)
    for section, field in (("detection", "model"), ("demographics", "face_detector")):
        value = (cfg.get(section) or {}).get(field)
        if value and not os.path.isabs(value):
            cfg[section][field] = os.path.join(weights_dir, os.path.basename(value))
    reid = cfg.setdefault("reid", {})
    weights = str(reid.get("weights") or "auto").strip()
    if weights.lower() in ("auto", ""):
        # models.ReIDExtractor would resolve `auto` against the CWD, and the
        # worker's CWD is not the project root - resolve it here instead.
        weights = os.path.join(weights_dir, f"{reid['model']}_msmt17.pth")
    elif not os.path.isabs(weights):
        weights = os.path.join(weights_dir, os.path.basename(weights))
    reid["weights"] = weights

    # Rendering is OFF by default and is turned on per job, never globally:
    # `storage.allow_result_video` is only the permission to OFFER the option,
    # and PersonAnalysis / VehicleAnalysis set `video.render` for the one job
    # whose user ticked the box.
    storage = cfg.setdefault("storage", {})
    storage["allow_result_video"] = bool(storage.get("allow_result_video", True))
    cfg.setdefault("video", {})["render"] = False

    _cache[path] = cfg
    return cfg


def _apply_video_root():
    """Point the storage roots at `storage.video_root`.

    Done once at import so `from analyzer import UPLOAD_ROOT` gives every module
    the same answer. An unreadable config leaves the data/ defaults in place
    rather than breaking the import.
    """
    global VIDEO_ROOT, UPLOAD_ROOT, RESULT_ROOT
    try:
        root = (load_config().get("storage") or {}).get("video_root")
    except Exception as exc:                      # noqa: BLE001 - see docstring
        print(f"[analyzer] cannot read storage.video_root ({exc}); "
              f"falling back to {DATA_DIR}")
        return
    if not root:
        return
    VIDEO_ROOT = Path(root)
    UPLOAD_ROOT = VIDEO_ROOT / "uploads"
    RESULT_ROOT = VIDEO_ROOT / "results"


_apply_video_root()


__all__ = [
    "PROJECT_ROOT", "SERVER_ROOT", "CONFIG_PATH", "DATA_DIR", "STATIC_DIR",
    "VIDEO_ROOT", "UPLOAD_ROOT", "RESULT_ROOT",
    "RESULT_VIDEO", "FRAME_IMAGE", "TRACKS_FILE", "TRACKS_RAW",
    "ensure_dirs", "upload_dir", "result_dir", "result_video_path",
    "tracks_path", "load_config",
]
