"""Entry / Exit ROI results, recomputed from stored track history.

This is the second half of person analysis. GPU analysis writes every visible
person's bbox to `tracks.jsonl.gz` at ~5 Hz of video time, keyed by the FINAL
ReID identity; this module replays those boxes through the same state machine
the live pipeline used and produces the entry/exit results.

Nothing here decodes video or touches the GPU:

    tracks.jsonl.gz
      -> bbox  -> pipeline.person_ground_point()   (the bottom-band rule)
      -> pipeline.ROIMonitor                        (debounced transitions)
      -> per-identity de-duplication                (the same rule as live)
      -> ENTRY / EXIT events, per-person times, summary

So the ROI can be drawn, redrawn and re-applied any number of times after the
source video has been deleted.

Timing note
-----------
The live monitor debounces in FRAMES (`enter_frames: 3` at 30 fps = 0.1 s). The
stored history is only 5 Hz, so reusing 3 there would mean 0.6 s - six times
stricter. The frame thresholds are therefore converted to seconds using the
analysed video's fps and back into samples at the storage rate, which keeps the
behaviour approximately equivalent.
"""

from __future__ import annotations

import gzip
import json
import math
import os

from . import PROJECT_ROOT, TRACKS_FILE  # noqa: F401  (sys.path bootstrap)

import pipeline as core

ROIMonitor = core.ROIMonitor
person_ground_point = core.person_ground_point
point_in_polygon = core.point_in_polygon


def read_tracks(path):
    """Yield one decoded sample per line. Streaming - the file is never held."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def validate_tracks(path, max_lines=None):
    """(ok, detail) - the gate that decides whether the source may be deleted."""
    if not os.path.exists(path):
        return False, f"{os.path.basename(path)} does not exist"
    if os.path.getsize(path) <= 0:
        return False, f"{os.path.basename(path)} is empty"
    lines = people = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "frame" not in rec or "time" not in rec or "people" not in rec:
                    return False, f"line {lines + 1} is missing frame/time/people"
                for entry in rec["people"]:
                    box = entry.get("bbox")
                    if not isinstance(box, list) or len(box) != 4:
                        return False, f"line {lines + 1} has a malformed bbox"
                    people += 1
                lines += 1
                if max_lines and lines >= max_lines:
                    break
    except (OSError, EOFError, json.JSONDecodeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if lines == 0:
        return False, "no track samples were written"
    return True, f"{lines} samples, {people} boxes"


def replay_line(tracks_file, line, inside, roi_cfg, person_cfg,
                analysis_fps=30.0, sample_fps=5.0):
    """Replay stored boxes across ONE line: outside->inside enters, the reverse exits.

    Same shape of output as the polygon replay, so the database, the summary and
    the dashboard need no special case.
    """
    band = float((person_cfg or {}).get("ground_band_ratio",
                                        core.DEFAULT_GROUND_BAND_RATIO))
    crossing = core.build_crossing_line(
        {**(roi_cfg or {}), "line": line, "inside": inside},
        analysis_fps=analysis_fps, sample_fps=sample_fps)
    if crossing is None:
        return [], {}
    min_repeat = float((roi_cfg or {}).get("min_repeat_seconds", 2.0))

    events = []
    last_fired = {}                  # (owner, direction) -> time of last count
    for rec in read_tracks(tracks_file):
        t_now = float(rec["time"])
        frame_idx = int(rec.get("frame", 0))
        for entry in rec["people"]:
            tid = int(entry["local_track_id"])
            point = person_ground_point(entry["bbox"], band)
            direction = crossing.update(tid, point, t_now)
            if not direction:
                continue
            owner = entry.get("person_name") or f"track:{tid}"
            key = (owner, direction)
            last = last_fired.get(key)
            if last is not None and t_now - last < min_repeat:
                continue             # same person, same direction, moments ago
            last_fired[key] = t_now
            events.append({
                "time": round(t_now, 3),
                "frame": frame_idx,
                "event": direction,          # "entry" | "exit"
                "local_track_id": tid,
                "person_name": entry.get("person_name"),
                "point": [round(point[0], 1), round(point[1], 1)],
                "counted": True,             # crossing IS the event
                "matched": True,
            })

    events.sort(key=lambda e: e["time"])
    times = {}
    for ev in events:
        name = ev.get("person_name")
        if not name:
            continue
        slot = times.setdefault(name, {"entry_time": None, "exit_time": None})
        if ev["event"] == "entry":
            if slot["entry_time"] is None:
                slot["entry_time"] = ev["time"]      # first crossing in
        else:
            slot["exit_time"] = ev["time"]           # last crossing out
    return events, times


def count_mode(roi_cfg):
    """Which counting rule this job uses - see pipeline.classify_roi_events."""
    mode = str((roi_cfg or {}).get("count_mode", "transition")).lower()
    return mode if mode in core.ROI_COUNT_MODES else "transition"


def _thresholds(roi_cfg, analysis_fps, sample_fps):
    """Frame thresholds tuned at the analysis rate, restated at the sample rate."""
    analysis_fps = float(analysis_fps or 30.0) or 30.0
    sample_fps = float(sample_fps or 5.0) or 5.0
    enter_seconds = float(roi_cfg.get("enter_frames", 3)) / analysis_fps
    exit_seconds = float(roi_cfg.get("exit_frames", 5)) / analysis_fps
    return (max(1, int(math.ceil(enter_seconds * sample_fps))),
            max(1, int(math.ceil(exit_seconds * sample_fps))),
            enter_seconds, exit_seconds)


def replay(tracks_file, entry_roi, exit_roi, roi_cfg, person_cfg,
           analysis_fps=30.0, sample_fps=5.0):
    """Run the ROI state machine over the stored history.

    Returns (events, per_person_times). Events carry the same shape the live
    pipeline emitted, so every downstream consumer is unchanged.
    """
    band = float((person_cfg or {}).get("ground_band_ratio",
                                        core.DEFAULT_GROUND_BAND_RATIO))
    enter_n, exit_n, enter_s, exit_s = _thresholds(roi_cfg, analysis_fps, sample_fps)
    mode = count_mode(roi_cfg)
    require_order = bool(roi_cfg.get("require_entry_before_exit", False))
    min_repeat = float(roi_cfg.get("min_repeat_seconds", 2.0))
    first_inside = bool(roi_cfg.get("count_first_seen_inside", False))

    monitors = []
    for kind, polygon in (("entry", entry_roi), ("exit", exit_roi)):
        if polygon and len(polygon) >= 3:
            monitors.append(ROIMonitor(kind, polygon, enter_frames=enter_n,
                                       exit_frames=exit_n,
                                       count_first_seen_inside=first_inside))
    if not monitors:
        return [], {}

    events = []
    last_fired = {}          # (owner, kind) -> time of the last counted event
    seen_tracks = set()

    for rec in read_tracks(tracks_file):
        t_now = float(rec["time"])
        frame_idx = int(rec.get("frame", 0))
        present = set()
        for entry in rec["people"]:
            tid = int(entry["local_track_id"])
            present.add(tid)
            seen_tracks.add(tid)
            point = person_ground_point(entry["bbox"], band)
            owner = entry.get("person_name") or f"track:{tid}"
            for monitor in monitors:
                if not monitor.update(tid, point, t_now):
                    continue
                key = (owner, monitor.kind)
                last = last_fired.get(key)
                if last is not None and t_now - last < min_repeat:
                    continue          # same identity, same ROI, moments ago
                last_fired[key] = t_now
                events.append({
                    "time": round(t_now, 3),
                    "frame": frame_idx,
                    "event": monitor.kind,
                    "local_track_id": tid,
                    "person_name": entry.get("person_name"),
                    "point": [round(point[0], 1), round(point[1], 1)],
                })
        # a track that vanished for good should not keep its lane state around
        for monitor in monitors:
            for tid in list(monitor.states):
                if tid not in present and tid not in seen_tracks:
                    monitor.drop(tid)

    # Which visits count: the pipeline's own rule, applied to the final names
    # the track file already carries.
    events = core.classify_roi_events(events, mode, require_order)

    times = {}
    for ev in events:
        name = ev.get("person_name")
        if not name or not ev.get("counted", True):
            continue
        slot = times.setdefault(name, {"entry_time": None, "exit_time": None})
        if ev["event"] == "entry":
            if slot["entry_time"] is None:
                slot["entry_time"] = ev["time"]          # first counted entry
        else:
            slot["exit_time"] = ev["time"]               # last counted exit
    return events, times


def summarize(events, persons, roi_cfg, groups, bucket_seconds, duration,
              base_summary, bucket_fn):
    """Fold ROI results into the ROI-independent summary produced at analysis."""
    require_order = bool(roi_cfg.get("require_entry_before_exit", False))
    mode = count_mode(roi_cfg)
    entries = [e for e in events
               if e["event"] == "entry" and e.get("counted", True)]
    all_exits = [e for e in events if e["event"] == "exit"]
    exits = [e for e in all_exits if e.get("counted", True)]
    uncounted = [e for e in events if not e.get("counted", True)]

    entered = {e["person_name"] for e in entries if e.get("person_name")}
    counted = [p for p in persons if p["person_name"] in entered] or persons

    gender = {"male": 0, "female": 0, "unknown": 0}
    for p in counted:
        g = (p.get("estimated_gender") or "unknown").lower()
        gender[g if g in gender else "unknown"] += 1
    ages = {label: 0 for label in groups}
    ages["unknown"] = 0
    for p in counted:
        label = p.get("age_group")
        ages[label if label in ages else "unknown"] += 1

    summary = dict(base_summary)
    summary.update({
        "roi_configured": True,
        "entry_count": len(entries),
        "exit_count": len(exits),
        # visits that were recorded but did not count - in transition mode,
        # somebody who reached a region without coming from the other one
        "uncounted_visit_count": len(uncounted),
        "unmatched_exit_count": len(all_exits) - len(exits),
        "roi_count_mode": mode,
        "require_entry_before_exit": require_order,
        "unique_persons_entered": len(entered),
        "gender": gender,
        "age_groups": ages,
        "time_bucket_seconds": bucket_seconds,
        "entries_by_time": bucket_fn([e["time"] for e in entries],
                                     bucket_seconds, duration),
        "exits_by_time": bucket_fn([e["time"] for e in exits],
                                   bucket_seconds, duration),
    })
    return summary
