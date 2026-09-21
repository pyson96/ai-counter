"""Single-GPU analysis worker.

Runs as its OWN process, separate from FastAPI, and takes one queued job at a
time in FIFO order. Two modes:

    python worker.py                 supervisor: claim jobs, run each one in a
                                     child process, never die because a job did
    python worker.py --run-job ID    the child: actually analyse that one job

The child-process split is deliberate. A CUDA OOM, a corrupt frame that takes
OpenCV down, or a hard crash inside a CUDA kernel kills the child and nothing
else; the supervisor records FAILED and moves to the next job, and the GPU
memory is returned by the OS. It also means the web server keeps serving
uploads no matter what the GPU is doing.

Job lifecycle handled here:

    QUEUED -> PROCESSING -> RESULT_VERIFYING -> COMPLETED
                                              -> source video deleted
                                              -> upload SOURCE_DELETED
    any step failing        -> FAILED, and the source video is KEPT for retry
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import database as db
from analyzer import (TRACKS_FILE, ensure_dirs, load_config, result_dir as job_result_dir,
                      result_video_path, upload_dir)

HERE = os.path.dirname(os.path.abspath(__file__))

PERSON_FILES = ("summary.json", "persons.json", "person_events.json", "job_metadata.json")
VEHICLE_FILES = ("summary.json", "vehicles.json", "vehicle_events.json", "job_metadata.json")


# ---------------------------------------------------------------------------
# the child: one job
# ---------------------------------------------------------------------------

def run_job(job_id):
    cfg = load_config()
    ensure_dirs()
    job = db.get_job(job_id)
    if job is None:
        print(f"[worker] no such job {job_id}", file=sys.stderr)
        return 2
    upload = db.get_upload(job["upload_id"])
    if upload is None:
        _fail(job_id, "upload record is gone")
        return 1
    source = upload["source_path"]
    if not source or not os.path.exists(source):
        _fail(job_id, f"source video missing: {source}")
        return 1

    result_dir = job["result_dir"] or str(job_result_dir(job_id))
    os.makedirs(result_dir, exist_ok=True)
    db.update_job(job_id, status="PROCESSING", started_at=time.time(),
                  error_message=None, result_dir=result_dir)
    db.update_upload(job["upload_id"], status="PROCESSING")

    info = json.loads(upload["video_info"]) if upload["video_info"] else {}
    total_frames = int(info.get("frame_count") or 0)
    db.update_job(job_id, total_frames=total_frames)

    last = [0.0]

    def progress(current, total):
        total = int(total or total_frames or 0)
        pct = (current / total * 100.0) if total else 0.0
        now = time.time()
        if now - last[0] < 0.5:
            return
        last[0] = now
        db.update_job(job_id, analysis_progress=round(min(pct, 99.9), 2),
                      current_frame=int(current), total_frames=total)

    # Rendering is opt-in per job and costs one extra decode of the whole video,
    # so it only happens when the user ticked the box AND the config allows it.
    want_video = bool(job["create_video"]) and bool(
        (cfg.get("storage") or {}).get("allow_result_video", True))
    if job["create_video"] and not want_video:
        print("[worker] result video requested but storage.allow_result_video is "
              "false - rendering skipped", file=sys.stderr)

    try:
        print(f"[worker] {job_id}: {job['analysis_type']} analysis of {source}"
              f"{' (+ result video)' if want_video else ''}")
        if job["analysis_type"] == "person":
            from analyzer.pipeline import run_person_analysis

            # No ROI here by design: Entry/Exit are applied afterwards from the
            # stored track history (analyzer.pipeline.recalculate_roi).
            result = run_person_analysis(cfg, source, result_dir,
                                         progress_cb=progress, job_id=job_id,
                                         create_video=want_video)
            db.save_person_results(job_id, result["persons"], result["events"])
        else:
            from analyzer.vehicle_pipeline import run_vehicle_analysis

            result = run_vehicle_analysis(
                cfg, source, result_dir, progress_cb=progress, job_id=job_id,
                weights_dir=cfg["storage"]["weights_dir"], create_video=want_video)
            db.save_vehicle_results(job_id, result["vehicles"], result["events"])
    except Exception as exc:
        traceback.print_exc()
        # Where it failed matters as much as what failed: a bare
        # "KeyError: 'CoralBadger'" in the UI told us nothing about which of
        # seven call sites raised it. Carry the last few frames along.
        frames = traceback.extract_tb(exc.__traceback__)[-3:]
        where = " <- ".join(f"{f.filename.rsplit(chr(92), 1)[-1]}:{f.lineno} {f.name}"
                            for f in reversed(frames))
        _fail(job_id, f"{type(exc).__name__}: {exc}  @ {where}")
        return 1

    # -- verify before anything is deleted ---------------------------------
    db.update_job(job_id, status="RESULT_VERIFYING", analysis_progress=99.9)
    ok, problem = verify_results(job_id, job["analysis_type"], result_dir,
                                 result["summary"], want_video)
    if not ok:
        _fail(job_id, f"result verification failed: {problem}")
        return 1

    db.update_job(job_id, status="COMPLETED", analysis_progress=100.0,
                  completed_at=time.time(), summary=json.dumps(result["summary"]),
                  current_frame=int(result["summary"].get("frames_processed") or 0))

    # -- only now is the source video allowed to go ------------------------
    storage = cfg.get("storage") or {}
    if storage.get("delete_source_after_success", True) and not storage.get("keep_source_video", False):
        cleanup_job(job["upload_id"], source, result_dir)
        db.update_upload(job["upload_id"], status="SOURCE_DELETED", source_path=None)
        print(f"[worker] {job_id}: verified -> source video deleted")
    else:
        db.update_upload(job["upload_id"], status="COMPLETED")
        print(f"[worker] {job_id}: verified (source kept by config)")
    return 0


def _fail(job_id, message):
    print(f"[worker] {job_id} FAILED: {message}", file=sys.stderr)
    db.update_job(job_id, status="FAILED", error_message=message[:2000],
                  completed_at=time.time())
    job = db.get_job(job_id)
    if job:
        # The source video is deliberately NOT deleted on failure - it is the
        # only way to diagnose the cause or retry the job.
        db.update_upload(job["upload_id"], status="FAILED", error_message=message[:2000])


def verify_results(job_id, analysis_type, result_dir, summary, want_video=False):
    """Everything that must hold before the source video may be deleted."""
    """Everything that must hold before the source video may be deleted."""
    required = PERSON_FILES if analysis_type == "person" else VEHICLE_FILES
    for name in required:
        path = os.path.join(result_dir, name)
        if not os.path.exists(path):
            return False, f"missing {name}"
        if os.path.getsize(path) <= 2:
            return False, f"{name} is empty"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                json.load(fh)
        except Exception as exc:
            return False, f"{name} does not parse: {exc}"

    with open(os.path.join(result_dir, "summary.json"), "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("job_id") != job_id:
        return False, "summary.json job_id mismatch"
    for field in ("frames_processed",):
        if field not in (payload.get("summary") or {}):
            return False, f"summary.{field} missing"
    if analysis_type == "person":
        for field in ("gender", "age_groups", "roi_configured"):
            if field not in payload["summary"]:
                return False, f"summary.{field} missing"
    else:
        for field in ("entry_count", "exit_count"):
            if field not in payload["summary"]:
                return False, f"summary.{field} missing"

    if int(summary.get("frames_processed", 0)) <= 0:
        return False, "no frames were processed"

    rows, events = db.count_results(job_id, analysis_type)
    if analysis_type == "person":
        # A person job is verified WITHOUT Entry/Exit: no ROI exists yet, so a
        # null entry_count and an empty person_events table are the correct
        # state, not a failure. What must hold is the track history, because it
        # is the only thing that can produce ROI results once the source video
        # is gone.
        from analyzer.roi_replay import validate_tracks

        ok, detail = validate_tracks(os.path.join(result_dir, TRACKS_FILE))
        if not ok:
            return False, f"track history: {detail}"
        if rows == 0 and int(summary.get("unique_persons_detected", 0)) > 0:
            return False, "people were detected but no person rows were committed"
    else:
        entries = int(summary.get("entry_count") or 0)
        exits = int(summary.get("exit_count") or 0)
        if entries + exits > 0 and events == 0:
            return False, "summary counts events but no event rows were committed"
        if entries > 0 and rows == 0:
            return False, "entries counted but no vehicle rows were committed"

    # the result video: required when the job asked for one, forbidden otherwise
    videos = []
    for root, _dirs, files in os.walk(result_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                videos.append(os.path.join(root, name))
    if want_video:
        path = str(result_video_path(job_id))
        if not os.path.exists(path):
            return False, "a result video was requested but none was produced"
        if os.path.getsize(path) < 1024:
            return False, "the result video is empty"
        import cv2

        cap = cv2.VideoCapture(path)
        readable = cap.isOpened() and cap.read()[0]
        cap.release()
        if not readable:
            return False, "the result video cannot be decoded"
    elif videos:
        return False, (f"a result video was produced ({os.path.basename(videos[0])}) "
                       f"but this job did not ask for one")
    return True, ""


def cleanup_job(upload_id, source, result_dir):
    """The source video, the whole upload directory, and every temporary
    analysis artifact. Results - JSON, the still, tracks.jsonl.gz and any
    rendered video - are kept."""
    # the upload directory holds both upload.part and the assembled source
    updir = upload_dir(upload_id)
    if updir.exists():
        shutil.rmtree(updir, ignore_errors=True)
    elif source and os.path.exists(source):
        try:
            os.remove(source)
        except OSError as exc:
            print(f"[worker] could not delete {source}: {exc}", file=sys.stderr)

    # raw track file (normally already consumed by finalize_tracks), debug crops
    for name in ("tracks.raw.jsonl",):
        path = os.path.join(result_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    for sub in ("raw/reid_debug", "frames", "crops"):
        path = os.path.join(result_dir, *sub.split("/"))
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# the supervisor
# ---------------------------------------------------------------------------

def supervise(poll_seconds=None, once=False):
    cfg = load_config()
    ensure_dirs()
    db.init_db()
    poll = float(poll_seconds or (cfg.get("analysis") or {}).get("poll_seconds", 2.0))
    max_jobs = int((cfg.get("analysis") or {}).get("max_gpu_jobs", 1))
    if max_jobs != 1:
        print(f"[worker] analysis.max_gpu_jobs is {max_jobs}; this worker still runs "
              f"one job at a time - one GPU, one video.")

    requeued = db.reset_stale_jobs()
    if requeued:
        print(f"[worker] re-queued {len(requeued)} job(s) left PROCESSING by a previous run")

    print(f"[worker] watching the queue every {poll:.1f}s  (pid {os.getpid()})")
    while True:
        job = db.claim_next_job()
        if job is None:
            if once:
                return 0
            time.sleep(poll)
            continue

        job_id = job["job_id"]
        print(f"[worker] --> {job_id}  {job['analysis_type']}  {job['filename']}")
        started = time.time()
        cmd = [sys.executable, os.path.join(HERE, "worker.py"), "--run-job", job_id]
        try:
            code = subprocess.call(cmd, cwd=HERE)
        except Exception as exc:
            code = -1
            print(f"[worker] could not spawn the job process: {exc}", file=sys.stderr)

        row = db.get_job(job_id)
        if code != 0 and row and row["status"] not in ("FAILED", "COMPLETED"):
            # the child died without recording why (segfault, OOM-killer, ...)
            _fail(job_id, f"analysis process exited with code {code}")
        print(f"[worker] <-- {job_id} finished in {time.time() - started:.1f}s "
              f"(status {(db.get_job(job_id) or {}).get('status')})")
        if once:
            return 0 if code == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Single-GPU analysis worker")
    ap.add_argument("--run-job", default=None, help="run exactly this job, then exit")
    ap.add_argument("--once", action="store_true", help="process at most one queued job")
    ap.add_argument("--poll", type=float, default=None, help="queue poll interval in seconds")
    args = ap.parse_args(argv)

    if args.run_job:
        return run_job(args.run_job)
    return supervise(args.poll, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
