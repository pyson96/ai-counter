"""FastAPI front end for the video analysis server (default port 7802).

This process never touches the GPU and never calls an Analyzer. It accepts
resumable chunked uploads, validates the finished video, stores job metadata and
hands work to the queue; worker.py picks jobs up in its own process. That split
is what lets Videos B, C and D keep uploading while Video A is on the GPU, and
what stops a CUDA failure from taking the web server down with it.

    python server.py            # http://0.0.0.0:7802
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import database as db
from analyzer import (FRAME_IMAGE, SERVER_ROOT, STATIC_DIR, TRACKS_FILE, ensure_dirs,
                      load_config, result_dir as job_result_dir,
                      result_video_path, upload_dir)

CFG = load_config()
ensure_dirs()
db.init_db()

app = FastAPI(title="Video Analysis Server", version="1.0")

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
WRITE_BUFFER = 8 * 1024 * 1024      # RAM ceiling per in-flight chunk request


def _safe_filename(name):
    name = os.path.basename(name or "video.mp4").strip()
    name = SAFE_NAME.sub("_", name)
    return name[:180] or "video.mp4"


def _make_sparse(fh):
    """Ask NTFS to treat this file as sparse. No-op elsewhere / on failure."""
    if os.name != "nt":
        return
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        FSCTL_SET_SPARSE = 0x000900C4
        handle = msvcrt.get_osfhandle(fh.fileno())
        returned = wintypes.DWORD()
        ctypes.windll.kernel32.DeviceIoControl(
            wintypes.HANDLE(handle), wintypes.DWORD(FSCTL_SET_SPARSE),
            None, 0, None, 0, ctypes.byref(returned), None)
    except Exception:                              # noqa: BLE001 - best effort
        pass


def _chunk_size():
    return int((CFG.get("upload") or {}).get("chunk_size_mb", 32)) * 1024 * 1024


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------

class UploadInit(BaseModel):
    filename: str
    file_size: int = Field(gt=0)
    chunk_size: int | None = None
    # which build of upload.js is asking. A phone that never reloaded the page
    # keeps running old code for days, and that is invisible from the server
    # unless the client says so.
    client_version: str | None = None


class JobCreate(BaseModel):
    upload_id: str
    analysis_type: str                       # "person" | "vehicle"
    start: bool = False                      # vehicle jobs can queue immediately
    create_video: bool = False               # opt-in annotated result video


class ROIConfig(BaseModel):
    """Where the counting boundary is, applied to a finished job.

    Two shapes are accepted. A `line` with an `inside` side is the current one:
    crossing it outward is an exit, inward an entry. The Entry/Exit polygon pair
    is kept so results produced before the change can still be recomputed.
    """
    line: list[list[float]] | None = None
    inside: int | None = None
    entry_roi: list[list[float]] | None = None
    exit_roi: list[list[float]] | None = None


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------

@app.post("/api/uploads/init")
def upload_init(body: UploadInit):
    """Start (or resume) an upload and tell the browser where to continue from."""
    filename = _safe_filename(body.filename)
    chunk_size = int(body.chunk_size or _chunk_size())
    max_bytes = int((CFG.get("upload") or {}).get("max_file_gb", 200)) * 1024 ** 3
    if body.file_size > max_bytes:
        raise HTTPException(413, f"file larger than the configured limit ({max_bytes} bytes)")

    # Resume whatever is already on disk for this file, even if the client now
    # proposes a different chunk size - the client adopts the size the server
    # reports, and refusing to match here used to start a SECOND upload from
    # zero (and pre-allocate a second full-size .part) every time the mobile
    # build changed its preferred chunk size.
    print(f"[upload] init {filename!r} "
          f"{body.file_size / 1073741824:.2f} GB  client={body.client_version or 'OLD (버전 미전송)'}",
          flush=True)

    existing = db.find_resumable_upload(filename, body.file_size)
    if existing and os.path.exists(existing["part_path"]):
        return _upload_payload(existing, resumed=True)

    upload_id = uuid.uuid4().hex[:16]
    total_chunks = max(1, -(-body.file_size // chunk_size))
    updir = upload_dir(upload_id)
    updir.mkdir(parents=True, exist_ok=True)
    part_path = str(updir / "upload.part")
    # Pre-create the file so chunks can be written at their own offset in any
    # order - that is what makes an interrupted upload resumable. It is marked
    # SPARSE first: NTFS otherwise commits the whole final size immediately, so
    # three abandoned 48 GB attempts really did occupy 144 GB of disk.
    with open(part_path, "wb") as fh:
        _make_sparse(fh)
        fh.truncate(body.file_size)
    row = db.create_upload(upload_id, filename, body.file_size, chunk_size,
                           total_chunks, part_path)
    return _upload_payload(row, resumed=False)


def _upload_payload(row, resumed=False):
    chunk_map = row["chunk_map"] or ""
    missing = [i for i, bit in enumerate(chunk_map) if bit == "0"]
    return {
        "upload_id": row["upload_id"],
        "filename": row["filename"],
        "file_size": row["file_size"],
        "chunk_size": row["chunk_size"],
        "total_chunks": row["total_chunks"],
        "received_bytes": row["received_bytes"],
        "upload_progress": row["upload_progress"],
        "status": row["status"],
        "resumed": resumed,
        "received_chunks": row["total_chunks"] - len(missing),
        # NOT truncated: a 48 GB file at 8 MB is 5679 chunks, and cutting the
        # list at 5000 made the client treat the tail as already received - it
        # would stop early and then fail completion forever.
        "missing_chunks": missing,
        "next_chunk": missing[0] if missing else None,
        "created_at": row["created_at"],
    }


@app.put("/api/uploads/{upload_id}/chunk")
async def upload_chunk(upload_id: str, request: Request, index: int = Query(..., ge=0)):
    """Stream one chunk straight to its offset in the .part file.

    The request body is never buffered whole: it is read in stream slices and
    flushed to disk in <= WRITE_BUFFER blocks, so a 64 MB chunk of a 35 GB file
    costs 8 MB of RAM, not 35 GB.
    """
    row = db.get_upload(upload_id)
    if row is None:
        raise HTTPException(404, "unknown upload_id")
    if row["status"] not in ("UPLOADING", "FAILED"):
        raise HTTPException(409, f"upload is {row['status']}, not accepting chunks")
    if index >= row["total_chunks"]:
        raise HTTPException(400, "chunk index out of range")

    offset = index * row["chunk_size"]
    expected = min(row["chunk_size"], row["file_size"] - offset)
    part_path = row["part_path"]

    written = 0
    buffer = bytearray()

    def _flush(payload, at):
        with open(part_path, "r+b") as fh:
            fh.seek(at)
            fh.write(payload)

    try:
        async for piece in request.stream():
            if not piece:
                continue
            buffer.extend(piece)
            if len(buffer) >= WRITE_BUFFER:
                payload = bytes(buffer)
                buffer.clear()
                if written + len(payload) > expected:
                    raise HTTPException(400, "chunk larger than declared chunk_size")
                await run_in_threadpool(_flush, payload, offset + written)
                written += len(payload)
        if buffer:
            payload = bytes(buffer)
            if written + len(payload) > expected:
                raise HTTPException(400, "chunk larger than declared chunk_size")
            await run_in_threadpool(_flush, payload, offset + written)
            written += len(payload)
    except HTTPException:
        raise
    except Exception as exc:                      # network died mid-chunk
        # The chunk bit is NOT set, so the browser simply sends this chunk again.
        raise HTTPException(499, f"chunk aborted after {written} bytes: {exc}")

    if written != expected:
        # A short chunk means the connection dropped; leave the bit clear.
        raise HTTPException(400, f"expected {expected} bytes, received {written}")

    row = await run_in_threadpool(db.mark_chunk_received, upload_id, index, written)
    return {
        "upload_id": upload_id,
        "index": index,
        "received_bytes": row["received_bytes"],
        "upload_progress": row["upload_progress"],
    }


class ClientLog(BaseModel):
    event: str
    detail: str | None = None
    upload_id: str | None = None
    version: str | None = None


@app.post("/api/net-test")
async def net_test(request: Request):
    """Swallow a body of any size and report how much actually arrived.

    A phone that polls /api/health happily but cannot PUT an 8 MB chunk is
    telling us about the network path, not about the upload logic. This finds
    the ceiling directly - no real upload is touched, nothing is written.
    """
    total = 0
    async for piece in request.stream():
        total += len(piece)
    who = request.client.host if request.client else "?"
    print(f"[net-test {who}] {total} bytes 수신", flush=True)
    return {"received": total}


@app.post("/api/client-log")
async def client_log(body: ClientLog, request: Request):
    """Diagnostics from the browser.

    A phone's console is unreachable, and an upload that dies inside the browser
    leaves nothing on the server at all - which is how a stuck 30 GB upload
    turned into hours of guesswork. Anything the client knows about a failure it
    reports here, and it lands in the same log as everything else.
    """
    who = request.client.host if request.client else "?"
    print(f"[client {body.version or '?'} {who}] {body.event} "
          f"{body.upload_id or ''} {(body.detail or '')[:400]}", flush=True)
    return {"ok": True}


@app.get("/api/uploads/{upload_id}/status")
def upload_status(upload_id: str):
    row = db.get_upload(upload_id)
    if row is None:
        raise HTTPException(404, "unknown upload_id")
    payload = _upload_payload(row)
    payload["video_info"] = json.loads(row["video_info"]) if row["video_info"] else None
    payload["error_message"] = row["error_message"]
    return payload


@app.post("/api/uploads/{upload_id}/complete")
async def upload_complete(upload_id: str):
    """Assemble, size-check and probe the video before any job may be created."""
    row = db.get_upload(upload_id)
    if row is None:
        raise HTTPException(404, "unknown upload_id")
    if row["status"] in ("UPLOADED", "WAITING_CONFIG", "QUEUED", "PROCESSING", "COMPLETED"):
        return upload_status(upload_id)

    missing = [i for i, bit in enumerate(row["chunk_map"]) if bit == "0"]
    if missing:
        raise HTTPException(409, {"error": "upload incomplete",
                                  "missing_chunks": missing[:5000],
                                  "next_chunk": missing[0]})

    db.update_upload(upload_id, status="UPLOAD_VERIFYING", error_message=None)
    result = await run_in_threadpool(_finalise_upload, upload_id)
    if not result["ok"]:
        db.update_upload(upload_id, status="FAILED", error_message=result["error"])
        raise HTTPException(400, result["error"])
    return upload_status(upload_id)


def _finalise_upload(upload_id):
    row = db.get_upload(upload_id)
    part_path, size = row["part_path"], row["file_size"]
    try:
        actual = os.path.getsize(part_path)
    except OSError as exc:
        return {"ok": False, "error": f"source file missing: {exc}"}
    if actual != size:
        return {"ok": False, "error": f"size mismatch: on disk {actual}, declared {size}"}

    # <video_root>/uploads/<upload_id>/source<ext> - one directory per upload,
    # so cleanup is a single rmtree and two uploads can never collide.
    ext = os.path.splitext(row["filename"])[1].lower() or ".mp4"
    dest = str(upload_dir(upload_id) / f"source{ext}")
    try:
        os.replace(part_path, dest)
    except OSError:
        shutil.move(part_path, dest)

    info = probe_video(dest)
    if not info.get("ok"):
        return {"ok": False, "error": info.get("error", "video validation failed")}

    db.update_upload(upload_id, status="UPLOADED", source_path=dest,
                     video_info=json.dumps(info), error_message=None)
    return {"ok": True}


def probe_video(path):
    """Validate the uploaded file really is a decodable video.

    OpenCV is the authority because it is what the analysis pipelines decode
    with - if OpenCV cannot read a frame, the job would fail anyway. ffprobe is
    consulted as well when it is installed, for the container/codec detail.
    """
    import cv2

    if not os.path.exists(path):
        return {"ok": False, "error": "file not found after assembly"}
    size = os.path.getsize(path)
    if size <= 0:
        return {"ok": False, "error": "file is empty"}

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return {"ok": False, "error": "video cannot be opened (unsupported or corrupt container)"}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"ok": False, "error": "video opened but no frame could be decoded"}
    if width <= 0 or height <= 0:
        return {"ok": False, "error": f"invalid resolution {width}x{height}"}
    if fps <= 0:
        fps = 30.0

    info = {
        "ok": True,
        "file_size": size,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frames,
        "duration_seconds": round(frames / fps, 2) if frames else None,
        "probe": "opencv",
    }
    info.update(_ffprobe(path))
    return info


def _ffprobe(path):
    exe = shutil.which("ffprobe")
    if not exe:
        return {"ffprobe": "not installed"}
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height,nb_frames,avg_frame_rate:format=format_name,duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return {"ffprobe": f"failed: {out.stderr.strip()[:200]}"}
        data = json.loads(out.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        return {
            "codec": stream.get("codec_name"),
            "container": fmt.get("format_name"),
            "ffprobe_duration": float(fmt["duration"]) if fmt.get("duration") else None,
            "ffprobe": "ok",
        }
    except Exception as exc:
        return {"ffprobe": f"error: {exc}"}


@app.get("/api/uploads")
def list_uploads():
    """Every registered video plus whatever its job is doing, for the UI list."""
    uploads = db.list_uploads()
    jobs = {j["upload_id"]: j for j in db.list_jobs()}
    out = []
    for u in uploads:
        payload = _upload_payload(u)
        payload.pop("missing_chunks", None)
        payload["video_info"] = json.loads(u["video_info"]) if u["video_info"] else None
        payload["error_message"] = u["error_message"]
        job = jobs.get(u["upload_id"])
        payload["job"] = _job_payload(job) if job else None
        out.append(payload)
    return {"uploads": out}


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: str):
    row = db.get_upload(upload_id)
    if row is None:
        raise HTTPException(404, "unknown upload_id")
    job = db.get_job_for_upload(upload_id)
    if job and job["status"] in ("QUEUED", "PROCESSING", "RESULT_VERIFYING"):
        raise HTTPException(409, "job is queued or running")
    updir = upload_dir(upload_id)
    if updir.exists():
        shutil.rmtree(updir, ignore_errors=True)
    for path in (row["part_path"], row["source_path"]):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    db.delete_upload(upload_id)
    return {"deleted": upload_id}


@app.get("/api/uploads/{upload_id}/frame")
def upload_frame(upload_id: str, at: float = Query(0.0, ge=0.0)):
    """A representative JPEG frame, for drawing the Entry / Exit ROIs on."""
    import cv2

    row = db.get_upload(upload_id)
    if row is None or not row["source_path"] or not os.path.exists(row["source_path"]):
        raise HTTPException(404, "source video not available")
    cap = cv2.VideoCapture(row["source_path"])
    if not cap.isOpened():
        cap.release()
        raise HTTPException(400, "cannot open video")
    if at > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000.0)
    ok, frame = cap.read()
    if not ok:                                   # asked past the end - take frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, "cannot decode a frame")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(500, "cannot encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def _job_payload(job):
    if job is None:
        return None
    out = dict(job)
    out["entry_roi"] = json.loads(job["entry_roi"]) if job.get("entry_roi") else None
    out["exit_roi"] = json.loads(job["exit_roi"]) if job.get("exit_roi") else None
    out["crossing_line"] = (json.loads(job["crossing_line"])
                            if job.get("crossing_line") else None)
    out["summary"] = json.loads(job["summary"]) if job.get("summary") else None
    return out


@app.post("/api/jobs")
def create_job(body: JobCreate):
    if body.analysis_type not in ("person", "vehicle"):
        raise HTTPException(400, "analysis_type must be 'person' or 'vehicle'")
    upload = db.get_upload(body.upload_id)
    if upload is None:
        raise HTTPException(404, "unknown upload_id")
    if upload["status"] not in ("UPLOADED", "WAITING_CONFIG"):
        raise HTTPException(409, f"upload is {upload['status']}, must be UPLOADED")

    existing = db.get_job_for_upload(body.upload_id)
    if existing and existing["status"] in ("QUEUED", "PROCESSING", "RESULT_VERIFYING", "COMPLETED"):
        raise HTTPException(409, f"job {existing['job_id']} already exists for this upload")

    job_id = f"job_{int(time.time() * 1000) % 10 ** 9:09d}"
    result_dir = str(job_result_dir(job_id))
    os.makedirs(result_dir, exist_ok=True)

    # Neither analysis type needs configuration before the GPU runs any more:
    # vehicle uses the whole frame, and person ROIs are applied afterwards from
    # the stored track history.
    status = "QUEUED" if body.start else "WAITING_CONFIG"
    job = db.create_job(job_id, body.upload_id, upload["filename"], body.analysis_type,
                        status, result_dir, create_video=body.create_video)
    # mirror it onto the upload so the card does not say WAITING_CONFIG for a
    # job that is already queued and has nothing left to configure
    db.update_upload(body.upload_id, status=status)
    return _job_payload(job)


@app.post("/api/jobs/{job_id}/person-roi")
async def set_person_roi(job_id: str, body: ROIConfig):
    """Apply Entry / Exit polygons to a FINISHED person job.

    This is post-processing: it replays the stored track history through the
    ROI state machine. No decoder, no YOLO, no SOLIDER, no MiVOLO, and the
    source video is not needed - it has usually been deleted by now. Call it
    again with different polygons as often as you like; each call fully
    replaces the previous ROI-derived results and leaves age, gender and
    visibility untouched.
    """
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    if job["analysis_type"] != "person":
        raise HTTPException(400, "ROIs only apply to person analysis")
    if job["status"] != "COMPLETED":
        raise HTTPException(409, f"job is {job['status']}; ROI can only be applied "
                                 f"to a completed analysis")
    if body.line is not None:
        if len(body.line) < 2:
            raise HTTPException(400, "line needs at least 2 points")
        if any(len(p) != 2 for p in body.line):
            raise HTTPException(400, "line points must be [x, y]")
        inside = 1 if int(body.inside or 1) >= 0 else -1
        result = await run_in_threadpool(_apply_line, job, body.line, inside)
    else:
        for name, polygon in (("entry_roi", body.entry_roi), ("exit_roi", body.exit_roi)):
            if not polygon or len(polygon) < 3:
                raise HTTPException(400, f"{name} needs at least 3 points")
            if any(len(p) != 2 for p in polygon):
                raise HTTPException(400, f"{name} points must be [x, y]")
        result = await run_in_threadpool(_apply_roi, job, body.entry_roi, body.exit_roi)
    if not result["ok"]:
        raise HTTPException(409, result["error"])
    return _job_payload(db.get_job(job_id))


def _apply_line(job, line, inside):
    from analyzer.pipeline import recalculate_line

    job_id = job["job_id"]
    try:
        out = recalculate_line(CFG, job["result_dir"], line, inside, job_id)
    except Exception as exc:                       # noqa: BLE001 - reported below
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    db.save_person_roi_results(job_id, out["events"], out["times"])
    db.update_job(job_id,
                  crossing_line=json.dumps({"line": line, "inside": inside}),
                  summary=json.dumps(out["summary"]))
    return {"ok": True}


def _apply_roi(job, entry_roi, exit_roi):
    from analyzer.pipeline import recalculate_roi

    job_id = job["job_id"]
    try:
        out = recalculate_roi(CFG, job["result_dir"], entry_roi, exit_roi, job_id)
    except Exception as exc:                       # noqa: BLE001 - reported below
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    db.save_person_roi_results(job_id, out["events"], out["times"])
    db.update_job(job_id,
                  entry_roi=json.dumps(entry_roi),
                  exit_roi=json.dumps(exit_roi),
                  summary=json.dumps(out["summary"]))
    return {"ok": True}


class StartJob(BaseModel):
    create_video: bool | None = None


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, body: StartJob | None = None):
    """Queue a configured job (vehicle jobs, or a person job whose ROIs are set)."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    if job["status"] not in ("WAITING_CONFIG", "FAILED"):
        raise HTTPException(409, f"job is {job['status']}")
    upload = db.get_upload(job["upload_id"])
    if upload is None or not upload["source_path"] or not os.path.exists(upload["source_path"]):
        raise HTTPException(409, "source video is no longer on disk")
    job = db.queue_job(job_id, create_video=None if body is None else body.create_video)
    db.update_upload(job["upload_id"], status="QUEUED")
    return _job_payload(job)


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": [_job_payload(j) for j in db.list_jobs()],
            "running": _job_payload(db.running_job())}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    payload = _job_payload(job)
    if job["status"] == "QUEUED":
        payload["queue_position"] = next(
            (j["queue_position"] for j in db.list_jobs() if j["job_id"] == job_id), None)
    return payload


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    """The structured result: summary.json plus the per-record rows from SQLite."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    summary_path = os.path.join(job["result_dir"] or "", "summary.json")
    if not os.path.exists(summary_path):
        raise HTTPException(409, f"no result yet (job is {job['status']})")
    with open(summary_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows, events = db.fetch_results(job_id, job["analysis_type"])
    payload["events"] = events
    payload["records"] = rows
    payload["status"] = job["status"]
    payload["filename"] = job["filename"]
    payload["result_dir"] = job["result_dir"]

    payload["roi_configured"] = bool((payload.get("summary") or {}).get("roi_configured"))
    payload["entry_roi"] = json.loads(job["entry_roi"]) if job["entry_roi"] else None
    payload["exit_roi"] = json.loads(job["exit_roi"]) if job["exit_roi"] else None
    payload["crossing_line"] = (json.loads(job["crossing_line"])
                                if job.get("crossing_line") else None)
    payload["tracks_available"] = os.path.exists(
        os.path.join(job["result_dir"] or "", TRACKS_FILE))

    heatmap_path = os.path.join(job["result_dir"] or "", "heatmap.json")
    if os.path.exists(heatmap_path):
        with open(heatmap_path, "r", encoding="utf-8") as fh:
            payload["heatmap"] = json.load(fh).get("heatmap")
    # the dwell picture draws the same boundary the counts used, so the two can
    # never tell different stories
    if payload.get("heatmap") and payload["crossing_line"]:
        payload["heatmap"]["crossing_line"] = payload["crossing_line"].get("line")
        payload["heatmap"]["inside_side"] = payload["crossing_line"].get("inside", 1)
    video_path = str(result_video_path(job_id))
    payload["video_url"] = f"/api/jobs/{job_id}/video" if os.path.exists(video_path) else None
    payload["result_video"] = payload["video_url"]
    payload["frame_url"] = (f"/api/jobs/{job_id}/frame"
                            if os.path.exists(os.path.join(job["result_dir"] or "",
                                                           FRAME_IMAGE)) else None)
    meta_path = os.path.join(job["result_dir"] or "", "job_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            payload["metadata"] = json.load(fh)
    return payload


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    """The annotated result video, when the job opted into one.

    FileResponse serves Range requests, so a browser can seek without pulling
    the whole file down first.
    """
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    path = str(result_video_path(job_id))
    if not os.path.exists(path):
        raise HTTPException(404, "this job did not produce a result video")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/api/jobs/{job_id}/frame")
def job_frame(job_id: str):
    """The video's first frame, kept as a still so the result dashboard still
    has a backdrop long after the source video was deleted."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    path = os.path.join(job["result_dir"] or "", FRAME_IMAGE)
    if not os.path.exists(path):
        raise HTTPException(404, "no frame stored for this job")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/jobs/{job_id}/result/{name}")
def job_result_file(job_id: str, name: str):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    if not re.fullmatch(r"[A-Za-z0-9_]+\.json", name):
        raise HTTPException(400, "only the JSON result files are served")
    path = os.path.join(job["result_dir"] or "", name)
    if not os.path.exists(path):
        raise HTTPException(404, name)
    return FileResponse(path, media_type="application/json")


@app.get("/result")
def result_page():
    """Dashboard page. Declared before the StaticFiles mount on "/" so this
    route wins; the job is chosen with ?job=job_000123."""
    return FileResponse(os.path.join(STATIC_DIR, "result.html"),
                        media_type="text/html")


@app.get("/api/health")
def health():
    running = db.running_job()
    return {
        "ok": True,
        "port": int((CFG.get("server") or {}).get("port", 7802)),
        "https_port": int((CFG.get("server") or {}).get("https_port", 0) or 0),
        "max_gpu_jobs": int((CFG.get("analysis") or {}).get("max_gpu_jobs", 1)),
        "allow_result_video": bool(
            (CFG.get("storage") or {}).get("allow_result_video", True)),
        "running_job": running["job_id"] if running else None,
        "queued": sum(1 for j in db.list_jobs() if j["status"] == "QUEUED"),
    }


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """One bad request must never take the upload server down."""
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


class NoCacheStatic(StaticFiles):
    """Static files that always revalidate.

    Starlette sends only ETag/Last-Modified, and a phone browser will happily
    serve a cached script for days without asking - which is how a fixed
    upload.js failed to reach the one device that needed it (the phone was
    still slicing 32 MB chunks long after the 8 MB build shipped). These files
    are a few KB; correctness beats a saved round-trip.
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


app.mount("/", NoCacheStatic(directory=STATIC_DIR, html=True), name="static")


def local_ips():
    """Every IPv4 this machine answers on - the certificate's SAN list."""
    import socket

    out = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            out.add(info[4][0])
    except OSError:
        pass
    try:                                    # the address used to reach the LAN
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        out.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(out)


def ensure_cert(cert_dir, extra_hosts=()):
    """A self-signed certificate covering this machine's addresses.

    HTTPS is not decoration here. navigator.wakeLock - the only thing that stops
    a phone blanking its screen part-way through a multi-hour upload - exists
    only in a SECURE CONTEXT, and a plain http:// LAN address is not one
    (verified: isSecureContext is false on http://172.30.1.x, and the API is
    absent entirely). A self-signed cert makes the origin secure; the phone asks
    once whether to trust it.
    """
    import ipaddress
    import socket
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = cert_dir / "server.crt", cert_dir / "server.key"
    marker = cert_dir / "server.sans"

    ips = local_ips()
    extra = [str(x) for x in (extra_hosts or []) if str(x).strip()]
    want = ",".join(ips + extra)
    if (cert_path.exists() and key_path.exists() and marker.exists()
            and marker.read_text(encoding="utf-8") == want):
        return str(cert_path), str(key_path)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, socket.gethostname())])
    alt = [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())]
    alt += [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    # the address the phone actually types - a public IP behind port forwarding,
    # or a hostname - must be in the certificate or the browser rejects it
    for host in extra:
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            alt.append(x509.DNSName(host))
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))

    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    marker.write_text(want, encoding="utf-8")
    print(f"[server] self-signed certificate for {want}")
    return str(cert_path), str(key_path)


async def tls_sniffing_proxy(listen_host, listen_port, http_port, https_port):
    """Accept HTTP and HTTPS on the SAME port.

    The router forwards one external port per machine straight to 7802, and
    changing that is not something the server can do for itself. So 7802 peeks
    at the first byte instead: 0x16 is a TLS handshake record and goes to the
    TLS listener, anything else is plain HTTP. That way

        http://<lan-ip>:7802     keeps working, no certificate prompt
        https://<public-ip>:7802 is a SECURE context, so navigator.wakeLock
                                 exists and the screen stays on

    without touching the port forwarding at all.
    """
    import asyncio

    async def pipe(reader, writer, prefix=b""):
        try:
            if prefix:
                writer.write(prefix)
                await writer.drain()
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError,
                OSError):
            pass
        finally:
            try:
                writer.close()
            except OSError:
                pass

    async def handle(client_reader, client_writer):
        try:
            first = await asyncio.wait_for(client_reader.read(1), timeout=30)
        except (asyncio.TimeoutError, OSError):
            client_writer.close()
            return
        if not first:
            client_writer.close()
            return
        target = https_port if first[0] == 0x16 else http_port
        try:
            up_reader, up_writer = await asyncio.open_connection("127.0.0.1", target)
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(
            pipe(client_reader, up_writer, first),
            pipe(up_reader, client_writer),
        )

    server = await asyncio.start_server(handle, listen_host, listen_port)
    return server


def main():
    import asyncio

    import uvicorn

    from analyzer import RESULT_ROOT, UPLOAD_ROOT

    scfg = CFG.get("server") or {}
    host, port = scfg.get("host", "0.0.0.0"), int(scfg.get("port", 7802))
    https_port = int(scfg.get("https_port", 0) or 0)
    extra = scfg.get("cert_hosts") or []

    # uvicorn binds loopback-only; the public listener is the sniffing proxy,
    # so one forwarded port serves both protocols
    inner_http, inner_https = 17802, 17803
    crt = key = None
    if https_port or extra:
        try:
            crt, key = ensure_cert(
                Path(SERVER_ROOT) / (scfg.get("cert_dir") or "data/certs"), extra)
        except Exception as exc:                  # noqa: BLE001 - HTTP still works
            print(f"[server] HTTPS disabled ({type(exc).__name__}: {exc})")

    servers = [uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=inner_http, log_level="info",
        timeout_keep_alive=120))]
    if crt:
        servers.append(uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=inner_https, log_level="warning",
            timeout_keep_alive=120, ssl_certfile=crt, ssl_keyfile=key)))

    print(f"[server] port {port}: http + https (같은 포트에서 둘 다 받음)")
    for ip in local_ips():
        if ip != "127.0.0.1":
            print(f"[server]   http://{ip}:{port}   https://{ip}:{port}")
    for h in extra:
        print(f"[server]   https://{h}:{port}   <- 휴대폰은 https 로")
    print(f"[server] uploads -> {UPLOAD_ROOT}")
    print(f"[server] results -> {RESULT_ROOT}   database -> {db.DB_PATH}")

    async def serve_all():
        proxies = [await tls_sniffing_proxy(host, port, inner_http,
                                            inner_https if crt else inner_http)]
        if https_port and https_port != port:
            # a second, https-only port for anyone who prefers it explicit
            proxies.append(await tls_sniffing_proxy(host, https_port, inner_http,
                                                    inner_https if crt else inner_http))
        await asyncio.gather(*(s.serve() for s in servers),
                             *(p.serve_forever() for p in proxies))

    asyncio.run(serve_all())


if __name__ == "__main__":
    main()
