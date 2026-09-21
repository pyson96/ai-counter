"""SQLite storage for uploads, analysis jobs and their structured results.

One file, one connection per call. SQLite in WAL mode handles the two writers
this system has (the FastAPI process and the GPU worker process) perfectly well
at this write rate, which is why there is no Redis/Celery anywhere near it.

State lifecycle
---------------
    uploads.status : UPLOADING -> UPLOAD_VERIFYING -> UPLOADED -> WAITING_CONFIG
                     -> (job runs) -> SOURCE_DELETED
                     with FAILED reachable from any of them
    jobs.status    : QUEUED -> PROCESSING -> RESULT_VERIFYING -> COMPLETED
                     with FAILED reachable from any of them
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(HERE, "data", "analysis.db")


def _resolve_db_path():
    """`storage.db_path` if the config sets one, else the historical location.

    Reading the config here keeps the database a deployment choice rather than a
    hard-coded path - this machine keeps it on the same volume as the videos,
    while an install that never sets the key is unaffected.
    """
    try:
        from analyzer import load_config

        path = (load_config().get("storage") or {}).get("db_path")
        if path:
            return str(path)
    except Exception as exc:                      # noqa: BLE001 - see below
        # A broken or missing config must not make the database unreachable;
        # fall back to where it has always been.
        print(f"[database] could not read storage.db_path ({exc}); "
              f"using {DEFAULT_DB_PATH}")
    return DEFAULT_DB_PATH


DB_PATH = _resolve_db_path()

UPLOAD_STATES = (
    "UPLOADING", "UPLOAD_VERIFYING", "UPLOADED", "WAITING_CONFIG",
    "QUEUED", "PROCESSING", "COMPLETED", "SOURCE_DELETED", "FAILED",
)
JOB_STATES = (
    "WAITING_CONFIG", "QUEUED", "PROCESSING", "RESULT_VERIFYING",
    "COMPLETED", "FAILED", "CANCELLED",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    upload_id       TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    file_size       INTEGER NOT NULL,
    received_bytes  INTEGER NOT NULL DEFAULT 0,
    chunk_size      INTEGER NOT NULL,
    total_chunks    INTEGER NOT NULL,
    chunk_map       TEXT NOT NULL DEFAULT '',   -- '0'/'1' per chunk
    upload_progress REAL NOT NULL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'UPLOADING',
    part_path       TEXT,
    source_path     TEXT,
    video_info      TEXT,                       -- JSON from validation
    error_message   TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    upload_id         TEXT NOT NULL,
    filename          TEXT NOT NULL,
    analysis_type     TEXT NOT NULL,            -- 'person' | 'vehicle'
    status            TEXT NOT NULL DEFAULT 'WAITING_CONFIG',
    priority          INTEGER NOT NULL DEFAULT 0,  -- reserved; FIFO for now
    queue_seq         INTEGER,                  -- FIFO order once queued
    entry_roi         TEXT,                     -- JSON polygon, video pixels (legacy)
    exit_roi          TEXT,
    crossing_line     TEXT,                     -- JSON {"line": [[x,y]...], "inside": 1|-1}
    create_video      INTEGER NOT NULL DEFAULT 0,  -- opt-in annotated result video
    analysis_progress REAL NOT NULL DEFAULT 0.0,
    current_frame     INTEGER NOT NULL DEFAULT 0,
    total_frames      INTEGER NOT NULL DEFAULT 0,
    result_dir        TEXT,
    summary           TEXT,                     -- JSON summary once completed
    error_message     TEXT,
    created_at        REAL NOT NULL,
    started_at        REAL,
    completed_at      REAL,
    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id)
);

CREATE TABLE IF NOT EXISTS person_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id            TEXT NOT NULL,
    person_name       TEXT NOT NULL,
    entry_time        REAL,
    exit_time         REAL,
    first_seen        REAL,
    last_seen         REAL,
    estimated_age     REAL,
    age_group         TEXT,
    estimated_gender  TEXT,
    gender_confidence REAL,
    visible_seconds   REAL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS person_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         TEXT NOT NULL,
    person_name    TEXT,
    event          TEXT NOT NULL,               -- 'entry' | 'exit'
    video_time     REAL NOT NULL,
    frame_index    INTEGER,
    local_track_id INTEGER,
    ground_x       REAL,
    ground_y       REAL,
    matched        INTEGER NOT NULL DEFAULT 1,  -- exit that follows an entry
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS vehicle_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL,
    plate_number TEXT NOT NULL,
    entry_time   REAL,
    exit_time    REAL,
    first_seen   REAL,
    last_seen    REAL,
    status       TEXT NOT NULL,                 -- ENTERED|LEFT_FRAME|EXITED
    reads        INTEGER NOT NULL DEFAULT 0,
    vote_share   REAL,
    vehicle_class TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS vehicle_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL,
    plate_number TEXT NOT NULL,
    event        TEXT NOT NULL,                 -- ENTRY|LEFT_FRAME|EXIT
    video_time   REAL NOT NULL,
    frame_index  INTEGER,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority DESC, queue_seq);
CREATE INDEX IF NOT EXISTS idx_person_results_job ON person_results(job_id);
CREATE INDEX IF NOT EXISTS idx_person_events_job ON person_events(job_id);
CREATE INDEX IF NOT EXISTS idx_vehicle_results_job ON vehicle_results(job_id);
CREATE INDEX IF NOT EXISTS idx_vehicle_events_job ON vehicle_events(job_id);
"""


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def cursor(write=False):
    conn = _connect()
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        if write:
            conn.execute("COMMIT")
    except Exception:
        if write:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        raise
    finally:
        conn.close()


def init_db():
    with cursor() as conn:
        conn.executescript(SCHEMA)
        # in-place migration for databases created before the opt-in video
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        if "create_video" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN create_video INTEGER NOT NULL DEFAULT 0")
        pe = {r["name"] for r in conn.execute("PRAGMA table_info(person_events)").fetchall()}
        jb = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        if jb and "crossing_line" not in jb:
            conn.execute("ALTER TABLE jobs ADD COLUMN crossing_line TEXT")

        if pe and "matched" not in pe:
            conn.execute("ALTER TABLE person_events ADD COLUMN matched "
                         "INTEGER NOT NULL DEFAULT 1")
    return DB_PATH


def _row(r):
    return None if r is None else dict(r)


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------

def create_upload(upload_id, filename, file_size, chunk_size, total_chunks, part_path):
    now = time.time()
    with cursor(write=True) as conn:
        conn.execute(
            """INSERT INTO uploads (upload_id, filename, file_size, received_bytes,
                                    chunk_size, total_chunks, chunk_map, upload_progress,
                                    status, part_path, created_at, updated_at)
               VALUES (?,?,?,0,?,?,?,0.0,'UPLOADING',?,?,?)""",
            (upload_id, filename, file_size, chunk_size, total_chunks,
             "0" * total_chunks, part_path, now, now),
        )
    return get_upload(upload_id)


def get_upload(upload_id):
    with cursor() as conn:
        return _row(conn.execute("SELECT * FROM uploads WHERE upload_id=?",
                                 (upload_id,)).fetchone())


def find_resumable_upload(filename, file_size):
    """An interrupted upload of the same file is resumed, not restarted."""
    with cursor() as conn:
        return _row(conn.execute(
            """SELECT * FROM uploads
               WHERE filename=? AND file_size=? AND status='UPLOADING'
               ORDER BY created_at DESC LIMIT 1""",
            (filename, file_size)).fetchone())


def list_uploads():
    with cursor() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM uploads ORDER BY created_at DESC").fetchall()]


def mark_chunk_received(upload_id, index, nbytes):
    """Flip one bit of the chunk map and recompute progress, atomically.

    The map is the source of truth for resume: received_bytes alone cannot tell
    the browser WHICH chunk is missing after a mid-chunk disconnect.
    """
    with cursor(write=True) as conn:
        row = conn.execute("SELECT chunk_map, file_size, chunk_size, total_chunks "
                           "FROM uploads WHERE upload_id=?", (upload_id,)).fetchone()
        if row is None:
            return None
        chunks = list(row["chunk_map"])
        if index < 0 or index >= len(chunks):
            return None
        chunks[index] = "1"
        chunk_map = "".join(chunks)
        received = 0
        size, csize, total = row["file_size"], row["chunk_size"], row["total_chunks"]
        for i, bit in enumerate(chunk_map):
            if bit == "1":
                received += min(csize, size - i * csize) if i == total - 1 else csize
        progress = (received / size * 100.0) if size else 0.0
        conn.execute(
            "UPDATE uploads SET chunk_map=?, received_bytes=?, upload_progress=?, "
            "updated_at=? WHERE upload_id=?",
            (chunk_map, received, round(progress, 2), time.time(), upload_id),
        )
    return get_upload(upload_id)


def update_upload(upload_id, **fields):
    if not fields:
        return get_upload(upload_id)
    fields["updated_at"] = time.time()
    sets = ", ".join(f"{k}=?" for k in fields)
    with cursor(write=True) as conn:
        conn.execute(f"UPDATE uploads SET {sets} WHERE upload_id=?",
                     (*fields.values(), upload_id))
    return get_upload(upload_id)


def delete_upload(upload_id):
    with cursor(write=True) as conn:
        conn.execute("DELETE FROM uploads WHERE upload_id=?", (upload_id,))


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def create_job(job_id, upload_id, filename, analysis_type, status, result_dir,
               entry_roi=None, exit_roi=None, create_video=False):
    now = time.time()
    with cursor(write=True) as conn:
        seq = None
        if status == "QUEUED":
            seq = (conn.execute("SELECT COALESCE(MAX(queue_seq), 0) FROM jobs")
                   .fetchone()[0]) + 1
        conn.execute(
            """INSERT INTO jobs (job_id, upload_id, filename, analysis_type, status,
                                 queue_seq, entry_roi, exit_roi, create_video,
                                 result_dir, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (job_id, upload_id, filename, analysis_type, status, seq,
             json.dumps(entry_roi) if entry_roi else None,
             json.dumps(exit_roi) if exit_roi else None,
             1 if create_video else 0,
             result_dir, now),
        )
    return get_job(job_id)


def get_job(job_id):
    with cursor() as conn:
        return _row(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())


def get_job_for_upload(upload_id):
    with cursor() as conn:
        return _row(conn.execute(
            "SELECT * FROM jobs WHERE upload_id=? ORDER BY created_at DESC LIMIT 1",
            (upload_id,)).fetchone())


def list_jobs():
    with cursor() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC").fetchall()]
    queued = sorted((r for r in rows if r["status"] == "QUEUED"),
                    key=lambda r: (-r["priority"], r["queue_seq"] or 0))
    positions = {r["job_id"]: i + 1 for i, r in enumerate(queued)}
    for r in rows:
        r["queue_position"] = positions.get(r["job_id"])
    return rows


def update_job(job_id, **fields):
    if not fields:
        return get_job(job_id)
    sets = ", ".join(f"{k}=?" for k in fields)
    with cursor(write=True) as conn:
        conn.execute(f"UPDATE jobs SET {sets} WHERE job_id=?", (*fields.values(), job_id))
    return get_job(job_id)


def queue_job(job_id, entry_roi=None, exit_roi=None, create_video=None):
    with cursor(write=True) as conn:
        seq = (conn.execute("SELECT COALESCE(MAX(queue_seq), 0) FROM jobs")
               .fetchone()[0]) + 1
        sets = ["status='QUEUED'", "queue_seq=?", "error_message=NULL"]
        params = [seq]
        if entry_roi is not None:
            sets.append("entry_roi=?")
            params.append(json.dumps(entry_roi))
        if exit_roi is not None:
            sets.append("exit_roi=?")
            params.append(json.dumps(exit_roi))
        if create_video is not None:
            sets.append("create_video=?")
            params.append(1 if create_video else 0)
        params.append(job_id)
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id=?", params)
    return get_job(job_id)


def claim_next_job():
    """FIFO claim of one queued job. The IMMEDIATE transaction is what keeps a
    second worker process from ever picking up the same job."""
    with cursor(write=True) as conn:
        row = conn.execute(
            """SELECT * FROM jobs WHERE status='QUEUED'
               ORDER BY priority DESC, queue_seq ASC, created_at ASC LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE jobs SET status='PROCESSING', started_at=?, analysis_progress=0.0 "
            "WHERE job_id=?", (time.time(), row["job_id"]))
        job = dict(row)
    job["status"] = "PROCESSING"
    return job


def running_job():
    with cursor() as conn:
        return _row(conn.execute(
            "SELECT * FROM jobs WHERE status IN ('PROCESSING','RESULT_VERIFYING') "
            "ORDER BY started_at LIMIT 1").fetchone())


def reset_stale_jobs():
    """A worker killed mid-job leaves a PROCESSING row behind; re-queue it."""
    with cursor(write=True) as conn:
        rows = conn.execute(
            "SELECT job_id FROM jobs WHERE status IN ('PROCESSING','RESULT_VERIFYING')"
        ).fetchall()
        for r in rows:
            seq = (conn.execute("SELECT COALESCE(MAX(queue_seq),0) FROM jobs")
                   .fetchone()[0]) + 1
            conn.execute("UPDATE jobs SET status='QUEUED', queue_seq=?, started_at=NULL, "
                         "analysis_progress=0.0, current_frame=0 WHERE job_id=?",
                         (seq, r["job_id"]))
        return [r["job_id"] for r in rows]


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def save_person_results(job_id, persons, events):
    """One transaction: results and events land together or not at all."""
    with cursor(write=True) as conn:
        conn.execute("DELETE FROM person_results WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM person_events WHERE job_id=?", (job_id,))
        conn.executemany(
            """INSERT INTO person_results (job_id, person_name, entry_time, exit_time,
                   first_seen, last_seen, estimated_age, age_group, estimated_gender,
                   gender_confidence, visible_seconds)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [(job_id, p["person_name"], p.get("entry_time"), p.get("exit_time"),
              p.get("first_seen"), p.get("last_seen"), p.get("estimated_age"),
              p.get("age_group"), p.get("estimated_gender"),
              p.get("gender_confidence"), p.get("total_visible_seconds"))
             for p in persons],
        )
        conn.executemany(
            """INSERT INTO person_events (job_id, person_name, event, video_time,
                   frame_index, local_track_id, ground_x, ground_y, matched)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [(job_id, e.get("person_name"), e["event"], e["time"], e.get("frame"),
              e.get("local_track_id"),
              (e.get("point") or [None, None])[0], (e.get("point") or [None, None])[1],
              1 if e.get("matched", True) else 0)
             for e in events],
        )


def save_vehicle_results(job_id, vehicles, events):
    with cursor(write=True) as conn:
        conn.execute("DELETE FROM vehicle_results WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM vehicle_events WHERE job_id=?", (job_id,))
        conn.executemany(
            """INSERT INTO vehicle_results (job_id, plate_number, entry_time, exit_time,
                   first_seen, last_seen, status, reads, vote_share, vehicle_class)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [(job_id, v["plate_number"], v.get("entry_time"), v.get("exit_time"),
              v.get("first_seen"), v.get("last_seen"), v.get("status", "UNKNOWN"),
              v.get("reads", 0), v.get("vote_share"), v.get("vehicle_class"))
             for v in vehicles],
        )
        conn.executemany(
            """INSERT INTO vehicle_events (job_id, plate_number, event, video_time, frame_index)
               VALUES (?,?,?,?,?)""",
            [(job_id, e["plate_number"], e["event"], e["time"], e.get("frame"))
             for e in events],
        )


def save_person_roi_results(job_id, events, times):
    """Apply a recalculated ROI to an existing person job.

    Deliberately NOT save_person_results(): that one rebuilds person_results
    from scratch and would throw away the age, gender and visibility figures
    that ROI has nothing to do with. This replaces the events and rewrites only
    entry_time / exit_time, in one transaction.
    """
    with cursor(write=True) as conn:
        conn.execute("DELETE FROM person_events WHERE job_id=?", (job_id,))
        conn.executemany(
            """INSERT INTO person_events (job_id, person_name, event, video_time,
                   frame_index, local_track_id, ground_x, ground_y, matched)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [(job_id, e.get("person_name"), e["event"], e["time"], e.get("frame"),
              e.get("local_track_id"),
              (e.get("point") or [None, None])[0], (e.get("point") or [None, None])[1],
              1 if e.get("matched", True) else 0)
             for e in events],
        )
        # every person is reset first, so a person who no longer crosses the new
        # ROI does not keep the times from the previous one
        conn.execute("UPDATE person_results SET entry_time=NULL, exit_time=NULL "
                     "WHERE job_id=?", (job_id,))
        conn.executemany(
            "UPDATE person_results SET entry_time=?, exit_time=? "
            "WHERE job_id=? AND person_name=?",
            [(slot.get("entry_time"), slot.get("exit_time"), job_id, name)
             for name, slot in (times or {}).items()],
        )


def count_results(job_id, analysis_type):
    table = "person_results" if analysis_type == "person" else "vehicle_results"
    etable = "person_events" if analysis_type == "person" else "vehicle_events"
    with cursor() as conn:
        n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE job_id=?", (job_id,)).fetchone()[0]
        m = conn.execute(f"SELECT COUNT(*) FROM {etable} WHERE job_id=?", (job_id,)).fetchone()[0]
    return n, m


def fetch_results(job_id, analysis_type):
    table = "person_results" if analysis_type == "person" else "vehicle_results"
    etable = "person_events" if analysis_type == "person" else "vehicle_events"
    with cursor() as conn:
        rows = [dict(r) for r in conn.execute(
            f"SELECT * FROM {table} WHERE job_id=? ORDER BY id", (job_id,)).fetchall()]
        events = [dict(r) for r in conn.execute(
            f"SELECT * FROM {etable} WHERE job_id=? ORDER BY video_time", (job_id,)).fetchall()]
    return rows, events


if __name__ == "__main__":
    print("initialised", init_db())
