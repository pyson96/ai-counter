"""Vehicle analysis for one uploaded video: full-frame ROI, plate state machine.

The detection / recognition stack is the project's existing one, reused as-is
from <root>/detect_car.py - YOLO11x vehicles (tracked by ByteTrack), the
plate-finetuned YOLO11x on the upscaled vehicle crop, PP-OCRv5 korean as the
recogniser, and the per-track confidence-weighted plate vote. Nothing here
swaps a model out.

What this module adds:

    * no ROI configuration and no renderer - the whole frame is the ROI and no
      result video is written
    * the ENTRY / LEFT_FRAME / EXIT state machine keyed on the STABILISED plate
      text rather than on the tracker id, so one car that picks up three track
      ids is still one vehicle

          UNKNOWN --first stable appearance--> ENTERED      (event ENTRY)
          ENTERED --gone for lost_seconds---->  LEFT_FRAME  (event LEFT_FRAME)
          LEFT_FRAME --same plate seen again--> EXITED       (event EXIT)
          EXITED  --gone for lost_seconds---->  LEFT_FRAME   (next sighting is
                                                              an ENTRY again)

      A single missed detection frame can never produce LEFT_FRAME: the vehicle
      has to be absent for `vehicle.lost_seconds` of video time.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from . import PROJECT_ROOT

import detect_car as core          # the project-root ALPR pipeline

PlateReader = core.PlateReader
plate_score = core.plate_score
normalise_plate = core.normalise_plate
merge_votes = core.merge_votes

ENTRY, EXIT, LEFT_FRAME = "ENTRY", "EXIT", "LEFT_FRAME"
S_ENTERED, S_LEFT_FRAME, S_EXITED = "ENTERED", "LEFT_FRAME", "EXITED"


# Entry blue #3987e5 / Exit orange #d95926, in BGR - the same pair the person
# renderer, the ROI editor and the result dashboard all use.
COUNT_COLORS = {"entry": (229, 135, 57), "exit": (38, 89, 217)}


def draw_vehicle_counts(canvas, entries, exits):
    """Running ENTRY / EXIT tally, bottom-left of the rendered frame."""
    h, w = canvas.shape[:2]
    pad, bar_h, width = 14, 54, 330
    y0, x0 = h - bar_h - pad, pad
    region = canvas[y0:y0 + bar_h, x0:x0 + min(width, w - 2 * pad)]
    if region.size:
        shade = np.full_like(region, 16)
        cv2.addWeighted(shade, 0.94, region, 0.06, 0, region)
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + bar_h), (70, 70, 70), 1)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(canvas, f"ENTRY {entries}", (x0 + 14, y0 + 36), font, 0.95,
                COUNT_COLORS["entry"], 2, cv2.LINE_AA)
    cv2.putText(canvas, f"EXIT {exits}", (x0 + 180, y0 + 36), font, 0.95,
                COUNT_COLORS["exit"], 2, cv2.LINE_AA)


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def build_args(cfg, video_path, weights_dir):
    """The argparse.Namespace detect_car.PlateReader expects."""
    vcfg = cfg.get("vehicle") or {}
    weights = _abs(weights_dir)
    return argparse.Namespace(
        input=video_path,
        output=None,
        cpu=not bool((cfg.get("device") or {}).get("cuda", True)),
        quiet=True,
        no_render=True,                       # never build the Renderer
        font=None,
        vehicle_weights=os.path.join(weights, "yolo11x.pt"),
        plate_weights=os.path.join(weights, "license_plate_yolo11x.pt"),
        ocr_dir=os.path.join(weights, "ocr"),
        imgsz=int(vcfg.get("imgsz", 960)),
        vehicle_conf=float(vcfg.get("vehicle_conf", 0.35)),
        min_vehicle_size=int(vcfg.get("min_vehicle_size", 48)),
        crop_size=int(vcfg.get("crop_size", 640)),
        plate_imgsz=int(vcfg.get("plate_imgsz", 640)),
        plate_conf=float(vcfg.get("plate_conf", 0.25)),
        plate_interval=int(vcfg.get("plate_interval", 2)),
        batch=int(vcfg.get("batch", 8)),
        ocr_height=int(vcfg.get("ocr_height", 96)),
        ocr_conf=float(vcfg.get("ocr_conf", 0.35)),
        min_quality=float(vcfg.get("min_quality", 0.12)),
        ocr_interval=int(vcfg.get("ocr_interval", 3)),
        max_reads=int(vcfg.get("max_reads", 12)),
        start_frame=int((cfg.get("video") or {}).get("start_frame", 0)),
        max_frames=int((cfg.get("video") or {}).get("max_frames", 0)),
    )


class PlateStateMachine:
    """ENTRY / LEFT_FRAME / EXIT, per stabilised plate number."""

    def __init__(self, lost_frames):
        self.lost_frames = max(1, int(lost_frames))
        self.plates = {}          # plate -> record
        self.events = []

    def _record(self, plate, t_now, frame_idx):
        rec = self.plates.get(plate)
        if rec is None:
            rec = self.plates[plate] = {
                "plate_number": plate,
                "status": None,
                "entry_time": None,
                "exit_time": None,
                "first_seen": t_now,
                "last_seen": t_now,
                "last_seen_frame": frame_idx,
                "pending": ENTRY,      # what the NEXT sighting means
                "sightings": 0,
                "reads": 0,
                "vote_share": None,
                "vehicle_class": None,
            }
        return rec

    def _emit(self, plate, event, t_now, frame_idx):
        self.events.append({
            "plate_number": plate,
            "event": event,
            "time": round(float(t_now), 3),
            "frame": int(frame_idx),
        })

    def seen(self, plate, t_now, frame_idx, first_seen=None, meta=None):
        """One frame in which this plate is visible somewhere in the frame."""
        rec = self._record(plate, t_now, frame_idx)
        if meta:
            rec.update({k: v for k, v in meta.items() if v is not None})
        rec["last_seen"] = t_now
        rec["last_seen_frame"] = frame_idx
        rec["sightings"] += 1

        if rec["status"] is None:
            # brand new plate -> this is an ENTRY. The entry time is when the
            # car first showed up on screen, not when OCR finally settled.
            at = t_now if first_seen is None else min(first_seen, t_now)
            rec["first_seen"] = at
            rec["status"] = S_ENTERED
            rec["entry_time"] = round(float(at), 3)
            rec["pending"] = EXIT
            self._emit(plate, ENTRY, at, frame_idx)
        elif rec["status"] == S_LEFT_FRAME:
            if rec["pending"] == EXIT:
                rec["status"] = S_EXITED
                rec["exit_time"] = round(float(t_now), 3)
                rec["pending"] = ENTRY
                self._emit(plate, EXIT, t_now, frame_idx)
            else:
                rec["status"] = S_ENTERED
                rec["entry_time"] = round(float(t_now), 3)
                rec["exit_time"] = None
                rec["pending"] = EXIT
                self._emit(plate, ENTRY, t_now, frame_idx)
        # ENTERED / EXITED and still visible -> no event, by design

    def sweep(self, frame_idx, fps):
        """Retire plates that have been absent long enough to have really left."""
        for plate, rec in self.plates.items():
            if rec["status"] in (None, S_LEFT_FRAME):
                continue
            if frame_idx - rec["last_seen_frame"] < self.lost_frames:
                continue
            rec["status"] = S_LEFT_FRAME
            self._emit(plate, LEFT_FRAME, rec["last_seen_frame"] / fps, rec["last_seen_frame"])

    def records(self):
        out = []
        for rec in sorted(self.plates.values(), key=lambda r: r["first_seen"]):
            item = dict(rec)
            item.pop("pending", None)
            item.pop("last_seen_frame", None)
            item["first_seen"] = round(float(item["first_seen"]), 3)
            item["last_seen"] = round(float(item["last_seen"]), 3)
            item["status"] = item["status"] or "UNKNOWN"
            out.append(item)
        return out


class VehicleAnalysis(PlateReader):
    """PlateReader without the renderer, driving the plate state machine."""

    RESULT_VIDEO = "result.mp4"

    def __init__(self, cfg, video_path, result_dir, progress_cb=None, job_id="job",
                 weights_dir="weights", create_video=False):
        self.cfg = cfg
        self.result_dir = result_dir
        self.progress_cb = progress_cb
        self.job_id = job_id
        self.video_path = video_path
        self.frame_image = None
        self.create_video = bool(create_video)
        self.video_path_out = None
        args = build_args(cfg, video_path, weights_dir)
        # the Renderer needs a Hangul TTF; only build it when a video was asked for
        args.no_render = not self.create_video
        super().__init__(args)

        vcfg = cfg.get("vehicle") or {}
        self.min_plate_score = float(vcfg.get("min_plate_score", 0.8))
        self.min_reads = int(vcfg.get("min_reads", 2))
        self.lost_seconds = float(vcfg.get("lost_seconds", 3.0))
        self.time_bucket = float(vcfg.get("time_bucket_seconds", 300))
        self.progress_interval = int((cfg.get("analysis") or {}).get("progress_interval", 50))

    # -- the stabilised plate for one tracked vehicle ----------------------
    def _stable_plate(self, track):
        text, share = track.text
        if not text:
            return None, 0.0
        if plate_score(text) < self.min_plate_score:
            return None, share
        if track.reads < self.min_reads:
            return None, share
        return text, share

    # -- driver ------------------------------------------------------------
    def run(self):
        args = self.args
        from .pipeline import save_first_frame

        # kept as the result dashboard's backdrop, before the source is deleted
        self.frame_image = save_first_frame(self.video_path, self.result_dir)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        if args.max_frames:
            total = min(total or args.max_frames, args.max_frames)

        machine = PlateStateMachine(max(1, round(self.lost_seconds * fps)))
        timeline = [] if self.create_video else None
        started = time.time()
        idx = 0
        print(f"[vehicle] {self.video_path} @ {fps:.2f}fps {total} frames "
              f"(full-frame ROI, no result video)")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and idx >= args.max_frames:
                break
            t_now = idx / fps

            vehicles = self.detect_vehicles(frame)
            for tid, cls_name, box in vehicles:
                track = self.tracks.get(tid)
                if track is None:
                    track = self.tracks[tid] = core.Vehicle(tid, cls_name, idx)
                track.last_frame = idx
                track.box = box

            if idx % args.plate_interval == 0 and vehicles:
                plates = self.detect_plates(frame, vehicles)
                for tid, box in plates.items():
                    track = self.tracks[tid]
                    track.plate_box = box[:4]
                    track.plate_frame = idx
                    if not self._should_read(track, idx):
                        continue
                    crop, text, conf = self.read(frame, box)
                    if crop is None:
                        continue
                    quality = core.crop_quality(crop)
                    if quality < args.min_quality:
                        continue          # too blurry to trust - do not pollute the vote
                    track.last_read = idx
                    if text and conf >= args.ocr_conf:
                        track.vote(text, conf, quality, crop)

            # every vehicle whose plate has settled counts as visible this frame,
            # even on frames where OCR was not attempted at all
            for tid, cls_name, _ in vehicles:
                track = self.tracks[tid]
                plate, share = self._stable_plate(track)
                if not plate:
                    continue
                machine.seen(plate, t_now, idx, first_seen=track.first_frame / fps,
                             meta={"reads": track.reads, "vote_share": round(share, 3),
                                   "vehicle_class": cls_name})

            machine.sweep(idx, fps)

            if timeline is not None:
                timeline.append([(tid, self.tracks[tid].box, self.tracks[tid].plate_box)
                                 for tid, _, _ in vehicles])

            idx += 1
            if idx % self.progress_interval == 0:
                rate = idx / max(time.time() - started, 1e-6)
                print(f"  ... {idx}/{total} frames ({rate:.1f} fps)  "
                      f"plates={len(machine.plates)}")
                if self.progress_cb is not None:
                    self.progress_cb(idx, total)

        cap.release()
        # close anything still on screen when the video ends
        machine.lost_frames = 0
        machine.sweep(idx + 1, fps)
        elapsed = time.time() - started
        if timeline is not None:
            self._render(timeline, machine, fps)
        return self._build_output(machine, idx, fps, elapsed)

    # -- pass 2: the annotated video (only when the user asked for one) -----
    def _render(self, timeline, machine, fps):
        """Draw every frame with the FINAL plate numbers and the settled
        entry/exit tally, reusing the project's plate renderer and its panel.

        Two passes for the same reason the CLI uses two: a plate's number is
        only settled once the votes are in, so a single-pass render would show
        a number that is still wrong for the first half of every car.
        """
        from collections import deque

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot re-open video for rendering: {self.video_path}")
        if self.args.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.args.start_frame)
        ok, probe = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("cannot decode a frame for rendering")
        h, w = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.args.start_frame)

        self.video_path_out = os.path.join(self.result_dir, self.RESULT_VIDEO)
        writer = cv2.VideoWriter(self.video_path_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w + core.PANEL_W, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open video writer for {self.video_path_out}")

        events = sorted(machine.events, key=lambda e: int(e.get("frame", 0)))
        recent, last_visible = deque(maxlen=64), {}
        cursor = 0
        n_entry = n_exit = 0
        started = time.time()
        for offset, visible in enumerate(timeline):
            ok, frame = cap.read()
            if not ok:
                break
            idx = self.args.start_frame + offset
            while cursor < len(events) and int(events[cursor].get("frame", 0)) <= idx:
                if events[cursor]["event"] == ENTRY:
                    n_entry += 1
                elif events[cursor]["event"] == EXIT:
                    n_exit += 1
                cursor += 1

            for track in self.tracks.values():
                track.box = track.plate_box = None
            for tid, box, plate_box in visible:
                track = self.tracks[tid]
                track.box, track.plate_box = box, plate_box
                last_visible[tid] = idx
                if tid in recent:
                    recent.remove(tid)
                recent.appendleft(tid)

            # a car keeps its panel row for two seconds after it leaves frame
            panel = [self.tracks[t] for t in recent
                     if idx - last_visible[t] <= int(fps * 2)]
            canvas = self.renderer.draw(frame, panel, idx, fps,
                                        {"unique": len(machine.plates)})
            draw_vehicle_counts(canvas, n_entry, n_exit)
            writer.write(canvas)
            if (offset + 1) % 200 == 0:
                rate = (offset + 1) / max(time.time() - started, 1e-6)
                print(f"  ... rendering {offset + 1}/{len(timeline)} ({rate:.1f} fps)")

        cap.release()
        writer.release()
        print(f"[vehicle] rendered -> {self.video_path_out} "
              f"in {time.time() - started:.1f}s")

    # -- structured output -------------------------------------------------
    def _build_output(self, machine, frames, fps, elapsed):
        from .pipeline import _bucket_histogram, _write

        vehicles = machine.records()
        events = machine.events
        duration = frames / fps if fps else 0.0

        entries = [e for e in events if e["event"] == ENTRY]
        exits = [e for e in events if e["event"] == EXIT]
        inside = sum(1 for v in vehicles
                     if v["entry_time"] is not None
                     and (v["exit_time"] is None or v["exit_time"] < v["entry_time"]))

        summary = {
            "entry_count": len(entries),
            "exit_count": len(exits),
            "total_vehicle_entries": len(entries),
            "total_vehicle_exits": len(exits),
            "current_entered_vehicle_count": inside,
            "unique_plates": len(vehicles),
            "vehicles_tracked": len(self.tracks),
            "time_bucket_seconds": self.time_bucket,
            "entries_by_time": _bucket_histogram([e["time"] for e in entries],
                                                 self.time_bucket, duration),
            "exits_by_time": _bucket_histogram([e["time"] for e in exits],
                                               self.time_bucket, duration),
            "video_duration_seconds": round(duration, 2),
            "frames_processed": frames,
            "fps": round(fps, 3),
        }

        metadata = {
            "job_id": self.job_id,
            "analysis_type": "vehicle",
            "source_video": os.path.basename(self.video_path),
            "processing_seconds": round(elapsed, 1),
            "processing_fps": round(frames / max(elapsed, 1e-6), 2),
            "roi": "full_frame",
            "frame_image": self.frame_image,
            "result_video": self.RESULT_VIDEO if self.create_video else None,
            "lost_seconds": self.lost_seconds,
            "min_plate_score": self.min_plate_score,
            "min_reads": self.min_reads,
            "models": {
                "vehicle_detector": os.path.basename(self.args.vehicle_weights),
                "plate_detector": os.path.basename(self.args.plate_weights),
                "ocr": "PP-OCRv5 korean (onnxruntime)",
                "tracker": "bytetrack",
            },
        }

        payload = {
            "job_id": self.job_id,
            "analysis_type": "vehicle",
            "summary": summary,
            "vehicles": vehicles,
            "frame_image": self.frame_image,
            "result_video": self.RESULT_VIDEO if self.create_video else None,
        }

        os.makedirs(self.result_dir, exist_ok=True)
        _write(os.path.join(self.result_dir, "summary.json"), payload)
        _write(os.path.join(self.result_dir, "vehicles.json"),
               {"job_id": self.job_id, "vehicles": vehicles})
        _write(os.path.join(self.result_dir, "vehicle_events.json"),
               {"job_id": self.job_id, "events": events})
        _write(os.path.join(self.result_dir, "job_metadata.json"), metadata)

        return {
            "analysis_type": "vehicle",
            "summary": summary,
            "vehicles": vehicles,
            "events": events,
            "frame_image": self.frame_image,
            "result_video": self.video_path_out,
            "metadata": metadata,
        }


def run_vehicle_analysis(cfg, video_path, result_dir, progress_cb=None, job_id="job",
                         weights_dir="weights", create_video=False):
    return VehicleAnalysis(cfg, video_path, result_dir, progress_cb, job_id,
                           weights_dir, create_video).run()
