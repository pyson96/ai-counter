"""Entry point.

    python main.py                       # analyse test.mp4 with config.yaml
    python main.py --download-weights    # fetch the model checkpoints first
    python main.py --max-frames 300      # quick debug run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

WEIGHTS = {
    # SOLIDER-REID, Swin-Small fine-tuned on MSMT17 (mAP 76.9 / Rank-1 90.8)
    "solider_swin_small_msmt17.pth": ("gdrive", "1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b"),
    "solider_swin_tiny_msmt17.pth": ("gdrive", "10YLhMbwvmxZl3gTVo2BN_828SKZHdCjr"),
    "solider_swin_base_msmt17.pth": ("gdrive", "1Y-RFAYdT56vnMjwxH1Ym3DVhZzZuMQZs"),
    "yolo11x.pt": ("ultralytics", "yolo11x.pt"),
    # face+person detector used to feed MiVOLO with face crops
    "yolov8x_person_face.pt": ("hf", ("iitolstykh/YOLO-Face-Person-Detector", "yolov8x_person_face.pt")),
}


def download_weights(names, out_dir="weights"):
    import shutil

    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst):
            print(f"[weights] {name} already present")
            continue
        kind, ref = WEIGHTS[name]
        print(f"[weights] downloading {name} ({kind}) ...")
        if kind == "gdrive":
            import gdown

            gdown.download(id=ref, output=dst, quiet=False)
        elif kind == "hf":
            from huggingface_hub import hf_hub_download

            shutil.copy(hf_hub_download(ref[0], ref[1]), dst)
        elif kind == "ultralytics":
            from ultralytics import YOLO

            YOLO(ref)  # ultralytics downloads into the cwd
            shutil.move(ref, dst)
        print(f"[weights] -> {dst}")
    # MiVOLO v2 itself is pulled from the HuggingFace hub into the local HF cache
    print("[weights] done. MiVOLO v2 is fetched into the HF cache on first run.")


def load_config(path, overrides):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, value in overrides.items():
        if value is None:
            continue
        section, _, field = key.partition(".")
        cfg[section][field] = value
    os.makedirs(cfg["video"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["debug"]["reid_debug_dir"], exist_ok=True)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local person detection / tracking / ReID / demographics")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default=None, help="override video.input")
    ap.add_argument("--output", default=None, help="override video.output")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--no-demographics", action="store_true", help="skip age/gender estimation")
    ap.add_argument("--cpu", action="store_true", help="force CPU (slow)")
    ap.add_argument("--quiet", action="store_true", help="do not print per-event lines")
    ap.add_argument("--download-weights", action="store_true", help="fetch checkpoints and exit")
    args = ap.parse_args(argv)

    if args.download_weights:
        download_weights(["yolo11x.pt", "yolov8x_person_face.pt", "solider_swin_base_msmt17.pth"])
        return 0

    cfg = load_config(
        args.config,
        {
            "video.input": args.input,
            "video.output": args.output,
            "video.max_frames": args.max_frames,
            "video.start_frame": args.start_frame,
        },
    )
    if args.no_demographics:
        cfg["demographics"]["enabled"] = False
    if args.cpu:
        cfg["device"]["cuda"] = False
        cfg["device"]["fp16"] = False
    if args.quiet:
        cfg["debug"]["verbose"] = False

    if not os.path.exists(cfg["video"]["input"]):
        print(f"input video not found: {cfg['video']['input']}", file=sys.stderr)
        return 2

    import pipeline

    result = pipeline.Analyzer(cfg).run()

    perf_path = os.path.join(cfg["video"]["output_dir"], "performance.json")
    with open(perf_path, "w", encoding="utf-8") as fh:
        json.dump(result["performance"], fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
