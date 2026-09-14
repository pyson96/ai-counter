"""Model wrappers for the local person-analysis pipeline.

Everything in here is a thin, GPU-friendly wrapper around a locally stored
checkpoint. Nothing contacts a cloud API at inference time.

  PersonDetector        YOLO11-X + ByteTrack/BoT-SORT   (detection + local track ids)
  FaceDetector          YOLOv8-X person/face            (face crops for MiVOLO)
  ReIDExtractor         SOLIDER Swin                    (appearance embeddings)
  DemographicsEstimator MiVOLO v2                       (age + gender, face AND body)

The SOLIDER backbone definition itself lives in solider_swin.py because it is
verbatim third-party model code.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import torch.nn as nn

import solider_swin

# MiVOLO pins timm==0.8.13.dev0; if a newer timm is installed, provide the two
# symbols its model factory expects so the import still succeeds.
try:  # pragma: no cover - only exercised on mismatched timm installs
    from timm.models import _factory as _timm_factory
    from timm.models import _helpers as _timm_helpers
    from timm.models import _pretrained as _timm_pretrained

    if not hasattr(_timm_helpers, "remap_checkpoint"):
        _timm_helpers.remap_checkpoint = (
            lambda model, sd, allow_reshape=True: _timm_helpers.remap_state_dict(sd, model, allow_reshape)
        )
    if not hasattr(_timm_pretrained, "split_model_name_tag"):
        _timm_pretrained.split_model_name_tag = _timm_factory.split_model_name_tag
except Exception:
    pass


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
class Stats:
    """Accumulates wall-clock time and call counts per pipeline stage."""

    def __init__(self):
        self.total = {}
        self.calls = {}
        self.items = {}

    @contextmanager
    def track(self, name, items=1):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.total[name] = self.total.get(name, 0.0) + dt
            self.calls[name] = self.calls.get(name, 0) + 1
            self.items[name] = self.items.get(name, 0) + items

    def report(self):
        return {
            name: {
                "total_s": round(self.total[name], 3),
                "calls": self.calls[name],
                "items": self.items[name],
                "ms_per_call": round(1000 * self.total[name] / max(1, self.calls[name]), 2),
                "ms_per_item": round(1000 * self.total[name] / max(1, self.items[name]), 2),
            }
            for name in sorted(self.total)
        }


def resolve_device(cfg):
    want_cuda = cfg["device"].get("cuda", True)
    if want_cuda and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available - falling back to CPU (this will be slow).")
        return torch.device("cpu")
    return torch.device("cuda:0" if want_cuda else "cpu")


# ---------------------------------------------------------------------------
# person detection + local tracking
# ---------------------------------------------------------------------------
def _write_tracker_config(cfg, out_dir):
    """Copy the packaged ultralytics tracker yaml and force our own track buffer.

    The buffer is kept SHORT on purpose: long-term identity must come from the
    ReID gallery, not from a tracker that simply refuses to forget.
    """
    import ultralytics
    import yaml as _yaml

    custom = cfg["tracking"].get("config_path") or ""
    if custom:
        return custom
    name = cfg["tracking"].get("tracker", "bytetrack")
    src = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", f"{name}.yaml")
    if not os.path.exists(src):
        raise FileNotFoundError(f"unknown tracker {name!r} (expected {src})")
    with open(src, "r", encoding="utf-8") as fh:
        data = _yaml.safe_load(fh)
    tc = cfg["tracking"]
    data["track_buffer"] = int(tc.get("max_age_frames", 30))
    data["track_low_thresh"] = float(tc.get("detect_low_conf", 0.1))
    data["track_high_thresh"] = float(cfg["detection"]["confidence"])
    data["new_track_thresh"] = float(tc.get("new_track_conf", 0.45))
    data["match_thresh"] = float(tc.get("match_thresh", 0.8))
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "_tracker.yaml")
    with open(dst, "w", encoding="utf-8") as fh:
        _yaml.safe_dump(data, fh)
    return dst


def _precision_kwargs(use_fp16):
    """FP16 flag for ultralytics: newer versions use `quantize=16`, older ones `half=True`."""
    if not use_fp16:
        return {}
    try:
        from ultralytics.utils import DEFAULT_CFG_DICT

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:
        pass
    return {"half": True}


class PersonDetector:
    """YOLO person detector with an attached short-term tracker."""

    def __init__(self, cfg, device, stats):
        from ultralytics import YOLO

        det = cfg["detection"]
        self.model = YOLO(det["model"])
        self.device = device
        self.stats = stats
        # The detector runs at the LOW threshold; ByteTrack applies
        # detection.confidence itself as its high threshold (see the tracker yaml).
        self.conf = float(cfg["tracking"].get("detect_low_conf", det["confidence"]))
        self.iou = float(det["iou"])
        self.imgsz = int(det["imgsz"])
        self.max_det = int(det.get("max_det", 100))
        self.min_h = int(det.get("min_box_height", 0))
        self.prec = _precision_kwargs(bool(cfg["device"].get("fp16", True)) and device.type == "cuda")
        self.tracker_cfg = _write_tracker_config(cfg, cfg["video"]["output_dir"])
        self.model.to(device)

    def track(self, frame):
        """Run detection + local tracking on one BGR frame.

        Returns a list of dicts: xyxy (float32[4]), conf, local_track_id.
        Detections the tracker did not give an id to are dropped.
        """
        with self.stats.track("detector"):
            res = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_cfg,
                classes=[0],  # person only
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                device=self.device,
                **self.prec,
                verbose=False,
            )[0]

        out = []
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        for box, conf, tid in zip(xyxy, confs, ids):
            if box[3] - box[1] < self.min_h:
                continue
            out.append({"xyxy": box.astype(np.float32), "conf": float(conf), "local_track_id": int(tid)})
        return out


class FaceDetector:
    """YOLOv8-X person/face detector - we only keep the 'face' class."""

    FACE_CLASS = 1

    def __init__(self, cfg, device, stats):
        from ultralytics import YOLO

        self.model = YOLO(cfg["demographics"]["face_detector"])
        self.model.to(device)
        self.device = device
        self.stats = stats
        self.conf = float(cfg["demographics"].get("face_confidence", 0.4))
        self.prec = _precision_kwargs(bool(cfg["device"].get("fp16", True)) and device.type == "cuda")
        self.imgsz = int(cfg["detection"]["imgsz"])

    def detect(self, frame):
        with self.stats.track("face_detector"):
            res = self.model.predict(
                frame,
                classes=[self.FACE_CLASS],
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                **self.prec,
                verbose=False,
            )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 5), np.float32)
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()[:, None]
        return np.hstack([xyxy, conf]).astype(np.float32)


# ---------------------------------------------------------------------------
# ReID
# ---------------------------------------------------------------------------
_REID_ARCH = {
    "solider_swin_tiny": solider_swin.swin_tiny_patch4_window7_224,
    "solider_swin_small": solider_swin.swin_small_patch4_window7_224,
    "solider_swin_base": solider_swin.swin_base_patch4_window7_224,
}

# Feature width of each SOLIDER variant, used to check a checkpoint actually
# belongs to the architecture named in the config before trying to load it.
_REID_DIM = {"solider_swin_tiny": 768, "solider_swin_small": 768, "solider_swin_base": 1024}
_REID_DEPTH = {"solider_swin_tiny": 6, "solider_swin_small": 18, "solider_swin_base": 18}


def _identify_checkpoint(sd):
    """Work out which SOLIDER variant a state dict came from."""
    dim = sd["bottleneck.weight"].shape[0] if "bottleneck.weight" in sd else None
    depth = 1 + max((int(k.split(".")[4]) for k in sd if k.startswith("base.stages.2.blocks.")),
                    default=-1)
    for name in _REID_ARCH:
        if _REID_DIM[name] == dim and _REID_DEPTH[name] == depth:
            return name
    return None


class ReIDExtractor:
    """SOLIDER appearance embedding extractor.

    crop (BGR) -> resize 384x128 -> SOLIDER Swin -> (optional BNNeck) -> L2 norm
    """

    def __init__(self, cfg, device, stats):
        rc = cfg["reid"]
        name = rc["model"]
        if name not in _REID_ARCH:
            raise ValueError(f"unknown reid model {name!r}, expected one of {sorted(_REID_ARCH)}")
        # `weights: auto` (or blank) derives the path from the model name, so
        # switching backbone is a ONE line config change and the two can never
        # drift apart.
        weights = str(rc.get("weights") or "auto").strip()
        if weights.lower() in ("auto", ""):
            weights = os.path.join("weights", f"{name}_msmt17.pth")
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"ReID weights not found: {weights}\n"
                f"Run  python main.py --download-weights --reid-model {name}  (see README)."
            )

        self.device = device
        self.stats = stats
        self.h, self.w = int(rc["input_size"][0]), int(rc["input_size"][1])
        self.mean = np.array(rc["pixel_mean"], np.float32).reshape(1, 1, 3)
        self.std = np.array(rc["pixel_std"], np.float32).reshape(1, 1, 3)
        self.batch_size = int(rc.get("batch_size", 16))
        self.flip = bool(rc.get("flip_augment", True))
        self.neck_feat = rc.get("neck_feat", "before")
        self.amp = bool(cfg["device"].get("fp16", True)) and device.type == "cuda"

        sd = torch.load(weights, map_location="cpu", weights_only=True)
        actual = _identify_checkpoint(sd)
        if actual is not None and actual != name:
            raise ValueError(
                f"reid.model is {name!r} but {weights} is a {actual!r} checkpoint.\n"
                f"Set  reid.weights: auto  in config.yaml (recommended), or point it at "
                f"weights/{name}_msmt17.pth."
            )
        self.backbone = _REID_ARCH[name](
            img_size=(self.h, self.w), semantic_weight=float(rc.get("semantic_weight", 0.2))
        )
        self.backbone.load_state_dict(
            {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}, strict=True
        )
        self.backbone.to(device)
        self.backbone.eval()

        self.dim = self.backbone.num_features[-1]
        self.bottleneck = None
        if self.neck_feat == "after":
            bn_sd = {k[len("bottleneck."):]: v for k, v in sd.items() if k.startswith("bottleneck.")}
            self.bottleneck = nn.BatchNorm1d(self.dim)
            self.bottleneck.load_state_dict(bn_sd)
            self.bottleneck.to(device)
            self.bottleneck.eval()
        print(f"[ReID] {name} loaded from {weights} (dim={self.dim}, neck_feat={self.neck_feat})")

    def _preprocess(self, crops):
        batch = np.empty((len(crops), self.h, self.w, 3), np.float32)
        for i, crop in enumerate(crops):
            img = cv2.resize(crop, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch[i] = (img - self.mean) / self.std
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        return tensor.to(self.device, non_blocking=True)

    def _forward(self, tensor):
        feat, _ = self.backbone(tensor)
        if self.bottleneck is not None:
            feat = self.bottleneck(feat)
        return feat.float()

    @torch.inference_mode()
    def extract(self, crops):
        """crops: list of BGR uint8 arrays -> float32 [N, D], L2-normalised."""
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        feats = []
        with self.stats.track("reid", items=len(crops)):
            for i in range(0, len(crops), self.batch_size):
                tensor = self._preprocess(crops[i: i + self.batch_size])
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    feat = self._forward(tensor)
                    if self.flip:
                        feat = feat + self._forward(torch.flip(tensor, dims=[3]))
                feats.append(torch.nn.functional.normalize(feat.float(), dim=1).cpu().numpy())
        return np.concatenate(feats, 0).astype(np.float32)


# ---------------------------------------------------------------------------
# age + gender
# ---------------------------------------------------------------------------
class DemographicsEstimator:
    """MiVOLO v2 - one transformer taking BOTH the face and the body crop.

    Body-only inference is supported (the face crop may be None), which matters
    for surveillance footage where the face is often not visible.
    """

    GENDER = {0: "male", 1: "female"}

    def __init__(self, cfg, device, stats):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        dc = cfg["demographics"]
        repo = dc.get("hf_repo", "iitolstykh/mivolo_v2")
        self.device = device
        self.stats = stats
        self.batch_size = int(dc.get("batch_size", 8))
        self.amp = bool(cfg["device"].get("fp16", True)) and device.type == "cuda"

        self.processor = AutoImageProcessor.from_pretrained(repo, trust_remote_code=True)
        self.model = AutoModelForImageClassification.from_pretrained(repo, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.with_persons = bool(self.model.config.with_persons_model)
        print(f"[Demographics] {repo} loaded (face+body={self.with_persons})")

    @torch.inference_mode()
    def predict(self, face_crops, body_crops):
        """face_crops / body_crops: equal-length lists of BGR arrays or None.

        Returns a list of dicts: age, gender, gender_conf.
        """
        n = len(body_crops)
        if n == 0:
            return []
        results = []
        with self.stats.track("demographics", items=n):
            for i in range(0, n, self.batch_size):
                faces = list(face_crops[i: i + self.batch_size])
                bodies = list(body_crops[i: i + self.batch_size])
                faces_in = self.processor.preprocess(faces)["pixel_values"].to(self.device)
                bodies_in = self.processor.preprocess(bodies)["pixel_values"].to(self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    out = self.model(faces_input=faces_in, body_input=bodies_in, return_dict=True)
                ages = out.age_output.float().flatten().cpu().numpy()
                gidx = out.gender_class_idx.flatten().cpu().numpy()
                gprob = out.gender_probs.float().flatten().cpu().numpy()
                for age, gi, gp in zip(ages, gidx, gprob):
                    results.append(
                        {
                            "age": float(age),
                            "gender": self.GENDER.get(int(gi), "unknown"),
                            "gender_conf": float(gp),
                        }
                    )
        return results


class FaceEmbedder:
    """FaceNet (InceptionResnetV1 / VGGFace2) embedding of a face crop.

    Used ONLY to link the same person across re-entries WITHIN one video, as a
    secondary signal behind the body appearance. Nothing is compared against any
    external face database, no identity is looked up, and the embeddings live in
    memory for the duration of the run and are never written to the outputs.

    On surveillance footage faces are small (median ~46 px here) so this is a
    weak signal on its own - measured 0.20 recall vs 0.72 for the body at the
    same false-merge rate. It earns its place only in combination.
    """

    SIZE = 160

    def __init__(self, cfg, device, stats):
        from facenet_pytorch import InceptionResnetV1

        fc = cfg["reid"].get("face_fusion", {})
        self.device = device
        self.stats = stats
        self.min_pixels = int(fc.get("min_face_pixels", 24))
        self.batch_size = int(fc.get("batch_size", 32))
        self.amp = bool(cfg["device"].get("fp16", True)) and device.type == "cuda"
        self.model = InceptionResnetV1(pretrained="vggface2")
        self.model.to(device)
        self.model.eval()
        self.dim = 512
        print("[Face] facenet-pytorch InceptionResnetV1 (vggface2) loaded "
              "- within-video linking only")

    def usable(self, crop):
        return crop is not None and min(crop.shape[:2]) >= self.min_pixels

    @torch.inference_mode()
    def extract(self, crops):
        """crops: list of BGR face images -> float32 [N, 512], L2-normalised."""
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        out = []
        with self.stats.track("face_embed", items=len(crops)):
            for i in range(0, len(crops), self.batch_size):
                batch = np.stack([
                    cv2.cvtColor(cv2.resize(c, (self.SIZE, self.SIZE)), cv2.COLOR_BGR2RGB)
                    for c in crops[i: i + self.batch_size]
                ]).astype(np.float32)
                tensor = torch.from_numpy((batch - 127.5) / 128.0).permute(0, 3, 1, 2)
                tensor = tensor.to(self.device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    feat = self.model(tensor)
                out.append(torch.nn.functional.normalize(feat.float(), dim=1).cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)
