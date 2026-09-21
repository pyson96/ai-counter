"""Vehicle license plate detection + recognition (ALPR).

Two-stage detection, because plates in wide traffic shots are tiny (30-60 px):

    frame --> YOLO11x (vehicles, tracked)      # gives a stable ID per car
          --> YOLO11x-plate on each vehicle crop  # crop is upscaled first, so a
                                                  # 40 px plate becomes ~250 px
          --> PP-OCRv5 (korean) on the plate crop
          --> per-track voting over the whole video

Recognising a 40 px plate on a single frame is unreliable no matter which model
you use, so the text is never taken from one frame: every OCR hit votes for the
track it belongs to and the winner is what gets drawn.

    python detect_car.py                        # car_test.mp4 -> output_car/result.mp4
    python detect_car.py --max-frames 300       # quick debug run
    python detect_car.py --download-weights     # fetch the checkpoints first

Models (all local, no cloud APIs):
  vehicles  ultralytics YOLO11x            weights/yolo11x.pt
  plates    YOLO11x fine-tuned on plates   weights/license_plate_yolo11x.pt
            (morsetechlab/yolov11-license-plate-detection, the -v1x variant)
  OCR       PP-OCRv5 mobile det + the      weights/ocr/{det,korean_rec}.onnx
            korean PP-OCRv5 recogniser     (PaddlePaddle/*_onnx, run via onnxruntime)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

VEHICLE_WEIGHTS = "weights/yolo11x.pt"
PLATE_WEIGHTS = "weights/license_plate_yolo11x.pt"
OCR_DIR = "weights/ocr"

HF_FILES = {
    PLATE_WEIGHTS: ("morsetechlab/yolov11-license-plate-detection", "license-plate-finetune-v1x.pt"),
    f"{OCR_DIR}/det.onnx": ("PaddlePaddle/PP-OCRv5_mobile_det_onnx", "inference.onnx"),
    f"{OCR_DIR}/det.yml": ("PaddlePaddle/PP-OCRv5_mobile_det_onnx", "inference.yml"),
    f"{OCR_DIR}/korean_rec.onnx": ("PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx", "inference.onnx"),
    f"{OCR_DIR}/korean_rec.yml": ("PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx", "inference.yml"),
}

# COCO ids that count as a vehicle
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def download_weights():
    from huggingface_hub import hf_hub_download

    os.makedirs(OCR_DIR, exist_ok=True)
    for dst, (repo, name) in HF_FILES.items():
        if os.path.exists(dst):
            print(f"[weights] {dst} already present")
            continue
        print(f"[weights] downloading {repo}/{name} ...")
        shutil.copy(hf_hub_download(repo, name), dst)
        print(f"[weights] -> {dst}")
    if not os.path.exists(VEHICLE_WEIGHTS):
        from ultralytics import YOLO

        print("[weights] downloading yolo11x.pt ...")
        YOLO("yolo11x.pt")
        shutil.move("yolo11x.pt", VEHICLE_WEIGHTS)
    print("[weights] done")


# ---------------------------------------------------------------------------
# Korean plate grammar
#
# Used two ways: to repair OCR output (a digit slot can only hold a digit, so
# "O" there is a zero) and to score competing readings of the same car.
# ---------------------------------------------------------------------------

HANGUL = r"\uac00-\ud7a3"

PLATE_PATTERNS = [
    # 12가3456 / 123가4567  - every plate issued since 2004
    (re.compile(rf"^\d{{2,3}}[{HANGUL}]\d{{4}}$"), 1.00),
    # 서울12가3456           - the old region-prefixed format
    (re.compile(rf"^[{HANGUL}]{{2}}\d{{2}}[{HANGUL}]\d{{4}}$"), 1.00),
    # 외교123-456 and friends
    (re.compile(rf"^[{HANGUL}]{{2,3}}\d{{3,6}}$"), 0.80),
]

# characters the recogniser confuses in a slot that must be a digit
DIGIT_FIXUPS = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
})

_STRIP = re.compile(rf"[^0-9A-Za-z{HANGUL}]")

# A Korean plate can only ever contain these syllables. Masking the CTC output
# down to this alphabet suppresses the junk letters the recogniser bolts onto
# the ends of a reading ("F18 68192" -> "1868192").
PLATE_SYLLABLES = (
    "가나다라마"          # private
    "거너더러머버서어저"
    "고노도로모보소오조"
    "구누두루무부수우주"
    "바사아자"            # commercial
    "배"                  # delivery
    "하허호"              # rental
    "육해공국합"          # military
)
# the region prefix on pre-2004 plates ("서울12가3456")
REGION_SYLLABLES = "서울경기인천강원충남대전북부산광주제종세"
PLATE_ALPHABET = frozenset("0123456789" + PLATE_SYLLABLES + REGION_SYLLABLES)

UNKNOWN = "?"

# The syllable on a plate 40 px wide is roughly an 8 px glyph, and at that size
# the recogniser reports a Latin look-alike instead ("85아0527" -> "85a0527")
# with the true syllable nowhere in its top candidates. Guessing a syllable
# from that would be inventing a number plate, so the slot is marked unknown
# and the digits - which are read reliably - are kept.
_LATIN_SYLLABLE_SLOT = re.compile(r"^(\d{2,3})[A-Za-z]{1,2}(\d{4})$")


def normalise_plate(text: str) -> str:
    """Strip decoration and fold the usual OCR look-alikes into digits."""
    text = unicodedata.normalize("NFC", text)
    text = _STRIP.sub("", text)
    if not text:
        return ""
    slot = _LATIN_SYLLABLE_SLOT.match(text)
    if slot:
        return f"{slot.group(1)}{UNKNOWN}{slot.group(2)}"
    # Latin letters never legitimately appear on a Korean plate, so any that
    # survived are misread digits.
    return text.translate(DIGIT_FIXUPS)


def plate_score(text: str) -> float:
    """0.0 = not plate-shaped, 1.0 = a perfectly well-formed plate number."""
    if not text:
        return 0.0
    for pattern, weight in PLATE_PATTERNS:
        if pattern.match(text):
            return weight
    # every digit read, only the syllable missing - still a usable plate
    if re.match(rf"^\d{{2,3}}\{UNKNOWN}\d{{4}}$", text):
        return 0.90
    if re.match(rf"^[{HANGUL}]{{2}}\d{{2}}\{UNKNOWN}\d{{4}}$", text):
        return 0.90
    # partial credit: right shape, wrong length (a digit was dropped or doubled)
    if re.match(rf"^\d{{1,4}}[{HANGUL}]\d{{2,5}}$", text):
        return 0.45
    if re.search(rf"[{HANGUL}]", text) and sum(c.isdigit() for c in text) >= 4:
        return 0.25
    return 0.05 if len(text) >= 4 else 0.0


# ---------------------------------------------------------------------------
# PP-OCRv5 via onnxruntime
# ---------------------------------------------------------------------------


def _register_cuda_dlls():
    """Point onnxruntime at torch's bundled CUDA runtime.

    onnxruntime-gpu needs cublasLt64_12.dll / cudnn64_9.dll on the DLL search
    path. A machine with torch+cu121 installed already has them inside
    torch/lib, and there is no reason to make the user install the CUDA toolkit
    a second time just to run the recogniser on the GPU.
    """
    if not hasattr(os, "add_dll_directory"):  # not Windows
        return
    try:
        import torch

        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
    except Exception:
        pass


class PaddleOCR:
    """The two PP-OCRv5 stages we need: DB text detection + CTC recognition.

    PaddleOCR itself is not installed - these are the official ONNX exports run
    straight through onnxruntime, which keeps the dependency footprint to
    onnxruntime + pyclipper and lets the recogniser share the GPU with YOLO.
    """

    REC_HEIGHT = 48

    def __init__(self, ocr_dir=OCR_DIR, use_gpu=True, det_limit=736,
                 det_thresh=0.3, det_box_thresh=0.4, det_unclip=1.8):
        if use_gpu:
            _register_cuda_dlls()
        import onnxruntime as ort
        import yaml

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(os.path.join(ocr_dir, "det.onnx"), opts, providers=providers)
        self.rec = ort.InferenceSession(os.path.join(ocr_dir, "korean_rec.onnx"), opts, providers=providers)
        self.provider = self.rec.get_providers()[0]

        with open(os.path.join(ocr_dir, "korean_rec.yml"), "r", encoding="utf-8") as fh:
            chars = yaml.safe_load(fh)["PostProcess"]["character_dict"]
        # PaddleOCR's CTC layout: blank, then the dictionary, then a space.
        self.charset = ["<b>"] + list(chars) + [" "]
        self.plate_mask = np.array(
            [i == 0 or c in PLATE_ALPHABET for i, c in enumerate(self.charset)],
            dtype=np.float32,
        )

        self.det_limit = det_limit
        self.det_thresh = det_thresh
        self.det_box_thresh = det_box_thresh
        self.det_unclip = det_unclip

    # -- detection ---------------------------------------------------------

    def detect(self, img):
        """Return text-line boxes as (4, 2) float arrays, top row first."""
        h, w = img.shape[:2]
        scale = min(self.det_limit / max(h, w), 4.0)
        rh = max(32, int(round(h * scale / 32)) * 32)
        rw = max(32, int(round(w * scale / 32)) * 32)
        resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)

        x = resized.astype(np.float32) / 255.0
        x -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        x /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = x.transpose(2, 0, 1)[None]

        prob = self.det.run(None, {self.det.get_inputs()[0].name: x})[0][0, 0]
        return self._db_boxes(prob, w / rw, h / rh, w, h)

    def _db_boxes(self, prob, sx, sy, w, h):
        import pyclipper
        from shapely.geometry import Polygon

        mask = (prob > self.det_thresh).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours[:200]:
            if cv2.contourArea(contour) < 4:
                continue
            approx = cv2.approxPolyDP(contour, 0.002 * cv2.arcLength(contour, True), True).reshape(-1, 2)
            if len(approx) < 4:
                continue
            if self._region_score(prob, approx) < self.det_box_thresh:
                continue
            poly = Polygon(approx)
            if poly.length < 1e-6:
                continue
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(approx.astype(np.int64), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(poly.area * self.det_unclip / poly.length)
            if not expanded:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(np.array(expanded[0]).reshape(-1, 2)))
            box[:, 0] = np.clip(box[:, 0] * sx, 0, w - 1)
            box[:, 1] = np.clip(box[:, 1] * sy, 0, h - 1)
            side = min(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))
            if side < 3:
                continue
            boxes.append(_order_quad(box))
        # top-to-bottom, then left-to-right - the reading order of a 2-line plate
        boxes.sort(key=lambda b: (round(b[:, 1].mean() / 8), b[:, 0].mean()))
        return boxes

    @staticmethod
    def _region_score(prob, poly):
        h, w = prob.shape
        x0 = int(np.clip(np.floor(poly[:, 0].min()), 0, w - 1))
        x1 = int(np.clip(np.ceil(poly[:, 0].max()), 0, w - 1))
        y0 = int(np.clip(np.floor(poly[:, 1].min()), 0, h - 1))
        y1 = int(np.clip(np.ceil(poly[:, 1].max()), 0, h - 1))
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        shifted = poly.copy()
        shifted[:, 0] -= x0
        shifted[:, 1] -= y0
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
        if not mask.any():
            return 0.0
        return float(cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], mask)[0])

    # -- recognition -------------------------------------------------------

    def recognise(self, crops):
        """Run the CTC recogniser over a batch of line crops.

        Returns the per-crop probability matrices rather than strings, so the
        caller can decode them more than once (free and plate-constrained)
        without paying for a second forward pass.
        """
        if not crops:
            return []
        ratios = [c.shape[1] / max(c.shape[0], 1) for c in crops]
        width = int(np.clip(math.ceil(self.REC_HEIGHT * max(ratios) / 8) * 8, 64, 800))

        batch = np.zeros((len(crops), 3, self.REC_HEIGHT, width), dtype=np.float32)
        for i, crop in enumerate(crops):
            cw = min(width, max(8, int(math.ceil(self.REC_HEIGHT * ratios[i]))))
            resized = cv2.resize(crop, (cw, self.REC_HEIGHT), interpolation=cv2.INTER_CUBIC)
            x = resized.astype(np.float32) / 255.0
            x = (x - 0.5) / 0.5
            batch[i, :, :, :cw] = x.transpose(2, 0, 1)

        logits = self.rec.run(None, {self.rec.get_inputs()[0].name: batch})[0]
        if logits.min() < 0.0 or logits.max() > 1.0:  # not softmaxed by the graph
            logits = _softmax(logits)
        return [logits[i] for i in range(len(crops))]

    def _ctc_decode(self, probs, restrict=False):
        if restrict:
            probs = probs * self.plate_mask
        idx = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        chars, scores, prev = [], [], -1
        for t, k in enumerate(idx):
            if k != 0 and k != prev:
                chars.append(self.charset[k] if k < len(self.charset) else "")
                scores.append(conf[t])
            prev = k
        if not chars:
            return "", 0.0
        return "".join(chars).strip(), float(np.mean(scores))

    # -- the bit callers actually use --------------------------------------

    def read_plate(self, crop):
        """Read one plate crop. Returns (text, confidence) after normalisation.

        Several readings are raced against each other, because none of them
        wins on every plate: the crop as a single line (the common case), an
        unsharp-masked version (upscaled plates are soft, and sharpening
        recovers digits the plain read drops), the text-detector's lines joined
        together (tightens the framing and handles 2-line plates), and - only
        for crops too tall to be one line - a blind top/bottom split for when
        the detector merges the two lines.

        The winner is picked on Korean plate grammar first, OCR confidence
        second: a well-formed plate read at 0.6 beats a confident 0.95 of junk.
        """
        h, w = crop.shape[:2]
        candidates = self._read_batch(self._crop_variants(crop))

        boxes = self.detect(crop)
        if boxes:
            lines = [_warp_quad(crop, b) for b in boxes]
            candidates += self._read_lines(lines)
            candidates += self._read_lines([_sharpen(c) for c in lines if c is not None])

        if w / max(h, 1) < 2.0:  # too tall to be a single line of plate text
            half, pad = h // 2, max(1, h // 12)
            candidates += self._read_lines([crop[:half + pad], crop[half - pad:]])

        best = ("", 0.0, -1.0)
        for text, conf in candidates:
            norm = normalise_plate(text)
            if not norm:
                continue
            # a Korean plate is 7-8 characters; readings of a wildly different
            # length are a misfire even when the recogniser is sure of them
            length_fit = 1.0 if 6 <= len(norm) <= 8 else (0.5 if len(norm) == 5 else 0.0)
            rank = plate_score(norm) * 2.0 + conf + 0.3 * length_fit
            if rank > best[2]:
                best = (norm, conf, rank)
        return best[0], best[1]

    @staticmethod
    def _crop_variants(crop):
        """The plate box includes the plate's frame and mounting, so the text
        itself only fills part of it - and the recogniser squashes whatever it
        is given to 48 px tall, which leaves the characters far too small.
        Trimming the frame away roughly doubles the confidence, so several
        trims are tried and the grammar decides which one was right."""
        h, w = crop.shape[:2]
        variants = []
        for keep in (1.0, 0.72, 0.58):
            margin = int(h * (1.0 - keep) / 2)
            sub = crop[margin:h - margin] if margin else crop
            if sub.shape[0] < 8:
                continue
            variants.append(sub)
            variants.append(_sharpen(sub))
        return variants

    def _read_batch(self, images):
        """Recognise independent images in one forward pass - one reading each."""
        images = [i for i in images if i is not None and i.size and i.shape[0] >= 4 and i.shape[1] >= 4]
        if not images:
            return []
        readings = []
        for probs in self.recognise(images):
            for restrict in (False, True):
                text, conf = self._ctc_decode(probs, restrict)
                if text:
                    readings.append((text, conf))
        return readings

    def _read_lines(self, crops):
        """Recognise a stack of line crops as one string, twice: once with the
        full Korean dictionary and once masked to the plate alphabet."""
        crops = [c for c in crops if c is not None and c.size and c.shape[0] >= 4 and c.shape[1] >= 4]
        if not crops:
            return []
        probs = self.recognise(crops)
        readings = []
        for restrict in (False, True):
            decoded = [self._ctc_decode(p, restrict) for p in probs]
            texts = [t for t, _ in decoded if t]
            if texts:
                confs = [c for t, c in decoded if t]
                readings.append(("".join(texts), float(np.mean(confs))))
        return readings


def _sharpen(img, amount=1.8):
    """Unsharp mask - undoes some of the softness bicubic upscaling adds."""
    if img is None or img.size == 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, amount, blurred, 1.0 - amount, 0)


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _order_quad(box):
    """Order a quad as top-left, top-right, bottom-right, bottom-left."""
    box = box[np.argsort(box[:, 0])]
    left, right = box[:2], box[2:]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def _warp_quad(img, quad):
    w = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[3] - quad[2])))
    h = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    if w < 4 or h < 4:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (w, h))


# ---------------------------------------------------------------------------
# per-vehicle state
# ---------------------------------------------------------------------------


@dataclass
class Vehicle:
    tid: int
    cls_name: str
    first_frame: int
    last_frame: int = 0
    votes: Counter = field(default_factory=Counter)
    reads: int = 0
    best_crop: np.ndarray | None = None   # sharpest plate crop, for the panel
    best_crop_quality: float = -1.0
    box: tuple | None = None              # last vehicle box, in frame coords
    plate_box: tuple | None = None        # last plate box, in frame coords
    plate_frame: int = -999               # frame that box came from
    last_read: int = -999                 # frame of the last OCR attempt
    _winner: tuple | None = None          # cache, invalidated by vote()

    @property
    def text(self):
        """The winning reading and its share of the vote. Cached: the renderer
        asks every visible vehicle for this on every frame."""
        if self._winner is None:
            if not self.votes:
                return "", 0.0
            merged = merge_votes(self.votes)
            text, weight = merged.most_common(1)[0]
            self._winner = (text, weight / max(sum(merged.values()), 1e-6))
        return self._winner

    def vote(self, text, conf, quality, crop):
        """Weight a reading by OCR confidence, plate grammar and crop quality."""
        if quality > self.best_crop_quality:
            self.best_crop_quality = quality
            self.best_crop = crop.copy()
        if len(text) < 5:  # a stray character or two is not a plate reading
            return
        self.reads += 1
        self.votes[text] += conf * (0.25 + plate_score(text)) * (0.4 + 0.6 * quality)
        self._winner = None


def _completes(unknown: str, resolved: str) -> bool:
    """Is `resolved` the same plate as `unknown`, with the syllable filled in?"""
    if len(unknown) != len(resolved):
        return False
    return all(u == r or (u == UNKNOWN and "가" <= r <= "힣")
               for u, r in zip(unknown, resolved))


SYLLABLE_MERGE_SUPPORT = 0.25


def merge_votes(votes: Counter) -> Counter:
    """Fold "85?0527" into "85아0527" when some frame did resolve the syllable.

    Most frames of a distant car cannot resolve it, so without this the
    unknown-syllable reading would always out-vote the correct one.

    The resolved reading has to have earned its own support first, though: a
    single frame that hallucinated one syllable must not capture a dozen votes
    that only ever agreed on the digits. Below the support threshold the
    syllable stays unknown, which is the honest answer.
    """
    resolved = [t for t in votes if UNKNOWN not in t]
    merged = Counter()
    for text, weight in votes.items():
        if UNKNOWN in text:
            matches = [r for r in resolved
                       if _completes(text, r) and votes[r] >= SYLLABLE_MERGE_SUPPORT * weight]
            if len(matches) == 1:
                merged[matches[0]] += weight
                continue
        merged[text] += weight
    return merged


def crop_quality(crop):
    """Sharpness x size, squashed to 0..1. Blurry motion frames score low."""
    if crop is None or crop.size == 0:
        return 0.0
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = min(cv2.Laplacian(grey, cv2.CV_64F).var() / 400.0, 1.0)
    size = min(crop.shape[1] / 120.0, 1.0)
    return float(0.6 * sharp + 0.4 * size)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _settled(vehicle):
    """Has this vehicle's number been read often enough to trust it?"""
    text, _ = vehicle.text
    return bool(text) and plate_score(text) >= 0.8 and vehicle.reads >= 2


FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]

PANEL_W = 380
PANEL_ROWS = 6
THUMB_W, THUMB_H = 200, 74


class Renderer:
    """Composites the annotated frame plus the plate panel down the right side."""

    def __init__(self, font_path=None):
        from PIL import ImageFont

        path = font_path or next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
        if path is None:
            raise RuntimeError("no Korean-capable TTF found - pass --font")
        self.font_path = path
        self.big = ImageFont.truetype(path, 26)
        self.mid = ImageFont.truetype(path, 19)
        self.small = ImageFont.truetype(path, 14)

    def draw(self, frame, vehicles, frame_idx, fps, stats):
        from PIL import Image, ImageDraw

        h, w = frame.shape[:2]
        canvas = np.full((h, w + PANEL_W, 3), 24, dtype=np.uint8)
        canvas[:, :w] = frame

        text_jobs = []
        for v in vehicles:
            if v.box is not None:
                cv2.rectangle(canvas, v.box[:2], v.box[2:], (90, 90, 90), 1)
            if v.plate_box is None:
                continue
            x1, y1, x2, y2 = v.plate_box
            text, share = v.text
            colour = (0, 220, 60) if _settled(v) else (0, 170, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            if text:
                # label above the plate, or below it when the plate is near the top
                ty = y1 - 27 if y1 > 30 else y2 + 5
                # plates sit against bright bodywork, so the label needs a plate
                # of its own to stay readable
                width = int(len(text) * 12.5) + 10
                cv2.rectangle(canvas, (x1 - 3, ty - 2), (x1 + width, ty + 24), (20, 20, 20), -1)
                text_jobs.append((x1 + 2, ty, text, colour, self.mid, False))

        self._panel(canvas, w, h, vehicles, text_jobs, frame_idx, fps, stats)

        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        drawer = ImageDraw.Draw(image)
        for x, y, text, colour, font, shadow in text_jobs:
            rgb = (colour[2], colour[1], colour[0])
            if shadow:
                drawer.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
            drawer.text((x, y), text, font=font, fill=rgb)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def _panel(self, canvas, w, h, vehicles, text_jobs, frame_idx, fps, stats):
        px = w + 14
        cv2.line(canvas, (w, 0), (w, h), (60, 60, 60), 1)
        text_jobs.append((px, 10, "License Plates", (255, 255, 255), self.big, False))
        text_jobs.append((
            px, 44,
            f"frame {frame_idx}  |  {frame_idx / max(fps, 1e-6):6.1f}s  |  "
            f"plates {stats['unique']}",
            (170, 170, 170), self.small, False,
        ))

        y = 72
        rows = [v for v in vehicles if v.votes and v.best_crop is not None]
        for v in rows[:PANEL_ROWS]:
            text, share = v.text
            thumb = _fit(v.best_crop, THUMB_W, THUMB_H)
            th, tw = thumb.shape[:2]
            canvas[y:y + th, px:px + tw] = thumb
            cv2.rectangle(canvas, (px - 1, y - 1), (px + tw, y + th), (90, 90, 90), 1)

            colour = (0, 220, 60) if _settled(v) else (0, 170, 255)
            tx = px + tw + 12
            text_jobs.append((tx, y + 4, text or "...", colour, self.mid, False))
            text_jobs.append((
                tx, y + 30,
                f"#{v.tid} {v.cls_name}",
                (160, 160, 160), self.small, False,
            ))
            text_jobs.append((
                tx, y + 48,
                f"{share * 100:3.0f}% / {v.reads} reads",
                (160, 160, 160), self.small, False,
            ))
            y += th + 16


def _fit(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


class PlateReader:
    def __init__(self, args):
        from ultralytics import YOLO

        self.args = args
        self.device = "cpu" if args.cpu else 0
        self.vehicles_model = YOLO(args.vehicle_weights)
        self.plates_model = YOLO(args.plate_weights)
        self.ocr = PaddleOCR(args.ocr_dir, use_gpu=not args.cpu)
        # The renderer needs a Hangul TTF and PIL; an analysis-only run (the web
        # server never writes a result video) must not require either.
        self.renderer = None if getattr(args, "no_render", False) else Renderer(args.font)
        self.tracks: dict[int, Vehicle] = {}
        self.recent: deque[int] = deque(maxlen=64)
        print(f"[ocr] onnxruntime provider: {self.ocr.provider}")

    # -- stage 1 -----------------------------------------------------------

    def detect_vehicles(self, frame):
        result = self.vehicles_model.track(
            frame, persist=True, verbose=False, device=self.device,
            imgsz=self.args.imgsz, conf=self.args.vehicle_conf, iou=0.6,
            classes=sorted(VEHICLE_CLASSES), tracker="bytetrack.yaml",
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        out = []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), tid, cls in zip(xyxy, ids, clss):
            if (x2 - x1) < self.args.min_vehicle_size or (y2 - y1) < self.args.min_vehicle_size:
                continue
            out.append((tid, VEHICLE_CLASSES.get(cls, "vehicle"), (x1, y1, x2, y2)))
        return out

    # -- stage 2 -----------------------------------------------------------

    def detect_plates(self, frame, vehicles):
        """Find one plate per vehicle. The crops are upscaled first: the plate
        detector needs pixels, and a 40 px plate in the full frame has none."""
        if not vehicles:
            return {}
        h, w = frame.shape[:2]
        crops, meta = [], []
        for tid, _, (x1, y1, x2, y2) in vehicles:
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue
            scale = min(max(self.args.crop_size / max(crop.shape[1], 1), 1.0), 6.0)
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            crops.append(crop)
            meta.append((tid, max(0, x1), max(0, y1), scale))

        found = {}
        for start in range(0, len(crops), self.args.batch):
            chunk = crops[start:start + self.args.batch]
            results = self.plates_model.predict(
                chunk, verbose=False, device=self.device,
                imgsz=self.args.plate_imgsz, conf=self.args.plate_conf, iou=0.5, max_det=4,
            )
            for offset, result in enumerate(results):
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                tid, ox, oy, scale = meta[start + offset]
                confs = result.boxes.conf.cpu().numpy()
                box = result.boxes.xyxy.cpu().numpy()[int(confs.argmax())] / scale
                x1, y1, x2, y2 = (box + np.array([ox, oy, ox, oy])).astype(int)
                if x2 - x1 < 8 or y2 - y1 < 4:
                    continue
                # signage on the side of a truck is the usual false positive;
                # a real plate is wider than tall but never a long thin banner
                if not 0.9 <= (x2 - x1) / (y2 - y1) <= 6.5:
                    continue
                found[tid] = (int(x1), int(y1), int(x2), int(y2), float(confs.max()))
        return found

    # -- stage 3 -----------------------------------------------------------

    def read(self, frame, box):
        """Crop the plate out of the *original* frame and upscale it for OCR."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box[:4]
        # generous horizontal padding: the plate detector tends to clip the
        # first and last digit, and a clipped digit is a wrong plate number
        px = max(3, int((x2 - x1) * 0.10))
        py = max(2, int((y2 - y1) * 0.12))
        crop = frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 12:
            return None, None, 0.0
        scale = min(max(self.args.ocr_height / crop.shape[0], 1.0), 10.0)
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        text, conf = self.ocr.read_plate(big)
        return crop, text, conf

    # -- driver ------------------------------------------------------------

    def run(self):
        """Analyse the whole video, then draw it.

        Two passes on purpose. A plate's number is only settled once the votes
        are in, so a single-pass render would spend the first half of every car
        showing a number that is still wrong - the BMW at 77 s reads "52607883"
        for seventeen frames before it becomes "26다7883". The second pass costs
        one more decode of the video (seconds, against minutes of inference) and
        draws every car with the number the whole clip agreed on.
        """
        fps, size, limit = self._probe()
        started = time.time()
        timeline = self._analyse(limit, fps)
        analysed = time.time() - started
        self._render(timeline, fps, size, limit)
        return self.summary(fps, analysed, len(timeline))

    def _probe(self):
        cap = cv2.VideoCapture(self.args.input)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {self.args.input}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        limit = total if self.args.max_frames <= 0 else min(
            total, self.args.start_frame + self.args.max_frames)
        return fps, size, limit

    def _open(self):
        cap = cv2.VideoCapture(self.args.input)
        if self.args.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.args.start_frame)
        return cap

    def _analyse(self, limit, fps):
        """Pass 1: detect, track and read. Returns what was visible per frame."""
        args = self.args
        cap = self._open()
        timeline = []
        idx = args.start_frame
        started = time.time()
        while idx < limit:
            ok, frame = cap.read()
            if not ok:
                break

            vehicles = self.detect_vehicles(frame)
            for track in self.tracks.values():
                track.box = None
            for tid, cls_name, box in vehicles:
                track = self.tracks.get(tid)
                if track is None:
                    track = self.tracks[tid] = Vehicle(tid, cls_name, idx)
                track.last_frame = idx
                track.box = box
                if tid in self.recent:
                    self.recent.remove(tid)
                self.recent.appendleft(tid)

            if idx % args.plate_interval == 0:
                plates = self.detect_plates(frame, vehicles)
                for tid, box in plates.items():
                    track = self.tracks[tid]
                    track.plate_box = box[:4]
                    track.plate_frame = idx
                    if not self._should_read(track, idx):
                        continue
                    crop, text, conf = self.read(frame, box)
                    quality = crop_quality(crop)
                    if crop is None or quality < args.min_quality:
                        continue          # too blurry to trust - do not pollute the vote
                    track.last_read = idx  # an unreadable plate still costs an attempt
                    if text and conf >= args.ocr_conf:
                        track.vote(text, conf, quality, crop)
                    elif quality > track.best_crop_quality:
                        # nothing readable yet, but still show the driver a crop
                        track.best_crop = crop.copy()
                        track.best_crop_quality = quality

            # the box stays on screen between detector runs, but not forever
            for tid, _, _ in vehicles:
                track = self.tracks[tid]
                if idx - track.plate_frame > args.plate_interval * 3:
                    track.plate_box = None

            timeline.append([(tid, self.tracks[tid].box, self.tracks[tid].plate_box)
                             for tid, _, _ in vehicles])

            idx += 1
            if not args.quiet and idx % 50 == 0:
                done = idx - args.start_frame
                rate = done / max(time.time() - started, 1e-6)
                found = sum(1 for v in self.tracks.values() if plate_score(v.text[0]) >= 0.8)
                print(f"\r[analyse {idx}/{limit}] {rate:5.1f} fps  plates={found}",
                      end="", flush=True)

        cap.release()
        if not args.quiet:
            print()
        return timeline

    def _render(self, timeline, fps, size, limit):
        """Pass 2: draw every frame with the final, settled plate numbers."""
        args = self.args
        w, h = size
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w + PANEL_W, h))
        if not writer.isOpened():
            raise SystemExit(f"cannot open writer for {args.output}")

        # a plate counts from the moment its car first appears, since the number
        # drawn from frame one is the one the whole clip settled on
        identified = sorted(v.first_frame for v in self.tracks.values()
                            if plate_score(v.text[0]) >= 0.8)

        cap = self._open()
        recent = deque(maxlen=64)
        last_visible = {}
        started = time.time()
        for offset, visible in enumerate(timeline):
            ok, frame = cap.read()
            if not ok:
                break
            idx = args.start_frame + offset

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

            stats = {"unique": sum(1 for f in identified if f <= idx)}
            writer.write(self.renderer.draw(frame, panel, idx, fps, stats))

            if not args.quiet and (offset + 1) % 50 == 0:
                rate = (offset + 1) / max(time.time() - started, 1e-6)
                print(f"\r[render {idx + 1}/{limit}] {rate:5.1f} fps", end="", flush=True)

        cap.release()
        writer.release()
        if not args.quiet:
            print()

    def _should_read(self, track, idx):
        if idx - track.last_read < self.args.ocr_interval:
            return False
        # once a plate is well-formed and consistently voted for, back off
        text, share = track.text
        if plate_score(text) >= 0.8 and share > 0.7 and track.reads >= self.args.max_reads:
            return False
        return True

    def summary(self, fps, elapsed, frames):
        plates = []
        for v in sorted(self.tracks.values(), key=lambda t: t.first_frame):
            text, share = v.text
            if not text:
                continue
            plates.append({
                "track_id": int(v.tid),
                "class": v.cls_name,
                "plate": text,
                "valid_format": plate_score(text) >= 0.8,
                "vote_share": round(share, 3),
                "reads": v.reads,
                "first_seen_s": round(v.first_frame / fps, 2),
                "last_seen_s": round(v.last_frame / fps, 2),
                "syllable_unresolved": UNKNOWN in text,
                "alternatives": [t for t, _ in merge_votes(v.votes).most_common(4)][1:],
            })
        # one car can pick up more than one track id - it leaves frame behind a
        # pillar and comes back as a new track - so the same number shows up
        # twice. Roll those up, keeping the best-supported reading of each.
        unique = {}
        for p in plates:
            if not p["valid_format"]:
                continue
            best = unique.get(p["plate"])
            if best is None or p["reads"] > best["reads"]:
                unique[p["plate"]] = p
        rollup = [{"plate": p["plate"], "class": p["class"], "reads": p["reads"],
                   "first_seen_s": p["first_seen_s"],
                   "syllable_unresolved": p["syllable_unresolved"]}
                  for p in sorted(unique.values(), key=lambda x: x["first_seen_s"])]

        return {
            "video": self.args.input,
            "frames": frames,
            "elapsed_s": round(elapsed, 1),
            "fps": round(frames / max(elapsed, 1e-6), 2),
            "vehicles_tracked": len(self.tracks),
            "plates_read": len(plates),
            "plates_valid_format": sum(1 for p in plates if p["valid_format"]),
            "unique_plates": len(rollup),
            "plates_deduplicated": rollup,
            "plates": plates,
        }


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="License plate detection + Korean OCR")
    ap.add_argument("--input", default="car_test.mp4")
    ap.add_argument("--output", default="output_car/result.mp4")
    ap.add_argument("--json", default=None, help="default: <output dir>/plates.json")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--start-frame", type=int, default=0)

    ap.add_argument("--vehicle-weights", default=VEHICLE_WEIGHTS)
    ap.add_argument("--plate-weights", default=PLATE_WEIGHTS)
    ap.add_argument("--ocr-dir", default=OCR_DIR)
    ap.add_argument("--font", default=None, help="TTF with Hangul coverage")

    ap.add_argument("--imgsz", type=int, default=960, help="vehicle detector input size")
    ap.add_argument("--vehicle-conf", type=float, default=0.35)
    ap.add_argument("--min-vehicle-size", type=int, default=48,
                    help="ignore vehicles smaller than this (px) - their plate is unreadable")
    ap.add_argument("--crop-size", type=int, default=640,
                    help="vehicle crops are upscaled to at least this width before plate detection")
    ap.add_argument("--plate-imgsz", type=int, default=640)
    ap.add_argument("--plate-conf", type=float, default=0.25)
    ap.add_argument("--plate-interval", type=int, default=2, help="run plate detection every N frames")
    ap.add_argument("--batch", type=int, default=8, help="vehicle crops per plate-detector batch")

    ap.add_argument("--ocr-height", type=int, default=96, help="plate crops are upscaled to this height")
    ap.add_argument("--ocr-conf", type=float, default=0.35, help="minimum CTC confidence to cast a vote")
    ap.add_argument("--min-quality", type=float, default=0.12,
                    help="skip OCR on plate crops blurrier/smaller than this (0..1)")
    ap.add_argument("--ocr-interval", type=int, default=3, help="min frames between OCR runs per vehicle")
    ap.add_argument("--max-reads", type=int, default=12, help="stop reading a vehicle after N confident votes")

    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--download-weights", action="store_true")
    args = ap.parse_args(argv)

    if args.download_weights:
        download_weights()
        return 0

    for path in (args.vehicle_weights, args.plate_weights, os.path.join(args.ocr_dir, "korean_rec.onnx")):
        if not os.path.exists(path):
            print(f"missing {path} - run: python detect_car.py --download-weights", file=sys.stderr)
            return 2
    if not os.path.exists(args.input):
        print(f"input video not found: {args.input}", file=sys.stderr)
        return 2

    summary = PlateReader(args).run()

    json_path = args.json or os.path.join(os.path.dirname(args.output) or ".", "plates.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\nvideo   -> {args.output}")
    print(f"plates  -> {json_path}")
    print(f"{summary['vehicles_tracked']} vehicles tracked, {summary['plates_read']} read, "
          f"{summary['plates_valid_format']} well-formed -> {summary['unique_plates']} distinct "
          f"plates, in {summary['elapsed_s']}s ({summary['fps']} fps)")
    for p in summary["plates_deduplicated"]:
        mark = "?" if p["syllable_unresolved"] else " "
        print(f"  {mark} {p['plate']:<12} {p['class']:<10} {p['first_seen_s']:6.1f}s  "
              f"{p['reads']} reads")
    partial = [p for p in summary["plates"] if not p["valid_format"] and p["reads"] >= 5]
    if partial:
        print(f"  ({len(partial)} more read only partially - see {json_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
