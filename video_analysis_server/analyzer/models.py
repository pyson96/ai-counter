"""Re-export of the project-root model wrappers.

The detector / tracker / SOLIDER ReID / MiVOLO wrappers are NOT duplicated for
the web server - this module simply makes them importable as
`analyzer.models` so the server code reads naturally.

    <root>/models.py  ->  analyzer.models
"""

from __future__ import annotations

import importlib

from . import PROJECT_ROOT  # noqa: F401  (puts the project root on sys.path)

_root = importlib.import_module("models")

Stats = _root.Stats
resolve_device = _root.resolve_device
PersonDetector = _root.PersonDetector
FaceDetector = _root.FaceDetector
ReIDExtractor = _root.ReIDExtractor
DemographicsEstimator = _root.DemographicsEstimator
FaceEmbedder = _root.FaceEmbedder

__all__ = [
    "Stats", "resolve_device", "PersonDetector", "FaceDetector",
    "ReIDExtractor", "DemographicsEstimator", "FaceEmbedder",
]
