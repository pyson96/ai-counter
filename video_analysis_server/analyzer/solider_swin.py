"""Re-export of the project-root SOLIDER Swin backbone.

    <root>/solider_swin.py  ->  analyzer.solider_swin

The checkpoint loader in models.ReIDExtractor imports the root module directly;
this alias exists so the package layout matches the rest of the analyzer and so
`from analyzer import solider_swin` works.
"""

from __future__ import annotations

import importlib

from . import PROJECT_ROOT  # noqa: F401

_root = importlib.import_module("solider_swin")

for _name in dir(_root):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_root, _name)

del _name
