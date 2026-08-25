"""Tiny stub backend used by the smoke test.

It does NOT load any model — it just verifies that the (Patient × Timepoint)
loop, the NIfTI-path plumbing, the resume / flush logic, and the CSV/JSON
sinks all work end-to-end before we spend ~10 hours on real RadFM inference.

The smoke test in `tests/test_smoke_pipeline.py` swaps this in via
`--backend stub` after monkey-patching the ladder.
"""
from pathlib import Path

_LOADED = False

def load_model(device=None):
    global _LOADED
    _LOADED = True
    print(f"[stub] loaded (device hint = {device})")

def caption_volumes(t1c, t1n, t2f, t2w):
    if not _LOADED:
        raise RuntimeError("stub backend not loaded")
    # Cheap sanity check: every NIfTI path must exist.
    for p in (t1c, t1n, t2f, t2w):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    return ("STUB CAPTION — t1c=" + Path(t1c).name +
            "; t1n=" + Path(t1n).name +
            "; t2f=" + Path(t2f).name +
            "; t2w=" + Path(t2w).name)
