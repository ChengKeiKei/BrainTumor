"""MAIRA-2 backend stub.

Fill in `load_model()` and `caption_volumes(...)` when this rung of the
medical-VLM fallback ladder is selected.  Raise ImportError or
RuntimeError to skip to the next model in the ladder."""

def load_model():
    raise ImportError("MAIRA-2 not yet wired in — falling through ladder")

def caption_volumes(t1c, t1n, t2f, t2w):
    raise NotImplementedError
