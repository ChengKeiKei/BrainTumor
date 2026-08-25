from __future__ import annotations

from src.hybrid_inference import build_first_recurrence_artifact


if __name__ == "__main__":
    out = build_first_recurrence_artifact(force=True)
    print(out)

