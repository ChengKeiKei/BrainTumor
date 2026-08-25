"""MRI upload helpers and optional RadFM caption generation for Second Recurrence."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UI_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_ROOT = UI_ROOT.parent
RADFM_BACKEND = HANDOFF_ROOT / "second" / "VLM" / "vlm_backends"
RADFM_WEIGHTS = HANDOFF_ROOT / "second" / "VLM" / "RadFM_weights" / "pytorch_model.bin"


@dataclass
class CaptionResult:
    caption: str
    source: str
    warning: str = ""
    details: dict[str, Any] | None = None


def radfm_available() -> bool:
    return RADFM_WEIGHTS.exists() and (RADFM_BACKEND / "radfm_backend.py").exists()


def _suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower).suffix


def _is_nifti(name: str) -> bool:
    return _suffix(name) in {".nii", ".nii.gz"}


def _guess_modality(name: str) -> str | None:
    lower = name.lower()
    for key in ("t1c", "t1ce", "t1n", "t1", "t2f", "flair", "t2w", "t2"):
        if key in lower:
            if key in {"t1ce", "t1c"}:
                return "t1c"
            if key in {"t1n", "t1"}:
                return "t1n"
            if key in {"t2f", "flair"}:
                return "t2f"
            if key in {"t2w", "t2"}:
                return "t2w"
    return None


def save_uploads(uploads: list[Any], dest_dir: Path) -> dict[str, Path]:
    """Save Streamlit UploadedFile objects; map modality→path when possible."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    extras: list[Path] = []
    for idx, upload in enumerate(uploads):
        name = getattr(upload, "name", f"upload_{idx}")
        path = dest_dir / Path(name).name
        path.write_bytes(upload.getvalue())
        modality = _guess_modality(name)
        if modality and modality not in saved:
            saved[modality] = path
        else:
            extras.append(path)
    # Fill missing modality slots with extras in order
    for modality, path in zip(("t1c", "t1n", "t2f", "t2w"), extras):
        saved.setdefault(modality, path)
    return saved


def generate_caption_from_uploads(
    uploads: list[Any] | None,
    *,
    clinical_context: str = "",
    try_radfm: bool = True,
) -> CaptionResult:
    if not uploads:
        return CaptionResult(
            caption="",
            source="none",
            warning="No MRI files uploaded. Enter or paste an MRI/RadFM caption manually.",
        )

    names = [getattr(u, "name", "upload") for u in uploads]
    image_like = [n for n in names if _suffix(n) in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
    nifti_like = [n for n in names if _is_nifti(n)]

    if try_radfm and radfm_available() and nifti_like:
        try:
            if str(RADFM_BACKEND) not in sys.path:
                sys.path.insert(0, str(RADFM_BACKEND.parent))
            from vlm_backends.radfm_backend import caption_volumes, load_model  # type: ignore

            with tempfile.TemporaryDirectory(prefix="sr_mri_") as tmp:
                saved = save_uploads(uploads, Path(tmp))
                missing = [m for m in ("t1c", "t1n", "t2f", "t2w") if m not in saved]
                if missing:
                    return CaptionResult(
                        caption="",
                        source="radfm-incomplete",
                        warning=(
                            "RadFM expects four NIfTI modalities (t1c, t1n, t2f, t2w). "
                            f"Missing: {', '.join(missing)}. "
                            "Upload all four or paste a caption manually."
                        ),
                        details={"saved": {k: str(v) for k, v in saved.items()}, "files": names},
                    )
                load_model()
                caption = caption_volumes(
                    t1c=str(saved["t1c"]),
                    t1n=str(saved["t1n"]),
                    t2f=str(saved["t2f"]),
                    t2w=str(saved["t2w"]),
                    clinical_context=clinical_context or None,
                )
            return CaptionResult(
                caption=str(caption).strip(),
                source="radfm",
                warning="",
                details={"files": names},
            )
        except Exception as exc:  # noqa: BLE001 - surface any RadFM runtime failure
            return CaptionResult(
                caption=_fallback_caption(names, clinical_context),
                source="fallback-after-radfm-error",
                warning=f"RadFM failed ({type(exc).__name__}: {exc}). Using editable fallback caption.",
                details={"files": names},
            )

    if nifti_like and not radfm_available():
        return CaptionResult(
            caption=_fallback_caption(names, clinical_context),
            source="fallback-no-radfm-weights",
            warning=(
                "RadFM weights were not found under second/VLM/RadFM_weights/. "
                "Uploaded NIfTI files are recorded; edit the fallback caption or run setup_radfm.sh."
            ),
            details={"files": names, "nifti": nifti_like},
        )

    if image_like:
        return CaptionResult(
            caption=_fallback_caption(names, clinical_context, preview_note=True),
            source="image-preview-fallback",
            warning=(
                "2D image upload received. Full RadFM captioning expects 3D NIfTI volumes "
                "(t1c/t1n/t2f/t2w). Edit the generated draft caption below."
            ),
            details={"files": names, "images": image_like},
        )

    return CaptionResult(
        caption=_fallback_caption(names, clinical_context),
        source="fallback-unknown-format",
        warning="Unrecognized upload format. Edit the draft caption manually.",
        details={"files": names},
    )


def _fallback_caption(names: list[str], clinical_context: str = "", preview_note: bool = False) -> str:
    file_list = ", ".join(names)
    bits = [
        "Pre-second-recurrence MRI study uploaded for review.",
        f"Files: {file_list}.",
    ]
    if preview_note:
        bits.append("2D preview only; not a validated radiology report.")
    bits.append(
        "Findings pending radiologist/VLM confirmation: describe residual enhancement, "
        "edema/FLAIR change, and whether disease appears stable or progressing."
    )
    if clinical_context.strip():
        bits.append(f"Clinical context: {clinical_context.strip()}")
    return " ".join(bits)
