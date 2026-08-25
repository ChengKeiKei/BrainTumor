"""RadFM (13 B) backend for the medical-VLM caption pipeline.

Loads RadFM lazily (the model is module-cached after the first call so we pay
the ~50 GB load cost exactly once) and turns one (patient × timepoint) MRI
study — four NIfTI volumes (t1c, t1n, t2f, t2w) — into a free-text radiology
finding string suitable for Mistral's RAG prompt builder.

Setup:
    bash setup_radfm.sh        # downloads code + 50 GB weights + tokenizer

Inference:
    from vlm_backends.radfm_backend import load_model, caption_volumes
    load_model()                                       # one-shot
    cap = caption_volumes(t1c=..., t1n=..., t2f=..., t2w=...)

The implementation mirrors `RadFM_repo/Quick_demo/test.py` but generalised to
multi-volume 3D MRI input and patched for Apple-MPS / CPU fallback.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------------
# Constants & paths
# -----------------------------------------------------------------------------
HERE          = Path(__file__).resolve().parent
VLM_ROOT      = HERE.parent
RADFM_REPO    = VLM_ROOT / "RadFM_repo"
WEIGHTS_DIR   = VLM_ROOT / "RadFM_weights"
TOKENIZER_DIR = WEIGHTS_DIR / "Language_files"
WEIGHTS_BIN   = WEIGHTS_DIR / "pytorch_model.bin"

# RadFM's vision frontend was trained on 512×512 slices.  For 3D, the official
# ViT pads/interpolates to 32 depth slices.  We respect those defaults.
TARGET_H = 512
TARGET_W = 512
TARGET_D = 32

# One MRI study = 4 modalities = 4 image positions in the prompt.
MODALITIES = ("t1c", "t1n", "t2f", "t2w")

# Module-level singletons (populated by load_model()).
_MODEL          = None
_TOKENIZER      = None
_PADDING_TOKENS = None
_DEVICE         = None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_model(device: Optional[str] = None) -> None:
    """Load RadFM weights once and cache them at module scope.

    Raises ImportError if RadFM is not yet set up — caller (the fallback
    ladder driver) treats that as "skip to the next VLM in the ladder"."""
    global _MODEL, _TOKENIZER, _PADDING_TOKENS, _DEVICE
    if _MODEL is not None:
        return

    # --- Pre-flight checks (fail fast, with actionable messages) -------------
    if not RADFM_REPO.exists():
        raise ImportError(
            "RadFM repo missing — run `bash setup_radfm.sh` first "
            f"(expected at {RADFM_REPO})"
        )
    if not WEIGHTS_BIN.exists():
        raise ImportError(
            "RadFM weights missing — run `bash setup_radfm.sh` first "
            f"(expected at {WEIGHTS_BIN}, ~50 GB)"
        )
    if not (TOKENIZER_DIR / "tokenizer.model").exists():
        raise ImportError(
            "LLaMA-13B tokenizer files missing — run `bash setup_radfm.sh` "
            f"first (expected at {TOKENIZER_DIR})"
        )

    # --- Imports that depend on the cloned repo + heavy ML libs --------------
    sys.path.insert(0, str(RADFM_REPO / "Quick_demo"))
    import torch  # noqa: E402
    from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig  # noqa: E402

    # ----- Hack: avoid a 26 GB download of the base LLaMA-13B weights ---------
    # RadFM's MultiLLaMAForCausalLM calls LlamaForCausalLM.from_pretrained(path)
    # which normally requires the *full* HF model directory (config + weights).
    # We only have the config + tokenizer in `Language_files`; the *actual*
    # LLaMA-13B weights are inside RadFM's pytorch_model.bin (RadFM ships them
    # there because it fine-tuned the LLM end-to-end).  So we monkey-patch
    # from_pretrained to do a config-only random init — every weight gets
    # overwritten 30 lines later by `model.load_state_dict(ckpt)` anyway.
    _orig_from_pretrained = LlamaForCausalLM.from_pretrained

    @classmethod
    def _config_only_init(cls, path, *args, **kwargs):
        cfg = LlamaConfig.from_pretrained(path)
        return cls(cfg)
    LlamaForCausalLM.from_pretrained = _config_only_init
    # --------------------------------------------------------------------------

    try:
        from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM  # noqa: E402
    except Exception as e:
        LlamaForCausalLM.from_pretrained = _orig_from_pretrained
        raise ImportError(
            f"Could not import RadFM model code from {RADFM_REPO}/Quick_demo: {e}"
        ) from e

    # --- Device selection (MPS on Apple silicon, else CUDA, else CPU) --------
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            print("[RadFM] WARNING: falling back to CPU — generation will be very slow")
    _DEVICE = torch.device(device)
    print(f"[RadFM] device = {_DEVICE}")

    # --- Tokenizer with image-placeholder special tokens ---------------------
    print(f"[RadFM] loading tokenizer from {TOKENIZER_DIR} ...")
    text_tokenizer, image_padding_tokens = _build_tokenizer(
        str(TOKENIZER_DIR), max_img_size=100, image_num=32, _LlamaTokenizer=LlamaTokenizer
    )

    # --- Model ---------------------------------------------------------------
    print(f"[RadFM] building model skeleton from config (no base-LLM download) ...")
    model = MultiLLaMAForCausalLM(lang_model_path=str(TOKENIZER_DIR))
    # Restore the unpatched from_pretrained so we don't surprise downstream callers.
    LlamaForCausalLM.from_pretrained = _orig_from_pretrained

    print(f"[RadFM] loading 13B weights from {WEIGHTS_BIN} (~50 GB; takes a few minutes) ...")
    ckpt = torch.load(WEIGHTS_BIN, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"[RadFM] state-dict load: {len(missing)} missing, "
              f"{len(unexpected)} unexpected (expected — RadFM ships extra heads).")
    # MPS does not support fp16 for every op in the LLaMA stack; keep fp32 there.
    if _DEVICE.type == "cuda":
        model = model.half()
    model = model.to(_DEVICE).eval()
    print("[RadFM] model ready.")

    _MODEL          = model
    _TOKENIZER      = text_tokenizer
    _PADDING_TOKENS = image_padding_tokens


def caption_volumes(t1c: str, t1n: str, t2f: str, t2w: str,
                    max_new_tokens: int = 256,
                    clinical_context: str = "",
                    prompt_version: str = "v1") -> str:
    """Generate a free-text radiology finding for one MRI study.

    Parameters
    ----------
    t1c, t1n, t2f, t2w : str
        Paths to the four NIfTI volumes for one scan.
    max_new_tokens : int
        Generation budget.
    clinical_context : str
        Optional patient-context block (e.g. demographics + Dx + Tx history)
        that gets prepended to anchor RadFM in the right clinical scenario.
        Used when `prompt_version` is "v2_context" or "v3_structured" (both
        benefit from the anatomy / Dx anchor; without it RadFM still drifts
        toward extra-axial / meningioma labels on glioma scans).
    prompt_version : {"v1", "v2_context", "v3_structured"}
        - "v1" : original prompt (no patient context). Kept for backward
                 compatibility with the cached `mri_captions.csv`.
        - "v2_context" : anatomy-grounded free-text prompt that prepends the
                 patient's clinical context before the imaging question.
                 Reduces RadFM hallucinations such as "meningioma" /
                 "extra-axial" labels in our post-treatment glioma cohort.
        - "v3_structured" : same patient anchor as v2, but the question is a
                 fixed 7-item YES/NO/UNCLEAR checklist. The raw RadFM output
                 is post-processed by `parse_structured_caption()` into a
                 deterministic feature record so BioMistral receives an
                 auditable structured block instead of a paragraph. Bypasses
                 the v1 weak points (lobe accuracy 23 %, necrosis recall 12 %)
                 by only asking concepts where the v1 audit showed RadFM is
                 informative (enhancement / edema / mass effect / hemorrhage).
    """
    if _MODEL is None:
        load_model()

    import torch  # noqa: E402

    if prompt_version == "v3_structured":
        ctx = clinical_context.strip()
        anchor = (
            f"Background: {ctx} " if ctx else ""
        ) + (
            "This is a follow-up brain MRI of a confirmed glioblastoma patient. "
            "The four sequences provided are post-contrast T1 (T1c), pre-contrast "
            "T1 (T1n), T2 FLAIR (T2f), and T2-weighted (T2w). "
        )
        question = anchor + (
            "Answer EACH of the following questions with one line in the format "
            "'<n>. <YES|NO|UNCLEAR> — <one short reason>'. Do not add any other "
            "text. (1) Is contrast enhancement present? (2) Is intratumoral "
            "necrosis visible? (3) Is intratumoral hemorrhage present? "
            "(4) Is peritumoral edema present? (5) Is there mass effect or "
            "midline shift? (6) Are there multifocal enhancing lesions? "
            "(7) Is the enhancing lesion larger than expected for a typical "
            "post-treatment baseline?"
        )
    elif prompt_version == "v2_context":
        # Anatomy-grounded prompt — keep the imaging question short and
        # in RadFM's familiar Q&A style (the over-constrained v2-draft
        # version made RadFM echo the instructions instead of describing
        # the image). Patient context goes BEFORE the question so RadFM
        # treats it as background. Diagnosis lock is folded into one
        # short sentence right before the question, not as a list of
        # forbidden labels (RadFM was trained on factual radiology Q&A,
        # not on negative-instruction following).
        ctx = clinical_context.strip()
        if ctx:
            question = (
                f"Background: {ctx} "
                "This is a follow-up brain MRI of the same patient. "
                "The four sequences provided are post-contrast T1 (T1c), "
                "pre-contrast T1 (T1n), T2 FLAIR (T2f), and T2-weighted (T2w). "
                "Describe the post-treatment imaging findings — including "
                "tumor lobe location, size, contrast enhancement pattern, "
                "necrotic core, peritumoral edema, mass effect, and any "
                "evidence of progression."
            )
        else:
            # User asked for v2 but supplied no context — fall back to v1
            # so we never produce a worse prompt than the cache had.
            prompt_version = "v1"

    if prompt_version == "v1":
        question = (
            "You are reading a brain MRI of a glioma patient. "
            "The four sequences provided are post-contrast T1 (T1c), pre-contrast "
            "T1 (T1n), T2 FLAIR (T2f), and T2-weighted (T2w). "
            "Describe the tumor's appearance, location, size, edema, mass effect, "
            "enhancement pattern, and any concerning features for tumor progression."
        )

    # v3 generates short structured Q&A; give it more headroom than free-text v1/v2.
    if prompt_version == "v3_structured":
        max_new_tokens = max(max_new_tokens, 384)

    image_list = []
    for pos, (path, modality) in enumerate(zip(
        (t1c, t1n, t2f, t2w), MODALITIES
    )):
        image_list.append({"img_path": path, "position": pos, "modality": modality})

    text, vision_x = _build_inputs(question, image_list)
    vision_x = vision_x.to(_DEVICE)
    if _DEVICE.type == "cuda":
        vision_x = vision_x.half()

    with torch.no_grad():
        lang_x = _TOKENIZER(
            text, max_length=2048, truncation=True, return_tensors="pt"
        )["input_ids"].to(_DEVICE)

        gen = _MODEL.generate(lang_x, vision_x)
        out = _TOKENIZER.batch_decode(gen, skip_special_tokens=True)[0]

    # Strip the echoed question if RadFM repeats it.
    out = out.strip()
    if out.lower().startswith(question[:30].lower()):
        out = out[len(question):].lstrip(" :\n")

    # For v3, post-process the 7-item Q&A into a deterministic block.
    if prompt_version == "v3_structured":
        out = format_structured_block(parse_structured_caption(out))
    return out


# ---------------------------------------------------------------------------
# v3_structured parser & formatter
# ---------------------------------------------------------------------------
# The 7 concepts asked of RadFM, in order. Keys are the canonical field names
# stored in the structured caption block (and consumed by feature_render's
# VLM block reader).
V3_CONCEPTS = [
    "enhancement",
    "necrosis",
    "hemorrhage",
    "edema",
    "mass_effect",
    "multifocal",
    "size_vs_baseline",
]
V3_ALLOWED  = {"yes", "no", "unclear"}


def parse_structured_caption(raw: str) -> dict:
    """Extract the 7 YES/NO/UNCLEAR answers from a v3_structured RadFM output.

    RadFM is asked to return one line per concept in the form
        "1. YES — <short reason>"
    but small models drift, so we parse defensively:
      * try to match "<index>. <YES|NO|UNCLEAR>" first;
      * fall back to scanning for the keywords near the concept's index;
      * default to "unclear" if nothing matches.

    Returns
    -------
    dict[str, str]
        {concept_name: "yes"|"no"|"unclear"} for each of the 7 V3_CONCEPTS.
        Always returns all 7 keys (so downstream prompts have a stable schema).
    """
    import re

    out = {c: "unclear" for c in V3_CONCEPTS}
    if not raw:
        return out

    text = raw.strip()
    # Primary pattern: "1. YES — ..." or "(1) NO ...".
    pat = re.compile(
        r"\(?\s*(\d+)\s*[).:\-]\s*(YES|NO|UNCLEAR|YES\.|NO\.|UNCLEAR\.)",
        re.IGNORECASE,
    )
    matches = pat.findall(text)
    seen = set()
    for idx_str, ans in matches:
        try:
            idx = int(idx_str) - 1
        except ValueError:
            continue
        if 0 <= idx < len(V3_CONCEPTS) and idx not in seen:
            ans_clean = ans.strip(".").lower()
            if ans_clean in V3_ALLOWED:
                out[V3_CONCEPTS[idx]] = ans_clean
                seen.add(idx)

    # Fallback: if RadFM returned an unstructured paragraph but mentions the
    # keywords, do a simple presence/negation sweep concept-by-concept.
    if len(seen) < len(V3_CONCEPTS):
        keyword_map = {
            "enhancement":      ("enhancement", "enhancing"),
            "necrosis":         ("necrosis", "necrotic"),
            "hemorrhage":       ("hemorrhage", "haemorrhage", "blood product"),
            "edema":            ("edema", "oedema"),
            "mass_effect":      ("mass effect", "midline shift"),
            "multifocal":       ("multifocal", "multiple lesions", "satellite lesion"),
            "size_vs_baseline": ("larger", "increased size", "interval growth",
                                 "smaller", "decreased size"),
        }
        lower = text.lower()
        # Split into sentence-like chunks so a "no <other>" doesn't bleed into
        # the next concept's window (saw this on smoke test: "no hemorrhage.
        # Mild mass effect" was wrongly flipping mass effect → no).
        import re as _re
        sentences = [s.strip() for s in _re.split(r"[.;\n]+", lower) if s.strip()]
        for i, concept in enumerate(V3_CONCEPTS):
            if i in seen:
                continue
            kws = keyword_map[concept]
            hit_sentence = next(
                (s for s in sentences if any(kw in s for kw in kws)),
                None,
            )
            if hit_sentence is None:
                continue
            negated = any(neg in hit_sentence for neg in
                          ("no ", "not ", "without ", "absent",
                           "no evidence", "without evidence"))
            out[concept] = "no" if negated else "yes"
    return out


def format_structured_block(parsed: dict) -> str:
    """Render the parsed structured caption as a compact, LLM-friendly block.

    The string format is what gets stored in the `caption` column of
    `mri_captions_v3_structured.csv` and what BioMistral sees in the VLM
    block of the RAG prompt.
    """
    pretty = {
        "enhancement":      "contrast enhancement",
        "necrosis":         "necrotic core",
        "hemorrhage":       "intratumoral hemorrhage",
        "edema":            "peritumoral edema",
        "mass_effect":      "mass effect / midline shift",
        "multifocal":       "multifocal lesions",
        "size_vs_baseline": "larger than expected baseline",
    }
    parts = [f"{pretty[c]}: {parsed.get(c, 'unclear')}" for c in V3_CONCEPTS]
    return "Structured imaging findings — " + "; ".join(parts) + "."


# -----------------------------------------------------------------------------
# Helpers (kept private — internal RadFM glue)
# -----------------------------------------------------------------------------
def _build_tokenizer(tokenizer_path: str, max_img_size: int, image_num: int,
                     _LlamaTokenizer):
    """Replica of `Quick_demo/test.py::get_tokenizer`, with the upstream's
    string handling cleaned up (the original lost the angle brackets when the
    file was rendered as Markdown)."""
    text_tokenizer = _LlamaTokenizer.from_pretrained(tokenizer_path)
    special_tokens = ["<image>", "</image>"]
    image_padding_tokens = []

    for i in range(max_img_size):
        pads = ""
        for j in range(image_num):
            tok = f"<image{i}_{j}>"
            pads += tok
            special_tokens.append(tok)
        image_padding_tokens.append(pads)

    text_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    text_tokenizer.pad_token_id = 0
    text_tokenizer.bos_token_id = 1
    text_tokenizer.eos_token_id = 2
    return text_tokenizer, image_padding_tokens


def _build_inputs(question: str, image_list):
    """Convert one (question, [4 NIfTI paths]) pair to (text, vision_x).

    Returns
    -------
    text : str
        The question with image placeholders substituted at the requested
        char positions.
    vision_x : torch.Tensor
        Shape  (1, S, 3, TARGET_H, TARGET_W, TARGET_D)  where S = number of
        modalities we attached (4 here).  RGB is fake-replicated from the
        single grayscale MRI channel — RadFM's ViT was trained on RGB inputs
        and the redundancy is harmless.
    """
    import torch
    import torch.nn.functional as F
    import nibabel as nib
    import numpy as np

    chars = list(question)
    volumes = []
    for pad_idx, img in enumerate(image_list):
        vol = _load_nifti_to_tensor(img["img_path"])           # (1,1,H,W,D) fp32
        vol = F.interpolate(vol, size=(TARGET_H, TARGET_W, TARGET_D),
                            mode="trilinear", align_corners=False)
        vol = vol.expand(1, 3, -1, -1, -1).contiguous()         # fake-RGB
        volumes.append(vol)

        # Insert the image placeholder string at the requested char position.
        pos = img["position"]
        chars[pos] = "<image>" + _PADDING_TOKENS[pad_idx] + "</image>" + chars[pos]

    vision_x = torch.stack(volumes, dim=1)                     # (1, S, 3, H, W, D)
    text = "".join(chars)
    return text, vision_x


def _load_nifti_to_tensor(path):
    """Load a NIfTI volume and rescale to [0, 1].  Returns shape (1,1,H,W,D)."""
    import nibabel as nib
    import numpy as np
    import torch

    arr = nib.load(path).get_fdata().astype(np.float32)        # (H,W,D)
    lo, hi = np.percentile(arr, [1, 99])                       # robust window
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)        # (1,1,H,W,D)
    return t
