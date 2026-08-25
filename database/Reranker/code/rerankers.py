"""
rerankers.py — First-Progression reranker zoo (five rerankers, one interface).

Every class exposes the same method:

    rerank(query: str, documents: list[dict], top_k: int) -> list[dict]

where each `document` dict has at least `title` and `abstract`.
The returned list is a subset of `documents` sorted in descending relevance.

Lineup (must match Reranker/README.md — paper narrative):

  1. BeepPubMedBERTReranker  — **primary**, BEEP paper's official cross-encoder.
     PubMedBERT-base fine-tuned on TREC PM 2016 (Naik et al., Findings-NAACL 2022).
     Loads `First_Recur/BEEP/checkpoints/crossencoder-reranker.pt`.

  2. MiniLMReranker          — `cross-encoder/ms-marco-MiniLM-L-6-v2`.
     Fast general-domain baseline. See README for WoS/Scopus citation.

  3. MedCPTReranker          — `ncbi/MedCPT-Cross-Encoder`.
     Biomedical cross-encoder trained on PubMed user-click data.

  4. ColBERTv2Reranker       — `colbert-ir/colbertv2.0` via ragatouille.
     Late-interaction retriever run in rerank-only mode.

  5. BGEM3Reranker           — `BAAI/bge-reranker-v2-m3`.
     Strong general-domain multilingual reranker.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _concat_text(doc: dict, max_chars: int | None = None) -> str:
    text = (doc.get("title", "") + " " + doc.get("abstract", "")).strip()
    return text[:max_chars] if max_chars else text


def _install_ragatouille_optional_dependency_shims() -> None:
    """Provide tiny optional-dependency shims for RAGatouille rerank-only use.

    RAGatouille imports LangChain and LlamaIndex wrappers at module import
    time, even when we only need `RAGPretrainedModel.from_pretrained(...).rerank`.
    On Python 3.13/macOS, the full RAGatouille dependency set currently pulls
    `voyager`, which has no matching wheel in this environment. These shims
    satisfy the unused wrapper imports without changing ColBERT scoring.
    """
    if "langchain.retrievers.document_compressors.base" not in sys.modules:
        langchain = types.ModuleType("langchain")
        retrievers = types.ModuleType("langchain.retrievers")
        compressors = types.ModuleType("langchain.retrievers.document_compressors")
        base = types.ModuleType("langchain.retrievers.document_compressors.base")

        class BaseDocumentCompressor:  # pragma: no cover - import shim only
            pass

        base.BaseDocumentCompressor = BaseDocumentCompressor
        sys.modules.setdefault("langchain", langchain)
        sys.modules.setdefault("langchain.retrievers", retrievers)
        sys.modules.setdefault("langchain.retrievers.document_compressors", compressors)
        sys.modules.setdefault("langchain.retrievers.document_compressors.base", base)

    if "langchain_core.retrievers" not in sys.modules:
        langchain_core = types.ModuleType("langchain_core")
        core_retrievers = types.ModuleType("langchain_core.retrievers")
        callbacks = types.ModuleType("langchain_core.callbacks")
        callback_manager = types.ModuleType("langchain_core.callbacks.manager")
        documents = types.ModuleType("langchain_core.documents")

        class BaseRetriever:  # pragma: no cover - import shim only
            pass

        class CallbackManagerForRetrieverRun:  # pragma: no cover
            pass

        class Document:  # pragma: no cover
            def __init__(self, page_content: str, metadata: dict | None = None):
                self.page_content = page_content
                self.metadata = metadata or {}

        core_retrievers.BaseRetriever = BaseRetriever
        callback_manager.CallbackManagerForRetrieverRun = CallbackManagerForRetrieverRun
        callback_manager.Callbacks = object
        documents.Document = Document
        sys.modules.setdefault("langchain_core", langchain_core)
        sys.modules.setdefault("langchain_core.retrievers", core_retrievers)
        sys.modules.setdefault("langchain_core.callbacks", callbacks)
        sys.modules.setdefault("langchain_core.callbacks.manager", callback_manager)
        sys.modules.setdefault("langchain_core.documents", documents)

    if "llama_index" not in sys.modules and "llama_index.core" not in sys.modules:
        llama_index = types.ModuleType("llama_index")
        text_splitter = types.ModuleType("llama_index.text_splitter")
        core = types.ModuleType("llama_index.core")
        core_text_splitter = types.ModuleType("llama_index.core.text_splitter")

        class Document:  # pragma: no cover
            def __init__(self, text: str = "", **kwargs):
                self.text = text
                self.kwargs = kwargs

        class SentenceSplitter:  # pragma: no cover
            def __init__(self, chunk_size: int = 256, **kwargs):
                self.chunk_size = chunk_size

            def split_text(self, text: str):
                return [text]

            def get_nodes_from_documents(self, documents):
                return documents

        llama_index.Document = Document
        text_splitter.SentenceSplitter = SentenceSplitter
        core.Document = Document
        core_text_splitter.SentenceSplitter = SentenceSplitter
        sys.modules.setdefault("llama_index", llama_index)
        sys.modules.setdefault("llama_index.text_splitter", text_splitter)
        sys.modules.setdefault("llama_index.core", core)
        sys.modules.setdefault("llama_index.core.text_splitter", core_text_splitter)

    if "voyager" not in sys.modules:
        voyager = types.ModuleType("voyager")

        class Index:  # pragma: no cover - training/indexing shim only
            def __init__(self, *args, **kwargs):
                raise RuntimeError("voyager shim is only for ColBERT rerank-only imports")

        class Space:  # pragma: no cover
            Cosine = "cosine"
            InnerProduct = "inner_product"

        class StorageDataType:  # pragma: no cover
            Float32 = "float32"

        voyager.Index = Index
        voyager.Space = Space
        voyager.StorageDataType = StorageDataType
        sys.modules.setdefault("voyager", voyager)


# ---------------------------------------------------------------------------
# 1. BEEP PubMedBERT cross-encoder — primary
# ---------------------------------------------------------------------------

class BeepPubMedBERTReranker:
    """BEEP paper's official reranker.

    Naik et al., "Literature-Augmented Clinical Outcome Prediction",
    Findings of NAACL 2022. https://aclanthology.org/2022.findings-naacl.33/

    Implementation faithful to the BEEP training recipe:
    * Backbone: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
    * Adds `[ENTSEP]` special token and resizes embeddings before loading
      the checkpoint — matches the state-dict saved in the paper release
      (vocab size 30,523).
    * Two-class classifier head (Relevant vs Irrelevant). Relevance score
      is `softmax(logits)[:, 1]`.
    * Query is prefixed with an outcome question — the BEEP paper uses
      e.g. "What is the hospital mortality?". We use the recurrence
      variant consistent with `Retrieval/code/dense_retrieval_beep.py`.
    """

    DEFAULT_BACKBONE = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    DEFAULT_OUTCOME_QUESTION = "What is the probability of brain tumor recurrence? "
    DEFAULT_CHECKPOINT = str(
        Path(__file__).resolve().parents[2]
        / "BEEP" / "checkpoints" / "crossencoder-reranker.pt"
    )

    def __init__(
        self,
        checkpoint_path: str | None = None,
        backbone: str = DEFAULT_BACKBONE,
        outcome_question: str = DEFAULT_OUTCOME_QUESTION,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        self.device = device or _auto_device()
        self.outcome_question = outcome_question
        self.batch_size = batch_size or (8 if self.device == "mps" else 16)

        checkpoint_path = checkpoint_path or os.environ.get(
            "BEEP_RERANKER_CHECKPOINT", self.DEFAULT_CHECKPOINT
        )
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"BEEP reranker checkpoint not found at {checkpoint_path}.\n"
                "Copy it from BEEP-main/models/retrieval-models/, or download from:\n"
                "  https://ai2-s2-beep.s3.amazonaws.com/retrieval-models/crossencoder-reranker.pt"
            )
        self.checkpoint_path = checkpoint_path

        print(f"[beep-reranker] loading on {self.device} from {checkpoint_path}")
        label_vocab = {"Relevant": 1, "Irrelevant": 0}
        config = AutoConfig.from_pretrained(
            backbone,
            num_labels=len(label_vocab),
            label2id=label_vocab,
            id2label={i: l for l, i in label_vocab.items()},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            backbone, config=config
        )
        # BEEP adds a single [ENTSEP] token — must be present before loading
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[ENTSEP]"]}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[beep-reranker] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[beep-reranker] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

        self.model.to(self.device)
        self.model.eval()
        self._softmax = torch.nn.Softmax(dim=1)

    def _format_query(self, query: str) -> str:
        if not self.outcome_question or query.startswith(self.outcome_question):
            return query
        return self.outcome_question + query

    @torch.no_grad()
    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        q = self._format_query(query)
        pairs = [[q, _concat_text(d)] for d in documents]
        score_chunks = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            features = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**features).logits
            probs = self._softmax(logits)
            score_chunks.append(probs[:, 1].detach().cpu())
        scores = torch.cat(score_chunks, dim=0).numpy().reshape(-1)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 2. MiniLM cross-encoder — fast general-domain baseline
# ---------------------------------------------------------------------------

class MiniLMReranker:
    """cross-encoder/ms-marco-MiniLM-L-6-v2 — distilled MS MARCO cross-encoder.

    WoS/Scopus-indexed representative usage (within 5 years): see
    `Reranker/README.md` for the full citation.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        self.device = device or _auto_device()
        print(f"[minilm] loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        features = self.tokenizer(
            [[query, _concat_text(d)] for d in documents],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        scores = self.model(**features).logits.squeeze(-1).detach().cpu().numpy()
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 3. MedCPT — biomedical cross-encoder
# ---------------------------------------------------------------------------

class MedCPTReranker:
    """ncbi/MedCPT-Cross-Encoder — Jin et al., Bioinformatics 2023.

    Biomedical cross-encoder trained on 255M query-article pairs from
    PubMed user click logs. Fixed 512-token input, single relevance logit.
    """

    def __init__(
        self,
        model_name: str = "ncbi/MedCPT-Cross-Encoder",
        device: str | None = None,
    ):
        self.device = device or _auto_device()
        print(f"[medcpt] loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        features = self.tokenizer(
            [[query, _concat_text(d)] for d in documents],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        scores = self.model(**features).logits.squeeze(-1).detach().cpu().numpy()
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 4. ColBERTv2 — late-interaction reranker
# ---------------------------------------------------------------------------

class ColBERTv2Reranker:
    """colbert-ir/colbertv2.0 via ragatouille — late-interaction scoring.

    Install:  pip install ragatouille
    Santhanam et al., NAACL 2022 (arXiv:2112.01488).

    Notes on `query_maxlen`
    -----------------------
    Stanford ColBERTv2's default `query_maxlen` is 32 tokens, optimised for
    web-style MS-MARCO queries. Our production queries are full EHR
    narratives (Demographics + Diagnosis + Molecular + Treatment) that can
    easily exceed 400 tokens for Exp4. The 32-token default would silently
    truncate every query to roughly the demographics block, dropping all
    molecular and treatment context before MaxSim is computed. We override
    it to 256 — within ColBERTv2's safe operating range (the model is
    architecturally agnostic to the limit; only the [MASK] padding is
    extended) and consistent with the long-query recipes used by ColBERT
    follow-ups such as ColBERT-QA. See:
        https://github.com/stanford-futuredata/ColBERT/issues/166
    """

    MAX_CHARS = 1500           # ~250 words ≈ full PubMed abstract
    QUERY_MAXLEN = 256         # vs Stanford default of 32 — see docstring

    def __init__(self, query_maxlen: int | None = None):
        try:
            _install_ragatouille_optional_dependency_shims()
            from ragatouille import RAGPretrainedModel
        except ImportError as err:
            raise ImportError(
                "ragatouille not installed.\n"
                "Run: pip install ragatouille"
            ) from err
        try:
            import colbert.modeling.hf_colbert as hf_colbert
            original_class_factory = hf_colbert.class_factory

            def compat_class_factory(name_or_path):
                cls = original_class_factory(name_or_path)
                if not hasattr(cls, "all_tied_weights_keys"):
                    cls.all_tied_weights_keys = {}
                return cls

            hf_colbert.class_factory = compat_class_factory
            import colbert.modeling.base_colbert as base_colbert
            base_colbert.class_factory = compat_class_factory
        except Exception as err:  # pylint: disable=broad-except
            print(f"[colbert] WARNING: could not patch HF_ColBERT compatibility: {err}")
        print("[colbert] loading colbert-ir/colbertv2.0")
        self.model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

        qmax = query_maxlen if query_maxlen is not None else self.QUERY_MAXLEN
        self._set_query_maxlen(qmax)
        print(f"[colbert] query_maxlen set to {qmax} (Stanford default = 32)")

    def _set_query_maxlen(self, qmax: int) -> None:
        """Reach into the underlying Stanford ColBERT model to bump query_maxlen.

        RAGatouille 0.0.x does not expose this through `rerank()`, so we
        patch the inner Checkpoint + ColBERTConfig directly. Defensively
        no-op if the internal attribute layout changes in a future version.
        """
        try:
            inner = self.model.model
            if hasattr(inner, "config"):
                inner.config.query_maxlen = qmax
            ckpt = getattr(inner, "checkpoint", None) or getattr(inner, "inference_ckpt", None)
            if ckpt is not None:
                if hasattr(ckpt, "query_tokenizer"):
                    ckpt.query_tokenizer.query_maxlen = qmax
                if hasattr(ckpt, "config"):
                    ckpt.config.query_maxlen = qmax
        except Exception as err:  # pylint: disable=broad-except
            print(f"[colbert] WARNING: could not set query_maxlen={qmax}: {err}")

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        # Deduplicate passages — ColBERT degrades on duplicate strings.
        seen = set()
        unique_docs: list[dict] = []
        for doc in documents:
            text = _concat_text(doc, self.MAX_CHARS)
            if text not in seen:
                seen.add(text)
                unique_docs.append(doc)
        passages = [_concat_text(d, self.MAX_CHARS) for d in unique_docs]
        results = self.model.rerank(query=query, documents=passages, k=top_k)
        passage_to_doc = {p: d for p, d in zip(passages, unique_docs)}
        out: list[dict] = []
        for r in results:
            matched = passage_to_doc.get(r["content"])
            if matched is None:
                # Fallback on prefix match
                for p, d in passage_to_doc.items():
                    if p[:100] == r["content"][:100]:
                        matched = d
                        break
            if matched is not None:
                out.append(matched)
        return out[:top_k]


# ---------------------------------------------------------------------------
# 5. BGE-M3 — strong general-domain multilingual reranker
# ---------------------------------------------------------------------------

class BGEM3Reranker:
    """BAAI/bge-reranker-v2-m3 — Chen et al., arXiv:2402.03216 (2024).

    Strong multilingual cross-encoder. Single-logit output.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ):
        self.device = device or _auto_device()
        print(f"[bge-m3] loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        features = self.tokenizer(
            [[query, _concat_text(d)] for d in documents],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**features).logits
        if logits.ndim == 2 and logits.shape[-1] > 1:
            scores = logits[:, -1]
        else:
            scores = logits.squeeze(-1)
        scores = scores.detach().cpu().numpy().reshape(-1)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

RERANKER_REGISTRY = {
    "beep":    BeepPubMedBERTReranker,   # primary, faithful to paper
    "minilm":  MiniLMReranker,
    "medcpt":  MedCPTReranker,
    "colbert": ColBERTv2Reranker,
    "bge_m3":  BGEM3Reranker,
}


def get_reranker(name: str = "beep", **kwargs):
    """Construct a reranker by short name.

    Parameters
    ----------
    name : one of "beep" | "minilm" | "medcpt" | "colbert" | "bge_m3"
    **kwargs : forwarded to the underlying class constructor.
    """
    name = name.lower()
    if name not in RERANKER_REGISTRY:
        raise ValueError(
            f"Unknown reranker '{name}'. "
            f"Valid choices: {sorted(RERANKER_REGISTRY)}"
        )
    return RERANKER_REGISTRY[name](**kwargs)


__all__ = [
    "BeepPubMedBERTReranker",
    "MiniLMReranker",
    "MedCPTReranker",
    "ColBERTv2Reranker",
    "BGEM3Reranker",
    "RERANKER_REGISTRY",
    "get_reranker",
]
