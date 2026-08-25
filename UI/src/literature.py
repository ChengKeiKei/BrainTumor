from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


UI_ROOT = Path(__file__).resolve().parents[1]
RAG_ROOT = UI_ROOT.parent.parent
DEFAULT_CORPUS_CANDIDATES = [
    UI_ROOT / "data" / "pubmed_brain_tumor_recurrence_10000_complete_title_abstract.jsonl",
    RAG_ROOT / "Brain_KG" / "data" / "pubmed" / "pubmed_brain_tumor_recurrence_10000_complete_title_abstract.jsonl",
    RAG_ROOT / "BrainTumor" / "database" / "Retrieval" / "corpus" / "pubmed_brain_tumor_recurrence_10000_complete_title_abstract.jsonl",
]
CACHE_DIR = UI_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_CACHE = CACHE_DIR / "local_pubmed_tfidf.joblib"


@dataclass(frozen=True)
class LiteratureDoc:
    pmid: str
    title: str
    abstract: str
    source: str
    score: float = 0.0
    year: str = ""

    @property
    def citation(self) -> str:
        return f"PMID {self.pmid}" if self.pmid else self.source


def _clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def resolve_corpus_path() -> Path:
    for path in DEFAULT_CORPUS_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Local PubMed corpus not found. Place the JSONL under UI/data/ or restore "
        "Brain_KG/data/pubmed/pubmed_brain_tumor_recurrence_10000_complete_title_abstract.jsonl"
    )


def load_local_corpus(path: Path | None = None, limit: int | None = None) -> list[LiteratureDoc]:
    path = path or resolve_corpus_path()
    docs: list[LiteratureDoc] = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            title = _clean(rec.get("title"))
            abstract = _clean(rec.get("abstract"))
            if not abstract:
                continue
            docs.append(
                LiteratureDoc(
                    pmid=str(rec.get("pmid", "")),
                    title=title or "Untitled PubMed record",
                    abstract=abstract,
                    source="Local PubMed corpus",
                    year=str(rec.get("year", "")),
                )
            )
            if limit and len(docs) >= limit:
                break
    return docs


def _build_or_load_index(path: Path | None = None):
    path = path or resolve_corpus_path()
    meta = {"path": str(path), "mtime": path.stat().st_mtime}
    if INDEX_CACHE.exists():
        cached = joblib.load(INDEX_CACHE)
        if cached.get("meta") == meta:
            return cached["docs"], cached["vectorizer"], cached["matrix"]

    docs = load_local_corpus(path)
    texts = [f"{d.title}. {d.abstract}" for d in docs]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
    )
    matrix = vectorizer.fit_transform(texts)
    joblib.dump({"meta": meta, "docs": docs, "vectorizer": vectorizer, "matrix": matrix}, INDEX_CACHE)
    return docs, vectorizer, matrix


def retrieve_local_literature(query: str, top_k: int = 5) -> list[LiteratureDoc]:
    docs, vectorizer, matrix = _build_or_load_index()
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).ravel()
    top_idx = scores.argsort()[::-1][:top_k]
    out = []
    for idx in top_idx:
        d = docs[int(idx)]
        out.append(LiteratureDoc(d.pmid, d.title, d.abstract, d.source, float(scores[idx]), d.year))
    return out


def _pubmed_search_ids(query: str, retmax: int = 5) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "pub date",
    }
    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def _pubmed_fetch_details(pmids: Iterable[str]) -> list[LiteratureDoc]:
    ids = ",".join(pmids)
    if not ids:
        return []
    params = {"db": "pubmed", "id": ids, "retmode": "xml"}
    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    docs: list[LiteratureDoc] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _clean(article.findtext(".//PMID"))
        title = _clean(" ".join(article.findtext(".//ArticleTitle", default="").split()))
        abstract_parts = [node.text or "" for node in article.findall(".//AbstractText")]
        abstract = _clean(" ".join(abstract_parts))
        year = _clean(article.findtext(".//PubDate/Year"))
        if abstract:
            docs.append(LiteratureDoc(pmid, title or "Untitled PubMed record", abstract, "Live PubMed refresh", 1.0, year))
    return docs


def retrieve_live_pubmed(query: str, top_k: int = 5) -> tuple[list[LiteratureDoc], str]:
    try:
        pmids = _pubmed_search_ids(query, retmax=top_k)
        time.sleep(0.34)
        return _pubmed_fetch_details(pmids), ""
    except Exception as exc:
        return [], f"Live PubMed refresh failed: {type(exc).__name__}: {exc}"


def build_literature_context(docs: list[LiteratureDoc], max_chars_per_doc: int = 900) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        abstract = doc.abstract[:max_chars_per_doc]
        blocks.append(
            f"[{i}] {doc.title}\n"
            f"Source: {doc.source}; {doc.citation}; Year: {doc.year or 'NA'}; Score: {doc.score:.3f}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n".join(blocks)


def build_literature_context_beep(docs: list[LiteratureDoc], max_chars_per_doc: int = 500) -> str:
    """Compact BEEP-style literature block used in the training prompts."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        abstract = doc.abstract[:max_chars_per_doc].rstrip()
        blocks.append(f"[{i}] {doc.title} - {abstract}")
    return "\n".join(blocks) if blocks else "[none retrieved]"


def docs_to_frame(docs: list[LiteratureDoc]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": d.source,
                "pmid": d.pmid,
                "year": d.year,
                "score": round(d.score, 4),
                "title": d.title,
                "abstract_preview": d.abstract[:260] + ("..." if len(d.abstract) > 260 else ""),
            }
            for d in docs
        ]
    )
