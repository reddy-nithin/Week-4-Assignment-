"""
openfda_rag.py
API-first helpers for building a lightweight RAG pipeline over openFDA drug labels.
"""

import json
import os
import re
import html
import time
import pickle
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import faiss
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_MAX_LIMIT = 1000


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    field: str
    text: str


@dataclass
class SubChunk:
    chunk_id: str
    doc_id: str
    field: str
    text: str


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_field_value(value: Any) -> str:
    if isinstance(value, list):
        parts = [v for v in value if isinstance(v, str) and v.strip()]
        value = " ".join(parts)
    elif isinstance(value, str):
        value = value
    else:
        return ""
    return clean_text(value)


def pick_text_fields(
    record: Dict[str, Any],
    field_allowlist: Optional[List[str]],
    field_blocklist: Iterable[str],
    include_table_fields: bool,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in record.items():
        if field_allowlist and key not in field_allowlist:
            continue
        if key in field_blocklist:
            continue
        if (not include_table_fields) and key.endswith("_table"):
            continue
        text = normalize_field_value(value)
        if text:
            out[key] = text
    return out


def derive_doc_id(record: Dict[str, Any], idx: int) -> str:
    for key in ["id", "set_id"]:
        v = record.get(key)
        if v:
            return str(v)
    openfda = record.get("openfda") or {}
    for key in ["spl_id", "spl_set_id", "brand_name", "product_ndc", "application_number"]:
        v = openfda.get(key)
        if isinstance(v, list) and v:
            return str(v[0])
        if isinstance(v, str) and v:
            return v
    return f"record_{idx+1}"


def fixed_size_chunk(text: str, words_per_chunk: int, overlap: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text)]


def build_openfda_query(
    prompt: str, fields: Optional[Iterable[str]] = None, max_terms: int = 8
) -> str:
    """
    Convert a free-text prompt into an openFDA `search=field:term` query.
    """
    terms = [t for t in tokenize(prompt) if len(t) > 2][:max_terms]
    if not terms:
        return "_exists_:openfda"

    if not fields:
        return " ".join(terms)

    field_list = list(fields)
    groups = []
    for term in terms:
        group = " OR ".join(f"{field}:{term}" for field in field_list)
        groups.append(f"({group})")
    return " AND ".join(groups)


def _openfda_request(
    base_url: str, params: Dict[str, Any], timeout_s: int = 30
) -> Dict[str, Any]:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{base_url}?{query}" if query else base_url
    req = urllib.request.Request(url, headers={"User-Agent": "Week4-Assignment-RAG/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"openFDA HTTP error {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"openFDA request failed: {e.reason}") from e

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise RuntimeError("openFDA response was not valid JSON") from e

    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(f"openFDA API error: {msg}")

    return data


def fetch_openfda_records(
    search: str,
    api_key: Optional[str] = None,
    base_url: str = OPENFDA_BASE_URL,
    limit: int = 100,
    skip: int = 0,
    sort: Optional[str] = None,
    timeout_s: int = 30,
) -> List[Dict[str, Any]]:
    limit = min(max(1, limit), OPENFDA_MAX_LIMIT)
    params = {"search": search, "limit": limit, "skip": skip, "sort": sort}
    if api_key:
        params["api_key"] = api_key
    data = _openfda_request(base_url, params, timeout_s=timeout_s)
    return data.get("results", []) if isinstance(data, dict) else []


def iter_openfda_records(
    search: str,
    api_key: Optional[str] = None,
    base_url: str = OPENFDA_BASE_URL,
    limit: int = 100,
    max_records: Optional[int] = None,
    sort: Optional[str] = None,
    pause_s: float = 0.0,
    timeout_s: int = 30,
) -> Iterable[Dict[str, Any]]:
    limit = min(max(1, limit), OPENFDA_MAX_LIMIT)
    fetched = 0
    skip = 0

    while True:
        if max_records is not None and fetched >= max_records:
            return

        batch_limit = limit
        if max_records is not None:
            batch_limit = min(limit, max_records - fetched)
            if batch_limit <= 0:
                return

        results = fetch_openfda_records(
            search=search,
            api_key=api_key,
            base_url=base_url,
            limit=batch_limit,
            skip=skip,
            sort=sort,
            timeout_s=timeout_s,
        )
        if not results:
            return

        for rec in results:
            yield rec
            fetched += 1
            if max_records is not None and fetched >= max_records:
                return

        skip += len(results)
        if len(results) < batch_limit:
            return
        if pause_s:
            time.sleep(pause_s)


def _write_jsonl(path: str, items: List[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            if hasattr(it, "__dict__"):
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _read_jsonl_chunks(path: str, cls):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(cls(**obj))
    return out


def _build_faiss_ip(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index


def build_artifacts(
    api_search: str,
    output_dir: str = "preprocessed",
    field_allowlist: Optional[List[str]] = None,
    field_blocklist: Optional[Iterable[str]] = None,
    include_table_fields: bool = False,
    min_chars: int = 40,
    words_per_chunk: int = 250,
    overlap: int = 40,
    use_st: bool = True,
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    save: bool = True,
    save_vectorizer: bool = True,
    verbose: bool = True,
    api_key: Optional[str] = None,
    api_base_url: str = OPENFDA_BASE_URL,
    api_limit: int = 200,
    api_max_records: Optional[int] = None,
    api_sort: Optional[str] = None,
    api_pause_s: float = 0.0,
    api_timeout_s: int = 30,
) -> Dict[str, Any]:
    if not api_search:
        raise ValueError("api_search is required for openFDA ingestion.")

    blocklist = set(field_blocklist or [])
    record_chunks: List[TextChunk] = []
    records_count = 0

    for rec in iter_openfda_records(
        search=api_search,
        api_key=api_key,
        base_url=api_base_url,
        limit=api_limit,
        max_records=api_max_records,
        sort=api_sort,
        pause_s=api_pause_s,
        timeout_s=api_timeout_s,
    ):
        doc_id = derive_doc_id(rec, records_count)
        for field, text in pick_text_fields(
            rec, field_allowlist, blocklist, include_table_fields
        ).items():
            if len(text) < min_chars:
                continue
            chunk_id = f"{doc_id}::{field}"
            record_chunks.append(TextChunk(chunk_id, doc_id, field, text))
        records_count += 1

    sub_chunks: List[SubChunk] = []
    for rc in record_chunks:
        for j, t in enumerate(fixed_size_chunk(rc.text, words_per_chunk, overlap)):
            sub_chunks.append(SubChunk(f"{rc.chunk_id}::c{j+1}", rc.doc_id, rc.field, t))

    texts_A = [c.text for c in record_chunks]
    texts_B = [c.text for c in sub_chunks]

    tokens_A = [tokenize(t) for t in texts_A] if texts_A else []
    tokens_B = [tokenize(t) for t in texts_B] if texts_B else []

    bm25_A = BM25Okapi(tokens_A) if tokens_A else None
    bm25_B = BM25Okapi(tokens_B) if tokens_B else None

    embedder_type = "none"
    vectorizer = None
    vecs_A = None
    vecs_B = None

    if use_st and SentenceTransformer is not None and texts_A:
        embedder = SentenceTransformer(st_model)
        embedder_type = "sentence_transformers"
        vecs_A = embedder.encode(
            texts_A, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )
        vecs_B = embedder.encode(
            texts_B, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        ) if texts_B else None
    elif texts_A:
        embedder_type = "tfidf"
        tfidf_vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        combined = texts_A + texts_B
        tfidf_vec.fit(combined)
        vectorizer = tfidf_vec
        vecs_A = normalize(tfidf_vec.transform(texts_A)).toarray().astype(np.float32)
        vecs_B = (
            normalize(tfidf_vec.transform(texts_B)).toarray().astype(np.float32)
            if texts_B
            else None
        )

    faiss_A = _build_faiss_ip(vecs_A) if vecs_A is not None and len(texts_A) else None
    faiss_B = _build_faiss_ip(vecs_B) if vecs_B is not None and len(texts_B) else None

    source_files = [
        {
            "path": api_base_url,
            "search": api_search,
            "limit": min(max(1, api_limit), OPENFDA_MAX_LIMIT),
            "sort": api_sort,
        }
    ]

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_type": "api",
        "source_files": source_files,
        "config": {
            "field_allowlist": field_allowlist,
            "field_blocklist": sorted(blocklist),
            "include_table_fields": include_table_fields,
            "max_records": api_max_records,
            "min_chars": min_chars,
            "words_per_chunk": words_per_chunk,
            "overlap": overlap,
            "use_st": use_st,
            "st_model": st_model,
            "api": {
                "base_url": api_base_url,
                "search": api_search,
                "limit": min(max(1, api_limit), OPENFDA_MAX_LIMIT),
                "sort": api_sort,
                "max_records": api_max_records,
                "pause_s": api_pause_s,
                "timeout_s": api_timeout_s,
            },
        },
        "counts": {
            "records": records_count,
            "record_chunks": len(record_chunks),
            "sub_chunks": len(sub_chunks),
        },
        "embedder": {
            "type": embedder_type,
            "model": st_model if embedder_type == "sentence_transformers" else None,
        },
        "artifacts": {},
    }

    if save:
        os.makedirs(output_dir, exist_ok=True)

        paths = {
            "record_chunks": os.path.join(output_dir, "record_chunks.jsonl"),
            "sub_chunks": os.path.join(output_dir, "sub_chunks.jsonl"),
            "bm25_record_tokens": os.path.join(output_dir, "bm25_record_tokens.json"),
            "bm25_sub_tokens": os.path.join(output_dir, "bm25_sub_tokens.json"),
            "faiss_record": os.path.join(output_dir, "faiss_record.index"),
            "faiss_sub": os.path.join(output_dir, "faiss_sub.index"),
            "vectorizer": os.path.join(output_dir, "tfidf_vectorizer.pkl"),
            "manifest": os.path.join(output_dir, "manifest.json"),
        }

        _write_jsonl(paths["record_chunks"], record_chunks)
        _write_jsonl(paths["sub_chunks"], sub_chunks)

        with open(paths["bm25_record_tokens"], "w", encoding="utf-8") as f:
            json.dump(tokens_A, f)
        with open(paths["bm25_sub_tokens"], "w", encoding="utf-8") as f:
            json.dump(tokens_B, f)

        if faiss_A is not None:
            faiss.write_index(faiss_A, paths["faiss_record"])
        if faiss_B is not None:
            faiss.write_index(faiss_B, paths["faiss_sub"])

        if vectorizer is not None and save_vectorizer:
            with open(paths["vectorizer"], "wb") as f:
                pickle.dump(vectorizer, f)

        manifest["artifacts"] = {
            "record_chunks": paths["record_chunks"],
            "sub_chunks": paths["sub_chunks"],
            "bm25_record_tokens": paths["bm25_record_tokens"],
            "bm25_sub_tokens": paths["bm25_sub_tokens"],
            "faiss_record": paths["faiss_record"] if faiss_A is not None else None,
            "faiss_sub": paths["faiss_sub"] if faiss_B is not None else None,
            "vectorizer": paths["vectorizer"] if vectorizer is not None else None,
        }

        with open(paths["manifest"], "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if verbose:
        print("Records:", records_count)
        print("Text chunks:", len(record_chunks))
        print("Sub-chunks:", len(sub_chunks))
        print("Embedder:", embedder_type)
        if save:
            print("Artifacts saved to:", output_dir)

    return {
        "record_chunks": record_chunks,
        "sub_chunks": sub_chunks,
        "bm25_A": bm25_A,
        "bm25_B": bm25_B,
        "faiss_A": faiss_A,
        "faiss_B": faiss_B,
        "tokens_A": tokens_A,
        "tokens_B": tokens_B,
        "vectorizer": vectorizer,
        "manifest": manifest,
    }


# --- Usage (Week 4.ipynb) ---
# 1) Set OPENFDA_* config values (search, limit, max records).
# 2) Define OPENFDA_FIELD_ALLOWLIST / OPENFDA_FIELD_BLOCKLIST.
# 3) Fetch records with iter_openfda_records or build_artifacts(...).
# 4) Use the notebook's retrieve_from_api(...) to fetch -> chunk -> retrieve -> LLM.
# 5) Run the demo loop to generate answers with citations.


def load_artifacts(output_dir: str = "preprocessed", load_vectorizer: bool = True) -> Dict[str, Any]:
    manifest_path = os.path.join(output_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found in {output_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    record_chunks = _read_jsonl_chunks(
        os.path.join(output_dir, "record_chunks.jsonl"), TextChunk
    )
    sub_chunks = _read_jsonl_chunks(
        os.path.join(output_dir, "sub_chunks.jsonl"), SubChunk
    )

    with open(os.path.join(output_dir, "bm25_record_tokens.json"), "r", encoding="utf-8") as f:
        tokens_A = json.load(f)
    with open(os.path.join(output_dir, "bm25_sub_tokens.json"), "r", encoding="utf-8") as f:
        tokens_B = json.load(f)

    bm25_A = BM25Okapi(tokens_A) if tokens_A else None
    bm25_B = BM25Okapi(tokens_B) if tokens_B else None

    faiss_A = None
    faiss_B = None
    faiss_record_path = os.path.join(output_dir, "faiss_record.index")
    faiss_sub_path = os.path.join(output_dir, "faiss_sub.index")
    if os.path.exists(faiss_record_path):
        faiss_A = faiss.read_index(faiss_record_path)
    if os.path.exists(faiss_sub_path):
        faiss_B = faiss.read_index(faiss_sub_path)

    vectorizer = None
    if (
        load_vectorizer
        and manifest.get("embedder", {}).get("type") == "tfidf"
        and os.path.exists(os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    ):
        with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

    return {
        "record_chunks": record_chunks,
        "sub_chunks": sub_chunks,
        "bm25_A": bm25_A,
        "bm25_B": bm25_B,
        "faiss_A": faiss_A,
        "faiss_B": faiss_B,
        "tokens_A": tokens_A,
        "tokens_B": tokens_B,
        "vectorizer": vectorizer,
        "manifest": manifest,
    }
