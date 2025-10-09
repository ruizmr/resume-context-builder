from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from kb.db import STATE_DIR, get_engine, fetch_all_chunks, compute_hash

try:
    from pynndescent import NNDescent  # type: ignore
except Exception:  # pragma: no cover
    NNDescent = None

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore


ARTIFACTS_DIR = STATE_DIR / "artifacts"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _artifact_paths() -> Dict[str, Path]:
    d = ARTIFACTS_DIR
    return {
        "tfidf": d / "tfidf_vectorizer.pkl",
        "count": d / "count_vectorizer.pkl",
        "svd": d / "svd.pkl",
        "tfidf_matrix": d / "tfidf_matrix.npz",
        "doc_vectors": d / "doc_vectors.npy",
        "ids": d / "ids.json",
        "meta": d / "meta.json",
        "ann": d / "ann_index.pkl",
        "sig": d / "corpus_sig.txt",
    }


def _compute_corpus_signature(docs: List[Tuple[int, str, str, str]]) -> str:
    # Hash over (id, path, chunk_name, content_hash) for stability
    from hashlib import sha256
    h = sha256()
    for cid, path, cname, content in docs:
        h.update(str(int(cid)).encode("utf-8"))
        h.update((path or "").encode("utf-8"))
        h.update((cname or "").encode("utf-8"))
        h.update(compute_hash(content).encode("utf-8"))
    return h.hexdigest()


def build_and_save_artifacts(*, enable_ann: bool = True, svd_components_cap: int = 256) -> None:
    engine = get_engine()
    docs = fetch_all_chunks(engine)
    if not docs:
        return
    ids = [int(d[0]) for d in docs]
    paths = [d[1] for d in docs]
    names = [d[2] for d in docs]
    texts = [d[3] for d in docs]
    sig = _compute_corpus_signature(docs)

    # Vectorizers
    tfidf = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000,
        stop_words="english",
        sublinear_tf=True,
        max_df=0.95,
        norm="l2",
    )
    tfidf_matrix = tfidf.fit_transform(texts)

    count_vec = CountVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000,
        stop_words="english",
        max_df=0.95,
    )
    _ = count_vec.fit(texts)

    # SVD + normalized reduced vectors
    n_comp = min(int(svd_components_cap), max(16, min(tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)))
    svd: TruncatedSVD | None = None
    doc_vecs: np.ndarray | None = None
    if n_comp >= 2:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        doc_vecs = svd.fit_transform(tfidf_matrix).astype(np.float32, copy=False)
        norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_vecs = (doc_vecs / norms).astype(np.float32, copy=False)

    # ANN index (best-effort)
    ann_index = None
    if enable_ann and NNDescent is not None and doc_vecs is not None and doc_vecs.shape[0] >= 10:
        try:
            n_nbrs = int(min(64, max(10, doc_vecs.shape[0] - 1)))
            ann_index = NNDescent(doc_vecs, metric="cosine", n_neighbors=n_nbrs, random_state=42)
        except Exception:
            ann_index = None

    p = _artifact_paths()
    # Save artifacts atomically
    _atomic_write_bytes(p["tfidf"], pickle.dumps(tfidf))
    _atomic_write_bytes(p["count"], pickle.dumps(count_vec))
    if svd is not None:
        _atomic_write_bytes(p["svd"], pickle.dumps(svd))
    if sp is not None:
        p["tfidf_matrix"].parent.mkdir(parents=True, exist_ok=True)
        try:
            sp.save_npz(str(p["tfidf_matrix"]), tfidf_matrix)
        except Exception:
            pass
    if doc_vecs is not None:
        p["doc_vectors"].parent.mkdir(parents=True, exist_ok=True)
        np.save(p["doc_vectors"], doc_vecs)
    _atomic_write_text(p["ids"], json.dumps(ids))
    _atomic_write_text(p["meta"], json.dumps([[paths[i], names[i]] for i in range(len(ids))]))
    _atomic_write_text(p["sig"], sig)
    if ann_index is not None:
        try:
            _atomic_write_bytes(p["ann"], pickle.dumps(ann_index))
        except Exception:
            # Non-fatal; ANN will be rebuilt on demand
            pass


def load_artifacts() -> Dict[str, Any] | None:
    p = _artifact_paths()
    try:
        if not p["tfidf"].exists() or not p["count"].exists():
            return None
        tfidf = pickle.loads(p["tfidf"].read_bytes())
        count_vec = pickle.loads(p["count"].read_bytes())
        svd = pickle.loads(p["svd"].read_bytes()) if p["svd"].exists() else None
        tfidf_matrix = None
        if sp is not None and p["tfidf_matrix"].exists():
            try:
                tfidf_matrix = sp.load_npz(str(p["tfidf_matrix"]))
            except Exception:
                tfidf_matrix = None
        doc_vecs = None
        if p["doc_vectors"].exists():
            try:
                doc_vecs = np.load(p["doc_vectors"])
            except Exception:
                doc_vecs = None
        ids = json.loads(p["ids"].read_text(encoding="utf-8")) if p["ids"].exists() else []
        meta = json.loads(p["meta"].read_text(encoding="utf-8")) if p["meta"].exists() else []
        ann_index = None
        if p["ann"].exists():
            try:
                ann_index = pickle.loads(p["ann"].read_bytes())
            except Exception:
                ann_index = None
        sig = p["sig"].read_text(encoding="utf-8").strip() if p["sig"].exists() else None
        return {
            "tfidf": tfidf,
            "count": count_vec,
            "svd": svd,
            "tfidf_matrix": tfidf_matrix,
            "doc_vectors": doc_vecs,
            "ids": ids,
            "meta": meta,
            "ann_index": ann_index,
            "sig": sig,
        }
    except Exception:
        return None


