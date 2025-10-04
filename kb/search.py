from __future__ import annotations

from typing import List, Tuple
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

try:
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover - optional
    hnswlib = None


class HybridSearcher:
	def __init__(self):
		self.vectorizer = TfidfVectorizer(
			strip_accents="unicode",
			lowercase=True,
			ngram_range=(1, 2),
			max_features=50000,
			stop_words="english",
			sublinear_tf=True,
			max_df=0.95,
			norm="l2",
		)
		self.matrix = None
		self.ids: List[int] = []
		self.texts: List[str] = []
		self.meta: List[Tuple[str, str]] = []  # path, chunk_name
		# HNSW over SVD-reduced TF-IDF (optional)
		self.svd: TruncatedSVD | None = None
		self.doc_vectors_reduced: np.ndarray | None = None
		self.hnsw_index = None
		self.hnsw_dim: int = 0

	def fit(self, docs: List[Tuple[int, str, str, str]]):
		# docs: (id, path, chunk_name, content)
		self.ids = [d[0] for d in docs]
		self.meta = [(d[1], d[2]) for d in docs]
		self.texts = [d[3] for d in docs]
		if self.texts:
			self.matrix = self.vectorizer.fit_transform(self.texts)
			# Build HNSW if available and enough docs
			if hnswlib is not None and self.matrix.shape[0] >= 10 and self.matrix.shape[1] >= 2:
				# Choose a safe number of components
				n_comp = min(256, max(16, min(self.matrix.shape[0] - 1, self.matrix.shape[1] - 1)))
				try:
					self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
					doc_vecs = self.svd.fit_transform(self.matrix).astype(np.float32, copy=False)
					self.doc_vectors_reduced = doc_vecs
					idx = hnswlib.Index(space='cosine', dim=n_comp)
					idx.init_index(max_elements=doc_vecs.shape[0], ef_construction=200, M=32)
					idx.add_items(doc_vecs, np.arange(doc_vecs.shape[0]))
					idx.set_ef(100)
					self.hnsw_index = idx
					self.hnsw_dim = n_comp
				except Exception:
					# Fall back silently if SVD/HNSW fails
					self.svd = None
					self.doc_vectors_reduced = None
					self.hnsw_index = None

	def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str, str, str, str]]:
		"""Return (id, score, path, chunk_name, snippet, full_text)."""
		if not self.texts:
			return []
		query = (query or "").strip()
		if not query:
			return []
		q_vec = self.vectorizer.transform([query])
		k = min(max(1, top_k), len(self.ids))
		if k <= 0:
			return []
		candidate_idx: np.ndarray
		# Use HNSW for candidate recall if available
		if self.hnsw_index is not None and self.svd is not None and self.doc_vectors_reduced is not None:
			try:
				q_red = self.svd.transform(q_vec).astype(np.float32, copy=False)
				cand_k = min(max(10, k * 3), len(self.ids))
				labels, distances = self.hnsw_index.knn_query(q_red, k=cand_k)
				candidate_idx = labels[0]
			except Exception:
				candidate_idx = np.arange(len(self.ids))
		else:
			candidate_idx = np.arange(len(self.ids))

		# Exact re-ranking on candidates using linear kernel
		if candidate_idx.size == 0:
			return []
		cand_matrix = self.matrix[candidate_idx]
		scores = linear_kernel(q_vec, cand_matrix).ravel()
		order = np.argsort(scores)[::-1][:k]
		ordered_idx = candidate_idx[order]
		results: List[Tuple[int, float, str, str, str, str]] = []
		for idx in ordered_idx:
			score = float(scores[np.where(candidate_idx == idx)[0][0]]) if candidate_idx.ndim == 1 else float(scores[idx])
			cid = self.ids[idx]
			path, cname = self.meta[idx]
			text = self.texts[idx]
			# Build a better snippet centered around query terms and avoid mid-word cuts
			q_terms = [t for t in (query.lower().split()) if len(t) >= 2]
			pos = -1
			for t in q_terms:
				p = text.lower().find(t)
				if p != -1:
					pos = p
					break
			start = 0
			if pos != -1:
				start = max(0, pos - 120)
			end = min(len(text), start + 400)
			snippet_raw = text[start:end]
			# Expand to nearest whitespace to avoid cutting words
			try:
				# trim leading partial word
				if start > 0 and not snippet_raw[:1].isspace():
					lead = snippet_raw.find(" ")
					if lead != -1 and lead < 40:
						snippet_raw = snippet_raw[lead + 1:]
				# trim trailing partial word
				if end < len(text) and not snippet_raw[-1:].isspace():
					tail = snippet_raw.rfind(" ")
					if tail != -1 and (len(snippet_raw) - tail) < 40:
						snippet_raw = snippet_raw[:tail]
			except Exception:
				pass
			snippet = snippet_raw.strip()
			results.append((cid, score, path, cname, snippet, text))
		return results


