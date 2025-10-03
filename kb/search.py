from __future__ import annotations

from typing import List, Tuple
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


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

	def fit(self, docs: List[Tuple[int, str, str, str]]):
		# docs: (id, path, chunk_name, content)
		self.ids = [d[0] for d in docs]
		self.meta = [(d[1], d[2]) for d in docs]
		self.texts = [d[3] for d in docs]
		if self.texts:
			self.matrix = self.vectorizer.fit_transform(self.texts)

	def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str, str, str, str]]:
		"""Return (id, score, path, chunk_name, snippet, full_text)."""
		if not self.texts:
			return []
		query = (query or "").strip()
		if not query:
			return []
		q_vec = self.vectorizer.transform([query])
		scores = linear_kernel(q_vec, self.matrix).ravel()
		k = min(top_k, scores.shape[0])
		if k <= 0:
			return []
		# Fast top-k selection
		if k < scores.shape[0]:
			candidate_idx = np.argpartition(scores, -k)[-k:]
			ordered_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
		else:
			ordered_idx = np.argsort(scores)[::-1]
		results: List[Tuple[int, float, str, str, str, str]] = []
		for idx in ordered_idx[:k]:
			score = scores[idx]
			cid = self.ids[idx]
			path, cname = self.meta[idx]
			text = self.texts[idx]
			snippet = text[:400]
			results.append((cid, float(score), path, cname, snippet, text))
		return results


