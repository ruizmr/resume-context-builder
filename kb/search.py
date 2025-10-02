from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridSearcher:
	def __init__(self):
		self.vectorizer = TfidfVectorizer(
			strip_accents="unicode",
			lowercase=True,
			ngram_range=(1, 2),
			max_features=50000,
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

	def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str, str, str]]:
		"""Return (id, score, path, chunk_name, snippet)."""
		if not self.texts:
			return []
		q_vec = self.vectorizer.transform([query])
		scores = cosine_similarity(q_vec, self.matrix).ravel()
		ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
		results: List[Tuple[int, float, str, str, str]] = []
		for idx, score in ranked:
			cid = self.ids[idx]
			path, cname = self.meta[idx]
			text = self.texts[idx]
			snippet = text[:400]
			results.append((cid, float(score), path, cname, snippet))
		return results


