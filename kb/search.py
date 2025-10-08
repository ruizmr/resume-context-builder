from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from kb.db import get_engine, persist_chunk_vectors, fetch_all_chunks, fetch_chunk_vectors

try:
    from pynndescent import NNDescent  # type: ignore
except Exception:  # pragma: no cover - optional
    NNDescent = None


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
		# CountVectorizer for BM25
		self.count_vectorizer = CountVectorizer(
			strip_accents="unicode",
			lowercase=True,
			ngram_range=(1, 2),
			max_features=50000,
			stop_words="english",
			max_df=0.95,
		)
		self.count_matrix = None
		self.doc_lengths: np.ndarray | None = None
		self.avg_doc_length: float = 0.0
		self.bm25_idf: np.ndarray | None = None
		self.bm25_k1: float = 1.2
		self.bm25_b: float = 0.75
		self.ids: List[int] = []
		self.texts: List[str] = []
		self.meta: List[Tuple[str, str]] = []  # path, chunk_name
		# ANN over SVD-reduced TF-IDF (optional)
		self.svd: TruncatedSVD | None = None
		self.doc_vectors_reduced: np.ndarray | None = None
		self.ann_index = None
		# Map of file path -> list of matrix indices ordered by chunk number
		self.path_to_ordered_indices: Dict[str, List[int]] = {}
		# Cache settings
		self.max_pgvector_cache: int = 10000  # max vectors to hydrate from DB on fit

	def fit(self, docs: List[Tuple[int, str, str, str]]):
		# docs: (id, path, chunk_name, content)
		self.ids = [d[0] for d in docs]
		self.meta = [(d[1], d[2]) for d in docs]
		self.texts = [d[3] for d in docs]
		if self.texts:
			self.matrix = self.vectorizer.fit_transform(self.texts)
			# Build BM25 structures
			try:
				self.count_matrix = self.count_vectorizer.fit_transform(self.texts)
				n_docs = self.count_matrix.shape[0]
				# document frequency per term
				df = (self.count_matrix > 0).astype(np.int32).sum(axis=0)
				# convert to 1D array
				df = np.asarray(df).ravel()
				# Okapi BM25 IDF with +1 to keep positivity
				self.bm25_idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
				# doc lengths and avg length
				self.doc_lengths = np.asarray(self.count_matrix.sum(axis=1)).ravel()
				self.avg_doc_length = float(self.doc_lengths.mean() if self.doc_lengths.size else 0.0)
			except Exception:
				self.count_matrix = None
				self.doc_lengths = None
				self.avg_doc_length = 0.0
				self.bm25_idf = None
			# Build path -> ordered matrix index list for neighbor sequencing
			order_map: Dict[str, List[Tuple[int, int]]] = {}
			for m_idx, (path, cname) in enumerate(self.meta):
				m = re.search(r"(\d+)$", cname)  # expect names like 'part12'
				chunk_num = int(m.group(1)) if m else (m_idx + 1)
				order_map.setdefault(path, []).append((chunk_num, m_idx))
			self.path_to_ordered_indices = {
				p: [mi for _, mi in sorted(pairs, key=lambda x: x[0])] for p, pairs in order_map.items()
			}
			# Try to hydrate SVD vectors from pgvector if Postgres available
			engine = get_engine()
			pgvec: Dict[int, List[float]] = {}
			try:
				if engine.dialect.name == "postgresql" and len(self.ids) <= int(max(1, self.max_pgvector_cache)):
					pg = fetch_chunk_vectors(engine, self.ids)
					if pg:
						pgvec = {cid: vec for cid, vec in pg.items() if vec}
			except Exception:
				pgvec = {}
			# Build ANN (PyNNDescent) if available and enough docs
			if NNDescent is not None and self.matrix.shape[0] >= 10 and self.matrix.shape[1] >= 2:
				# Choose a safe number of components
				n_comp = min(256, max(16, min(self.matrix.shape[0] - 1, self.matrix.shape[1] - 1)))
				try:
					self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
					if len(pgvec) == len(self.ids):
						# Use hydrated vectors (assumed same dimensionality); if mismatch, fallback to fit
						try:
							doc_vecs = np.array([pgvec.get(cid, []) for cid in self.ids], dtype=np.float32)
							if doc_vecs.ndim != 2 or doc_vecs.shape[1] != n_comp:
								raise ValueError("pgvector dim mismatch")
						except Exception:
							doc_vecs = self.svd.fit_transform(self.matrix).astype(np.float32, copy=False)
					else:
						doc_vecs = self.svd.fit_transform(self.matrix).astype(np.float32, copy=False)
					# L2-normalize SVD vectors to make cosine distances meaningful
					norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
					norms[norms == 0] = 1.0
					doc_vecs = doc_vecs / norms
					self.doc_vectors_reduced = doc_vecs
					# Build NN-Descent index over reduced vectors
					try:
						# Use a reasonable neighbor graph size; cap by dataset size
						n_nbrs = int(min(64, max(10, doc_vecs.shape[0] - 1)))
						self.ann_index = NNDescent(doc_vecs, metric="cosine", n_neighbors=n_nbrs, random_state=42)
					except Exception:
						self.ann_index = None
				except Exception:
					# Fall back silently if SVD/ANN build fails
					self.svd = None
					self.doc_vectors_reduced = None
					self.ann_index = None
			# Persist vectors to pgvector (best-effort) when Postgres present
			try:
				if engine.dialect.name == "postgresql" and self.doc_vectors_reduced is not None:
					rows: List[Tuple[int, List[float]]] = []
					for i, cid in enumerate(self.ids):
						vec = self.doc_vectors_reduced[i].astype(np.float32).tolist()
						rows.append((int(cid), vec))
					persist_chunk_vectors(engine, rows)
			except Exception:
				pass

	def search(
		self,
		query: str,
		top_k: int = 5,
		neighbors: int = 0,
		sequence: bool = False,
		*,
		use_ann: bool = True,
		bm25_weight: float = 0.45,
		cand_multiplier: int = 5,
		lsa_weight: float = 0.2,
		tfidf_metric: str = "cosine",
		ann_weight: float = 0.15,
		# ANN query-time controls (legacy names retained for compatibility)
		ann_ef_factor: float = 2.0,
		ann_ef_min: int = 50,
		ann_ef_max: int = 400,
		# Surgical improvements (optional)
		mmr_diversify: bool = True,
		mmr_lambda: float = 0.2,
		phrase_boost: float = 0.1,
		enable_rare_term_filter: bool = True,
		rare_idf_threshold: float = 3.0,
		min_score: float | None = None,
	) -> List[Tuple[int, float, str, str, str, str]]:
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
		# Compute effective minimum score with a built-in floor
		try:
			base_floor = 0.005
			ms = float(min_score) if min_score is not None else base_floor
			if ms < base_floor:
				ms = base_floor
		except Exception:
			ms = 0.005
		candidate_idx: np.ndarray
		# Use ANN index for candidate recall if available
		ann_scores = None
		if use_ann and self.ann_index is not None and self.svd is not None and self.doc_vectors_reduced is not None:
			try:
				q_red = self.svd.transform(q_vec).astype(np.float32, copy=False)
				cand_k = min(max(10, k * max(1, int(cand_multiplier))), len(self.ids))
				# Normalize the query vector for cosine consistency
				qn = np.linalg.norm(q_red)
				if qn > 0:
					q_red = q_red / qn
				labels, distances = self.ann_index.query(q_red, k=cand_k)
				candidate_idx = labels[0]
				# Convert cosine distances to similarity in [0,1]
				try:
					ann_scores = 1.0 - distances[0]
					ann_scores = np.clip(ann_scores, 0.0, 1.0)
				except Exception:
					ann_scores = None
			except Exception:
				candidate_idx = np.arange(len(self.ids))
		else:
			candidate_idx = np.arange(len(self.ids))

		# Optional rare-term candidate filtering using BM25 vocabulary/IDF
		if enable_rare_term_filter and self.count_matrix is not None and self.bm25_idf is not None and getattr(self.count_vectorizer, 'vocabulary_', None):
			try:
				an = self.count_vectorizer.build_analyzer()
				q_tokens = an(query)
				vocab = self.count_vectorizer.vocabulary_ or {}
				q_idxs = [vocab[t] for t in q_tokens if t in vocab]
				if q_idxs:
					q_idxs = sorted(set(q_idxs))
					idf_vals = self.bm25_idf[np.array(q_idxs, dtype=int)]
					# Define rare tokens by IDF threshold or presence of digits
					rare_mask = (idf_vals >= float(rare_idf_threshold))
					if not np.any(rare_mask):
						# Also treat any token with a digit as rare/specific
						rare_idx = [q_idxs[i] for i, tok in enumerate([t for t in q_tokens if t in vocab]) if any(ch.isdigit() for ch in tok)]
					else:
						rare_idx = [q_idxs[i] for i, r in enumerate(rare_mask) if bool(r)]
					if rare_idx:
						cm_sub = self.count_matrix[candidate_idx][:, rare_idx]
						has_any = np.asarray((cm_sub > 0).sum(axis=1)).ravel() > 0
						if np.any(has_any):
							candidate_idx_filtered = candidate_idx[has_any]
							# Keep ANN scores in sync when present
							if ann_scores is not None and ann_scores.shape[0] == candidate_idx.shape[0]:
								ann_scores = ann_scores[has_any]
							candidate_idx = candidate_idx_filtered
			except Exception:
				pass

		# Exact re-ranking on candidates using linear kernel
		if candidate_idx.size == 0:
			return []
		cand_matrix = self.matrix[candidate_idx]
		# Base TF-IDF similarity (cosine via dot product on L2-normalized vectors)
		cand_scores = linear_kernel(q_vec, cand_matrix).ravel()
		if (tfidf_metric or "cosine").lower() == "l2":
			# Convert Euclidean distance to similarity in [0,1]; assumes L2-normalized vectors
			# d = sqrt(2 - 2 * cos), sim = 1 - d/2
			with np.errstate(invalid='ignore'):
				d = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cand_scores))
				cand_scores = 1.0 - (d / 2.0)
		# BM25 scoring on the same candidate set
		bm25_scores = None
		if (
			self.count_matrix is not None
			and self.bm25_idf is not None
			and self.doc_lengths is not None
			and self.avg_doc_length > 0.0
		):
			# Build analyzer to get query tokens consistent with CountVectorizer
			analyzer = self.count_vectorizer.build_analyzer()
			q_tokens = analyzer(query)
			# Map tokens to indices and keep unique indices
			vocab = self.count_vectorizer.vocabulary_ or {}
			q_indices = [vocab[t] for t in q_tokens if t in vocab]
			if q_indices:
				q_indices = sorted(set(q_indices))
				# Slice the candidate rows for these columns
				cm_cand = self.count_matrix[candidate_idx][:, q_indices]
				dl = self.doc_lengths[candidate_idx]
				# Convert to dense small array for arithmetic
				f = cm_cand.toarray().astype(np.float32, copy=False)
				idf_vec = self.bm25_idf[np.array(q_indices, dtype=int)].astype(np.float32, copy=False)
				# BM25 formula
				k1 = float(self.bm25_k1)
				b = float(self.bm25_b)
				denom_base = k1 * (1.0 - b + b * (dl / self.avg_doc_length + 1e-12))
				# broadcasting over terms
				denom = f + denom_base[:, None]
				sat = (f * (k1 + 1.0)) / (denom + 1e-12)
				bm25_raw = (sat * idf_vec[None, :]).sum(axis=1)
				bm25_scores = bm25_raw

		# Optional LSA/SVD cosine similarity on candidates
		lsa_scores = None
		if self.svd is not None and self.doc_vectors_reduced is not None:
			try:
				q_red = self.svd.transform(q_vec).astype(np.float32, copy=False)
				qn = np.linalg.norm(q_red)
				if qn > 0:
					q_red = q_red / qn
				cand_red = self.doc_vectors_reduced[candidate_idx]
				# Normalize candidate rows
				cn = np.linalg.norm(cand_red, axis=1, keepdims=True)
				cn[cn == 0] = 1.0
				cand_red = cand_red / cn
				lsa_scores = (cand_red @ q_red.T).ravel()
			except Exception:
				lsa_scores = None

		# Normalize scores to [0,1] per candidate set for fusion
		def _minmax_norm(arr: np.ndarray) -> np.ndarray:
			if arr is None:
				return None  # type: ignore
			amin = float(np.min(arr))
			amax = float(np.max(arr))
			if amax <= amin + 1e-12:
				return np.zeros_like(arr)
			return (arr - amin) / max(1e-12, (amax - amin))

		cand_scores_norm = _minmax_norm(cand_scores)
		bm25_norm = _minmax_norm(bm25_scores) if bm25_scores is not None else None
		lsa_norm = _minmax_norm(lsa_scores) if lsa_scores is not None else None
		ann_norm = _minmax_norm(ann_scores) if ann_scores is not None else None
		# Weights: allocate bm25_weight to BM25, lsa_weight to LSA, ann_weight to ANN distance, remainder to TF-IDF
		w_bm25 = float(max(0.0, min(1.0, bm25_weight)))
		w_lsa = float(max(0.0, min(1.0, lsa_weight)))
		w_ann = float(max(0.0, min(1.0, ann_weight)))
		w_tfidf = max(0.0, 1.0 - w_bm25 - w_lsa - w_ann)
		if bm25_norm is None:
			w_tfidf += w_bm25
			w_bm25 = 0.0
		if lsa_norm is None:
			w_tfidf += w_lsa
			w_lsa = 0.0
		if ann_norm is None:
			w_tfidf += w_ann
			w_ann = 0.0
		# Renormalize weights to sum to 1.0
		wsum = max(1e-12, (w_bm25 + w_lsa + w_ann + w_tfidf))
		w_bm25 /= wsum
		w_lsa /= wsum
		w_ann /= wsum
		w_tfidf /= wsum
		final_scores = (w_tfidf * cand_scores_norm)
		if bm25_norm is not None:
			final_scores = final_scores + (w_bm25 * bm25_norm)
		if lsa_norm is not None:
			final_scores = final_scores + (w_lsa * lsa_norm)
		if ann_norm is not None:
			final_scores = final_scores + (w_ann * ann_norm)
		# If scores are uniformly low, short-circuit using effective min score
		try:
			if final_scores is None or final_scores.size == 0 or float(np.max(final_scores)) < float(ms):
				return []
		except Exception:
			pass
		# Order by fused score (with optional MMR diversification)
		if mmr_diversify and candidate_idx.shape[0] > 1 and k > 1:
			try:
				# Pairwise similarity over TF-IDF space for diversity
				cand_matrix = self.matrix[candidate_idx]
				sim = linear_kernel(cand_matrix, cand_matrix)
				selected_positions: List[int] = []
				remaining = list(range(candidate_idx.shape[0]))
				lam = float(max(0.0, min(1.0, mmr_lambda)))
				for _ in range(min(k, len(remaining))):
					if not selected_positions:
						best = int(np.argmax(final_scores))
					else:
						mmr_scores = []
						for j in remaining:
							if j in selected_positions:
								mmr_scores.append(-1e9)
								continue
							div_penalty = 0.0
							if selected_positions:
								div_penalty = max(sim[j, sel] for sel in selected_positions)
							mmr_scores.append(lam * float(final_scores[j]) - (1.0 - lam) * float(div_penalty))
						best = int(remaining[int(np.argmax(np.array(mmr_scores)))]) if remaining else int(np.argmax(final_scores))
					selected_positions.append(best)
					remaining.remove(best)
				order = np.array(selected_positions, dtype=int)
				ordered_idx = candidate_idx[order]
			except Exception:
				order = np.argsort(final_scores)[::-1][:k]
				ordered_idx = candidate_idx[order]
		else:
			order = np.argsort(final_scores)[::-1][:k]
			ordered_idx = candidate_idx[order]
		# Optionally compute scores for all docs if we need neighbor sequencing
		all_scores = linear_kernel(q_vec, self.matrix).ravel()

		# Phrase boost (quoted phrases in query)
		if phrase_boost and phrase_boost > 0.0:
			try:
				phrases = []
				for m in re.finditer(r'"([^"]+)"', query):
					ph = m.group(1).strip().lower()
					if ph:
						phrases.append(ph)
				if phrases:
					boosts = np.zeros_like(final_scores)
					lb_ph = [p.lower() for p in phrases]
					for pos, idx in enumerate(candidate_idx):
						text_l = (self.texts[idx] or "").lower()
						cnt = 0
						for ph in lb_ph:
							if ph in text_l:
								cnt += 1
						if cnt:
							boosts[pos] = float(cnt)
					if np.any(boosts > 0):
						boosts = boosts / float(np.max(boosts))
						final_scores = np.clip(final_scores + (float(phrase_boost) * boosts), 0.0, 1.0)
			except Exception:
				pass

		# Expand with neighbors if requested
		selected: List[int] = []
		seen = set()
		if neighbors and neighbors > 0 and self.path_to_ordered_indices:
			for base_idx in ordered_idx:
				if base_idx in seen:
					continue
				seen.add(base_idx)
				selected.append(base_idx)
				path, _ = self.meta[base_idx]
				order_list = self.path_to_ordered_indices.get(path, [])
				try:
					pos = order_list.index(base_idx)
				except ValueError:
					pos = -1
				if pos != -1:
					start = max(0, pos - int(neighbors))
					end = min(len(order_list), pos + int(neighbors) + 1)
					for ni in order_list[start:end]:
						if ni not in seen:
							seen.add(ni)
							selected.append(ni)
		else:
			selected = list(ordered_idx)

		# Reorder results for continuity if requested
		final_indices: List[int]
		if sequence and selected:
			# Group by path; order groups by best score within group
			by_path: Dict[str, List[int]] = {}
			for idx in selected:
				p, _ = self.meta[idx]
				by_path.setdefault(p, []).append(idx)
			# Sort each group's indices by their sequential order
			for p, idxs in by_path.items():
				order_list = self.path_to_ordered_indices.get(p, [])
				idxs.sort(key=lambda x: (order_list.index(x) if x in order_list else x))
			# Order documents by their best (highest) score among selected
			doc_order = sorted(by_path.keys(), key=lambda p: max(all_scores[i] for i in by_path[p]), reverse=True)
			final_indices = []
			for p in doc_order:
				final_indices.extend(by_path[p])
		else:
			# Keep relevance order, but ensure neighbors stay close to their seed order
			seed_order = list(ordered_idx)
			final_indices = []
			added = set()
			for seed in seed_order:
				for idx in selected:
					if idx == seed and idx not in added:
						final_indices.append(idx)
						added.add(idx)
						# append immediate neighbors for this seed in document order
						p, _ = self.meta[idx]
						order_list = self.path_to_ordered_indices.get(p, [])
						if idx in order_list:
							pos = order_list.index(idx)
							start = max(0, pos - int(neighbors))
							end = min(len(order_list), pos + int(neighbors) + 1)
							for ni in order_list[start:end]:
								if ni not in added and ni in selected:
									final_indices.append(ni)
									added.add(ni)

		# Apply a reasonable cap: allow neighbors around each of top_k seeds
		max_results = min(len(final_indices), max(k, min(len(self.ids), k * (2 * int(neighbors) + 1))))
		final_indices = final_indices[:max_results]
		results: List[Tuple[int, float, str, str, str, str]] = []
		for idx in final_indices:
			# Report fused score when available; map doc idx to candidate position
			try:
				cand_pos = int(np.where(candidate_idx == idx)[0][0])
				score = float(final_scores[cand_pos])
			except Exception:
				score = float(all_scores[idx])
			# Enforce effective minimum score
			if score < float(ms):
				continue
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


