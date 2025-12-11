import json
import random
import numpy as np
from typing import List, Dict, Tuple
from sklearn.datasets import fetch_20newsgroups
from kb.search import HybridSearcher
from kb.tuning import get_optimizer

def dcg_at_k(r: List[float], k: int) -> float:
    """Score is discounted cumulative gain at rank k."""
    r = np.asarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r: List[float], k: int) -> float:
    """Score is normalized discounted cumulative gain (NDCG) at rank k."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

def prepare_training_data(limit: int = 500) -> Tuple[List[Tuple[int, str, str, str]], List[Dict]]:
    """
    Load 20 Newsgroups dataset and generate (chunk, query) pairs.
    Returns: (chunks, queries)
    """
    print("Loading 20 Newsgroups dataset (this may download data)...")
    try:
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    except Exception as e:
        print(f"Failed to load 20newsgroups: {e}")
        return [], []

    chunks = []
    queries = []
    
    print(f"Processing {min(len(newsgroups.data), limit)} documents...")
    
    # Create chunks
    for i, text in enumerate(newsgroups.data[:limit]):
        if len(text) < 200:
            continue
        
        cid = i + 100000 # Offset IDs to avoid conflict with local DB
        path = f"newsgroup_{i}.txt"
        cname = "content"
        chunks.append((cid, path, cname, text))
        
        # Generate synthetic query: 
        # Strategy: take a random sentence or phrase from the middle
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 50]
        if lines:
            target_line = random.choice(lines)
            words = target_line.split()
            if len(words) > 5:
                # Take a slice of 5-10 words
                start = random.randint(0, max(0, len(words) - 10))
                q_text = " ".join(words[start:start+10])
                queries.append({
                    "query": q_text,
                    "relevant_id": cid
                })

    return chunks, queries

def train_optimizer(n_iter: int = 100):
    """
    Run MCTS training loop on the dataset.
    """
    chunks, dataset = prepare_training_data(limit=1000)
    if not chunks:
        print("No training data available.")
        return

    print(f"Training on {len(chunks)} documents with {len(dataset)} queries...")
    
    # Initialize searcher in memory (no DB persistence needed for training)
    searcher = HybridSearcher()
    searcher.fit(chunks)
    
    opt = get_optimizer()
    
    # Reset optimizer to clean state for training
    from kb.tuning import SearchOptimizer, ParameterNode
    opt.root = ParameterNode(SearchOptimizer.DEFAULTS.copy())
    opt.nodes_map = {opt._param_sig(opt.root.params): opt.root}
    
    scores = []
    
    for i in range(n_iter):
        # Pick a random query case
        case = random.choice(dataset)
        query = case["query"]
        target = case["relevant_id"]
        
        # 1. Suggest Params (High exploration during training)
        params = opt.suggest_parameters(exploration_prob=0.4)
        
        # 2. Search
        try:
            results = searcher.search(
                query,
                top_k=params.get("kb_top_k", 5),
                neighbors=params.get("kb_neighbors", 0),
                bm25_weight=params.get("kb_bm25_weight", 0.45),
                lsa_weight=params.get("kb_lsa_weight", 0.2),
                ann_weight=params.get("kb_ann_weight", 0.15),
                cand_multiplier=params.get("kb_cand_mult", 5),
                mmr_diversify=params.get("kb_mmr_diversify", True),
                mmr_lambda=params.get("kb_mmr_lambda", 0.2),
                phrase_boost=params.get("kb_phrase_boost", 0.1),
                min_score=params.get("kb_min_score", 0.005)
            )
        except Exception:
            results = []
            
        # 3. Calculate Reward (NDCG@5)
        # We only have one "relevant" document in this synthetic setup, so relevance is binary
        relevance_vector = []
        found = False
        for cid, score, _, _, _, _ in results:
            if cid == target:
                relevance_vector.append(1.0)
                found = True
            else:
                relevance_vector.append(0.0)
        
        # Pad if fewer results than k
        k = 5
        if len(relevance_vector) < k:
            relevance_vector.extend([0.0] * (k - len(relevance_vector)))
            
        # NDCG calculation
        # Ideal vector has 1.0 at index 0
        reward = ndcg_at_k(relevance_vector, k)
        
        # Penalize empty results severely
        if not results:
            reward = -0.5
            
        scores.append(reward)
        
        # 4. Update
        opt.update(params, reward)
        
        if i % 10 == 0:
            avg_score = sum(scores[-20:]) / max(1, len(scores[-20:]))
            print(f"Iter {i}: Reward={reward:.2f}, Moving Avg={avg_score:.2f}, Params={json.dumps(params)}")

    print("Training complete.")
    opt.save_model("kb/tuning_model.json")
    print("Saved model to kb/tuning_model.json")

if __name__ == "__main__":
    train_optimizer(n_iter=200)
