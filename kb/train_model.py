import os
import json
import zipfile
import tempfile
import random
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple

from kb.search import HybridSearcher
from kb.tuning import get_optimizer
from kb.score import ndcg_at_k

DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

def download_and_load_scifact() -> Tuple[List[Tuple[int, str, str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Downloads SciFact dataset.
    Returns:
        corpus: List[(id_int, path, title, text)]
        queries: Dict[query_id_str, text]
        qrels: Dict[query_id_str, Dict[doc_id_str, score]]
    """
    print(f"Downloading dataset from {DATASET_URL}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "scifact.zip"
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        
        base = Path(tmpdir) / "scifact"
        
        # Load Corpus
        corpus = []
        print("Loading corpus...")
        with open(base / "corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("_id")
                title = doc.get("title", "")
                text = doc.get("text", "")
                full_text = f"{title}\n\n{text}"
                # We map string ID to int for HybridSearcher compatibility
                try:
                    cid = int(doc_id)
                    corpus.append((cid, f"scifact_{cid}", title, full_text))
                except ValueError:
                    pass
        
        # Load Queries
        queries = {}
        print("Loading queries...")
        with open(base / "queries.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                queries[q.get("_id")] = q.get("text")
                
        # Load Qrels
        qrels = {}
        print("Loading qrels...")
        # Check both test and dev
        for split in ["test.tsv", "dev.tsv"]:
            p = base / "qrels" / split
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    next(f) # header
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            qid, docid, score = parts[0], parts[1], int(parts[2])
                            if score > 0:
                                if qid not in qrels: qrels[qid] = {}
                                qrels[qid][docid] = score
                                
        return corpus, queries, qrels

def train_optimizer():
    corpus, queries, qrels = download_and_load_scifact()
    print(f"Loaded {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels.")
    
    # 1. Fit Searcher
    print("Indexing corpus...")
    searcher = HybridSearcher()
    searcher.fit(corpus)
    
    # 2. Training Loop
    print("Starting MCTS training loop...")
    optimizer = get_optimizer()
    
    # Filter queries to those with qrels
    valid_qids = [k for k in queries.keys() if k in qrels]
    random.shuffle(valid_qids)
    
    # Run for N iterations (enough to converge parameters slightly)
    # We loop through queries multiple times if needed, but 1 pass over ~300 queries is decent
    # SciFact has small qrels, so we iterate carefully
    
    metrics = []
    
    # Exploration phase (high epsilon) -> Exploitation (low epsilon)
    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        random.shuffle(valid_qids)
        
        for i, qid in enumerate(valid_qids):
            query_text = queries[qid]
            rel_map = qrels[qid]
            
            # Select params
            # Anneal exploration: 0.5 -> 0.1
            progress = (epoch * len(valid_qids) + i) / (epochs * len(valid_qids))
            epsilon = 0.5 * (1 - progress) + 0.1
            
            params = optimizer.suggest_parameters(exploration_prob=epsilon)
            
            # Search
            results = searcher.search(
                query_text,
                top_k=max(10, params.get("kb_top_k", 10)), # Force higher k for eval visibility
                neighbors=params.get("kb_neighbors", 0),
                bm25_weight=params.get("kb_bm25_weight", 0.45),
                lsa_weight=params.get("kb_lsa_weight", 0.2),
                ann_weight=params.get("kb_ann_weight", 0.15),
                cand_multiplier=params.get("kb_cand_mult", 5),
                mmr_diversify=params.get("kb_mmr_diversify", True),
                mmr_lambda=params.get("kb_mmr_lambda", 0.2),
                phrase_boost=params.get("kb_phrase_boost", 0.1),
                min_score=0.0 # Disable threshold for eval to get ranking
            )
            
            # Convert results to binary relevance vector
            # rel_map keys are strings, result cids are ints
            retrieved_rels = []
            for cid, score, _, _, _, _ in results:
                is_rel = 1 if str(cid) in rel_map else 0
                retrieved_rels.append(is_rel)
            
            # Calculate Reward (NDCG@10)
            ground_truth_count = len(rel_map)
            score = ndcg_at_k(retrieved_rels, 10, ground_truth_count)
            
            # Update Optimizer
            # Map NDCG 0..1 to something slightly more punitive for 0
            # If score is high -> reward +1.0
            # If score is 0 -> reward -0.5
            reward = (score * 2.0) - 0.5
            
            optimizer.update(params, reward)
            
            metrics.append(score)
            if i % 20 == 0:
                avg = sum(metrics[-50:]) / max(1, len(metrics[-50:]))
                print(f"Step {i}: Avg NDCG@10: {avg:.3f} | Params: {json.dumps(params)}")

    # 3. Save
    out_path = Path("kb/tuning_model.json")
    print(f"Saving model to {out_path.absolute()}")
    optimizer.save_model(str(out_path.absolute()))
    
    # Print best path
    print("Root visits:", optimizer.root.visits)
    if optimizer.root.children:
        best = max(optimizer.root.children, key=lambda c: c.value / max(1, c.visits))
        print("Best Params found:", best.params)
        print("Best Value:", best.value / max(1, best.visits))

if __name__ == "__main__":
    train_optimizer()

