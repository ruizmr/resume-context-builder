import re
from typing import List, Dict, Set
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from kb.db import get_engine, fetch_chunks, upsert_meta_node, link_chunk_to_meta, fetch_chunk_by_id

def extract_keywords(texts: List[str], top_n: int = 5) -> List[List[str]]:
    """Extract top TF-IDF keywords for each text."""
    if not texts:
        return []
    try:
        # standard stop words + custom noise
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        matrix = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()
        
        results = []
        for row in matrix:
            # Sort indices by score
            indices = row.indices
            data = row.data
            sorted_items = sorted(zip(indices, data), key=lambda x: x[1], reverse=True)
            keywords = [feature_names[idx] for idx, score in sorted_items[:top_n]]
            results.append(keywords)
        return results
    except Exception:
        # Fallback if corpus too small for TF-IDF
        return [[] for _ in texts]

def build_meta_graph(chunk_ids: List[int]) -> None:
    """
    Build graph connections for the given chunk IDs.
    Extracts entities/keywords and links chunks to MetaNodes.
    """
    if not chunk_ids:
        return
    engine = get_engine()
    
    # 1. Fetch content
    # We need to fetch by specific IDs, but db.py only has fetch_chunk_by_id (single)
    # or fetch_chunks (paginated). Let's iterate or add a batch fetch.
    # For now, we iterate (optimization: add batch fetch to db.py later).
    chunks_data = []
    for cid in chunk_ids:
        row = fetch_chunk_by_id(engine, cid)
        if row:
            chunks_data.append(row)
            
    if not chunks_data:
        return

    texts = [c[3] for c in chunks_data]
    cids = [c[0] for c in chunks_data]

    # 2. Extract Keywords (Topics)
    keywords_list = extract_keywords(texts)

    # 3. Upsert Nodes and Edges
    for i, cid in enumerate(cids):
        # Link Topics
        for kw in keywords_list[i]:
            if len(kw) < 3 or kw.isnumeric(): 
                continue
            
            # Create/Get Node
            node_id = upsert_meta_node(engine, "topic", kw)
            
            # Link Chunk -> Node
            link_chunk_to_meta(engine, cid, node_id, weight=1.0)
            
        # Link Entities (Simple heuristic: Capitalized phrases? regex?)
        # For now, let's stick to keywords.
