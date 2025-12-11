import json
import random
from typing import Dict, Any

def create_synthetic_model():
    """
    Generate a synthetic pretrained model that biases towards defaults.
    """
    # Defaults from tuning.py
    DEFAULTS = {
        "kb_top_k": 5,
        "kb_min_score": 0.005,
        "kb_neighbors": 0,
        "kb_bm25_weight": 0.45,
        "kb_lsa_weight": 0.2,
        "kb_ann_weight": 0.15,
        "kb_cand_mult": 5,
        "kb_mmr_diversify": True,
        "kb_mmr_lambda": 0.2,
        "kb_phrase_boost": 0.1,
    }
    
    # Root node with high visits and positive value
    root = {
        "params": DEFAULTS,
        "visits": 100,
        "value": 50.0,
        "children": []
    }
    
    # Add some variations as children
    # 1. High recall (more neighbors, lower threshold)
    v1 = DEFAULTS.copy()
    v1["kb_top_k"] = 10
    v1["kb_neighbors"] = 1
    root["children"].append({
        "params": v1,
        "visits": 20,
        "value": 5.0,
        "children": []
    })
    
    # 2. Precision (high bm25)
    v2 = DEFAULTS.copy()
    v2["kb_bm25_weight"] = 0.8
    root["children"].append({
        "params": v2,
        "visits": 20,
        "value": 2.0,
        "children": []
    })

    with open("kb/tuning_model.json", "w", encoding="utf-8") as f:
        json.dump(root, f, indent=2)
    print("Created kb/tuning_model.json")

if __name__ == "__main__":
    create_synthetic_model()

