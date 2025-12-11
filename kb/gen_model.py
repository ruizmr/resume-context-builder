import json
from kb.tuning import ParameterNode, SearchOptimizer

def generate_pretrained_model():
    """
    Construct a synthetic MCTS tree with robust priors.
    We simulate visits to the default node and some reasonable variations.
    """
    defaults = SearchOptimizer.DEFAULTS
    
    # Root: Give it solid visits to represent a safe baseline
    root = ParameterNode(defaults)
    root.visits = 50
    root.value = 25.0  # Net positive feedback
    
    # Variation 1: Higher recall (top_k=10, cand_mult=8)
    v1_params = defaults.copy()
    v1_params["kb_top_k"] = 10
    v1_params["kb_cand_mult"] = 8
    v1 = ParameterNode(v1_params, parent=root)
    v1.visits = 10
    v1.value = 4.0
    root.children.append(v1)
    
    # Variation 2: Higher precision (bm25=0.8, lsa=0.1)
    v2_params = defaults.copy()
    v2_params["kb_bm25_weight"] = 0.8
    v2_params["kb_lsa_weight"] = 0.1
    v2 = ParameterNode(v2_params, parent=root)
    v2.visits = 10
    v2.value = 6.0 # Slightly better
    root.children.append(v2)
    
    # Variation 3: Semantic heavy (ann=0.4, lsa=0.4, bm25=0.2)
    v3_params = defaults.copy()
    v3_params["kb_ann_weight"] = 0.4
    v3_params["kb_lsa_weight"] = 0.4
    v3_params["kb_bm25_weight"] = 0.2
    v3 = ParameterNode(v3_params, parent=root)
    v3.visits = 10
    v3.value = 3.0
    root.children.append(v3)
    
    data = root.to_dict()
    
    with open("kb/tuning_model.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print("Generated kb/tuning_model.json")

if __name__ == "__main__":
    generate_pretrained_model()

