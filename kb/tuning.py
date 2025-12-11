import math
import random
import json
import importlib.resources as resources
from typing import Dict, List, Any, Optional
from kb.db import fetch_search_history, fetch_feedback_for_query

class ParameterNode:
    def __init__(self, params: Dict[str, Any], parent=None):
        self.params = params
        self.parent = parent
        self.children: List['ParameterNode'] = []
        self.visits = 0
        self.value = 0.0  # Cumulative reward

    def uct_score(self, total_visits: int, exploration_weight: float = 1.41) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(total_visits) / self.visits)

    def to_dict(self):
        """Serialize for storage."""
        return {
            "params": self.params,
            "visits": self.visits,
            "value": self.value,
            "children": [c.to_dict() for c in self.children]
        }

    @classmethod
    def from_dict(cls, data: Dict, parent=None) -> 'ParameterNode':
        node = cls(data["params"], parent)
        node.visits = data["visits"]
        node.value = data["value"]
        node.children = [cls.from_dict(c, parent=node) for c in data.get("children", [])]
        return node

class SearchOptimizer:
    """
    Monte Carlo Tree Search (MCTS) inspired optimizer for search parameters.
    Treats parameter selection as a tree search where:
    - Nodes are parameter configurations.
    - Expansion adds slight perturbations to parameters.
    - Selection uses UCT.
    - Simulation is replaced by actual user feedback (online learning).
    """

    # Default baseline parameters
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

    # parameter bounds and step sizes for mutation
    RANGES = {
        "kb_top_k": (1, 20, int),
        "kb_min_score": (0.0, 0.5, float),
        "kb_neighbors": (0, 5, int),
        "kb_bm25_weight": (0.0, 1.0, float),
        "kb_lsa_weight": (0.0, 1.0, float),
        "kb_ann_weight": (0.0, 1.0, float),
        "kb_cand_mult": (1, 10, int),
        "kb_mmr_lambda": (0.0, 1.0, float),
        "kb_phrase_boost": (0.0, 1.0, float),
    }

    def __init__(self):
        self.root = ParameterNode(self.DEFAULTS.copy())
        self.nodes_map = {} # Signature -> Node to reuse states
        self.nodes_map[self._param_sig(self.root.params)] = self.root
        
        # 1. Try to load pretrained model from package
        self._load_pretrained()
        
        # 2. Hydrate/Update from local history (user specific tuning)
        self._hydrate_from_history()

    def _param_sig(self, params: Dict) -> str:
        # Canonical string for params
        return json.dumps(params, sort_keys=True)

    def _load_pretrained(self):
        """Load static pretrained model if available."""
        try:
            # Look for tuning_model.json in kb package
            ref = resources.files("kb").joinpath("tuning_model.json")
            if ref and ref.is_file():
                data = json.loads(ref.read_text(encoding="utf-8"))
                # Replace root with loaded tree
                self.root = ParameterNode.from_dict(data)
                # Rebuild map
                self.nodes_map = {}
                self._rebuild_map(self.root)
        except Exception:
            pass

    def _rebuild_map(self, node: ParameterNode):
        sig = self._param_sig(node.params)
        self.nodes_map[sig] = node
        for c in node.children:
            self._rebuild_map(c)

    def _hydrate_from_history(self):
        """Rebuild limited tree stats from DB history."""
        try:
            history = fetch_search_history(limit=500)
            for h in history:
                q = h["query"]
                params = h["params"]
                # Try to get feedback
                feedback = fetch_feedback_for_query(q)
                if not feedback:
                    continue
                
                # Simple reward: +1 for each positive, -1 for negative
                reward = sum(score for _, score in feedback)
                
                # Find or create node
                sig = self._param_sig(params)
                if sig not in self.nodes_map:
                    node = ParameterNode(params, parent=self.root) # simplify: flat attach to root for hydration
                    self.root.children.append(node)
                    self.nodes_map[sig] = node
                else:
                    node = self.nodes_map[sig]
                
                node.visits += 1
                node.value += reward
                self.root.visits += 1
        except Exception:
            pass

    def save_model(self, path: str):
        """Save current tree state to JSON."""
        data = self.root.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def suggest_parameters(self, exploration_prob: float = 0.2) -> Dict[str, Any]:
        """Select parameters for the next search query."""
        
        # Epsilon-greedy: sometimes purely explore
        if random.random() < exploration_prob:
             return self._mutate(self.root.params)

        # Selection: Traverse best UCT path
        node = self.root
        # Depth limit 
        for _ in range(3):
            if not node.children:
                # If leaf, expand
                if node.visits > 0:
                    self._expand(node)
                    if node.children:
                        node = random.choice(node.children)
                break
            
            # Choose best child
            best_child = max(node.children, key=lambda c: c.uct_score(node.visits))
            node = best_child
        
        return node.params

    def _expand(self, node: ParameterNode, num_children=3):
        """Create mutated children."""
        for _ in range(num_children):
            new_params = self._mutate(node.params)
            sig = self._param_sig(new_params)
            if sig not in self.nodes_map:
                child = ParameterNode(new_params, parent=node)
                node.children.append(child)
                self.nodes_map[sig] = child

    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb one random parameter."""
        new_p = params.copy()
        keys = list(self.RANGES.keys())
        k = random.choice(keys)
        min_v, max_v, dtype = self.RANGES[k]
        
        curr = new_p.get(k, self.DEFAULTS.get(k))
        
        if dtype == int:
            step = 1 if random.random() < 0.5 else -1
            curr += step
        else:
            step = 0.05 if random.random() < 0.5 else -0.05
            curr += step
        
        # Clamp
        if dtype == int:
            curr = max(min_v, min(max_v, int(round(curr))))
        else:
            curr = max(min_v, min(max_v, float(curr)))
            
        new_p[k] = curr
        return new_p

    def update(self, params: Dict[str, Any], reward: float):
        """Update stats for a parameter set (Backpropagation)."""
        sig = self._param_sig(params)
        node = self.nodes_map.get(sig)
        if not node:
            # If we used params not in tree (e.g. random explore), attach to root
            node = ParameterNode(params, parent=self.root)
            self.root.children.append(node)
            self.nodes_map[sig] = node
        
        # Backprop
        curr = node
        while curr:
            curr.visits += 1
            curr.value += reward
            curr = curr.parent

# Singleton instance
_optimizer = None
def get_optimizer():
    global _optimizer
    if _optimizer is None:
        _optimizer = SearchOptimizer()
    return _optimizer

