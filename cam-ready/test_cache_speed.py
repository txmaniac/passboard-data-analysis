import sys
import os
import time
import string
import numpy as np
import random
from functools import lru_cache

# Ensure we can load risk-estimation-libs
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "risk-estimation-libs"))

from ann_density import ANNDensity
from markov_entropy_risk import MarkovModel
from simulator import SimulationEngine
from typist import TemplateTypist
from policy import NullPolicy, ThresholdPolicy

class RobustTemplateTypist(TemplateTypist):
    def next_char(self, prefix, allowed_chars=None):
        choice = super().next_char(prefix, allowed_chars)
        if choice is None and self.idx < len(self.current_plan):
            if allowed_chars:
                choice = np.random.choice(list(allowed_chars))
                self.idx += 1
                return choice
        return choice

def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    ann = ANNDensity(k=20, file_path=None, limit=None, embedding_type='tfidf')
    ann.load_model(models_dir)
    
    mm = MarkovModel(n=4)
    mm.load(os.path.join(models_dir, "markov_model.json"))
    return ann, mm

def get_optimized_risk_evaluator(ann_model, mm_model):
    cache = {}
    
    def evaluator(prefix, candidates):
        # Use tuple to be hashable, assume candidates is always the same 70 chars.
        if prefix in cache:
            return cache[prefix]
            
        queries = [prefix + c for c in candidates]
        distances = ann_model.query(queries, k=20)
        density_risks = ann_model.get_risk_percentile(distances)
        
        prefix_entropy = 0.0
        for i in range(len(prefix)):
            p_probs = mm_model.predict_next_probs(prefix[:i])
            p = p_probs.get(prefix[i], 1e-9)
            prefix_entropy -= np.log2(p)
            
        next_probs = mm_model.predict_next_probs(prefix)
        
        final_risks = {}
        new_len = len(prefix) + 1
        for i, c in enumerate(candidates):
            p = next_probs.get(c, 1e-9)
            char_ent = -np.log2(p)
            new_ent_rate = (prefix_entropy + char_ent) / new_len
            ent_risk = max(0, min(100, (1.0 - (new_ent_rate / 8.0)) * 100))
            
            combined = 0.5 * density_risks[i] + 0.5 * ent_risk
            final_risks[c] = combined
            
        cache[prefix] = final_risks
        return final_risks
        
    return evaluator

def main():
    ann, mm = load_models()
    risk_func = get_optimized_risk_evaluator(ann, mm)
    engine = SimulationEngine(risk_func)
    
    proactive_policy = ThresholdPolicy(threshold=80.0, min_escape_percent=0.10)
    
    start_time = time.time()
    N = 100
    print(f"Testing caching speed with N={N}...")
    
    for seed in range(N):
        random.seed(seed)
        np.random.seed(seed)
        t = RobustTemplateTypist()
        engine.run_trajectory(t, proactive_policy, target_length=12)
        
    elapsed = time.time() - start_time
    print(f"Time taken for N={N}: {elapsed:.2f} seconds.")
    print(f"Estimated time for N=5000: {elapsed * 50:.2f} seconds.")

if __name__ == "__main__":
    main()
