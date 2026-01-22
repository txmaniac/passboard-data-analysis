import os
import random
import pandas as pd
import numpy as np
import tqdm
from markov_entropy_risk import MarkovModel
from ann_density import ANNDensity
from typist import MarkovTypist, TemplateTypist
from policy import NullPolicy, ThresholdPolicy
from simulator import SimulationEngine

# Config
DATA_FILE = "rockyou.txt"
# For PoC speed, we limit the dataset size.
# In a real full-scale run, we'd use the whole file.
LIMIT_TOTAL = 200000 
TRAIN_TEST_SPLIT = 0.9 # 90% for Risk Engine, 10% for Typist
TARGET_LENGTH = 8 # Simulation length
NUM_TRAJECTORIES = 10000 # Small batch for verification (Plan said 10k, but I'll default to 100 for dev run, user can increase)

def prepare_data():
    """Split rockyou.txt into risk (90%) and typist (10%) sets."""
    if os.path.exists("rockyou_risk.txt") and os.path.exists("rockyou_typist.txt"):
        print("Data split already exists.")
        return

    print("Loading data for splitting...")
    # Locate valid path
    paths = ["rockyou.txt", "../rockyou.txt", "clustering-analysis/rockyou.txt", "../clustering-analysis/rockyou.txt"]
    src = None
    for p in paths:
        if os.path.exists(p):
            src = p
            break
            
    if not src:
        raise FileNotFoundError("rockyou.txt not found")
        
    with open(src, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
        
    if LIMIT_TOTAL:
        lines = random.sample(lines, min(len(lines), LIMIT_TOTAL))
        
    # Split
    split_idx = int(len(lines) * TRAIN_TEST_SPLIT)
    risk_data = lines[:split_idx]
    typist_data = lines[split_idx:]
    
    print(f"Splitting {len(lines)} items: {len(risk_data)} (Risk) / {len(typist_data)} (Typist)")
    
    with open("rockyou_risk.txt", "w") as f:
        f.write("\n".join(risk_data))
        
    with open("rockyou_typist.txt", "w") as f:
        f.write("\n".join(typist_data))

def train_risk_engine():
    print("Training Risk Engine (Defender)...")
    # 1. Density Model
    ann = ANNDensity(k=20, file_path="rockyou_risk.txt", limit=None, embedding_type='tfidf')
    ann.fit(backend='faiss')
    
    # 2. Entropy Model (for Evaluation/Risk)
    mm_risk = MarkovModel(n=4)
    with open("rockyou_risk.txt", "r", encoding="utf-8", errors="ignore") as f:
        data = f.read().splitlines()
    mm_risk.train(data)
    
    return ann, mm_risk

def train_typist_model():
    print("Training Typist Model (Attacker User)...")
    mm_typist = MarkovModel(n=4)
    with open("rockyou_typist.txt", "r", encoding="utf-8", errors="ignore") as f:
        data = f.read().splitlines()
    mm_typist.train(data)
    return mm_typist

def get_risk_evaluator(ann_model, mm_model):
    def evaluator(prefix, candidates):
        # 1. Density Risk
        # Formulate full strings: prefix + c
        queries = [prefix + c for c in candidates]
        
        # Batch query ANNDensity
        # Returns distances to k-th neighbor
        distances = ann_model.query(queries, k=20)
        
        # Convert to Percentile Risk (0-100)
        # Low Dist = High Risk
        density_risks = ann_model.get_risk_percentile(distances)
        
        # 2. Entropy Risk
        # Calculate entropy rate of the CANDIDATE string (prefix+c)
        # Entropy Rate = TotalEntropy / Len
        # Actually, simpler: Measure how surprising 'c' is given 'prefix'?
        # Plan said "Entropy Risk: Linear mapping of Entropy Rate (0-100)"
        # Does it mean the rate of the *whole string so far*? Yes usually.
        
        entropy_risks = []
        for c in candidates:
             # This might be slow unoptimized loop, but fine for PoC
             full_str = prefix + c
             rate = mm_model.calculate_entropy_rate(full_str)
             # Map 0-8 bits to 100-0 Risk
             # 0 bits (highly predictable) -> 100 Risk
             # 8 bits (random) -> 0 Risk
             r = (1.0 - (rate / 8.0)) * 100.0
             r = max(0, min(100, r))
             entropy_risks.append(r)
             
        # 3. Combine
        final_risks = {}
        for i, c in enumerate(candidates):
            combined = 0.5 * density_risks[i] + 0.5 * entropy_risks[i]
            final_risks[c] = combined
            
        return final_risks
        
    return evaluator

if __name__ == "__main__":
    # 1. Setup
    prepare_data()
    
    # 2. Train Models
    ann_risk, mm_risk = train_risk_engine()
    mm_typist = train_typist_model()
    
    # 3. Risk Evaluator
    risk_func = get_risk_evaluator(ann_risk, mm_risk)
    
    # 4. Engine
    engine = SimulationEngine(risk_func)
    
    # 5. Experiments
    results = []
    
    scenarios = [
        ("Markov", MarkovTypist(mm_typist)),
        ("Template", TemplateTypist())
    ]
    
    conditions = [
        ("Baseline", NullPolicy()),
        ("Blocking_T80", ThresholdPolicy(threshold=80.0, min_escape_percent=0.10))
    ]
    
    print(f"\nStarting Simulation ({NUM_TRAJECTORIES} trajectories per condition)...")
    
    for sc_name, typist in scenarios:
        for cond_name, policy in conditions:
            print(f"Running {sc_name} x {cond_name}...")
            
            for i in tqdm.tqdm(range(NUM_TRAJECTORIES)):
                # Run trajectory
                stats = engine.run_trajectory(typist, policy, target_length=TARGET_LENGTH)
                
                # Calculate final scores for logging (using Risk Model)
                final_pwd = stats['password']
                final_dist = ann_risk.query([final_pwd])[0]
                final_density_risk = ann_risk.get_risk_percentile([final_dist])[0]
                final_entropy_rate = mm_risk.calculate_entropy_rate(final_pwd)
                
                results.append({
                    "scenario": sc_name,
                    "condition": cond_name,
                    "password": final_pwd,
                    "length": stats['length'],
                    "blocked_events": stats['blocked_steps'],
                    "avg_step_risk": stats['avg_step_risk'],
                    "final_density_risk": final_density_risk,
                    "final_entropy_rate": final_entropy_rate
                })
                
    # 6. Save
    df = pd.DataFrame(results)
    df.to_csv("simulation_results.csv", index=False)
    print("\nSimulation Complete. Results saved to simulation_results.csv")
    print(df.groupby(['scenario', 'condition'])[['blocked_events', 'final_density_risk', 'avg_step_risk']].mean())
