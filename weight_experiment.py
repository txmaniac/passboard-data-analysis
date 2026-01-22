import os
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("risk-estimation-libs")
from ann_density import ANNDensity
from policy import ThresholdPolicy
from markov_entropy_risk import MarkovModel
from simulator import SimulationEngine
from typist import TemplateTypist
from policy import ThresholdPolicy

# We reuse the logic from run_simulation but parameterize weights
def get_weighted_risk_evaluator(ann_model, mm_model, w_density=0.5):
    w_entropy = 1.0 - w_density
    
    def evaluator(prefix, candidates):
        queries = [prefix + c for c in candidates]
        distances = ann_model.query(queries, k=20)
        density_risks = ann_model.get_risk_percentile(distances)
        
        entropy_risks = []
        for c in candidates:
             full_str = prefix + c
             rate = mm_model.calculate_entropy_rate(full_str)
             r = (1.0 - (rate / 8.0)) * 100.0
             r = max(0, min(100, r))
             entropy_risks.append(r)
             
        final_risks = {}
        for i, c in enumerate(candidates):
            combined = w_density * density_risks[i] + w_entropy * entropy_risks[i]
            final_risks[c] = combined
            
        return final_risks
    return evaluator

def run_weight_experiment():
    print("Loading Models...")
    # Load cached models for speed
    try:
        ann = ANNDensity(k=20, file_path=None, limit=None, embedding_type='tfidf')
        ann.load_model("models")
        mm = MarkovModel(n=4)
        mm.load("models/markov_model.json")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # User Typist: Template (since they are the ones needing blocking)
    typist = TemplateTypist()
    
    # Policy: Blocking T80
    policy = ThresholdPolicy(threshold=80.0, min_escape_percent=0.10)
    
    weights = [0.0, 0.2, 0.5, 0.8, 1.0]
    results = []
    
    NUM_TRAJ = 50 # Sufficient for trend (Reduced for speed)
    TARGET_LEN = 8
    
    print(f"Running Weight Analysis (N={NUM_TRAJ})...")
    
    for w in weights:
        print(f"Testing w_density={w}...")
        risk_func = get_weighted_risk_evaluator(ann, mm, w_density=w)
        engine = SimulationEngine(risk_func)
        
        for i in tqdm.tqdm(range(NUM_TRAJ)):
            stats = engine.run_trajectory(typist, policy, target_length=TARGET_LEN)
            
            # Record outcomes
            # We want to check "True" quality.
            # Let's record Density Risk and Entropy Risk separately to see trade-offs.
            final_pwd = stats['password']
            
            # Calculate final metrics independent of weight
            d_dist = ann.query([final_pwd])[0]
            d_risk = ann.get_risk_percentile([d_dist])[0]
            
            e_rate = mm.calculate_entropy_rate(final_pwd)
            e_risk = max(0, min(100, (1.0 - (e_rate / 8.0)) * 100))
            
            results.append({
                "weight_density": w,
                "blocked_events": stats['blocked_steps'],
                "final_density_risk": d_risk,
                "final_entropy_risk": e_risk,
                "final_combined_eval": 0.5 * d_risk + 0.5 * e_risk # Neutral eval
            })

    df = pd.DataFrame(results)
    df.to_csv("weight_experiment_results.csv", index=False)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # 1. Effect on Neutral Combined Score
    plt.subplot(2, 2, 1)
    sns.lineplot(x="weight_density", y="final_combined_eval", data=df, marker="o")
    plt.title("Impact on Final Password Quality (Neutral Score)")
    plt.ylabel("Combined Risk (Lower is Better)")
    plt.xlabel("Weight given to Density (vs Entropy)")
    plt.grid(True)
    
    # 2. Effect on Density Risk
    plt.subplot(2, 2, 2)
    sns.lineplot(x="weight_density", y="final_density_risk", data=df, marker="o", color="red")
    plt.title("Impact on Density Risk")
    plt.ylabel("Density Risk")
    plt.xlabel("Weight given to Density")
    plt.grid(True)
    
    # 3. Effect on Entropy Risk
    plt.subplot(2, 2, 3)
    sns.lineplot(x="weight_density", y="final_entropy_risk", data=df, marker="o", color="blue")
    plt.title("Impact on Entropy Risk")
    plt.ylabel("Entropy Risk")
    plt.xlabel("Weight given to Density")
    plt.grid(True)
    
    # 4. Cost (Blocking Events)
    plt.subplot(2, 2, 4)
    sns.lineplot(x="weight_density", y="blocked_events", data=df, marker="o", color="green")
    plt.title("Cost: Interventions per Password")
    plt.ylabel("Avg Blocked Events")
    plt.xlabel("Weight given to Density")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("weight_experiment_plots.png")
    print("Saved plots to weight_experiment_plots.png")

if __name__ == "__main__":
    run_weight_experiment()
