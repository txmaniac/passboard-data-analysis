import sys
import os
import math
import random
import string
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we can load risk-estimation-libs
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "risk-estimation-libs"))

from ann_density import ANNDensity
from markov_entropy_risk import MarkovModel
from simulator import SimulationEngine
from typist import MarkovTypist, TemplateTypist
from policy import NullPolicy, ThresholdPolicy

class RobustTemplateTypist(TemplateTypist):
    """
    A TemplateTypist that resolves stuck trajectories by choosing randomly from 
    allowed characters if the planned character class is blocked.
    """
    def next_char(self, prefix, allowed_chars=None):
        choice = super().next_char(prefix, allowed_chars)
        if choice is None and self.idx < len(self.current_plan):
            if allowed_chars:
                choice = np.random.choice(allowed_chars)
                self.idx += 1
                return choice
        return choice

def load_models():
    print("Loading pre-trained models (FAISS density index & Markov model)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    ann = ANNDensity(k=20, file_path=None, limit=None, embedding_type='tfidf')
    ann.load_model(models_dir)
    
    mm = MarkovModel(n=4)
    mm.load(os.path.join(models_dir, "markov_model.json"))
    return ann, mm

def get_optimized_risk_evaluator(ann_model, mm_model):
    """
    Highly optimized risk evaluator. Instead of re-calculating the entropy 
    of the entire string prefix + c for each candidate c (which is O(L^2)), 
    we compute the prefix's cumulative entropy once and update it in O(1) time.
    """
    def evaluator(prefix, candidates):
        queries = [prefix + c for c in candidates]
        
        # 1. Batch query FAISS density (extremely fast C++ lookup)
        distances = ann_model.query(queries, k=20)
        density_risks = ann_model.get_risk_percentile(distances)
        
        # 2. Optimized Entropy Calculation
        # Compute prefix entropy once
        prefix_entropy = 0.0
        for i in range(len(prefix)):
            p_probs = mm_model.predict_next_probs(prefix[:i])
            p = p_probs.get(prefix[i], 1e-9)
            prefix_entropy -= math.log2(p)
            
        # Get probabilities for the next step directly
        next_probs = mm_model.predict_next_probs(prefix)
        
        entropy_risks = []
        new_len = len(prefix) + 1
        for c in candidates:
            p = next_probs.get(c, 1e-9)
            char_ent = -math.log2(p)
            new_ent_rate = (prefix_entropy + char_ent) / new_len
            # Risk maps 0-8 entropy rate to 100-0% risk score
            r = max(0, min(100, (1.0 - (new_ent_rate / 8.0)) * 100))
            entropy_risks.append(r)
            
        # 3. Combine Risks
        final_risks = {}
        for i, c in enumerate(candidates):
            combined = 0.5 * density_risks[i] + 0.5 * entropy_risks[i]
            final_risks[c] = combined
        return final_risks
    return evaluator

def main():
    parser = argparse.ArgumentParser(description="HumSec2026 Password Security Simulation & Plotting Analysis")
    parser.add_argument("--num-sims", type=int, default=500, help="Number of trajectories per cohort (default: 500)")
    parser.add_argument("--threshold", type=float, default=80.0, help="Risk threshold for intervention (default: 80.0)")
    args = parser.parse_args()

    N = args.num_sims
    T = args.threshold
    
    # Setup plots folder
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Models
    ann, mm = load_models()
    risk_func = get_optimized_risk_evaluator(ann, mm)
    engine = SimulationEngine(risk_func)
    
    # Policies
    null_policy = NullPolicy()
    proactive_policy = ThresholdPolicy(threshold=T, min_escape_percent=0.10)
    
    lengths = [8, 10, 12, 14]
    typist_configs = [
        ("Markov Typist", lambda: MarkovTypist(mm)),
        ("Template Typist", lambda: RobustTemplateTypist())
    ]
    
    raw_data = []
    
    print(f"\nRunning HumSec2026 Simulation Experiments (N={N}, Threshold={T})...")
    
    for typist_name, typist_factory in typist_configs:
        print(f"\nCohort: {typist_name}")
        
        # --- 1. Baseline ---
        print("  Running Baseline (No Policy)...")
        for i in range(N):
            target_L = int(np.random.choice(lengths))
            t = typist_factory()
            stats = engine.run_trajectory(t, null_policy, target_length=target_L)
            pwd = stats['password']
            avg_risk = stats['avg_step_risk']
            ent_rate = mm.calculate_entropy_rate(pwd)
            
            raw_data.append({
                "Typist": typist_name,
                "Policy": "Baseline",
                "Password": pwd,
                "Length": len(pwd),
                "Final_Risk": avg_risk,
                "Entropy_Rate": ent_rate,
                "Attempts": 1,
                "Total_Keystrokes": len(pwd),
                "Discarded_Keystrokes": 0,
                "Blocked_Steps": 0,
                "Success": True
            })
            
        # --- 2. Post-hoc Rejection ---
        print("  Running Post-hoc Rejection (Check on Submit)...")
        for i in range(N):
            target_L = int(np.random.choice(lengths))
            attempts = 0
            success = False
            total_keys = 0
            final_risk = 0.0
            final_pwd = ""
            
            while attempts < 10 and not success:
                attempts += 1
                t = typist_factory()
                stats = engine.run_trajectory(t, null_policy, target_length=target_L)
                total_keys += stats['length']
                risk = stats['avg_step_risk']
                
                if risk <= T:
                    success = True
                    final_risk = risk
                    final_pwd = stats['password']
                else:
                    final_risk = risk
                    final_pwd = stats['password']
            
            ent_rate = mm.calculate_entropy_rate(final_pwd)
            raw_data.append({
                "Typist": typist_name,
                "Policy": "Post-hoc Rejection",
                "Password": final_pwd,
                "Length": len(final_pwd),
                "Final_Risk": final_risk,
                "Entropy_Rate": ent_rate,
                "Attempts": attempts,
                "Total_Keystrokes": total_keys,
                "Discarded_Keystrokes": total_keys - len(final_pwd),
                "Blocked_Steps": 0,
                "Success": success
            })
            
        # --- 3. Proactive Blocking ---
        print("  Running Proactive Intervention (Passboard)...")
        for i in range(N):
            target_L = int(np.random.choice(lengths))
            t = typist_factory()
            stats = engine.run_trajectory(t, proactive_policy, target_length=target_L)
            pwd = stats['password']
            avg_risk = stats['avg_step_risk']
            ent_rate = mm.calculate_entropy_rate(pwd)
            
            raw_data.append({
                "Typist": typist_name,
                "Policy": "Proactive Intervention",
                "Password": pwd,
                "Length": len(pwd),
                "Final_Risk": avg_risk,
                "Entropy_Rate": ent_rate,
                "Attempts": 1,
                "Total_Keystrokes": len(pwd),
                "Discarded_Keystrokes": 0,
                "Blocked_Steps": stats['blocked_steps'],
                "Success": True
            })

    # Save to DataFrame
    df = pd.DataFrame(raw_data)
    df.to_csv(os.path.join(output_dir, "humsec_simulation_data.csv"), index=False)
    print(f"\nRaw simulation data saved to {output_dir}/humsec_simulation_data.csv")

    # Generate Summary Metrics
    summary = df.groupby(["Typist", "Policy"]).agg(
        Mean_Risk=("Final_Risk", "mean"),
        Max_Risk=("Final_Risk", "max"),
        Unsafe_Pct=("Final_Risk", lambda x: sum(x > T) / len(x) * 100),
        Mean_Entropy_Rate=("Entropy_Rate", "mean"),
        Avg_Attempts=("Attempts", "mean"),
        Mean_Total_Keystrokes=("Total_Keystrokes", "mean"),
        Mean_Discarded_Keys=("Discarded_Keystrokes", "mean"),
        Mean_Blocked_Steps=("Blocked_Steps", "mean"),
        Failure_Rate_Pct=("Success", lambda x: sum(~x) / len(x) * 100)
    ).reset_index()
    
    summary.to_csv(os.path.join(output_dir, "humsec_summary.csv"), index=False)
    print(f"Summary metrics saved to {output_dir}/humsec_summary.csv")
    
    print("\nSummary Metrics Table:")
    print(summary.to_string(index=False))

    # --- PLOT GENERATION ---
    print("\nGenerating Paper-Quality Plots...")
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Plot 1: Security CDF (Comparison of Risk Cumulative Probability)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = {"Baseline": "#e74c3c", "Post-hoc Rejection": "#3498db", "Proactive Intervention": "#2ecc71"}
    
    for idx, (typist_name, _) in enumerate(typist_configs):
        ax = axes[idx]
        subset = df[df["Typist"] == typist_name]
        for policy in colors:
            pol_sub = subset[subset["Policy"] == policy]
            sns.ecdfplot(data=pol_sub, x="Final_Risk", label=policy, color=colors[policy], linewidth=2.5, ax=ax)
        
        ax.axvline(x=T, color="gray", linestyle=":", label="Security Threshold (80)")
        ax.set_title(f"{typist_name} Cohort")
        ax.set_xlabel("Average Password Risk Score (Lower is Safer)")
        if idx == 0:
            ax.set_ylabel("Cumulative Probability")
        ax.set_xlim(0, 100)
        ax.legend()
        
    plt.suptitle("Security Evaluation: Cumulative Distribution Function (CDF) of Password Risk", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "humsec_security_cdf.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/humsec_security_cdf.png")

    # Plot 2: Usability and Frustration (Total vs Discarded Keystrokes)
    # We display Boxplots showing usability overhead (Total Keystrokes needed to get accepted)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for idx, (typist_name, _) in enumerate(typist_configs):
        ax = axes[idx]
        subset = df[df["Typist"] == typist_name]
        
        # Create a melted dataframe for showing usability
        # We look at 'Total_Keystrokes' (work input)
        sns.boxplot(
            data=subset,
            x="Policy",
            y="Total_Keystrokes",
            palette=["#f1c40f", "#3498db", "#2ecc71"],
            ax=ax
        )
        ax.set_title(f"{typist_name} Cohort")
        ax.set_xlabel("Policy System")
        if idx == 0:
            ax.set_ylabel("Total Keystrokes Typed per Success")
            
    plt.suptitle("Usability Evaluation: Typing Effort (Keystroke Count) Comparison", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "humsec_usability_keystrokes.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/humsec_usability_keystrokes.png")

    # Plot 3: Entropy comparison (Strength Enhancement)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    for idx, (typist_name, _) in enumerate(typist_configs):
        ax = axes[idx]
        subset = df[df["Typist"] == typist_name]
        
        sns.barplot(
            data=subset,
            x="Policy",
            y="Entropy_Rate",
            palette=["#e74c3c", "#3498db", "#2ecc71"],
            estimator=np.mean,
            errorbar="sd",
            capsize=0.1,
            ax=ax
        )
        ax.set_title(f"{typist_name} Cohort")
        ax.set_xlabel("Policy System")
        ax.set_ylabel("Shannon Entropy Rate (Bits/Char)")
        
    plt.suptitle("Security Strength: Average Password Shannon Entropy Rate", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "humsec_entropy_rates.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/humsec_entropy_rates.png")
    
    print("\nAll plots generated and saved successfully.")

if __name__ == "__main__":
    main()
