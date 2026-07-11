import sys
import os
import argparse
import numpy as np
import pandas as pd
import random
import string

# Ensure we can load risk-estimation-libs
sys.path.append("risk-estimation-libs")

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
    print("Loading pre-trained models (this may take a few seconds due to 6GB FAISS index)...")
    ann = ANNDensity(k=20, file_path=None, limit=None, embedding_type='tfidf')
    ann.load_model("models")
    mm = MarkovModel(n=4)
    mm.load(os.path.join("models", "markov_model.json"))
    return ann, mm

def get_risk_evaluator(ann_model, mm_model):
    def evaluator(prefix, candidates):
        queries = [prefix + c for c in candidates]
        distances = ann_model.query(queries, k=20)
        density_risks = ann_model.get_risk_percentile(distances)
        entropy_risks = []
        for c in candidates:
             rate = mm_model.calculate_entropy_rate(prefix + c)
             r = max(0, min(100, (1.0 - (rate / 8.0)) * 100))
             entropy_risks.append(r)
        
        final_risks = {}
        for i, c in enumerate(candidates):
            combined = 0.5 * density_risks[i] + 0.5 * entropy_risks[i]
            final_risks[c] = combined
        return final_risks
    return evaluator

def main():
    parser = argparse.ArgumentParser(description="Security vs Usability Analysis: Post-hoc Rejection vs Proactive Blocking")
    parser.add_argument("--num-sims", type=int, default=100, help="Number of trajectories to simulate per cohort (default: 100)")
    parser.add_argument("--threshold", type=float, default=80.0, help="Risk threshold for blocking/rejection (default: 80.0)")
    args = parser.parse_args()

    N = args.num_sims
    T = args.threshold

    ann, mm = load_models()
    risk_func = get_risk_evaluator(ann, mm)
    engine = SimulationEngine(risk_func)
    
    null_policy = NullPolicy()
    proactive_policy = ThresholdPolicy(threshold=T, min_escape_percent=0.10)
    
    lengths = [8, 10, 12, 14]
    typists = [
        ("Markov Typist", lambda: MarkovTypist(mm)),
        ("Template Typist", lambda: RobustTemplateTypist())
    ]
    
    results = []

    print(f"\n=======================================================")
    print(f"RUNNING SECURITY ANALYSIS (N={N}, Threshold={T})")
    print(f"=======================================================")

    for typist_name, typist_factory in typists:
        print(f"\nSimulating {typist_name} cohort...")
        
        # 1. Baseline (No Intervention)
        print("  [1/3] Simulating Baseline (No Policy)...")
        baseline_risks = []
        baseline_entropies = []
        for _ in range(N):
            target_L = int(np.random.choice(lengths))
            t = typist_factory()
            stats = engine.run_trajectory(t, null_policy, target_length=target_L)
            baseline_risks.append(stats['avg_step_risk'])
            baseline_entropies.append(mm.calculate_entropy_rate(stats['password']))
            
        # 2. Post-hoc Rejection
        # The user types a password. If final risk > T, reject and re-type.
        # Max 10 attempts.
        print("  [2/3] Simulating Post-hoc Rejection (Check on Submit)...")
        posthoc_risks = []
        posthoc_attempts = []
        posthoc_total_keystrokes = []
        posthoc_discarded_keystrokes = []
        posthoc_entropies = []
        posthoc_failures = 0
        
        for _ in range(N):
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
                    
            posthoc_total_keystrokes.append(total_keys)
            posthoc_discarded_keystrokes.append(total_keys - len(final_pwd))
            posthoc_attempts.append(attempts)
            posthoc_risks.append(final_risk)
            posthoc_entropies.append(mm.calculate_entropy_rate(final_pwd))
            if not success:
                posthoc_failures += 1
                
        # 3. Proactive Intervention (Per-Keystroke Blocking)
        print("  [3/3] Simulating Proactive Intervention (Passboard)...")
        proactive_risks = []
        proactive_blocked_steps = []
        proactive_total_keystrokes = []
        proactive_entropies = []
        
        for _ in range(N):
            target_L = int(np.random.choice(lengths))
            t = typist_factory()
            stats = engine.run_trajectory(t, proactive_policy, target_length=target_L)
            proactive_risks.append(stats['avg_step_risk'])
            proactive_blocked_steps.append(stats['blocked_steps'])
            proactive_total_keystrokes.append(stats['length'])
            proactive_entropies.append(mm.calculate_entropy_rate(stats['password']))
            
        # Compile Metrics
        # Baseline
        mean_base_risk = np.mean(baseline_risks)
        max_base_risk = np.max(baseline_risks)
        pct_base_unsafe = sum(1 for r in baseline_risks if r > T) / N * 100
        mean_base_ent = np.mean(baseline_entropies)
        
        # Post-hoc
        mean_ph_risk = np.mean(posthoc_risks)
        max_ph_risk = np.max(posthoc_risks)
        mean_attempts = np.mean(posthoc_attempts)
        mean_ph_keys = np.mean(posthoc_total_keystrokes)
        ph_fail_rate = (posthoc_failures / N) * 100
        # Passwords that bypass security (either due to failure or relaxation)
        pct_ph_unsafe = sum(1 for r in posthoc_risks if r > T) / N * 100 
        mean_ph_ent = np.mean(posthoc_entropies)
        eff_ph = np.mean(lengths) / mean_ph_keys
        
        # Proactive
        mean_pr_risk = np.mean(proactive_risks)
        max_pr_risk = np.max(proactive_risks)
        mean_pr_keys = np.mean(proactive_total_keystrokes)
        mean_pr_blocked = np.mean(proactive_blocked_steps)
        # Passwords that bypass security (due to escape route relaxation)
        pct_pr_unsafe = sum(1 for r in proactive_risks if r > T) / N * 100
        mean_pr_ent = np.mean(proactive_entropies)
        eff_pr = np.mean(lengths) / mean_pr_keys
        
        # Frustration metric
        mean_ph_discarded = np.mean(posthoc_discarded_keystrokes)
        
        results.append({
            "Typist": typist_name,
            "Policy": "Baseline",
            "Mean Risk": mean_base_risk,
            "Max Risk": max_base_risk,
            "Unsafe %": pct_base_unsafe,
            "Mean Entropy": mean_base_ent,
            "Avg Attempts": 1.0,
            "Total Keystrokes": np.mean(lengths),
            "Discarded Keystrokes": 0.0,
            "Blocked Keystrokes": 0.0,
            "Efficiency": 1.0,
            "Frustration Index": 0.0
        })
        
        results.append({
            "Typist": typist_name,
            "Policy": "Post-hoc Rejection",
            "Mean Risk": mean_ph_risk,
            "Max Risk": max_ph_risk,
            "Unsafe %": pct_ph_unsafe, 
            "Mean Entropy": mean_ph_ent,
            "Avg Attempts": mean_attempts,
            "Total Keystrokes": mean_ph_keys,
            "Discarded Keystrokes": mean_ph_discarded,
            "Blocked Keystrokes": 0.0,
            "Efficiency": eff_ph,
            "Frustration Index": mean_ph_discarded  
        })
        
        results.append({
            "Typist": typist_name,
            "Policy": "Proactive Intervention (Ours)",
            "Mean Risk": mean_pr_risk,
            "Max Risk": max_pr_risk,
            "Unsafe %": pct_pr_unsafe,
            "Mean Entropy": mean_pr_ent,
            "Avg Attempts": 1.0,
            "Total Keystrokes": mean_pr_keys,
            "Discarded Keystrokes": 0.0,
            "Blocked Keystrokes": mean_pr_blocked,
            "Efficiency": eff_pr,
            "Frustration Index": mean_pr_blocked 
        })

    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("COMPARATIVE SECURITY & USABILITY ANALYSIS RESULTS")
    print("="*80)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    for typist_name, _ in typists:
        print(f"\nCohort: {typist_name}")
        cohort_df = df[df["Typist"] == typist_name].drop(columns=["Typist"])
        print(cohort_df.to_string(index=False))
        print("-" * 80)
        
    # Export csv
    df.to_csv("security_usability_analysis_results.csv", index=False)
    print("\nResults exported successfully to: security_usability_analysis_results.csv")

if __name__ == "__main__":
    main()
