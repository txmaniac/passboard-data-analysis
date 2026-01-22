import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from zxcvbn import zxcvbn
from scipy.stats import spearmanr
import tqdm

def compare_zxcvbn(csv_path="simulation_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Loading simulation results...")
    df = pd.read_csv(csv_path)
    
    # Compute Combined Risk
    df['final_entropy_risk'] = (1.0 - (df['final_entropy_rate'] / 8.0)) * 100.0
    df['final_entropy_risk'] = df['final_entropy_risk'].clip(0, 100)
    df['combined_risk'] = 0.5 * df['final_density_risk'] + 0.5 * df['final_entropy_risk']
    
    # Compute ZXCVBN Scores
    print(f"Computing ZXCVBN scores for {len(df)} passwords...")
    zxcvbn_scores = []
    # Using tqdm for progress
    for pwd in tqdm.tqdm(df['password'].astype(str)):
        try:
            res = zxcvbn(pwd)
            zxcvbn_scores.append(res['score'])
        except Exception:
            zxcvbn_scores.append(0) # Fail safe
            
    df['zxcvbn_score'] = zxcvbn_scores
    
    # Analysis
    # Correlation (Risk should be INVERSELY correlated with ZXCVBN score)
    # Risk: High (100) = Bad
    # ZXCVBN: High (4) = Good
    # So we expect NEGATIVE correlation.
    corr, pval = spearmanr(df['combined_risk'], df['zxcvbn_score'])
    print(f"\nSpearman Correlation (Risk vs ZXCVBN): {corr:.4f} (p={pval:.4e})")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Boxplot of Risk Score grouped by ZXCVBN Score
    sns.boxplot(x='zxcvbn_score', y='combined_risk', data=df, palette="viridis")
    
    plt.title(f"Combined Risk Score vs ZXCVBN Score\n(Spearman Correlation: {corr:.2f})")
    plt.xlabel("ZXCVBN Score (0=Weakest, 4=Strongest)")
    plt.ylabel("Combined Risk Score (0=Safe, 100=Risky)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, 100)
    
    # Invert Y axis? No, keep standard. High Risk = High Y. Low ZXCVBN = Low X.
    # We expect a downward trend.
    
    out_file = "zxcvbn_comparison.png"
    plt.savefig(out_file)
    print(f"Saved comparison plot to {out_file}")
    
    # Summary Table
    print("\nMean Risk Score per ZXCVBN Level:")
    print(df.groupby('zxcvbn_score')['combined_risk'].agg(['mean', 'std', 'count']))

if __name__ == "__main__":
    compare_zxcvbn()
