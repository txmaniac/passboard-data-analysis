import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_security_analysis(csv_path="simulation_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Compute Derived Metrics
    # Entropy Rate -> Entropy Risk (0-100)
    # Risk = (1 - Rate/8.0) * 100
    df['final_entropy_risk'] = (1.0 - (df['final_entropy_rate'] / 8.0)) * 100.0
    df['final_entropy_risk'] = df['final_entropy_risk'].clip(0, 100)
    
    # Combined Risk
    df['combined_risk'] = 0.5 * df['final_density_risk'] + 0.5 * df['final_entropy_risk']
    
    scenarios = df['scenario'].unique()
    
    # Setup Figure: 3 cols (Density, Entropy, Combined) x N Rows (Scenarios)
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(15, 5 * len(scenarios)), constrained_layout=True)
    
    if len(scenarios) == 1:
        axes = np.array([axes]) # Ensure 2D array
        
    print("\n--- Security Analysis Report ---")
    
    for i, sc in enumerate(scenarios):
        sc_data = df[df['scenario'] == sc]
        conditions = sc_data['condition'].unique()
        
        print(f"\nScenario: {sc}")
        
        # Determine Axes for this row
        ax_dens = axes[i, 0]
        ax_ent = axes[i, 1]
        ax_comb = axes[i, 2]
        
        metrics = [
            ('final_density_risk', "Density Risk", ax_dens),
            ('final_entropy_risk', "Entropy Risk", ax_ent),
            ('combined_risk', "Combined Risk", ax_comb)
        ]
        
        for metric_col, title, ax in metrics:
            ax.set_title(f"{sc} - {title} CDF")
            ax.set_xlabel("Risk Score (0=Safe, 100=Risky)")
            ax.set_ylabel("CDF")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            
            for cond in conditions:
                cond_data = sc_data[sc_data['condition'] == cond]
                values = cond_data[metric_col].dropna()
                
                # CDF Plot
                sorted_vals = np.sort(values)
                yvals = np.arange(len(sorted_vals)) / float(len(sorted_vals) - 1)
                
                # Get stats for label
                mean_val = values.mean()
                label = f"{cond} (Mean: {mean_val:.1f})"
                
                lw = 2 if 'Blocking' in cond else 1.5
                ls = '-' if 'Blocking' in cond else '--'
                
                ax.plot(sorted_vals, yvals, label=label, linewidth=lw, linestyle=ls)
                
                # Print stats
                if metric_col == 'combined_risk':
                    print(f"  {cond:15s}: Mean Combined Risk = {mean_val:.2f}, Median = {values.median():.2f}")
            
            ax.legend(loc='lower right')
            
    plt.suptitle("Security Analysis: Distribution of Risk Scores", fontsize=16)
    
    out_file = "security_analysis_plots.png"
    plt.savefig(out_file)
    print(f"\nSaved plots to {out_file}")

if __name__ == "__main__":
    plot_security_analysis()
