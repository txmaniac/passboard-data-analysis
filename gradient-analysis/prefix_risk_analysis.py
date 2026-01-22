import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import string
from scipy.stats import percentileofscore

# Add parent directory to path to import scalable-density-analysis
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from scalable_density_analysis.ann_density import ANNDensity
except ImportError:
    # Handle hyphenated directory name
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scalable-density-analysis'))
    from ann_density import ANNDensity

def get_keyboard_layout():
    # QWERTY layout coordinates for visualization
    # Row 1
    keys = {
        'q': (0, 2), 'w': (1, 2), 'e': (2, 2), 'r': (3, 2), 't': (4, 2), 'y': (5, 2), 'u': (6, 2), 'i': (7, 2), 'o': (8, 2), 'p': (9, 2),
        'a': (0.5, 1), 's': (1.5, 1), 'd': (2.5, 1), 'f': (3.5, 1), 'g': (4.5, 1), 'h': (5.5, 1), 'j': (6.5, 1), 'k': (7.5, 1), 'l': (8.5, 1),
        'z': (1, 0), 'x': (2, 0), 'c': (3, 0), 'v': (4, 0), 'b': (5, 0), 'n': (6, 0), 'm': (7, 0)
    }
    # Add digits if we want full keyboard, but usually QWERTY analysis focuses on letters.
    # User requested: lower/upper letters, digits. 
    # For heatmap, digits are usually a separate row.
    # Let's add digits row above QWERTY
    for i, d in enumerate("1234567890"):
        keys[d] = (i, 3)
        
    return keys

class RiskAnalyzer:
    def __init__(self, limit=100000):
        self.limit = limit
        self.k = 20
        # Locate rockyou.txt
        if os.path.exists("clustering-analysis/rockyou.txt"):
            fpath = "clustering-analysis/rockyou.txt"
        elif os.path.exists("gradient-analysis/rockyou.txt"):
            fpath = "gradient-analysis/rockyou.txt"
        elif os.path.exists("rockyou.txt"):
            fpath = "rockyou.txt"
        else:
             # Fallback if running from subdir
             fpath = "../clustering-analysis/rockyou.txt"

        # Use TF-IDF for density reference
        self.density_model = ANNDensity(limit=limit, k=self.k, embedding_type='tfidf', file_path=fpath)
        
    def fit(self):
        print(f"Training density model on {self.limit} passwords...")
        self.density_model.fit(backend='faiss')
        
        # Compute reference distribution of distances (sorted)
        # fit() builds index but doesn't store all distances automatically unless fit_plot called.
        # We need the distribution of distances for the training set to compute percentiles.
        print("Computing reference distance distribution...")
        # We can query a subset of the training data to estimate the distribution
        sample_size = min(100000, self.limit)
        idx = np.random.choice(self.density_model.data_transformed.shape[0], sample_size, replace=False)
        sample_vecs = self.density_model.data_transformed[idx]
        
        # Query k-th NN distances for this sample
        # Since self is included in index, we search for k+1 and take k-th column
        D, I = self.density_model.index.search(sample_vecs, self.k + 1)
        distances = np.sqrt(D)
        self.reference_distances = distances[:, self.k]
        self.reference_distances.sort()
        
        # Interpretation: Lower distance = Higher Density = Higher "Risk" (of being common)
        # If we define Risk R(p) as Density Percentile:
        # Distance d. Percentile of d in reference.
        # Low distance -> Top 1% of density (99th percentile risk).
        # We want R(p) \in [0, 100].
        # Let's say R(p) = 100 * P(RefDist >= d(p)).
        # If d(p) is 0 (very dense), RefDist >= 0 is 100%. Risk = 100.
        # If d(p) is huge (very sparse), RefDist >= d is 0%. Risk = 0.
        
    def get_risk(self, distance):
        # Calculate percentile of passwords that have a LARGER distance (are sparser).
        # Which effectively means the percentage of passwords that are LESS DENSE than this one.
        # High Risk = High Density = Small Distance.
        # Reference distances are sorted (small to large).
        # usage: np.searchsorted(a, v)
        # index i means i elements are smaller than distance.
        # (N-i) elements are larger or equal.
        # P(Ref >= d) = (N - i) / N
        
        idx = np.searchsorted(self.reference_distances, distance, side='left')
        n = len(self.reference_distances)
        risk = 100.0 * (n - idx) / n
        return risk

    def analyze_prefixes(self, prefixes):
        candidates = string.ascii_lowercase + string.digits
        # Optionally add uppercase
        # candidates += string.ascii_uppercase
        
        results = {}
        
        print(f"Analyzing {len(prefixes)} prefixes...")
        
        for p in prefixes:
            # 1. Base Risk R(p)
            queries = [p] + [p + c for c in candidates]
            distances = self.density_model.query(queries, k=self.k)
            # distances is array of shape (len, k), but query returns (len, k) distances?
            # Waait, query returns distances to k NNs. We want the k-th NN distance.
            # My query method returns array shape (N, k).
            # We want the last column (k-th neighbor distance).
            
            # Since query returns distances to k NNs, the measure is usually the k-th distance.
            # (Note: for external query, 'self' is not in training set, so we search k neighbors directly.
            # The k-th neighbor in training set is the density estimator).
            
            # Distance of p
            d_p = distances[0][-1] # Last column (k-th)
            r_p = self.get_risk(d_p)
            
            p_res = {
                'base_risk': r_p,
                'deltas': {},
                'risks': {}
            }
            
            # 2. Candidates
            for i, c in enumerate(candidates):
                d_cand = distances[i+1][-1]
                r_cand = self.get_risk(d_cand)
                delta = r_cand - r_p
                p_res['deltas'][c] = delta
                p_res['risks'][c] = r_cand
            
            results[p] = p_res
            
        return results

    def plot_heatmap(self, prefix, risk_data):
        layout = get_keyboard_layout()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Normalize deltas for color map
        # Delta ranges from -100 to +100 theoretically.
        # Red = Positive Delta (Risk Increase). Green = Negative Delta (Risk Decrease/Safe).
        
        deltas = risk_data['deltas']
        
        for key, pos in layout.items():
            if key not in deltas:
                continue
                
            delta = deltas[key]
            
            # Color logic
            # Safe (Green): Delta <= 0. Risky (Red): Delta > 0.
            # Color intensity based on magnitude.
            if delta > 0:
                # Red
                alpha = min(1.0, delta / 50.0) # Cap at 50% change
                color = (1.0, 0, 0, alpha)
                edge = 'red'
            else:
                # Green
                alpha = min(1.0, abs(delta) / 50.0)
                color = (0, 1.0, 0, alpha)
                edge = 'green'
                
            # Draw Key
            rect = patches.Rectangle((pos[0], pos[1]), 0.9, 0.9, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(pos[0] + 0.45, pos[1] + 0.45, key.upper(), ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(pos[0] + 0.45, pos[1] + 0.15, f"{delta:+.0f}", ha='center', va='center', fontsize=7)

        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 4.5)
        ax.axis('off')
        ax.set_title(f"Risk Heatmap for prefix: '{prefix}' (Base Risk: {risk_data['base_risk']:.1f})")
        
        out_file = f"heatmap_risk_{prefix}.png"
        plt.savefig(out_file)
        print(f"Saved heatmap to {out_file}")
        plt.close()

if __name__ == "__main__":
    analyzer = RiskAnalyzer(limit=5000000)
    analyzer.fit()
    
    # 1. Define Prefixes
    prefixes = [
        "pass", "admin", "love", "123456", "football", # Weak
        "dragon", "master", "michael", # Common
        "qwer", "asdf", "qwert", "asdfg" # Structural
    ]
    
    # 2. Analyze
    results = analyzer.analyze_prefixes(prefixes)
    
    # 3. Visualize & Report
    print("\n--- Escape Route Statistics ---")
    safe_counts = []
    
    for p, res in results.items():
        deltas = list(res['deltas'].values())
        safe_moves = [d for d in deltas if d <= 0]
        fraction_safe = len(safe_moves) / len(deltas)
        safe_counts.append(fraction_safe)
        
        print(f"Prefix: '{p:<10}' | Base Risk: {res['base_risk']:5.1f}% | Median Delta: {np.median(deltas):+5.1f} | Safe Moves: {fraction_safe*100:.0f}%")
        
        # Plot heatmap for subset
        if p in prefixes:
            analyzer.plot_heatmap(p, res)
            
    median_safe = np.median(safe_counts)
    iqr_safe = np.percentile(safe_counts, 75) - np.percentile(safe_counts, 25)
    
    print(f"\nOverall Median Safe Fraction: {median_safe*100:.1f}% (IQR: {iqr_safe*100:.1f}%)")
    
    # 4. Escape Route Logic Plot
    plt.figure(figsize=(8, 4))
    # Bar chart for each prefix
    names = list(results.keys())
    values = safe_counts
    
    # Sort by safe fraction
    sorted_pairs = sorted(zip(names, values), key=lambda x: x[1])
    sorted_names, sorted_values = zip(*sorted_pairs)
    
    bars = plt.barh(sorted_names, sorted_values, color='skyblue')
    plt.axvline(median_safe, color='red', linestyle='--', label=f'Median: {median_safe:.2f}')
    
    plt.title("Escape Route Analysis: Fraction of Safe Moves (Delta R <= 0)")
    plt.xlabel("Fraction of Next Keys that Reduce Risk")
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    
    # Add values
    for rect, val in zip(bars, sorted_values):
        plt.text(val + 0.01, rect.get_y() + rect.get_height()/2, f"{val:.0%}", va='center')
        
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig("escape_stats.png")
    print("Saved escape stats plot to escape_stats.png")
