import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Add parent directory to path to import scalable-density-analysis
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import KDistanceDensity dynamically
def import_k_distance_module():
    spec = importlib.util.spec_from_file_location("k_nn_distance", "gradient-analysis/k-nn-distance.py")
    if spec is None:
        spec = importlib.util.spec_from_file_location("k_nn_distance", "k-nn-distance.py")
    if spec is None:
        # Absolute path fallback
        import os
        cwd = os.getcwd()
        path = os.path.join(cwd, "gradient-analysis", "k-nn-distance.py")
        spec = importlib.util.spec_from_file_location("k_nn_distance", path)
        
    module = importlib.util.module_from_spec(spec)
    sys.modules["k_nn_distance"] = module
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    knn_module = import_k_distance_module()
    
    LIMIT = 50000
    K = 20
    
    # Locate rockyou.txt
    if os.path.exists("clustering-analysis/rockyou.txt"):
        fpath = "clustering-analysis/rockyou.txt"
    elif os.path.exists("gradient-analysis/rockyou.txt"):
        fpath = "gradient-analysis/rockyou.txt"
    elif os.path.exists("rockyou.txt"):
        fpath = "rockyou.txt"
    else:
        fpath = "rockyou.txt" # Fallback

    print(f"--- Density Spectrum Analysis (N={LIMIT}) ---")
    
    # Subclass to extract unsorted distances and data
    class SpectrumDensity(knn_module.KDistanceDensity):
        def get_data_with_distances(self):
            print(f"Preprocessing data ({self.embedding_type})...")
            self.pipeline = self.preprocess_data()
            self.data_transformed = self.pipeline.fit_transform(self.data)
            
            print(f"Computing {self.k}-Nearest Neighbors...")
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=self.k+1, n_jobs=-1)
            neigh.fit(self.data_transformed)
            
            distances, indices = neigh.kneighbors(self.data_transformed)
            
            # Return k-th distance and valid passwords
            return distances[:, self.k], np.array(self.data)

    # Initialize first to load data
    algo = SpectrumDensity(k=K, limit=LIMIT, embedding_type='tfidf', file_path=fpath)
    
    # Deduplicate data to ensure we measure structural density, not frequency
    # (Unless we want to measure frequency, but "123456" appearing 500 times is different from "123456" being close to "123457")
    # For a "Spectrum" of the language, dedup is usually preferred.
    
    unique_passwords = list(set(algo.data))
    print(f"Deduplicated dataset: {len(algo.data)} -> {len(unique_passwords)}")
    
    # We must reload/re-fit on the unique dataset
    # Easiest way is to instantiate a new algo with explicit data (if API allows)
    # The current API loads from file. We can override self.data before calling get_data_with_distances.
    
    algo.data = unique_passwords
    # Limit to N if needed (random sample of uniques)
    if len(algo.data) > LIMIT:
        import random
        algo.data = random.sample(algo.data, LIMIT)
        
    distances, passwords = algo.get_data_with_distances()
    
    # Sort by distance (Dense -> Sparse)
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_passwords = passwords[sorted_indices]
    
    # Plotting
    plt.figure(figsize=(14, 8))
    
    # Plot the curve
    plt.plot(np.arange(len(sorted_distances)), sorted_distances, color='#4A90E2', linewidth=2, label=f'{K}-NN Distance')
    
    # Identify Regions
    n = len(sorted_distances)
    indices = {
        'Dense (Top 1%)': slice(0, int(n*0.01)),
        'Common (10-15%)': slice(int(n*0.10), int(n*0.15)),
        'Median (45-55%)': slice(int(n*0.45), int(n*0.55)),
        'Sparse (90-100%)': slice(int(n*0.90), n)
    }
    
    colors = ['green', 'yellowgreen', 'orange', 'red']
    
    print("\n--- Density Spectrum Examples ---")
    
    y_text_pos = 0.8
    x_text_pos = 0.05
    
    for i, (label, sl) in enumerate(indices.items()):
        region_pwds = sorted_passwords[sl]
        region_dists = sorted_distances[sl]
        
        # Pick 5 random examples from this region
        examples = np.random.choice(region_pwds, 5, replace=False)
        avg_dist = np.mean(region_dists)
        
        print(f"\n[{label}] Avg Dist: {avg_dist:.4f}")
        for ex in examples:
            print(f"  - {ex}")
            
        # Add text box to plot
        example_str = "\n".join(examples)
        box_text = f"{label}\nAvg Dist: {avg_dist:.2f}\n\n{example_str}"
        
        plt.text(x_text_pos + (i * 0.25), 0.2, box_text, 
                 transform=plt.gca().transAxes, 
                 fontsize=9, verticalalignment='bottom',
                 bbox=dict(boxstyle="round,pad=0.5", fc=colors[i], alpha=0.3))
                 
        # Highlight region on curve
        plt.axvspan(sl.start, sl.stop, color=colors[i], alpha=0.1)

    plt.title(f"Password Density Spectrum (Unique N={len(sorted_distances)})", fontsize=14)
    plt.xlabel(f"Rank (0 = Densest, {n} = Sparsest)", fontsize=12)
    plt.ylabel(f"{K}-NN Distance (TF-IDF)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    outfile = "density_spectrum.png"
    plt.savefig(outfile)
    print(f"\nSaved annotated plot to {outfile}")
