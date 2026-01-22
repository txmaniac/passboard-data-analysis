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
    
    LIMIT = 100000
    K = 20
    
    # Locate rockyou.txt
    if os.path.exists("clustering-analysis/rockyou.txt"):
        fpath = "clustering-analysis/rockyou.txt"
    elif os.path.exists("gradient-analysis/rockyou.txt"):
        fpath = "gradient-analysis/rockyou.txt"
    else:
        fpath = "rockyou.txt" 

    print(f"--- Histogram Peaks Analysis (N={LIMIT}) ---")
    
    class HistPeaks(knn_module.KDistanceDensity):
        def get_data_with_distances(self):
            print(f"Preprocessing data ({self.embedding_type})...")
            self.pipeline = self.preprocess_data()
            self.data_transformed = self.pipeline.fit_transform(self.data)
            
            print(f"Computing {self.k}-Nearest Neighbors...")
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=self.k+1, n_jobs=-1)
            neigh.fit(self.data_transformed)
            
            distances, indices = neigh.kneighbors(self.data_transformed)
            return distances[:, self.k], np.array(self.data)

    # Initialize
    viz = HistPeaks(k=K, limit=LIMIT, embedding_type='tfidf', file_path=fpath)
    
    # Dedup
    unique_passwords = list(set(viz.data))
    print(f"Deduplicated dataset: {len(viz.data)} -> {len(unique_passwords)}")
    viz.data = unique_passwords
    if len(viz.data) > LIMIT:
        import random
        viz.data = random.sample(viz.data, LIMIT)
        
    distances, passwords = viz.get_data_with_distances()
    
    # Plotting
    plt.figure(figsize=(12, 7))
    
    # Histogram
    counts, bins, patches = plt.hist(distances, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Find Regions based on Percentiles
    p10 = np.percentile(distances, 10)
    p45 = np.percentile(distances, 45)
    p55 = np.percentile(distances, 55)
    p90 = np.percentile(distances, 90)
    
    # Collect examples
    dense_examples = passwords[distances <= p10]
    median_examples = passwords[(distances >= p45) & (distances <= p55)]
    sparse_examples = passwords[distances >= p90]
    
    # Sample
    dense_sample = np.random.choice(dense_examples, 5, replace=False)
    median_sample = np.random.choice(median_examples, 5, replace=False)
    sparse_sample = np.random.choice(sparse_examples, 5, replace=False)
    
    print("\n--- Low Distance (Dense) Examples ---")
    for p in dense_sample: print(p)

    print("\n--- Median Distance (Common) Examples ---")
    for p in median_sample: print(p)
    
    print("\n--- High Distance (Sparse) Examples ---")
    for p in sparse_sample: print(p)
    
    # Annotate on Plot
    
    # 1. Dense Box (Left)
    dense_text = "Dense / High Risk\n(Low Distance)\n\n" + "\n".join(dense_sample)
    plt.text(0.02, 0.75, dense_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="red", alpha=0.15))

    # 2. Median Box (Center)
    median_text = "Median / Medium Risk\n(Avg Distance)\n\n" + "\n".join(median_sample)
    plt.text(0.37, 0.95, median_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="orange", alpha=0.15))
             
    # 3. Sparse Box (Right)
    sparse_text = "Sparse / Low Risk\n(High Distance)\n\n" + "\n".join(sparse_sample)
    plt.text(0.75, 0.75, sparse_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="green", alpha=0.15))
             
    # Arrows to regions
    # Arrow to dense (left tail)
    plt.annotate('Dense Region', xy=(p10/2, max(counts)*0.2), xytext=(p10, max(counts)*0.5),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=8)

    # Arrow to median (center)
    mid_x = (p45 + p55) / 2
    plt.annotate('Median Region', xy=(mid_x, max(counts)*0.8), xytext=(mid_x, max(counts)*0.6),
                 arrowprops=dict(facecolor='orange', shrink=0.05), fontsize=8, ha='center')
                 
    # Arrow to sparse (right tail)
    avg_sparse =  (p90 + max(distances))/2
    plt.annotate('Sparse Region', xy=(avg_sparse, max(counts)*0.2), 
                 xytext=(p90, max(counts)*0.5),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=8)

    plt.title(f"Distribution of K-NN Distances with Examples (N={len(distances)})", fontsize=14)
    plt.xlabel(f"{K}-NN Distance", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    outfile = "histogram_peaks.png"
    plt.savefig(outfile)
    print(f"\nSaved visualization to {outfile}")
