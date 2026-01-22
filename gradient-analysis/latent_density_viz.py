import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from sklearn.manifold import TSNE

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
    
    # 20k limit for t-SNE performance
    LIMIT = 20000
    K = 20
    
    # Locate rockyou.txt
    if os.path.exists("clustering-analysis/rockyou.txt"):
        fpath = "clustering-analysis/rockyou.txt"
    elif os.path.exists("gradient-analysis/rockyou.txt"):
        fpath = "gradient-analysis/rockyou.txt"
    else:
        fpath = "rockyou.txt" 

    print(f"--- Latent Density Visualization (N={LIMIT}) ---")
    
    class LatentViz(knn_module.KDistanceDensity):
        def get_vectors_and_distances(self):
            print(f"Preprocessing data ({self.embedding_type})...")
            self.pipeline = self.preprocess_data()
            self.data_transformed = self.pipeline.fit_transform(self.data)
            
            print(f"Computing {self.k}-Nearest Neighbors...")
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=self.k+1, n_jobs=-1)
            neigh.fit(self.data_transformed)
            
            distances, indices = neigh.kneighbors(self.data_transformed)
            
            # Return vectors, k-th distance, and passwords
            return self.data_transformed, distances[:, self.k], np.array(self.data)

    # Initialize
    viz = LatentViz(k=K, limit=LIMIT, embedding_type='tfidf', file_path=fpath)
    
    # Dedup
    unique_passwords = list(set(viz.data))
    print(f"Deduplicated dataset: {len(viz.data)} -> {len(unique_passwords)}")
    viz.data = unique_passwords
    if len(viz.data) > LIMIT:
        import random
        viz.data = random.sample(viz.data, LIMIT)
        
    vectors, distances, passwords = viz.get_vectors_and_distances()
    
    # t-SNE Projection
    print("Running t-SNE (this may take a moment)...")
    # Removing n_iter just in case, and n_jobs (supported in recent sklearn but safer to remove if erroring)
    # Actually n_jobs is supported in 0.22+. The error was specifically n_iter.
    # Wait, usually n_iter IS the parameter.
    # Maybe the user has a weird version? I'll remove both to be safe.
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Scatter plot colored by distance
    # Low distance = High Density = Red (Hot)
    # High distance = Low Density = Blue (Cold)
    # We invert the colormap 'jet' or 'plasma' so hot=dense
    
    # Plotting
    plt.figure(figsize=(14, 8))
    
    # Sort indices by distance (Low -> High)
    sorted_idx = np.argsort(distances)
    n = len(distances)
    
    # Define Groups based on sorted rank
    # Dense: Top 10% (Lowest distances)
    dense_mask = np.zeros(n, dtype=bool)
    dense_mask[sorted_idx[:int(n*0.10)]] = True
    
    # Sparse: Bottom 10% (Highest distances)
    sparse_mask = np.zeros(n, dtype=bool)
    sparse_mask[sorted_idx[int(n*0.90):]] = True
    
    # Medium: The rest
    medium_mask = ~(dense_mask | sparse_mask)
    
    print("Plotting groups...")
    
    # 1. Plot Medium first (Background)
    plt.scatter(vectors_2d[medium_mask, 0], vectors_2d[medium_mask, 1],
                c='lightgray', s=10, alpha=0.3, label='Medium Density (10-90%)')
                
    # 2. Plot Sparse (Outliers)
    plt.scatter(vectors_2d[sparse_mask, 0], vectors_2d[sparse_mask, 1],
                c='blue', s=15, alpha=0.6, label='Sparse Region (Bottom 10%)')
                
    # 3. Plot Dense (Clusters)
    plt.scatter(vectors_2d[dense_mask, 0], vectors_2d[dense_mask, 1],
                c='red', s=20, alpha=0.8, label='Dense Region (Top 10%)')
                
    plt.legend(markerscale=2)
    plt.title(f"Latent Space (t-SNE) by Density Group (K={K})\nDense Passwords (Red) form Clusters; Sparse Passwords (Blue) are Outliers", fontsize=14)
    plt.axis('off')
    
    outfile = "latent_density_groups.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {outfile}")
