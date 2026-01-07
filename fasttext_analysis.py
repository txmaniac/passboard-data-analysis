import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import FastText
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from pyclustertend import hopkins, ivat
import random
import warnings

# Suppress warnings for clearer output
warnings.filterwarnings("ignore")

def load_passwords(filepath="rockyou.txt", limit=None):
    print(f"Loading passwords from {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        passwords = f.read().splitlines()
    if limit:
        passwords = passwords[:limit]
    print(f"Loaded {len(passwords)} passwords.")
    return passwords

def train_fasttext(passwords, vector_size=100, window=3, min_count=1):
    print("Training FastText model...")
    # Train on list of passwords treating them as "sentences"
    # This allows FastText to learn from the character n-grams of the passwords
    model = FastText(vector_size=vector_size, window=window, min_count=min_count, sentences=[passwords], epochs=10)
    print("FastText training complete.")
    return model

def vectorize_passwords(model, passwords):
    print("Vectorizing passwords...")
    vectors = []
    for pwd in passwords:
        try:
            vectors.append(model.wv[pwd])
        except:
            pass 
    return np.array(vectors)

def find_optimal_eps(X, k=10):
    print(f"Calculating Optimal Epsilon (k={k})...")
    neigh = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])
    
    # K-Distance Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title(f"K-Distance Graph (k={k}) - FastText")
    plt.xlabel("Points Sorted by Distance")
    plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
    plt.grid(True, alpha=0.3)
    plt.savefig('fasttext_kdist.png')
    plt.close()
    
    # Automatic "Elbow" / Knee detection
    # Find the point with maximum distance from the line connecting start and end
    n_points = len(k_distances)
    all_coords = np.vstack((range(n_points), k_distances)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    best_idx = np.argmax(dist_to_line)
    
    optimal_eps = k_distances[best_idx]
    print(f"Optimal Epsilon found: {optimal_eps:.4f}")
    return optimal_eps

def main():
    # 1. Load Data
    train_size = 100000 
    analysis_size = 50000 
    
    all_passwords = load_passwords(limit=train_size)
    
    # 2. Train Model
    model = train_fasttext(all_passwords, vector_size=50) 

    # 3. Prepare Analysis Vector Set
    print(f"Selecting {analysis_size} passwords for analysis...")
    target_passwords = random.sample(all_passwords, min(analysis_size, len(all_passwords)))
    X = vectorize_passwords(model, target_passwords)
    
    # 4. Normalize for Euclidean-based DBSCAN (Equivalent to Cosine)
    print("Normalizing vectors (L2)...")
    X = normalize(X, norm='l2')
    print(f"Analysis Data Shape: {X.shape}")

    # 5. Hopkins & iVAT (ON SUBSET ONLY - 50k is too slow for O(N^2))
    subset_size = 2000
    print(f"\n--- Cluster Tendency (Subset n={subset_size}) ---")
    X_subset = X[:subset_size]
    
    # Hopkins
    df = pd.DataFrame(X_subset)
    hopkins_score = hopkins(df, X_subset.shape[0])
    print(f"Hopkins Statistic: {hopkins_score:.4f}")

    # iVAT
    print("Generating iVAT Heatmap...")
    ivat(X_subset)
    plt.title(f"iVAT (n={subset_size}) - FastText")
    plt.savefig('fasttext_ivat_subset.png')
    plt.close()
    print("Saved fasttext_ivat_subset.png")

    # 6. DBSCAN on FULL 50k
    print(f"\n--- DBSCAN Clustering (n={len(X)}) ---")
    
    # Find Epsilon
    eps = find_optimal_eps(X, k=10)
    
    print(f"Running DBSCAN (eps={eps:.4f}, min_samples=10)...")
    dbscan = DBSCAN(eps=eps, min_samples=10, metric='euclidean', n_jobs=-1)
    clusters = dbscan.fit_predict(X)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"Total Clusters Found: {n_clusters}")
    print(f"Noise Points (Outliers): {n_noise} ({n_noise/len(X)*100:.1f}%)")

    # 7. t-SNE Visualization with Cluster Colors
    print("\n--- t-SNE Visualization ---")
    # For 50k, we can run t-SNE but it might be slow. 
    # Use 'pca' init and fewer iterations if needed, or Barnes-Hut (default).
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, n_jobs=-1)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(14, 10))
    # Plot noise in grey, clusters in colormap
    # Create mask
    noise_mask = (clusters == -1)
    cluster_mask = ~noise_mask
    
    plt.scatter(X_embedded[noise_mask, 0], X_embedded[noise_mask, 1], 
                c='lightgrey', label='Noise', alpha=0.3, s=2)
    plt.scatter(X_embedded[cluster_mask, 0], X_embedded[cluster_mask, 1], 
                c=clusters[cluster_mask], cmap='Spectral', label='Clusters', alpha=0.6, s=5)
    
    plt.title(f"t-SNE Projection (n={len(X)}, Clusters={n_clusters}, Eps={eps:.3f})")
    plt.colorbar(label='Cluster ID')
    plt.legend(markerscale=5)
    plt.savefig('fasttext_tsne_50k.png')
    plt.close()
    print("Saved fasttext_tsne_50k.png")

if __name__ == "__main__":
    main()
