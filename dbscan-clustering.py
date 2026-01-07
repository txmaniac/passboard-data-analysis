from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings("ignore")

def load_passwords(filepath="rockyou.txt", limit=None):
    print(f"Loading passwords from {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        passwords = f.read().splitlines()
    if limit:
        passwords = passwords[:limit]
    print(f"Loaded {len(passwords)} passwords.")
    return passwords

def load_pipeline():
    return Pipeline([
            # 1. Convert text to fixed-size hashed counts (n-grams 2 to 4)
            ('hasher', HashingVectorizer(
                analyzer='char', 
                ngram_range=(2, 4), 
                n_features=512,  # The fixed size of your vector
                alternate_sign=True # Improves performance by reducing collision bias
            )),
            
            # 2. Apply TF-IDF weighting
            ('tfidf', TfidfTransformer(use_idf=True)),
            
            # 3. L2 Normalization so all vectors exist on a unit sphere (essential for Cosine Distance)
            ('normalizer', Normalizer(norm='l2'))
        ])

def find_optimal_eps(X, k=10):
    print(f"Calculating Optimal Epsilon (k={k})...")
    # For sparse TF-IDF vectors (already L2 normalized), Euclidean distance is equivalent to Cosine ranking
    # And it supports faster KD-trees/optimizations in many sklearn backend implementations (though sparse matrices usually fallback)
    neigh = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])
    
    # K-Distance Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title(f"K-Distance Graph (k={k}) - TF-IDF")
    plt.xlabel("Points Sorted by Distance")
    plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
    plt.grid(True, alpha=0.3)
    plt.savefig('tfidf_kdist.png')
    plt.close()
    
    # Automatic "Elbow" / Knee detection
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
    
    print(f"Selecting {analysis_size} passwords for analysis...")
    # Use same seed logic if possible, or just fixed random.sample
    random.seed(42)
    target_passwords = random.sample(all_passwords, min(analysis_size, len(all_passwords)))
    
    # 2. Vectorize
    print("Vectorizing with TF-IDF and Hashing...")
    X = load_pipeline().fit_transform(target_passwords)
    print(f"Analysis Data Shape: {X.shape}")
    
    # 3. Find Optimal Epsilon
    eps = find_optimal_eps(X, k=10)
    
    # 4. Run DBSCAN
    print(f"Running DBSCAN (eps={eps:.4f}, min_samples=10)...")
    dbscan = DBSCAN(
        eps=eps,
        min_samples=10,
        metric='euclidean', # Since vectors are L2 normalized, Euclidean matches Cosine ranking
        n_jobs=-1
    )
    clusters = dbscan.fit_predict(X)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"Total Clusters Found: {n_clusters}")
    print(f"Noise Points (Outliers): {n_noise} ({n_noise/len(target_passwords)*100:.1f}%)")
    
    # 5. t-SNE Visualization (Comparable to FastText)
    print("\n--- t-SNE Visualization ---")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, n_jobs=-1, metric='euclidean')
    # Use dense array for t-SNE if memory allows (50k x 512 floats is ~200MB, totally fine)
    # Sparse t-SNE in sklearn works but 'pca' init requires dense
    X_dense = X.toarray()
    X_embedded = tsne.fit_transform(X_dense)
    
    plt.figure(figsize=(14, 10))
    # Plot noise in grey, clusters in colormap
    noise_mask = (clusters == -1)
    cluster_mask = ~noise_mask
    
    plt.scatter(X_embedded[noise_mask, 0], X_embedded[noise_mask, 1], 
                c='lightgrey', label='Noise', alpha=0.3, s=2)
    plt.scatter(X_embedded[cluster_mask, 0], X_embedded[cluster_mask, 1], 
                c=clusters[cluster_mask], cmap='Spectral', label='Clusters', alpha=0.6, s=5)
    
    plt.title(f"t-SNE Projection (TF-IDF, n={analysis_size}, Clusters={n_clusters}, Eps={eps:.3f})")
    plt.colorbar(label='Cluster ID')
    plt.legend(markerscale=5)
    plt.savefig('tfidf_tsne_50k.png')
    plt.close()
    print("Saved tfidf_tsne_50k.png")

if __name__ == "__main__":
    main()