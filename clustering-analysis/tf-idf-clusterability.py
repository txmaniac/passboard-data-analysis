import warnings
warnings.filterwarnings("ignore")
from pyclustertend import hopkins
from random import sample
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
import umap
from matplotlib import pyplot as plt
import json
from collections import defaultdict

class PasswordNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [str(x).strip().lower() for x in X]

class Clusterability:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data(self.file_path, limit=None) # Load all, sample later
        self.pipeline = self.preprocess_data()

    def load_data(self, file_path, limit=None):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            if limit:
                self.data = sample(f.read().splitlines(), limit)
            else:
                self.data = f.read().splitlines()
        print(f"Loaded {len(self.data)} passwords.")

    def preprocess_data(
        self,
        n_components=50, # SVD components for dense representation
        ngram_range=(3, 5)
    ):
        steps = [
            ("normalize", PasswordNormalizer()),
            ("tfidf", TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, min_df=5)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ("l2norm", Normalizer(norm="l2")),
        ]
        return Pipeline(steps)

    def hopkins_statistic(self, n_samples=10000):
        print("Computing Hopkins Statistic...")
        hopkins_stats = []
        for i in range(5): # Reduced iterations for speed
            subset = sample(self.data, min(len(self.data), n_samples))
            vectors = self.pipeline.fit_transform(subset)
            h_stat = hopkins(vectors, min(vectors.shape[0], n_samples))
            hopkins_stats.append(h_stat)
        return np.mean(hopkins_stats), np.std(hopkins_stats)

if __name__ == "__main__":
    c_obj = Clusterability("rockyou.txt")
    
    # 1. Hopkins Statistic
    # Use a sample for speed
    mean_hopkins, std_hopkins = c_obj.hopkins_statistic()
    print(f"Hopkins Statistic: {mean_hopkins:.4f} (+/- {std_hopkins:.4f})")
    
    # 2. Prepare Data for Clustering (Use a subset if dataset is huge, e.g. 50k)
    SAMPLE_SIZE = 100000
    subset_data = sample(c_obj.data, min(len(c_obj.data), SAMPLE_SIZE))
    X_reduced = c_obj.pipeline.fit_transform(subset_data)
    
    # 3. HDBSCAN Clustering
    print(f"Running HDBSCAN (min_cluster_size=125, min_samples=5)...")
    clusterer = HDBSCAN(min_cluster_size=125, min_samples=5, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(X_reduced)
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster Distribution: {dict(zip(unique, counts))}")
    
    # 4. t-SNE Visualization
    print("Running t-SNE for visualization...")
    tsne = TSNE(n_components=3, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_reduced)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # Noise in grey, smaller, more transparent
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], X_tsne[mask, 2], c='lightgrey', label='Noise', s=5, alpha=0.3)
        else:
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], X_tsne[mask, 2], label=f'Cluster {label}', s=10, alpha=0.6)
            
    ax.set_title(f"3D t-SNE Visualization of Password Clusters (HDBSCAN)")
    # Legend might be huge if many clusters, so maybe limit it or just don't show all
    if len(unique_labels) < 20:
        ax.legend()
    else:
        print(f"Too many clusters ({len(unique_labels)}) for legend.")
        
    plt.savefig("tsne_clusters_3d.png")
    print("Saved 3D t-SNE plot to tsne_clusters_3d.png")
    plt.close()

    # 5. UMAP Visualization
    print("Running UMAP for visualization...")
    reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=-1)
    X_umap = reducer.fit_transform(X_reduced)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], X_umap[mask, 2], c='lightgrey', label='Noise', s=5, alpha=0.3)
        else:
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], X_umap[mask, 2], label=f'Cluster {label}', s=10, alpha=0.6)

    ax.set_title(f"3D UMAP Visualization of Password Clusters (HDBSCAN)")
    if len(unique_labels) < 20:
        ax.legend()
    
    plt.savefig("umap_clusters_3d.png")
    print("Saved 3D UMAP plot to umap_clusters_3d.png")
    plt.close()
