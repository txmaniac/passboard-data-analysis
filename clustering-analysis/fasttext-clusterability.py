import warnings
warnings.filterwarnings("ignore")
from pyclustertend import hopkins
from random import sample
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from gensim.models import FastText
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
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

class CharNGramTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, n=3):
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for s in X:
            if len(s) < self.n:
                out.append([s])
            else:
                out.append([s[i:i+self.n] for i in range(len(s)-self.n+1)])
        return out

class FastTextEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        vector_size=128,
        window=3,
        min_count=10,
        negative=10,
        epochs=20,
        workers=4
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        self.model = FastText(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            negative=self.negative,
            sg=0,                 # CBOW (smoother clusters)
            workers=self.workers,
            min_n=3,
            max_n=5,
        )
        self.model.build_vocab(X)
        self.model.train(
            X,
            total_examples=len(X),
            epochs=self.epochs
        )
        return self

    def transform(self, X):
        dim = self.vector_size
        vectors = np.zeros((len(X), dim), dtype=np.float32)

        for i, tokens in enumerate(X):
            valid = [self.model.wv[t] for t in tokens if t in self.model.wv]
            if valid:
                vectors[i] = np.mean(valid, axis=0)

        return vectors

class Clusterability:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data(self.file_path, limit=50000)
        self.pipeline = self.preprocess_data()

    def load_data(self, file_path, limit=None):
        with open(file_path, "r") as f:
            if limit:
                self.data = sample(f.read().splitlines(), limit)
            else:
                self.data = f.read().splitlines()

    def preprocess_data(
        self,
        dim=128,
        ngram_n=4,
        pca_dim=60   # set None if you want raw embedding
    ):
        steps = [
            ("normalize", PasswordNormalizer()),
            ("tokenize", CharNGramTokenizer(n=ngram_n)),
            ("fasttext", FastTextEmbedder(vector_size=dim)),
            ("l2norm", Normalizer(norm="l2")),
        ]

        if pca_dim is not None:
            steps.append(("pca", PCA(n_components=pca_dim, random_state=42)))
            steps.append(("l2norm2", Normalizer(norm="l2")))

        return Pipeline(steps)

    def hopkins_statistic(self):
        hopkins_stats = []
        for i in range(10):
            self.load_data(self.file_path, limit=10000)
            vectors = self.pipeline.fit_transform(self.data)
            hopkins_stats.append(hopkins(vectors, 10000))
        return np.mean(hopkins_stats), np.std(hopkins_stats)

if __name__ == "__main__":
    c_obj = Clusterability("rockyou.txt")
    pipeline = c_obj.pipeline

    X_reduced = pipeline.fit_transform(c_obj.data)
    # print(f"Hopkins Statistic: {c_obj.hopkins_statistic()}")

    clusterer = HDBSCAN(
        min_cluster_size=300, # Further reduced to 20
        min_samples=10,       # Further reduced to 5
        metric='euclidean',
        cluster_selection_method='eom',
        allow_single_cluster=True,
    )
    labels = clusterer.fit_predict(X_reduced)
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster Distribution: {dict(zip(unique, counts))}")
    # PLOT THE CLUSTERING
    plt.figure(figsize=(10, 7))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(
            X_reduced[mask, 0], 
            X_reduced[mask, 1], 
            label=f"Cluster {label}" if label != -1 else "Noise",
            s=10, 
            alpha=0.5
        )
    plt.title("HDBSCAN Clustering Visualization (PCA Projection)")
    plt.legend(loc="best", markerscale=3)
    plt.savefig("fasttext_pca_clusters.png")
    plt.close()

    # t-SNE Visualization
    print("Running t-SNE for visualization...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_reduced)

    plt.figure(figsize=(12, 8))
    for label in np.unique(labels):
        mask = labels == label
        if label == -1:
             plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c='lightgrey', label='Noise', s=5, alpha=0.3)
        else:
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f'Cluster {label}', s=10, alpha=0.6)

    plt.title("t-SNE Visualization of Password Clusters (FastText)")
    if len(np.unique(labels)) < 20:
        plt.legend(loc="best", markerscale=3)
    plt.savefig("fasttext_tsne_clusters.png")
    print("Saved t-SNE plot to fasttext_tsne_clusters.png")
    plt.close()

    # UMAP Visualization
    print("Running UMAP for visualization...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
    X_umap = reducer.fit_transform(X_reduced)

    plt.figure(figsize=(12, 8))
    for label in np.unique(labels):
        mask = labels == label
        if label == -1:
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], c='lightgrey', label='Noise', s=5, alpha=0.3)
        else:
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], label=f'Cluster {label}', s=10, alpha=0.6)

    plt.title("UMAP Visualization of Password Clusters (FastText)")
    if len(np.unique(labels)) < 20:
        plt.legend(loc="best", markerscale=3)
    plt.savefig("fasttext_umap_clusters.png")
    print("Saved UMAP plot to fasttext_umap_clusters.png")
    plt.close()

    # # COMPUTE THE DBCV SCORE
    # validity_index = validity_index(X_reduced.astype(np.float64), clusterer.labels_)
    # print(f"Validity Index: {validity_index}")

    # # Save the cluster labels and the passwords belonging to each cluster in a json file
    # cluster_labels = clusterer.labels_
    # cluster_info = defaultdict(list)
    # for i in range(len(cluster_labels)):
    #     cluster_info[int(cluster_labels[i])].append(c_obj.data[i])

    # with open("cluster_passwords.json", "w") as f:
    #     json.dump(cluster_info, f)