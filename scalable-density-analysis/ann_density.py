from random import sample
import time
import numpy as np
import faiss
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import FastText
import warnings

warnings.filterwarnings("ignore")

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


class ANNDensity:
    def __init__(self, k=20, file_path="gradient-analysis/rockyou.txt", limit=100000, embedding_type='tfidf'):
        self.k = k
        self.limit = limit
        self.embedding_type = embedding_type
        self.load_data(file_path)

    def load_data(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                if self.limit:
                    self.data = sample(f.read().splitlines(), self.limit)
                else:
                    self.data = f.read().splitlines()
            print(f"Loaded {len(self.data)} passwords.")
        except FileNotFoundError:
            # Fallback for running from different cwd
            alt_path = file_path.replace("../gradient-analysis/", "")
            with open(alt_path, "r", encoding="utf-8", errors="ignore") as f:
                if self.limit:
                    self.data = sample(f.read().splitlines(), self.limit)
                else:
                    self.data = f.read().splitlines()
            print(f"Loaded {len(self.data)} passwords form {alt_path}.")

    def preprocess_data(
        self,
        n_components=50, # SVD components for dense representation
        ngram_range=(3, 5),
        vector_size=128
    ):
        if self.embedding_type == 'fasttext':
            steps = [
                ("tokenize", CharNGramTokenizer(n=4)),
                ("fasttext", FastTextEmbedder(vector_size=vector_size)),
                ("l2norm", Normalizer(norm="l2")),
            ]
        else:
            steps = [
                ("tfidf", TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, min_df=5)),
                ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
                ("l2norm", Normalizer(norm="l2")),
            ]
        return Pipeline(steps)

    def fit(self, backend='faiss'):
        print(f"Preprocessing data ({self.embedding_type})...")
        start_time = time.time()
        self.pipeline = self.preprocess_data()
        self.data_transformed = self.pipeline.fit_transform(self.data)
        
        # Ensure float32
        self.data_transformed = self.data_transformed.astype(np.float32)
        print(f"Preprocessing done in {time.time() - start_time:.2f}s")
        
        d = self.data_transformed.shape[1]
        self.backend = backend
        
        if backend == 'rust':
            # Rust backend handles its own index building usually, 
            # but here it does k-nn search directly.
            # To support 'query_new', we might need to expose a build_index method in rust
            # or passes reference data every time (slow).
            # For now, we mainly support Faiss for flexible querying.
            pass
        
        if backend == 'faiss':
            # Faiss requires float32 numpy arrays (already ensured)
            # Using HNSW for Approximate Nearest Neighbors
            self.index = faiss.IndexHNSWFlat(d, 32) 
            self.index.add(self.data_transformed)
            print(f"Index built with {self.data_transformed.shape[0]} items.")

    def query(self, query_strings, k=None):
        """
        Query the nearest neighbors for new strings.
        Returns distances to k nearest neighbors.
        """
        if k is None:
            k = self.k
            
        # Transform query strings
        # We need to use the SAME pipeline
        # Note: TfidfVectorizer transforms strings.
        query_transformed = self.pipeline.transform(query_strings)
        query_transformed = query_transformed.astype(np.float32)
        
        if self.backend == 'faiss':
            D, I = self.index.search(query_transformed, k)
            distances = np.sqrt(D)
            return distances

        if self.backend == 'rust':
             # Fallback/TODO if needed
             raise NotImplementedError("Rust backend query not yet implemented for dynamic queries")

    def fit_plot(self, backend='faiss'):
        self.fit(backend)
        
        print(f"Searching {self.k}-Nearest Neighbors using {backend.upper()}...")
        start_time = time.time()
        
        if backend == 'rust':
            try:
                import rust_density
                self.k_distances = rust_density.compute_knn(self.data_transformed, self.k)
                self.k_distances.sort()
            except ImportError:
                print("Rust extension not found! Fallback to Faiss.")
                backend = 'faiss'
                self.backend = 'faiss'
                d = self.data_transformed.shape[1]
                self.index = faiss.IndexHNSWFlat(d, 32) 
                self.index.add(self.data_transformed)
        
        if backend == 'faiss':
            # Search for k+1 neighbors (self is included)
            D, I = self.index.search(self.data_transformed, self.k + 1)
            distances = np.sqrt(D)
            self.k_distances = distances[:, self.k]
            self.k_distances.sort()

        print(f"Search done in {time.time() - start_time:.2f}s")

        # Histogram
        plt.figure(figsize=(10, 5))
        plt.hist(self.k_distances, bins=50)
        plt.title(f"Histogram of Approximate {self.k}-NN Distances ({self.embedding_type}, {backend})")
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.savefig(f"ann_hist_{self.embedding_type}_{backend}.png")
        print(f"Saved histogram to ann_hist_{self.embedding_type}_{backend}.png")

        # Sorted K-Distance Graph
        plt.figure(figsize=(10, 5))
        plt.plot(self.k_distances)
        plt.title(f"Sorted approximate K-Distance Graph ({self.k}-NN, {self.embedding_type}, {backend})")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{self.k}-th Nearest Neighbor Distance")
        plt.grid(True)
        plt.savefig(f"ann_k_distance_{self.embedding_type}_{backend}.png")
        print(f"Saved plot to ann_k_distance_{self.embedding_type}_{backend}.png")

if __name__ == '__main__':
    # Using a larger limit to demonstrate scalability
    LIMIT = 1000000 
    
    print(f"--- Running Scalable ANN Analysis (Limit={LIMIT}) ---")
    
    print("\n[TF-IDF] - Faiss Backend")
    ann_tfidf = ANNDensity(limit=LIMIT, embedding_type='tfidf')
    ann_tfidf.fit_plot(backend='faiss')

    print("\n[TF-IDF] - Rust Backend")
    ann_tfidf.fit_plot(backend='rust')
    
    # print("\n[FastText]") 
    # Uncomment to run fasttext (might be slower to train)
    # ann_fasttext = ANNDensity(limit=LIMIT, embedding_type='fasttext')
    # ann_fasttext.fit_plot()
