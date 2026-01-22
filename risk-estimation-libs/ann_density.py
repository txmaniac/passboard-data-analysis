from random import sample
import time
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import FastText
import warnings
import warnings
import os
import joblib
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
    def __init__(self, k=20, file_path="rockyou.txt", limit=100000, embedding_type='tfidf'):
        self.k = k
        self.limit = limit
        self.embedding_type = embedding_type
        # Data loading is now explicit via load_data or passing to fit
        if file_path and os.path.exists(file_path):
            self.load_data(file_path)
        else:
            self.data = []

    def load_data(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                if self.limit:
                    self.data = sample(f.read().splitlines(), self.limit)
                else:
                    self.data = f.read().splitlines()
            print(f"Loaded {len(self.data)} passwords for ANNDensity.")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = []

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
        print(f"Fitting ANNDensity model ({self.embedding_type})...")
        if not self.data:
            raise ValueError("No data loaded for ANNDensity fit")
            
        start_time = time.time()
        self.pipeline = self.preprocess_data()
        self.data_transformed = self.pipeline.fit_transform(self.data)
        
        # Ensure float32
        self.data_transformed = self.data_transformed.astype(np.float32)
        print(f"Preprocessing done in {time.time() - start_time:.2f}s")
        
        d = self.data_transformed.shape[1]
        self.backend = backend
        
        if backend == 'faiss':
            self.index = faiss.IndexHNSWFlat(d, 32) 
            self.index.add(self.data_transformed)
            print(f"Faiss Index built with {self.data_transformed.shape[0]} items.")
            
        # Compute reference distance distribution for Percentile Risk
        print("Computing reference distance distribution...")
        # Since we use HNSW, we can query the index with the data itself
        # Search for k+1 neighbors (self included)
        D, I = self.index.search(self.data_transformed, self.k + 1)
        distances = np.sqrt(D)
        # The k-th neighbor distance is at column k
        self.reference_distances = distances[:, self.k]
        self.reference_distances.sort()
        print("Reference distribution computed.")

    def query(self, query_strings, k=None):
        """
        Query the nearest neighbors for new strings.
        Returns distances to k nearest neighbors.
        """
        if k is None:
            k = self.k
            
        if not query_strings:
            return np.array([])
            
        # Transform query strings
        query_transformed = self.pipeline.transform(query_strings)
        query_transformed = query_transformed.astype(np.float32)
        
        if self.backend == 'faiss':
            # We want the k-th neighbor distance. 
            # Note: For new queries, the 1st neighbor is NOT self.
            # So if we want the "k-th nearest neighbor distance", we ask for k neighbors
            # and take the last one? Or distance to the k-th neighbor in the set?
            # Standard definition: Distance to the k-th NN.
            # search returns sorted distances. column k-1 is the k-th neighbor.
            D, I = self.index.search(query_transformed, k)
            distances = np.sqrt(D)
            return distances[:, k-1] # k-th neighbor distance

    def get_risk_percentile(self, distances):
        """
        Convert distances to risk percentile (0-100).
        Low Distance = High Density = High Risk.
        High Distance = Low Density = Low Risk.
        
        Percentile calculation:
        We want to know: What % of the reference population is SPARSER than this?
        Or: What % of population is DENSER than this?
        
        If distance is 0 (very dense), it is denser than 100% of population -> Risk 100.
        If distance is large (sparse), it is denser than 0% of population -> Risk 0.
        
        So Risk = Percentile Rank of Distance (High Distance = High Rank) INVERTED?
        
        Standard np.searchsorted gives rank.
        Rank 0 -> Dist < min (Super Dense) -> We want High Risk.
        Rank N -> Dist > max (Super Sparse) -> We want Low Risk.
        
        So Risk = 100 * (1 - Rank / N).
        """
        ranks = np.searchsorted(self.reference_distances, distances)
        n = len(self.reference_distances)
        risk_scores = 100.0 * (1.0 - (ranks / n))
        return risk_scores

    def save_model(self, directory="models"):
        os.makedirs(directory, exist_ok=True)
        # Save Pipeline
        joblib.dump(self.pipeline, os.path.join(directory, "pipeline.joblib"))
        # Save Reference Distances
        np.save(os.path.join(directory, "ref_dists.npy"), self.reference_distances)
        # Save Faiss Index
        if self.backend == 'faiss':
            faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        print(f"Model saved to {directory}")

    def load_model(self, directory="models"):
        p_pipe = os.path.join(directory, "pipeline.joblib")
        p_dists = os.path.join(directory, "ref_dists.npy")
        p_index = os.path.join(directory, "faiss.index")
        
        if not (os.path.exists(p_pipe) and os.path.exists(p_dists) and os.path.exists(p_index)):
            raise FileNotFoundError("Model files not found")
            
        print(f"Loading model from {directory}...")
        self.pipeline = joblib.load(p_pipe)
        self.reference_distances = np.load(p_dists)
        self.index = faiss.read_index(p_index)
        self.backend = 'faiss'
        print("Model loaded successfully.")
