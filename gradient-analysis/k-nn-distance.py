from random import sample
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import FastText
import numpy as np

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


import warnings
warnings.filterwarnings("ignore")


class KDistanceDensity:
    def __init__(self, k=20, file_path="rockyou.txt", limit=10000, embedding_type='tfidf'):
        self.k = k
        self.limit = limit
        self.embedding_type = embedding_type
        self.load_data(file_path)

    def load_data(self, file_path: str):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            if self.limit:
                self.data = sample(f.read().splitlines(), self.limit)
            else:
                self.data = f.read().splitlines()

        print(f"Loaded {len(self.data)} passwords.")

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

    def fit_plot(self):
        self.nn = NearestNeighbors(n_neighbors=self.k)
        self.pipeline = self.preprocess_data()
        self.data = self.pipeline.fit_transform(self.data)

        self.nn.fit(self.data)
        self.k_distances = self.nn.kneighbors(self.data)[0][:, self.k-1]

        self.k_distances.sort()

        # histogram plot and percentile-based plot to show the k-distance distribution
        plt.hist(self.k_distances, bins=50)
        plt.show()

        plt.plot(self.k_distances)
        plt.show()

if __name__ == '__main__':
    print("Running with TF-IDF...")
    kdd_tfidf = KDistanceDensity(limit=500000, embedding_type='tfidf')
    kdd_tfidf.fit_plot()

    print("Running with FastText...")
    kdd_fasttext = KDistanceDensity(limit=500000, embedding_type='fasttext')
    kdd_fasttext.fit_plot()