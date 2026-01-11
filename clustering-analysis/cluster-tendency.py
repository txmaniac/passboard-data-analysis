# Class computing the Hopkins Statistic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyclustertend import hopkins, ivat, vat
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

class ClusterTendency:
    def __init__(self, data):
        self.data = data
        self.pipeline = self.load_pipeline()

    def load_pipeline(self):
        password_pipeline = Pipeline([
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
        return password_pipeline

    def fit_hopkins(self):
        X = self.pipeline.fit_transform(self.data)
        # pyclustertend expects a DataFrame to use .sample()
        df = pd.DataFrame(X.toarray())
        return hopkins(df, len(self.data))

    def fit_ivat(self):
        X = self.pipeline.fit_transform(self.data)
        # ivat plots the heatmap
        ivat(X.toarray())
        plt.savefig('ivat_heatmap.png')
        plt.close()
        return "Check ivat_heatmap.png"

    def fit_vat(self):
        X = self.pipeline.fit_transform(self.data)
        # vat plots the heatmap
        vat(X.toarray())
        plt.savefig('vat_heatmap.png')
        plt.close()
        return "Check vat_heatmap.png"

with open("rockyou.txt", "r", encoding="utf-8") as f:
    leaked_passwords_list = f.read().splitlines()

leaked_passwords_list = leaked_passwords_list[:500]
# score = ClusterTendency(leaked_passwords_list).fit_hopkins()

# print("Hopkins Statistic: ", score)

score = ClusterTendency(leaked_passwords_list).fit_vat()
print("IVAT Statistic: ", score)