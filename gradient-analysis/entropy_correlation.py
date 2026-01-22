import sys
import os
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import importlib.util

# Ensure we can import from clustering-analysis for Markov Model if needed, 
# but I'll reimplement/copy MarkovModel class to avoid path dependency issues across dirs
# or import it if the user prefers. I'll copy for robustness as requested by "Show me implementation plan and ask incase of doubts" 
# (Correction: Plan said "Import/Re-implement". Copying is safer for a standalone script).

class MarkovModel:
    def __init__(self, n, path=None):
        self.n = n
        self.path = path
        self.model = [defaultdict(lambda: defaultdict(int)) for _ in range(n)] if path is None else self.load(path)
        self.vocab_size = 256
        self.alpha = 1.0

    def load(self, path):
        print(f"Loading Markov Model from {path}...")
        with open(path, "r") as f:
            data = json.load(f)
            # data is a list of dicts. We need to convert inner dicts to defaultdict(int) 
            # or handle them carefully. Typically json loads as dict.
            # For read-only entropy calculation, dict is fine.
            return data
    
    def predict(self, prefix):
        # calculate the entropy of the prefix with backoff
        # Actually we want the total entropy of a full password. 
        # The original predict method calculates entropy of the *next character* given a prefix.
        # We need a method to sum this up for the whole string.
        pass

    def entropy(self, prefix, i):
        # Helper to calculate entropy of next char given prefix of length i
        if self.path:
             # When loaded from JSON, self.model[i] is a dict
             if prefix not in self.model[i]:
                 return math.log2(self.vocab_size)
             context_counts = self.model[i][prefix]
        else:
             context_counts = self.model[i][prefix]
             
        total_count = sum(context_counts.values())
        denominator = total_count + (self.vocab_size * self.alpha)
        
        entropy = 0.0
        
        # Contribution from observed suffixes
        for suffix_count in context_counts.values():
            prob = (suffix_count + self.alpha) / denominator
            entropy -= prob * math.log2(prob)
            
        # Contribution from unobserved suffixes
        observed_vocab_count = len(context_counts)
        unobserved_vocab_count = self.vocab_size - observed_vocab_count
        
        if unobserved_vocab_count > 0:
            prob_unobserved = self.alpha / denominator
            entropy -= unobserved_vocab_count * prob_unobserved * math.log2(prob_unobserved)
        
        return entropy

    def calculate_total_entropy(self, password):
        total_entropy = 0.0
        # For each character, calculate entropy given previous context
        for i in range(len(password)):
            # Determine max context length available
            # We want to use the largest n-gram model available (up to self.n)
            # Strategy: Try context of length n, if not found, backoff to n-1...
            # Actually, standard Markov model just uses order-n context.
            # But the original code implementation has a hierarchical model (list of models for 1..n).
            # We should use the 'predict' logic which backs off.
            
            # Context for character at index i is password[max(0, i-self.n):i]
            # But the original `train` method trained models for context lengths 1 to n.
            # So for the first char, context is empty? Original code handles 1 to n.
            # The `predict` method in original code takes a prefix and finds entropy of NEXT char.
            # So to measure entropy of password[i], we provide prefix password[:i].
            
            prefix = password[:i]
            char_entropy = self.get_next_char_entropy(prefix)
            total_entropy += char_entropy
            
        return total_entropy

    def get_next_char_entropy(self, prefix):
        # Copy of predict method logic but returning entropy
        for i in range(self.n - 1, -1, -1):
            n_gram_prefix = prefix[-(i + 1):]
            # Check if this prefix exists in the model of order (i+1)
            # model index i corresponds to n-gram size i+1 (length of prefix)
            if self.path:
                if i < len(self.model) and n_gram_prefix in self.model[i]:
                     return self.entropy(n_gram_prefix, i)
            else:
                 if n_gram_prefix in self.model[i]:
                     return self.entropy(n_gram_prefix, i)
        
        # If no context found, return max entropy (uniform)
        return math.log2(self.vocab_size)

# Import KDistanceDensity dynamically since filename has hyphens
def import_k_distance_module():
    spec = importlib.util.spec_from_file_location("k_nn_distance", "gradient-analysis/k-nn-distance.py")
    if spec is None:
        # Try relative path
        spec = importlib.util.spec_from_file_location("k_nn_distance", "k-nn-distance.py")
    if spec is None:
        # Try absolute
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
    
    # Configuration
    LIMIT = 1000000 
    K = 20
    MODEL_PATH = "clustering-analysis/markov_model.json" 
    # Fallback to absolute if needed
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "../clustering-analysis/markov_model.json"
    
    print(f"--- Entropy vs K-NN Distance Analysis (N={LIMIT}) ---")
    
    # 1. Compute K-NN Distances
    print("Computing K-NN Distances...")
    # Initialize KDistanceDensity
    # We need to execute the fit logic but capture the distances instead of just plotting
    # The class in k-nn-distance.py is KDistanceDensity.
    # It has compute_k_distance_graph that calculates self.k_distances (sorted).
    # But we need distances corresponding to specific passwords, not sorted!
    # Sorting loses the mapping to the password.
    # We need to modify or subclass to get unsorted distances.
    
    class CorrelationDensity(knn_module.KDistanceDensity):
        def get_unsorted_distances(self):
            print(f"Preprocessing data ({self.embedding_type})...")
            self.pipeline = self.preprocess_data()
            self.data_transformed = self.pipeline.fit_transform(self.data)
            
            print(f"Computing {self.k}-Nearest Neighbors...")
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=self.k+1, n_jobs=-1) # k+1 to include self
            neigh.fit(self.data_transformed)
            
            # distances is (N, k+1)
            distances, indices = neigh.kneighbors(self.data_transformed)
            
            # The distance to the k-th neighbor is at index k (since 0 is self)
            # We want this for every point.
            return distances[:, self.k], self.data # Return raw passwords too to ensure alignment

    # Use TF-IDF as default
    # Point to clustering-analysis/rockyou.txt or gradient-analysis/rockyou.txt
    if os.path.exists("clustering-analysis/rockyou.txt"):
        fpath = "clustering-analysis/rockyou.txt"
    elif os.path.exists("gradient-analysis/rockyou.txt"):
        fpath = "gradient-analysis/rockyou.txt"
    else:
        fpath = "rockyou.txt"
        
    kd = CorrelationDensity(k=K, limit=LIMIT, embedding_type='tfidf', file_path=fpath)
    # Warning: KDistanceDensity constructor signature: (self, k=20, file_path="rockyou.txt", limit=100000)
    # It doesn't take embedding_type in constructor in the original file? 
    # Wait, I updated it in previous turn? 
    # Let me check previous edits. Yes, I updated KDistanceDensity to take embedding_type.
    
    knn_dists, passwords = kd.get_unsorted_distances()
    
    # 2. Compute Entropies
    print("Computing Entropies...")
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Markov model not found at {MODEL_PATH}. Please train it first using markov-model.py")
        sys.exit(1)
        
    mm = MarkovModel(4, MODEL_PATH)
    
    entropies = []
    print("Calculating entropy for each password...")
    import tqdm
    entropies_per_char = []
    total_entropies = []
    
    for pwd in tqdm.tqdm(passwords):
        e = mm.calculate_total_entropy(pwd)
        total_entropies.append(e)
        if len(pwd) > 0:
            entropies_per_char.append(e / len(pwd))
        else:
            entropies_per_char.append(0)
        
    entropies_per_char = np.array(entropies_per_char)
    total_entropies = np.array(total_entropies)
    
    # 3. Correlation Analysis (Using Entropy Rate as requested)
    # Switch correlation to use Entropy Rate (bits/char)
    
    corr_p, _ = pearsonr(entropies_per_char, knn_dists)
    corr_s, _ = spearmanr(entropies_per_char, knn_dists)
    
    print(f"\nPearson Correlation (Rate): {corr_p:.4f}")
    print(f"Spearman Correlation (Rate): {corr_s:.4f}")
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(entropies_per_char, knn_dists, alpha=0.5, s=2)
    plt.title(f"Password Entropy Rate vs {K}-NN Distance (TF-IDF)\nPearson: {corr_p:.2f}, Spearman: {corr_s:.2f}")
    plt.xlabel("Entropy Rate (bits/char)")
    plt.ylabel(f"{K}-NN Distance")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 8) 
    
    out_file = "entropy_vs_knn.png"
    plt.savefig(out_file)
    print(f"Saved scatter plot to {out_file}")

    # 5. Entropy Distribution Histogram (Per Character)
    plt.figure(figsize=(10, 5))
    plt.hist(entropies_per_char, bins=50, range=(0, 8), color='skyblue', edgecolor='black')
    plt.title(f"Distribution of Entropy Rate (N={LIMIT})")
    plt.xlabel("Entropy Rate (bits/char)")
    plt.ylabel("Frequency")
    plt.xlim(0, 8)
    plt.grid(True, alpha=0.3)
    
    hist_file = "entropy_hist.png"
    plt.savefig(hist_file)
    print(f"Saved entropy histogram to {hist_file}")
    
    # 6. Entropy Risk Histogram (Linear Mapping)
    # User requested histogram instead of uniform percentile.
    # We map Entropy Rate (0-8 bits) to Risk (100-0).
    # Risk = (1 - EntropyRate / 8.0) * 100
    
    max_entropy = 8.0
    entropy_risks = (1.0 - (entropies_per_char / max_entropy)) * 100.0
    # Clip just in case
    entropy_risks = np.clip(entropy_risks, 0, 100)
    
    plt.figure(figsize=(10, 5))
    plt.hist(entropy_risks, bins=50, range=(0, 100), color='salmon', edgecolor='black')
    plt.title(f"Distribution of Entropy Risk (Linear) (N={LIMIT})")
    plt.xlabel("Entropy Risk Score (0=Safe, 100=Risky)")
    plt.ylabel("Frequency")
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3)
    
    risk_hist_file = "entropy_risk_hist.png"
    plt.savefig(risk_hist_file)
    print(f"Saved entropy risk histogram to {risk_hist_file}")
