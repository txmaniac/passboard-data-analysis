from collections import defaultdict
import json
import math
import tqdm

class MarkovModel:
    def __init__(self, n, path=None):
        self.n = n
        self.path = path
        self.model = [defaultdict(lambda: defaultdict(int)) for _ in range(n)] if path is None else self.load(path)
        self.vocab_size = 256
        self.alpha = 1.0
        self.entropy_threshold = 1.0

    def load(self, path):
        with open(path, "r") as f:
            return json.load(f)
    
    def train(self, words):
        if self.path:
            return
        
        for w in tqdm.tqdm(words, desc="Training Markov Model: "):
            # character level n-gram model
            for i in range(1, self.n + 1):
                if len(w) <= i:
                    continue
                for j in range(len(w) - i):
                    prefix = w[j:j + i]
                    suffix = w[j + i]
                    self.model[i - 1][prefix][suffix] += 1

        # write dict to json file
        with open("markov_model.json", "w") as f:
            json.dump(self.model, f)

    def predict(self, prefix):
        # calculate the entropy of the prefix with backoff
        for i in range(self.n - 1, -1, -1):
            n_gram_prefix = prefix[-(i + 1):]
            if n_gram_prefix in self.model[i]:
                return self.entropy(n_gram_prefix, i)
        
        return math.log2(self.vocab_size)

    def top_k_weak(self, prefix, k):
        # calculate the top k weak suffixes
        entropies = []
        for i in range(self.n - 1 , -1, -1):
            n_gram_prefix = prefix[-(i + 1):]
            if n_gram_prefix in self.model[i]:
                suffixes = self.model[i][n_gram_prefix]
                for suffix in suffixes:
                    entropies.append((suffix, self.predict(n_gram_prefix + suffix)))
                break
        
        entropies.sort(key=lambda x: x[1])
        return entropies[:k]
    
    def entropy(self, prefix, i):
        # calculate the entropy of the prefix
        total_count = sum(self.model[i][prefix].values())
        denominator = total_count + (self.vocab_size * self.alpha)
        
        entropy = 0.0
        
        # Contribution from observed suffixes
        for suffix_count in self.model[i][prefix].values():
            prob = (suffix_count + self.alpha) / denominator
            entropy -= prob * math.log2(prob)
            
        # Contribution from unobserved suffixes
        observed_vocab_count = len(self.model[i][prefix])
        unobserved_vocab_count = self.vocab_size - observed_vocab_count
        
        if unobserved_vocab_count > 0:
            prob_unobserved = self.alpha / denominator
            entropy -= unobserved_vocab_count * prob_unobserved * math.log2(prob_unobserved)
        
        return entropy

if __name__ == "__main__":
    # load words from file
    with open("rockyou.txt", "r", encoding="utf-8") as f:
        # for now I want to only cater this markov model to work on ascii characters only
        words = f.read().splitlines()
    
    # train markov model
    mm = MarkovModel(4, "markov_model.json")
    mm.train(words)
    
    while True:
        prefix = input("Enter prefix: ")
        if prefix == "exit":
            break
        print(f"Entropy: {mm.predict(prefix)}")
        print(f"Top 20 weak suffixes: {mm.top_k_weak(prefix, 20)}")