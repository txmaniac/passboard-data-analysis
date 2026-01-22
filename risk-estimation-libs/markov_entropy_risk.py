import json
import math
import random
import os
from collections import defaultdict

class MarkovModel:
    def __init__(self, n, path=None):
        self.n = n
        self.path = path
        # Model structure: model[order][prefix][suffix] = count
        # model[0] is order 1 (unigram of prefix length 0?) -> No, typically:
        # If n=4, we might store context lengths 1..4.
        # Original implementation used list of dicts.
        self.model = [defaultdict(lambda: defaultdict(int)) for _ in range(n)] 
        self.vocab_size = 256
        self.alpha = 1.0
        
        if path and os.path.exists(path):
            self.load(path)

    def train(self, passwords):
        """Train the model on a list of passwords."""
        print(f"Training Markov Model (N={self.n}) on {len(passwords)} passwords...")
        for pwd in passwords:
            # Add start/end tokens? Or just raw chars?
            # Standard password modeling often uses raw chars.
            # Let's stick to raw chars for now.
            for i in range(len(pwd)):
                char = pwd[i]
                # Train contexts of length 0 to n-1?
                # i.e., for pwd="abc", 'a' has context "", 'b' has "a", 'c' has "ab".
                for order in range(self.n):
                    # Context length = order
                    if i < order:
                        continue
                        
                    context = pwd[i-order:i]
                    self.model[order][context][char] += 1
                    
    def save(self, path):
        """Save model to JSON."""
        # Convert defaultdicts to dicts for JSON
        serializable_model = []
        for d in self.model:
            serializable_model.append({k: dict(v) for k, v in d.items()})
            
        with open(path, "w") as f:
            json.dump(serializable_model, f)
            
    def load(self, path):
        """Load model from JSON."""
        print(f"Loading Markov Model from {path}...")
        with open(path, "r") as f:
            self.model = json.load(f)
            # JSON keys are strings, values are dicts.
            # We don't necessarily need to convert back to defaultdict if we handle lookups safely.

    def predict_next_probs(self, prefix, temperature=1.0):
        """
        Return probability distribution for the next character given prefix.
        Uses the longest available context (up to n) with backoff.
        """
        # 1. Find longest matching context
        # Check model[n-1] (longest) down to model[0] (shortest)
        context_counts = None
        
        # Available context length from prefix
        max_order = min(self.n, len(prefix))
        
        # Try orders from max_order down to 1
        # indices in self.model: 0 -> order 1 context (len 1??)
        # Usually self.model[k] stores context of length k+1? Or k?
        # Let's assume self.model[k] stores context of length k+1 (so model[3] is 4-gram context? no)
        # Let's standardize:
        # self.model[0] -> context length 0 (Unigram model, P(c))? Or context length 1 (Bigram)?
        # In `train`: `order` goes 0..n-1. 
        # `context = pwd[i-order:i]`. value of `order` is length.
        # So self.model[0] has context length 0 (empty string). P(c).
        # self.model[1] has context length 1. P(c|prev).
        # self.model[n-1] has context length n-1.
        
        found_counts = None
        
        for order in range(self.n - 1, -1, -1):
            if order > len(prefix):
                continue # Cannot use context larger than prefix
                
            context = prefix[len(prefix)-order:]
            
            # Look up context
            # self.model is list of dicts/defaultdicts
            # If loaded from JSON, it's list of dicts.
            layer = self.model[order]
            if context in layer:
                found_counts = layer[context]
                break
                
        if not found_counts:
            # Fallback to uniform if even unigram (order 0) fails (shouldn't happen if trained properly)
            return {chr(c): 1.0/256 for c in range(256)} # Simplified
            
        # 2. Convert counts to probs with Temperature
        total = sum(found_counts.values())
        probs = {}
        
        for char, count in found_counts.items():
            # Apply temperature: p_i = (count / total) ^ (1/T)
            # Then renormalize.
            p = count / total
            if temperature != 1.0:
                p = math.pow(p, 1.0 / temperature)
            probs[char] = p
            
        # Renormalize
        prob_sum = sum(probs.values())
        for char in probs:
            probs[char] /= prob_sum
            
        return probs

    def calculate_entropy_rate(self, password):
        """Calculate entropy rate (bits/char) for the password."""
        if not password: return 0.0
        
        total_entropy = 0.0
        for i in range(len(password)):
            prefix = password[:i]
            char = password[i]
            
            # Get probs for next char
            probs = self.predict_next_probs(prefix, temperature=1.0)
            
            p = probs.get(char, 1e-9) # Small prob smoothing
            total_entropy -= math.log2(p)
            
        return total_entropy / len(password)
