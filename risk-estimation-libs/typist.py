import random
import string
import numpy as np
from abc import ABC, abstractmethod
from markov_entropy_risk import MarkovModel

printable_chars = list(string.printable.strip()) # "012...abc...ABC...!"

class Typist(ABC):
    @abstractmethod
    def next_char(self, prefix, allowed_chars=None):
        """
        Produce the next character given the current prefix.
        If allowed_chars is provided (blocking active), must choose from allowed set.
        Returns: char or None (if stuck/done)
        """
        pass

class MarkovTypist(Typist):
    def __init__(self, model: MarkovModel, temperature=1.0):
        self.model = model
        self.temperature = temperature
        
    def next_char(self, prefix, allowed_chars=None):
        # 1. Get full distribution for next char
        probs_map = self.model.predict_next_probs(prefix, self.temperature)
        
        candidates = list(probs_map.keys())
        probabilities = list(probs_map.values())
        
        # 2. Filter if Blocking is active
        if allowed_chars is not None:
            # Intersection of candidates and allowed
            allowed_set = set(allowed_chars)
            
            new_candidates = []
            new_probs = []
            
            for c, p in zip(candidates, probabilities):
                if c in allowed_set:
                    new_candidates.append(c)
                    new_probs.append(p)
            
            if not new_candidates:
                # BLOCKED FULLY: No preferred char is allowed.
                # Fallback: Pick ANY allowed char? Or fail?
                # Simulation rule: If stuck, maybe pick uniformly from allowed?
                # Or just abort. Let's pick uniformly from allowed to 'escape' hard blocks.
                if not allowed_chars:
                    return None # Truly impossible
                return random.choice(allowed_chars)
            
            # Renormalize (Resampling Strategy)
            total = sum(new_probs)
            candidates = new_candidates
            probabilities = [p/total for p in new_probs]
            
        # 3. Sample
        return np.random.choice(candidates, p=probabilities)

class TemplateTypist(Typist):
    def __init__(self):
        # Fixed templates for simulation
        self.templates = [
            ["word", "digit", "digit"],          # e.g., "love99"
            ["word", "digit", "symbol"],         # e.g., "pass1!"
            ["caps_word", "digit", "digit"],     # e.g., "Love99"
            ["word", "year"],                    # e.g., "admin2020"
        ]
        self.common_words = ["password", "iloveyou", "princess", "admin", "football", "monkey", "jessica", "charlie", "dragon"]
        self.years = [str(y) for y in range(1980, 2025)]
        
        # Current state
        self.current_template = None
        self.current_plan = ""
        self.idx = 0
        
    def start_new_password(self):
        """Generate a full planned string at start."""
        template = random.choice(self.templates)
        parts = []
        for token in template:
            if token == "word":
                parts.append(random.choice(self.common_words))
            elif token == "caps_word":
                parts.append(random.choice(self.common_words).capitalize())
            elif token == "digit":
                parts.append(random.choice(string.digits))
            elif token == "symbol":
                parts.append(random.choice("!@#$%^&*"))
            elif token == "year":
                parts.append(random.choice(self.years))
                
        self.current_plan = "".join(parts)
        self.idx = 0
        
    def next_char(self, prefix, allowed_chars=None):
        if self.idx >= len(self.current_plan):
            return None # Done
            
        intended = self.current_plan[self.idx]
        
        # Check blocking
        if allowed_chars is not None and intended not in allowed_chars:
            # Strategy: Local adjustment
            # If digit blocked, try another digit.
            # If symbol blocked, try another symbol.
            # If word char blocked... hard.
            
            # Simple heuristic: Try to find a substitution in same class
            if intended.isdigit():
                candidates = [c for c in string.digits if c in allowed_chars]
            elif intended in "!@#$%^&*":
                candidates = [c for c in "!@#$%^&*" if c in allowed_chars]
            elif intended.islower():
                candidates = [c for c in string.ascii_lowercase if c in allowed_chars]
            else:
                candidates = allowed_chars
                
            if candidates:
                # Deviation
                choice = random.choice(candidates)
                # Update plan? No, just emit this char. 
                # Ideally we should update future plan if this breaks word structure, 
                # but for simple char-by-char sim, we just emit.
                self.idx += 1
                return choice
            else:
                # Fully frustrated
                return None
        
        self.idx += 1
        return intended
