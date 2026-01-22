import numpy as np

class BlockingPolicy:
    def get_allowed_chars(self, prefix, candidates, risk_evaluator):
        """
        Return subset of candidates that are allowed.
        risk_evaluator: function(prefix, candidates) -> dict{char: score}
        """
        raise NotImplementedError

class NullPolicy(BlockingPolicy):
    def get_allowed_chars(self, prefix, candidates, risk_evaluator):
        # Baseline: Measure risk for logging, but allow everything
        # We might trigger risk_evaluator just to log risks if needed, 
        # but policy strictly returns all candidates.
        return candidates

class ThresholdPolicy(BlockingPolicy):
    def __init__(self, threshold=80.0, min_escape_percent=0.10):
        self.threshold = threshold
        self.min_escape_percent = min_escape_percent # e.g. 10% of keyboard must be open
        
    def get_allowed_chars(self, prefix, candidates, risk_evaluator):
        # 1. Compute risks
        risks_map = risk_evaluator(prefix, candidates)
        
        # 2. Sort candidates by risk (Ascending = Safe to Risky)
        # We want to keep low risk.
        sorted_candidates = sorted(candidates, key=lambda c: risks_map.get(c, 100))
        
        # 3. Apply Threshold
        allowed = [c for c in sorted_candidates if risks_map.get(c, 100) < self.threshold]
        
        # 4. Check Escape Route
        min_count = max(1, int(len(candidates) * self.min_escape_percent))
        
        if len(allowed) < min_count:
            # Must relax. Since we sorted by risk, just take the top-k safest.
            allowed = sorted_candidates[:min_count]
            
        return allowed
