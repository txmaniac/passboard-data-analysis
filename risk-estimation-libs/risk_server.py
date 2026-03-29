from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import os
import string
import numpy as np
from contextlib import asynccontextmanager

# Import our libs
from ann_density import ANNDensity
from markov_entropy_risk import MarkovModel

# Global Models
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Models on Startup
    print("Loading Risk Models...")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rockyou_risk_path = os.path.join(BASE_DIR, "rockyou_risk.txt")
    models_dir = os.path.join(BASE_DIR, "models")
    
    if not os.path.exists(rockyou_risk_path):
        # Fallback to main rockyou if split doesn't exist (sim run might not happened)
        # But we need training data.
        print(f"Warning: {rockyou_risk_path} not found. Using reduced functionality or creating empty?")
        # Just error out for now or define fallback
        fallback_path = os.path.join(BASE_DIR, "rockyou.txt")
        if os.path.exists(fallback_path):
             rockyou_risk_path = fallback_path
    
    # 1. ANNDensity
    try:
        print(f"Attempting to load cached models from {models_dir}...")
        ann = ANNDensity(k=20, file_path=None, limit=None, embedding_type='tfidf')
        ann.load_model(models_dir)
        print("Cached models loaded.")
    except Exception as e:
        print(f"Cache load failed ({e}). Training from scratch using {rockyou_risk_path}...")
        ann = ANNDensity(k=20, file_path=rockyou_risk_path, limit=None, embedding_type='tfidf')
        ann.fit(backend='faiss')
        # Auto-save for next time
        try:
            ann.save_model(models_dir)
        except Exception as se:
            print(f"Failed to save model: {se}")
            
    models['ann'] = ann
    
    # 2. Markov
    mm = MarkovModel(n=4)
    mm_path = os.path.join(models_dir, "markov_model.json")
    if os.path.exists(mm_path):
        try:
            print(f"Loading cached Markov model from {mm_path}...")
            mm.load(mm_path)
            print("Cached Markov loaded.")
        except Exception as e:
            print(f"Failed to load Markov ({e}). Training...")
            if os.path.exists(rockyou_risk_path):
                with open(rockyou_risk_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read().splitlines()
                mm.train(data)
                try:
                    mm.save(mm_path)
                except:
                    pass
    else:
        # Fallback train
        if os.path.exists(rockyou_risk_path):
            print(f"Training Markov from {rockyou_risk_path}...")
            with open(rockyou_risk_path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read().splitlines()
            mm.train(data)
            
    models['mm'] = mm
    
    print("Models Loaded.")
    yield
    # Cleanup
    models.clear()

app = FastAPI(lifespan=lifespan)

# CORS (important for Extension/Next.js access)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In prod, restrict to extension ID or localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BlockRequest(BaseModel):
    prefix: str
    candidates: Optional[List[str]] = None
    adaptive_enabled: bool = True

class BlockResponse(BaseModel):
    blocked_keys: List[str]
    threshold: float
    base_risk: float
    processing_time_ms: float

# Standard ASCII Candidates
ALL_CHARS = list(string.ascii_letters + string.digits + "!@#$%^&*")

@app.post("/predict_blocked", response_model=BlockResponse)
async def predict_blocked(req: BlockRequest):
    start_time = time.time()
    
    ann = models.get('ann')
    mm = models.get('mm')
    
    if not ann or not mm:
        raise HTTPException(status_code=503, detail="Models not loaded")

    candidates = req.candidates if req.candidates is not None else ALL_CHARS
    
    # 1. Calculate Base Risk of current prefix
    # Use the last few chars + context? Or risk of the prefix itself?
    # Usually "Risk of the prefix" means: How risky is what I typed so far?
    # ANNDensity works on strings.
    # If prefix is empty, Risk = 0.
    
    base_risk = 0.0
    if req.prefix:
        # Density Risk
        dist = ann.query([req.prefix], k=20)[0]
        dens_risk = ann.get_risk_percentile([dist])[0]
        
        # Entropy Risk
        # Rate of the whole prefix
        rate = mm.calculate_entropy_rate(req.prefix)
        ent_risk = max(0, min(100, (1.0 - (rate / 8.0)) * 100))
        
        base_risk = 0.5 * dens_risk + 0.5 * ent_risk
        
    # 2. Determine Threshold (Adaptive)
    # Default high threshold (permissive)
    # If Base Risk is high, lower the threshold (restrictive)
    # Formula from plan: T = 90 - (0.5 * BaseRisk)
    # Risk=0 -> T=90 (Block only >90 risk)
    # Risk=80 -> T=50 (Block >50 risk)
    
    if req.adaptive_enabled:
        threshold = 90.0 - (0.5 * base_risk)
        threshold = max(30.0, threshold) # Don't go too low (locking user out)
    else:
        threshold = 80.0
        
    # 3. Evaluate Candidates
    # We predict risk for (prefix + c)
    queries = [req.prefix + c for c in candidates]
    
    # 3a. Batch Density
    dists = ann.query(queries, k=20)
    dens_risks = ann.get_risk_percentile(dists)
    
    # 3b. Batch Entropy
    # Optimization: Use fast entropy update if possible, but full recalc is safe for 50 chars
    blocked = []
    
    for i, c in enumerate(candidates):
        # Entropy of NEW string
        full_str = queries[i]
        rate = mm.calculate_entropy_rate(full_str)
        ent_risk = max(0, min(100, (1.0 - (rate / 8.0)) * 100))
        
        combined_risk = 0.5 * dens_risks[i] + 0.5 * ent_risk
        
        if combined_risk > threshold:
            blocked.append(c)
            
    # 4. Escape Route
    # Ensure at least 10% allowed
    min_allowed = max(1, int(len(candidates) * 0.10))
    if len(candidates) - len(blocked) < min_allowed:
        # Too many blocked. Sort by risk and keep the 'safest' ones that were blocked?
        # Re-calc risks? We didn't store them all.
        # Ideally, we should have stored (char, risk).
        # Re-do simple relax: unblock random? No, unblock lowest risk.
        # Since we didn't store scores in the loop above for simplicity, let's just clear blocks
        # or implement proper sorting.
        # Let's Implement Correct Sorting to be robust.
        pass # To be fixed below
        
    # Re-implementation of Step 3 with sorting for Escape Route
    candidate_risks = []
    for i, c in enumerate(candidates):
        full_str = queries[i]
        rate = mm.calculate_entropy_rate(full_str)
        ent_risk = max(0, min(100, (1.0 - (rate / 8.0)) * 100))
        combined_risk = 0.5 * dens_risks[i] + 0.5 * ent_risk
        candidate_risks.append((c, combined_risk))
        
    # Sort by Risk (Low to High)
    candidate_risks.sort(key=lambda x: x[1])
    
    # Filter by Threshold
    final_blocked = []
    allowed_count = 0
    
    # We want to find the cut-off
    # Allowed: Risk <= Threshold
    # We also need AT LEAST min_allowed items.
    
    # If we just take top-K items as allowed where K >= min_allowed AND (Risk <= Threshold if possible)
    # The constraint "Risk < Threshold" is soft if "Count < Min" is hard.
    
    # Count how many pass threshold
    pass_threshold = sum(1 for _, r in candidate_risks if r <= threshold)
    
    # How many to allow?
    num_to_allow = max(pass_threshold, min_allowed)
    
    # The top `num_to_allow` are ALLOWED. The rest are BLOCKED.
    # Since list is sorted Low Risk -> High Risk, 
    # candidates[:num_to_allow] are Allowed.
    # candidates[num_to_allow:] are Blocked.
    
    blocked_part = candidate_risks[num_to_allow:]
    final_blocked = [c for c, _ in blocked_part]
    
    # Calculate processing time
    proc_time = (time.time() - start_time) * 1000
    
    return BlockResponse(
        blocked_keys=final_blocked,
        threshold=threshold,
        base_risk=base_risk,
        processing_time_ms=proc_time
    )

if __name__ == "__main__":
    import uvicorn
    # Run dev server
    uvicorn.run(app, host="0.0.0.0", port=8000)
