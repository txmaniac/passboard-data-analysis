from ann_density import ANNDensity
from markov_entropy_risk import MarkovModel
import os
import time

def train_and_save():
    print("Initializing Training...")
    rockyou_risk_path = "rockyou_risk.txt"
    os.makedirs("models", exist_ok=True)
    
    # 1. ANNDensity
    ann_files = ["models/pipeline.joblib", "models/ref_dists.npy", "models/faiss.index"]
    if all(os.path.exists(f) for f in ann_files):
        print("ANNDensity models already exist. Skipping training.")
    else:
        print("Training ANNDensity...")
        ann = ANNDensity(k=20, file_path=rockyou_risk_path, limit=None, embedding_type='tfidf')
        ann.fit(backend='faiss')
        print("Saving ANNDensity model...")
        ann.save_model("models")
    
    # 2. Markov
    mm_path = os.path.join("models", "markov_model.json")
    if os.path.exists(mm_path):
        print("Markov model already exists. Skipping training.")
    else:
        print("Training Markov Model (N=4)...")
        mm = MarkovModel(n=4)
        with open(rockyou_risk_path, "r", encoding="utf-8", errors="ignore") as f:
            # Load all data for Markov too
            data = f.read().splitlines()
        mm.train(data)
        print("Saving Markov Model...")
        mm.save(mm_path)
    
    print("Done.")

if __name__ == "__main__":
    train_and_save()
