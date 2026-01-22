import os
import random

SRC = "clustering-analysis/rockyou.txt"
RISK_FILE = "rockyou_risk.txt"
TYPIST_FILE = "rockyou_typist.txt"
TRAIN_TEST_SPLIT = 0.9

def regenerate():
    print(f"Reading {SRC}...")
    try:
        with open(SRC, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: {SRC} not found.")
        return

    total = len(lines)
    print(f"Total lines: {total}")
    
    # Shuffle? Rockyou is sorted by frequency.
    # Risk Engine should probably see the frequent ones?
    # Typist should also simulate frequent ones?
    # Standard practice: Shuffle to distribute.
    # But Rockyou is frequency list. Top is "123456".
    # If we shuffle, we mix them.
    # If we don't shuffle, Risk engine sees top 90%, Typist sees bottom 10% (obscure).
    # OR Risk sees top 12.6M, Typist sees 1.4M tail.
    # Ideally, we want a random split so Typist acts like general population.
    
    print("Shuffling...")
    random.seed(42) # Reproducible
    random.shuffle(lines)
    
    split_idx = int(total * TRAIN_TEST_SPLIT)
    risk_data = lines[:split_idx]
    typist_data = lines[split_idx:]
    
    print(f"Writing {len(risk_data)} to {RISK_FILE}...")
    with open(RISK_FILE, "w") as f:
        f.write("\n".join(risk_data))
        
    print(f"Writing {len(typist_data)} to {TYPIST_FILE}...")
    with open(TYPIST_FILE, "w") as f:
        f.write("\n".join(typist_data))
        
    print("Done.")

if __name__ == "__main__":
    regenerate()
