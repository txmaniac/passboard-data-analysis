import sys
sys.path.append("risk-estimation-libs")
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ann_density import ANNDensity
from markov_entropy_risk import MarkovModel
from simulator import SimulationEngine
from typist import MarkovTypist, TemplateTypist
from policy import NullPolicy, ThresholdPolicy
import string
import matplotlib.patches as patches

# Setup Fonts for "Paper Quality"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['figure.dpi'] = 300

def load_models():
    print("Loading Models...")
    ann = ANNDensity(k=20, file_path=None, limit=None, embedding_type='tfidf')
    ann.load_model("models")
    mm = MarkovModel(n=4)
    mm.load(os.path.join("models", "markov_model.json"))
    return ann, mm

def get_risk_evaluator(ann_model, mm_model):
    def evaluator(prefix, candidates):
        queries = [prefix + c for c in candidates]
        distances = ann_model.query(queries, k=20)
        density_risks = ann_model.get_risk_percentile(distances)
        entropy_risks = []
        for c in candidates:
             rate = mm_model.calculate_entropy_rate(prefix + c)
             r = max(0, min(100, (1.0 - (rate / 8.0)) * 100))
             entropy_risks.append(r)
        
        final_risks = {}
        for i, c in enumerate(candidates):
            combined = 0.5 * density_risks[i] + 0.5 * entropy_risks[i]
            final_risks[c] = combined
        return final_risks
    return evaluator

# 1. Reuse Density Distribution (Histogram of distances)
def plot_reuse_density(ann):
    print("Plotting 1: Reuse Density Distribution (Empirical)...")
    dists = ann.reference_distances
    # Sample 100k for Histogram
    sample_dists = np.random.choice(dists, 100000, replace=False)
    
    plt.figure(figsize=(8, 5))
    sns.histplot(sample_dists, kde=True, color="purple", bins=50)
    plt.title("Distribution of Password Reuse Density (k-NN Distance)")
    plt.xlabel("k-NN Distance (Lower = Higher Reuse Density)")
    plt.ylabel("Frequency")
    plt.axvline(x=np.percentile(sample_dists, 10), color='red', linestyle='--', label='High Risk (Top 10%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("paper_plot_1_reuse_density.png")
    plt.close()

# 2. Reuse Risk vs Predictability (Scatter)
def plot_risk_scatter(ann, mm):
    print("Plotting 2: Reuse Risk vs Predictability (Empirical)...")
    with open("rockyou_typist.txt", "r", encoding="utf-8", errors="ignore") as f:
        passwords = [line.strip() for line in f if len(line.strip()) > 0][:5000]
        
    risks = []
    entropies = []
    
    examples = {
        "password123": ("High Density", "High Predictability"), 
        "Summer2024!": ("Low Density", "High Predictability"), 
        "xP9#kQ2@": ("Low Density", "Low Predictability")
    }
    
    dists = ann.query(passwords, k=20)
    density_risks = ann.get_risk_percentile(dists)
    
    for i, pwd in enumerate(passwords):
        rate = mm.calculate_entropy_rate(pwd)
        entropies.append(rate)
        
    plt.figure(figsize=(8, 6))
    plt.scatter(density_risks, entropies, alpha=0.1, s=10, color="gray")
    
    for pwd, (desc1, desc2) in examples.items():
        d = ann.query([pwd], k=20)[0]
        dr = ann.get_risk_percentile([d])[0]
        er = mm.calculate_entropy_rate(pwd)
        plt.scatter([dr], [er], s=100, marker='x', label=f"{pwd}")
        plt.text(dr+2, er, pwd, fontsize=9, fontweight='bold')

    plt.title("Reuse Risk (Density) vs Predictability (Entropy)")
    plt.xlabel("Reuse/Density Risk Percentile (High = Common)")
    plt.ylabel("Entropy Rate (Low = Predictable)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("paper_plot_2_scatter.png")
    plt.close()

# 3. Risk Evolution (Line Plot)
def plot_risk_evolution(ann, mm):
    print("Plotting 3: Risk Evolution (Empirical)...")
    target = "password"
    prefixes = [target[:i] for i in range(1, len(target)+1)]
    
    risks = []
    for p in prefixes:
        d_dist = ann.query([p])[0]
        d_risk = ann.get_risk_percentile([d_dist])[0]
        rate = mm.calculate_entropy_rate(p)
        e_risk = max(0, min(100, (1.0 - (rate / 8.0)) * 100))
        combined = 0.5 * d_risk + 0.5 * e_risk
        risks.append(combined)
        
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(target)+1), risks, marker='o', linewidth=2, color="crimson")
    plt.xticks(range(1, len(target)+1), list(target))
    plt.title(f"Risk Evolution During Typing: '{target}'")
    plt.xlabel("Keystroke")
    plt.ylabel("Total Risk Score (0-100)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("paper_plot_3_evolution.png")
    plt.close()

# 4. Targeted Key Blocking (Keyboard Schematic)
def plot_keyboard_example(ann, mm):
    print("Plotting 4: Keyboard Blocking Example (Empirical Policy)...")
    rows = [
        "1234567890-=",
        "qwertyuiop[]",
        "asdfghjkl;'",
        "zxcvbnm,./"
    ]
    prefix = "pass"
    candidates = list(string.ascii_letters + string.digits + "!@#$%^&*")
    
    # Use ACTUAL Policy Logic
    risk_func = get_risk_evaluator(ann, mm)
    policy = ThresholdPolicy(threshold=80.0, min_escape_percent=0.10)
    
    # Get Allowed
    allowed = policy.get_allowed_chars(prefix, candidates, risk_func)
    blocked_chars = set(candidates) - set(allowed)
            
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_aspect('equal')
    ax.axis('off')
    
    key_w = 1
    key_h = 1
    spacing = 0.2
    
    y = len(rows) * (key_h + spacing)
    for row_idx, row in enumerate(rows):
        x = (row_idx * 0.5) 
        for char in row:
            if char in blocked_chars:
                face = "#ffcccc"
                edge = "red"
            else:
                face = "white"
                edge = "black"
                
            rect = patches.Rectangle((x, y), key_w, key_h, linewidth=1, edgecolor=edge, facecolor=face)
            ax.add_patch(rect)
            ax.text(x + key_w/2, y + key_h/2, char, ha='center', va='center', fontsize=12, fontweight='bold')
            x += key_w + spacing
        y -= (key_h + spacing)
        
    plt.title(f"Blocked Keys after prefix '{prefix}' (Red = Blocked)")
    plt.xlim(-1, 15)
    plt.ylim(0, 6)
    plt.savefig("paper_plot_4_keyboard.png")
    plt.close()

# 5. Intervention Timing (Empirical Simulation)
def plot_intervention_timing(ann, mm):
    print("Plotting 5: Intervention Timing (Empirical)...")
    
    files = ["rockyou_typist.txt", "../rockyou_typist.txt"]
    typist_file = next((f for f in files if os.path.exists(f)), None)
    
    mm_typist = MarkovModel(n=4)
    lengths = []
    if typist_file:
        with open(typist_file, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read().splitlines()
        mm_typist.train(data[:50000])
        # Collect lengths (4-20)
        # for p in data[:10000]:
        #     if 4 <= len(p) <= 20:
        #         lengths.append(len(p))

        for p in data[:10000]:
            if 8 <= len(p) <= 20:
                lengths.append(len(p))
        mm_typist.save(os.path.join("models", "mm_typist.json"))
    elif os.path.exists(os.path.join("models", "mm_typist.json")):
        mm_typist.load(os.path.join("models", "mm_typist.json"))

    if not lengths:
        lengths = [8, 10, 12]

    typist = MarkovTypist(mm_typist)
    risk_func = get_risk_evaluator(ann, mm)
    engine = SimulationEngine(risk_func)
    
    proactive_timings = []
    # Use Adaptive? Or T80? 
    # Paper plot 4 used T80. Let's stick to T80 for consistency.
    policy_block = ThresholdPolicy(threshold=80.0, min_escape_percent=0.10)
    
    print("Simulating Proactive Group (N=200 with natural lengths)...")
    for _ in range(200):
        target_L = int(np.random.choice(lengths))
        stats = engine.run_trajectory(typist, policy_block, target_length=target_L)
        if stats['first_blocked_at'] is not None:
            proactive_timings.append(stats['first_blocked_at'])
            
    post_hoc_timings = []
    policy_null = NullPolicy()
    
    print("Simulating Post-hoc Group (N=200 with natural lengths)...")
    for _ in range(200):
        target_L = int(np.random.choice(lengths))
        stats = engine.run_trajectory(typist, policy_null, target_length=target_L)
        # User Feedback: Post-hoc effectively intervenes/checks at the end, regardless of risk.
        post_hoc_timings.append(stats['length'])
            
    data = [proactive_timings, post_hoc_timings]
    
    plt.figure(figsize=(6, 6))
    plt.boxplot(data, labels=['Proactive (Real-time)', 'Post-hoc (On Submit)'], patch_artist=True)
    plt.ylabel("Keystroke Index of First Intervention")
    plt.title("Comparison of Intervention Position (Empirical)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("paper_plot_5_timing.png")
    plt.close()

# 6. Total Risk Distribution (Empirical Simulation)
def plot_outcomes_kde(ann, mm):
    print("Plotting 6: Risk Outcomes (Empirical)...")
    
    mm_typist = MarkovModel(n=4)
    files = ["rockyou_typist.txt", "../rockyou_typist.txt"]
    typist_file = next((f for f in files if os.path.exists(f)), None)
    if typist_file:
        with open(typist_file, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read().splitlines()
        mm_typist.train(data[:50000])
        
    typist = MarkovTypist(mm_typist)
    risk_func = get_risk_evaluator(ann, mm)
    engine = SimulationEngine(risk_func)
    
    baseline_risks = []
    proactive_risks = []
    
    print("Simulating Baseline for KDE (N=300)...")
    for _ in range(300):
        stats = engine.run_trajectory(typist, NullPolicy(), target_length=12)
        dist = ann.query([stats['password']])[0]
        r = ann.get_risk_percentile([dist])[0]
        baseline_risks.append(r)
        
    print("Simulating Proactive for KDE (N=300)...")
    for _ in range(300):
        stats = engine.run_trajectory(typist, ThresholdPolicy(threshold=80.0, min_escape_percent=0.10), target_length=12)
        dist = ann.query([stats['password']])[0]
        r = ann.get_risk_percentile([dist])[0]
        proactive_risks.append(r)
        
    plt.figure(figsize=(8, 5))
    sns.kdeplot(baseline_risks, label="Baseline (No Intervention)", fill=True, color="gray")
    sns.kdeplot(proactive_risks, label="Proactive Blocking", fill=True, color="blue")
    
    # "Post-hoc": Baseline passwords that were < 80.
    post_hoc_accepted = [r for r in baseline_risks if r <= 80.0]
    sns.kdeplot(post_hoc_accepted, label="Post-hoc (Accepted Subset)", fill=True, color="orange", linestyle="--")
        
    plt.title("Distribution of Final Password Density Risk")
    plt.xlabel("Density Risk Percentile")
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig("paper_plot_6_outcomes.png")
    plt.close()


# ... (Previous Code)

# 7. Risk Dynamics Analysis (C1, C2, C3)
def plot_risk_dynamics_analysis(ann, mm):
    print("Running Risk Dynamics Analysis (C1, C2, C3)...")
    
    # Setup Simulation (Natural Typing)
    files = ["rockyou_typist.txt", "../rockyou_typist.txt"]
    typist_file = next((f for f in files if os.path.exists(f)), None)
    
    mm_typist = MarkovModel(n=4)
    if typist_file:
        with open(typist_file, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read().splitlines()
        mm_typist.train(data[:50000])
        
    typist = MarkovTypist(mm_typist)
    risk_func = get_risk_evaluator(ann, mm)
    engine = SimulationEngine(risk_func)
    
    N = 300
    TARGET_LEN = 12
    traces = []
    
    traces_baseline = []
    print(f"Simulating Baseline (No Blocking) N={N}...")
    for _ in range(N):
        stats = engine.run_trajectory(typist, NullPolicy(), target_length=TARGET_LEN)
        if len(stats['risk_trace']) == TARGET_LEN:
             traces_baseline.append(stats['risk_trace'])
    traces_baseline = np.array(traces_baseline)

    traces_blocking = []
    policy_block = ThresholdPolicy(threshold=80.0, min_escape_percent=0.10)
    print(f"Simulating T80 Blocking N={N}...")
    for _ in range(N):
        stats = engine.run_trajectory(typist, policy_block, target_length=TARGET_LEN)
        if len(stats['risk_trace']) == TARGET_LEN:
             traces_blocking.append(stats['risk_trace'])
    traces_blocking = np.array(traces_blocking)
    
    # --- C1: Risk Evolution Comparison ---
    print("Plotting C1: Evolution Comparison...")
    plt.figure(figsize=(10, 6))
    
    x_axis = range(1, TARGET_LEN+1)

    # Baseline Stats
    mean_base = np.mean(traces_baseline, axis=0)
    std_base = np.std(traces_baseline, axis=0)
    
    # Blocking Stats
    mean_block = np.mean(traces_blocking, axis=0)
    std_block = np.std(traces_blocking, axis=0)
    
    # Plot Baseline
    plt.plot(x_axis, mean_base, color='blue', linewidth=3, label="No Blocking (Baseline)")
    plt.fill_between(x_axis, mean_base - std_base, mean_base + std_base, color='blue', alpha=0.1)
    
    # Plot Blocking
    plt.plot(x_axis, mean_block, color='green', linewidth=3, label="Proactive Blocking (T80)")
    plt.fill_between(x_axis, mean_block - std_block, mean_block + std_block, color='green', alpha=0.1)

    plt.title("Risk Evolution: Baseline vs. Proactive Blocking")
    plt.xlabel("Keystroke Index")
    plt.ylabel("Combined Risk Score")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("paper_plot_c1_evolution.png")
    plt.close()
    
    # --- C2: Gradients (Use Baseline for Dynamics Analysis) ---
    print("Plotting C2: Gradients (Baseline)...")
    traces = traces_baseline # Use baseline for general dynamics properties
    # Compute diffs along axis 1
    # diffs[:, i] = trace[:, i+1] - trace[:, i]
    # We want diffs for all steps. R_t - R_{t-1}. 
    # Pad start with 0? Or just take 1..12.
    # We can take diffs: trace[:, 1:] - trace[:, :-1]
    gradients = np.diff(traces, axis=1).flatten()
    
    plt.figure(figsize=(8, 5))
    sns.histplot(gradients, kde=True, color="green", bins=50)
    plt.title("Distribution of Risk Gradients (Step-to-Step Change)")
    plt.xlabel("Delta Risk (Risk[t] - Risk[t-1])")
    plt.ylabel("Frequency")
    
    # Highlight tails
    p95 = np.percentile(gradients, 95)
    plt.axvline(x=p95, color='red', linestyle='--', label=f'95th Percentile (+{p95:.1f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("paper_plot_c2_gradients.png")
    plt.close()
    
    # --- C3: Early Predictiveness Correlation ---
    print("Plotting C3: Early Predictiveness...")
    final_risks = traces[:, -1] # Risk at T=12
    correlations = []
    
    steps = range(1, TARGET_LEN+1)
    for t in steps:
        # Correlation between Risk at step t and Final Risk
        # traces[:, t-1] vs final_risks
        step_risks = traces[:, t-1]
        corr = np.corrcoef(step_risks, final_risks)[0, 1]
        correlations.append(corr)
        
    plt.figure(figsize=(8, 5))
    plt.plot(steps, correlations, marker='o', color='purple', linewidth=2)
    plt.title("Correlation of Early Risk with Final Risk")
    plt.xlabel("Keystroke Index (t)")
    plt.ylabel("Pearson Correlation with Final Risk (t=12)")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.xticks(steps)
    plt.tight_layout()
    plt.savefig("paper_plot_c3_correlation.png")
    plt.close()

# 8. Weak Password Evolution Demo (Qualitative)
def plot_weak_evolution_demo(ann, mm):
    print("Running Weak Evolution Demo...")
    
    # Setup Simulation
    files = ["rockyou_typist.txt", "../rockyou_typist.txt"]
    typist_file = next((f for f in files if os.path.exists(f)), None)
    
    mm_typist = MarkovModel(n=4)
    if typist_file:
        with open(typist_file, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read().splitlines()
        mm_typist.train(data[:50000])
        
    typist = MarkovTypist(mm_typist)
    risk_func = get_risk_evaluator(ann, mm)
    engine = SimulationEngine(risk_func)
    
    # Relaxed approach: Simulate batch, pick top 5 riskiest
    print("Simulating batch of 50 to find weak examples...")
    candidates = []
    
    for _ in range(50):
        stats = engine.run_trajectory(typist, NullPolicy(), target_length=12)
        trace = stats['risk_trace']
        pwd = stats['password']
        if len(trace) == 12:
            candidates.append((trace, pwd))
            
    # Sort by final risk (descending)
    candidates.sort(key=lambda x: x[0][-1], reverse=True)
    
    # Take top 6
    top_candidates = candidates[:6]
    weak_traces = [x[0] for x in top_candidates]
    weak_passwords = [x[1] for x in top_candidates]
    
    print(f"Selected {len(weak_traces)} weakest from batch.")
    
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(weak_traces))
    
    for i, trace in enumerate(weak_traces):
        pwd = weak_passwords[i]
        # Find steepest rise
        diffs = np.diff(trace)
        max_jump_idx = np.argmax(diffs) + 1 
        
        plt.plot(range(1, 13), trace, marker='o', linewidth=2, label=f"'{pwd}'", color=colors[i])
        
        # Mark the jump
        plt.scatter(max_jump_idx+1, trace[max_jump_idx], s=100, facecolors='none', edgecolors=colors[i], linestyle='--', zorder=10)
        
    plt.title("Risk Evolution of Weak Passwords (Empirical)")
    plt.xlabel("Keystroke Index")
    plt.ylabel("Combined Risk Score")
    plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label="High Risk Threshold")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("paper_plot_weak_evolution.png")
    plt.close()

    plt.tight_layout()
    plt.savefig("paper_plot_weak_evolution.png")
    plt.close()

# 9. Trajectory Divergence Analysis (Real Data Replay)
def plot_replay_divergence_analysis(ann, mm):
    print("Running Trajectory Divergence Analysis (C4) on Real Data Replay...")
    
    # 1. Load Real Passwords and Sample
    files = ["rockyou_typist.txt", "../rockyou_typist.txt"]
    typist_file = next((f for f in files if os.path.exists(f)), None)
    if not typist_file:
        print("Error: Typist file not found.")
        return

    with open(typist_file, "r", encoding="utf-8", errors="ignore") as f:
        all_lines = f.read().splitlines()
        
    # Sample 10 passwords of length 10
    target_len = 10
    candidates = [p for p in all_lines if len(p) == target_len]
    if len(candidates) < 10:
        candidates = all_lines # Fallback
        
    np.random.seed(42)
    selected_passwords = np.random.choice(candidates, 10, replace=False)
    print(f"Selected Sample: {selected_passwords}")
    
    risk_func = get_risk_evaluator(ann, mm)
    policy_block = ThresholdPolicy(threshold=80.0, min_escape_percent=0.10)
    all_chars = sorted(list(string.printable.strip()))
    
    traces_baseline = []
    traces_blocking = []
    
    for pwd in selected_passwords:
        # --- Baseline (Original Trajectory) ---
        trace_base = []
        for i in range(1, len(pwd)+1):
            prefix = pwd[:i]
            # Calculate step risk
            # engine.run_trajectory does this internally, let's replicate or use helper
            # Need 'allowed' to compute risk? No, risk is just property of prefix.
            # But we defined risk = 0.5*Density + 0.5*Entropy
            # Let's call risk_func directly
            
            # Note: risk_func signature is (prefix, candidates) -> map
            # We just need risk of the LAST char.
            # But the 'risk' in plots is the 'Combined Risk of the prefix'
            
            # Helper to get scalar risk for a full prefix
            r_map = risk_func(prefix[:-1], [prefix[-1]])
            r_val = r_map.get(prefix[-1], 0)
            trace_base.append(r_val)
        traces_baseline.append(trace_base)
        
        # --- Blocking (Replay with Substitution) ---
        current_prefix = ""
        trace_block = []
        
        np.random.seed(42) # Ensure deterministic substitution choices if needed
        
        for char in pwd:
            # 1. Check if intended char is allowed
            # Policy needs: prefix, candidates, risk_evaluator
            allowed = policy_block.get_allowed_chars(current_prefix, all_chars, risk_func)
            
            if char in allowed:
                next_char = char
            else:
                # BLOCKED! Substitute from allowed
                # "choose a character from the unblocked set"
                # We pick randomly to simulate generic deviation
                next_char = np.random.choice(allowed)
            
            current_prefix += next_char
            
            # 2. Compute Risk
            r_map = risk_func(current_prefix[:-1], [current_prefix[-1]])
            r_val = r_map.get(current_prefix[-1], 0)
            trace_block.append(r_val)
            
        traces_blocking.append(trace_block)
            
    traces_baseline = np.array(traces_baseline)
    traces_blocking = np.array(traces_blocking)
    
    # 3. Plot Comparison
    plt.figure(figsize=(10, 6))
    x_axis = range(1, target_len + 1)
    
    # Plot individual faint lines
    for t in traces_baseline:
        plt.plot(x_axis, t, color='red', alpha=0.1)
    for t in traces_blocking:
        plt.plot(x_axis, t, color='green', alpha=0.1)
        
    # Plot Averages
    mean_base = np.mean(traces_baseline, axis=0)
    mean_block = np.mean(traces_blocking, axis=0)
    
    plt.plot(x_axis, mean_base, color='red', linewidth=3, label="Baseline (Actual Passwords)")
    plt.plot(x_axis, mean_block, color='green', linewidth=3, label="With Blocking (Forced Substitution)")
    
    plt.title(f"Impact of Intervention on Real Passwords (N=10)")
    plt.xlabel("Keystroke Index")
    plt.ylabel("Combined Risk Score")
    plt.axhline(y=80, color='gray', linestyle='--', label="Threshold")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("paper_plot_c4_divergence.png")
    plt.close()


    
class RobustTemplateTypist(TemplateTypist):
    """
    A wrapper around TemplateTypist that guarantees a character is chosen 
    if blocked, preventing 'stuck' simulations.
    """
    def next_char(self, prefix, allowed_chars=None):
        # 1. Try normal specific substitution logic
        choice = super().next_char(prefix, allowed_chars)
        
        # 2. If stuck (choice is None) but we aren't finished with the plan
        if choice is None and self.idx < len(self.current_plan):
            # Fallback: Just pick ANY allowed character to continue trajectory
            if allowed_chars:
                choice = np.random.choice(allowed_chars)
                self.idx += 1 # Advance index even if we broke the template structure
                return choice
        return choice

# 10. Split Risk CDF Comparison (Markov vs Template)
def plot_risk_cdf_split(ann, mm):
    print("Running Split Risk CDF Comparison (N=200 each)...")
    
    risk_func = get_risk_evaluator(ann, mm)
    engine = SimulationEngine(risk_func)
    policy_null = NullPolicy()
    policy_block = ThresholdPolicy(threshold=80.0, min_escape_percent=0.10)
    
    scenarios = [
        ("Markov Typist", MarkovTypist(mm), "paper_plot_cdf_markov.png"),
        ("Template Typist", RobustTemplateTypist(), "paper_plot_cdf_template.png")
    ]
    
    for name, typist, filename in scenarios:
        print(f"  Simulating {name}...")
        risks_baseline = []
        risks_proactive = []
        
        lengths = [8, 10, 12, 14]
        
        # Run N=50 simulations
        for i in range(50):
            if i % 10 == 0:
                print(f"    {name}: {i}/50")
            
            target_L = int(np.random.choice(lengths))
            
            # Baseline
            stats_base = engine.run_trajectory(typist, policy_null, target_length=target_L)
            risks_baseline.append(stats_base['avg_step_risk'])
            
            # Proactive
            stats_block = engine.run_trajectory(typist, policy_block, target_length=target_L)
            risks_proactive.append(stats_block['avg_step_risk'])
            
        risks_baseline = np.array(risks_baseline)
        risks_proactive = np.array(risks_proactive)
        
        # Derive Post-hoc Rejection (Filter Baseline <= 80)
        # Note: If no passwords are rejected, this array equals baseline
        risks_posthoc = risks_baseline[risks_baseline <= 75.0]
        
        print(f"    {name} Results:")
        print(f"      Baseline Mean: {np.mean(risks_baseline):.2f}")
        print(f"      Post-hoc Mean: {np.mean(risks_posthoc):.2f} (Rejected {len(risks_baseline)-len(risks_posthoc)}/{len(risks_baseline)})")
        print(f"      Proactive Mean: {np.mean(risks_proactive):.2f}")
        
        # Plot CDF
        plt.figure(figsize=(10, 6))
        
        sns.ecdfplot(risks_baseline, label="Baseline (No Intervention)", color='red', linewidth=2)
        sns.ecdfplot(risks_posthoc, label="Post-hoc Rejection (Classic Check)", color='blue', linestyle='--', linewidth=2)
        sns.ecdfplot(risks_proactive, label="Proactive Blocking (Ours)", color='green', linewidth=2)
        
        plt.title(f"Cumulative Distribution of Risk ({name})")
        plt.xlabel("Total Risk Score (Lower is Safer)")
        plt.ylabel("Cumulative Probability")
        plt.axvline(x=80, color='gray', linestyle=':', label="Threshold (80)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    ann, mm = load_models()
    # plot_reuse_density(ann)
    # plot_risk_scatter(ann, mm)
    # plot_intervention_timing(ann, mm)
    # plot_outcomes_kde(ann, mm)
    # plot_risk_dynamics_analysis(ann, mm)
    # plot_replay_divergence_analysis(ann, mm)
    
    plot_risk_cdf_split(ann, mm)
    print("Split Risk CDF Comparison generated.")
