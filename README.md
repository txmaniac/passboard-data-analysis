# Passboard: Proactive Password Intervention Experiments

This repository contains the simulation framework, analysis scripts, and plotting utilities for evaluating **Passboard**—a proactive, per-keystroke password intervention mechanism—against traditional **Post-hoc Rejection** (on-submit checks) and **Baseline** policies.

## Project Agenda

Leaked password analysis (e.g., on the RockYou dataset) reveals that weak passwords do not cluster into discrete, separable categories. Instead, they form a **continuous, connected manifold in representation space**. 

Structural alterations (like appending digits or symbols, e.g., `admin` $\rightarrow$ `admin123!`) represent **smooth trajectories** traversing towards high-density (high-leakage-probability) regions of this manifold. Traditional post-hoc checkers reject passwords after the user has finished typing, causing high frustration and failing to prevent users from traversing to the boundary of acceptable risk. 

**Passboard** addresses this by modeling password creation as a trajectory in vector space. Using a per-keystroke blocking engine, it dynamically disables keys that would transition the input prefix into high-density/high-risk regions, guiding users into safe, high-entropy password regions.

---

## Repository Structure

*   `security_usability_analysis.py`: **[NEW]** Comparative simulation script evaluating security (final risk, safety failures), usability (attempts, keystrokes typed, discarded effort), and entropy across policies.
*   `paper_plots.py`: High-quality plot generator producing comparative graphs (CDFs, KDEs, Boxplots, and evolution curves) in PNG and SVG formats.
*   `Paper-Dets/`: Documents detailing the underlying password geometry and clusterability experiments.
*   `risk-estimation-libs/`: Subfolder containing the core simulation library:
    *   `ann_density.py`: Nearest-neighbor density estimation using a pre-trained FAISS index.
    *   `markov_entropy_risk.py`: N-gram Markov Model for character-by-character entropy rate estimation.
    *   `simulator.py`: Character-level simulation engine orchestrating typing trajectories.
    *   `typist.py`: Simulated user profiles (`MarkovTypist` and `TemplateTypist`).
    *   `policy.py`: Decision-making logic (`NullPolicy` and `ThresholdPolicy`).
*   `models/`: Cached models including the 6GB FAISS index (`faiss.index`) and Markov models.

---

## Running the Experiments

To run these experiments, ensure `uv` is installed, and execute the following commands from the project root directory.

### 1. Run the Security and Usability Analysis
To run the comparative simulation comparing Baseline, Post-hoc, and Proactive interventions:
```bash
uv run python security_usability_analysis.py --num-sims 100
```

#### Customized Runs
You can configure the simulation size and risk threshold:
```bash
# Run with 200 simulations and a tighter risk threshold of 75.0
uv run python security_usability_analysis.py --num-sims 200 --threshold 75.0
```

### 2. Generate Paper Plots
To regenerate the high-quality SVG and PNG comparative plots:
```bash
uv run python paper_plots.py
```
This produces plots such as:
*   `paper_plot_cdf_markov.svg` / `paper_plot_cdf_template.svg`: Cumulative distribution of risk.
*   `paper_plot_5_timing.svg`: Keystroke index of first intervention.
*   `paper_plot_6_outcomes.svg`: Risk outcome distributions.
