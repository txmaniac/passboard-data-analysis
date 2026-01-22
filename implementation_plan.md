# Offline Password Simulation Framework

Implement a framework to simulate password creation trajectories under different policies (Baseline vs. Blocking).

## Goals
1.  **Simulate Trajectories**: Character-by-character generation (Markov & Template strategies).
2.  **Apply Blocking**: Intervene at specific steps based on Risk Thresholds.
3.  **Collect Metrics**: Output datasets for comparison (Length, Density, Entropy, Blocked Events).

## Architecture

### [risk-estimation-libs]

#### [NEW] [typist.py](file:///Users/txmaniac/Documents/Passboard-Exps/risk-estimation-libs/typist.py)
Abstract base class `Typist` and implementations:
*   `MarkovTypist`:
    *   Loads `MarkovModel` (from `markov_entropy_risk.py` or JSON).
    *   `next_char(prefix, allowed_chars)`: Samples from $P(c|prefix)$ restricted to `allowed_chars`.
    *   Resampling strategy: Renormalize probabilities over `allowed_chars`.
*   `TemplateTypist`:
    *   Pre-defined templates (e.g., `["{word}{digit}{symbol}", "{word}{year}", "{leet_word}"]`).
    *   `next_char(prefix, allowed_chars)`: Follows template. If blocked, tries next best (e.g., different digit, different symbol, or aborts/deviates).

#### [NEW] [policy.py](file:///Users/txmaniac/Documents/Passboard-Exps/risk-estimation-libs/policy.py)
*   `BlockingPolicy` Interface.
*   `ThresholdPolicy`:
    *   Input: `RiskFunction` (callable), `threshold`, `min_allowed_percent` (Escape Route).
    *   `get_allowed_chars(prefix, candidates)`:
        1.  Compute risk for all candidates.
        2.  Filter candidates where risk > threshold.
        3.  Check escape route: If too few allowed, relax threshold until `min_allowed_percent` is met.

#### [NEW] [simulator.py](file:///Users/txmaniac/Documents/Passboard-Exps/risk-estimation-libs/simulator.py)
*   `SimulationEngine`:
    *   `run_trajectory(typist, policy, target_length)`:
        *   Loop until `target_length`.
        *   Get `allowed_chars` from `policy`.
        *   Get `char` from `typist`.
        *   Track "blocked events" (if preferred char was blocked).
        *   Return final password and stats.

#### [NEW] [run_simulation.py](file:///Users/txmaniac/Documents/Passboard-Exps/risk-estimation-libs/run_simulation.py)
*   **Setup**:
    *   **Data Preparation**:
        *   Split `rockyou.txt` into `rockyou_typist.txt` (50%) and `rockyou_risk.txt` (50%).
        *   **Typist Model**: Train `MarkovModel` on `rockyou_typist.txt`.
        *   **Risk Model**: Train/Fit `ANNDensity` and `MarkovEntropy` on `rockyou_risk.txt`.
    *   **Define Scenarios**:
        *   Scenario 1: `MarkovTypist` (Temperature=1.0).
        *   Scenario 2: `TemplateTypist` (List of common templates).
    *   **Define Conditions**:
        *   Baseline: `NullPolicy` (Allows all).
        *   Blocking: `ThresholdPolicy` (Risk > Threshold).
            *   **Risk Formula**: `Risk = 0.5 * DensityRisk + 0.5 * EntropyRisk`.
            *   `DensityRisk`: Percentile of K-NN distance (0-100).
            *   `EntropyRisk`: Linear mapping of Entropy Rate (0-100).
*   **Execution**:
    *   Run N=10,000 trajectories per combo.
    *   Save results to `simulation_results.csv` (cols: `scenario`, `condition`, `password`, `length`, `blocked_count`, `density_score`, `entropy_score`, `risk_score`).

## Implementation details
*   **Risk Function**: Linear combination `0.5 * Density + 0.5 * Entropy`.
*   **Data Split**: Essential to avoid testing on training data.
*   **Reproducibility**: `random.seed` per trajectory.

## Steps
1.  **Data Split**: Create script/function to split `rockyou.txt` -> `typist_train.txt`, `risk_ref.txt`.
2.  Implement `typist.py` (Markov & Template logic).
3.  Implement `policy.py` (Blocking logic with escape & linear risk).
4.  Implement `simulator.py` (Orchestration).
5.  Implement `run_simulation.py` (Experiment script).
