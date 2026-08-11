# Experimental Validation

## 0. Implementation and Parameterization

To ensure full reproducibility, we detail our pipeline and parameterization below:

* **Corpus & Preprocessing**: The system uses a subset of the RockYou dataset. During preprocessing, records were converted to lowercase ASCII, restricted to lengths 4–20, and deduplicated while preserving frequency counts to avoid artificially inflating structural density.
* **Neighborhood Density Model**: We represent passwords using character 4-grams with TF-IDF weighting. Density is measured using L2 distance to the $k=20$ nearest neighbors. The raw distance is normalized into a density risk score by calculating its percentile rank against the training distribution.
* **Predictability Model**: We utilize a Markov model of order 4 with add-one smoothing. Prefix probability is converted to a normalized entropy risk score via a linear mapping function.
* **Risk Normalization**: Both density and predictability are normalized to a 0–100 scale. The final combined model-estimated internal risk is computed as: $R = 0.5 \times Density\_Percentile + 0.5 \times Max(0, Min(100, (1 - Entropy\_Rate / 8) \times 100))$.
* **Proactive Intervention Algorithm**: The candidate action space consists of 70 characters (lowercase, uppercase, digits, symbols). At each keystroke, the system computes the risk of every continuation. If $R(prefix + c) > T$ (where $T=80$), $c$ is temporarily blocked. $T=80$ was selected heuristically to balance security and usability friction.
* **Simulation Protocol**: Trajectories are generated using simulated template-based typists. For comparative policies (Baseline, Post-hoc, Proactive), we employ $N=1000$ trajectories. The initial starting templates are paired using fixed random seeds across all conditions to ensure identical intent. When a simulated user encounters a blocked key under the Proactive policy, they resample uniformly from the remaining allowed characters of the same class (e.g., picking a different digit). For the Post-hoc baseline, the simulated user completes the entire password, and if the final $R > 80$, it is entirely rejected, and they begin a new attempt from scratch (up to a limit of 10 retries).

## 1. Topological Correlation: Density and Dataset Frequency

**Rationale:** We theorized that highly reused passwords form dense clusters in the topological space of password representations. To validate this, we needed to demonstrate a correlation between topological density (k-NN distance) and real-world reuse frequency proxy (dataset rank). We stratified our sample across the RockYou dataset to capture a representative distribution. Note that density is a proxy for structural commonness and leaked-password neighborhood risk, rather than direct proof of cross-service account-level reuse.

**Observations:** We observed a significant positive correlation (Pearson $r = 0.1182$, $p < 0.001$; Spearman $\rho = 0.1116$, $p < 0.001$). This confirms that as the topological density of a password's neighborhood increases (lower k-NN distance), its frequency of reuse proxy in real-world breaches also increases, validating topological sparsity as a proxy for password safety.

## 2. Structural Tightness of Dense vs. Sparse Neighborhoods

**Rationale:** If our intervention strategy is to block paths leading to dense clusters, we must ensure that these clusters represent minor structural variations of predictable base words (e.g., appending a '1' or '!'). We analyzed the Levenshtein edit distance between passwords and their nearest neighbors in both dense and sparse regions.

**Observations:** Our analysis revealed a stark contrast in structural tightness. Passwords in dense regions exhibited a mean edit distance of 1.95 edits to their nearest neighbors (e.g., `1234567b` neighbored by `1234567C`). In contrast, sparse regions exhibited a mean edit distance of 8.73 edits. Dense manifolds are characterized by highly predictable variations of the same weak password.

## 3. Predictive Power of Early Risk Trajectories

**Rationale:** The viability of interaction-time intervention hinges on the system's ability to accurately predict the final internal model risk of a password before the user finishes typing it. We simulated $N=200$ user trajectories and measured the correlation between the risk score at prefix lengths (3, 5, and 8 characters) and the final 12-character password risk.

**Observations:** The predictive power of the model scaled strongly with prefix length. The Pearson correlation ($r$) was modest at 3 characters ($0.1717$) but grew significantly to $0.3262$ at 5 characters, and reached a strong $0.6362$ by 8 characters. The system can confidently intervene at mid-length keystrokes, correcting the trajectory before submission.

## 4. Intervention Coverage and Usability Friction

**Rationale:** We hypothesized that our proactive engine would minimize usability friction by intervening selectively. We simulated $N=1000$ trajectories of a vulnerable typist and tracked the probability of intervention and the percentage of keyboard keys disabled.

**Observations:** The results demonstrated a highly selective intervention profile. On average, the system actively guided the user during only 5.99% of keystroke events, disabling an average of just 0.21% of candidate keys per step. The system intervened only 0.65 times per password creation, leaving most candidate actions available in the simulated action space. 

## 5. Frictionless Security vs. Post-Hoc Rejection

**Rationale:** We designed this experiment to benchmark the usability cost of our interaction-time proactive constraint system against a strict post-hoc rejection policy (rejecting completed passwords with internal risk > 80).

**Observations:** Users under the post-hoc policy faced a complete failure rate on initial weak attempts, averaging 1.09 submission attempts and wasting effort by typing 10.31 keystrokes before achieving a successful password with a mean accepted internal risk of 64.02. In contrast, trajectories under our proactive intervention achieved a 100% success rate on the first submission attempt (1.00 attempts), typing 10.40 keystrokes total, while achieving a strictly lower mean accepted internal risk of 56.45.

## 6. Comprehensive Security Comparison Across Paradigms

**Rationale:** We compared the risk profiles and independent guessability of three distinct cohorts ($N=1000$): a baseline vulnerable user (No policy), a vulnerable user under a Post-hoc Rejection policy ($R > 80$), and a vulnerable user guided by our Proactive Intervention. To avoid circular evaluation, we also measured the passwords using an independent attacker-facing evaluator: the `zxcvbn` guessability estimator.

**Observations:** The baseline cohort exhibited the highest risk profile (Mean Risk = 62.97, Max Risk = 79.87, Independent $log_{10}$ guesses = 4.39). While the Post-hoc rejection naturally achieved a lower maximum risk (79.98) due to submission-time rejection, it resulted in a higher mean internal risk (64.02) and negligible independent guessability improvement ($log_{10}$ guesses = 4.41). Passwords resulting from simulated user trajectories under proactive intervention eliminated the tail of high-risk passwords (Max Risk = 78.49), producing a comparable observed maximum internal risk to the post-hoc condition. Crucially, proactive intervention significantly improved independent security, raising the mean $log_{10}$ guesses to 6.65—proving external security relevance beyond our internal risk metric.

## 7. Threshold Sensitivity

**Rationale:** To demonstrate that our conclusions are not artifacts of an arbitrary cutoff, we ran a sensitivity analysis evaluating the intervention aggressiveness and final internal risk across five operating thresholds ($T \in \{40.0, 60.0, 80.0, 100.0, 120.0\}$) using $N=200$ seeded trajectories.

**Observations:** As expected, stricter thresholds (lower $T$) resulted in lower final model-estimated risk but disabled a progressively larger fraction of the keyboard. At $T=40.0$, average internal risk dropped to 18.95, but 6.66% of keys were disabled per step. At $T=100.0$, the policy effectively acted as a baseline with 0.00% keys disabled and 64.82 risk. $T=80.0$ emerged as a balanced operating point, disabling only 0.24% of keys while significantly reducing risk to 55.30.

## 8. Runtime and Memory Benchmark

**Rationale:** The deployability of interaction-time constraints depends on the system's ability to score every candidate action at every keystroke with imperceptible latency. We benchmarked the system across 500 simulated keystroke events, scoring all 70 candidates per update.

**Observations:** The system maintained a median full-keyboard update latency of 75.51 ms. Tail latencies remained tight, with p95 at 80.39 ms and a maximum observed latency of 117.65 ms. While the FAISS index footprint requires ~6 GB of memory, these sub-100-millisecond latencies prove the computational feasibility of real-time action-space constraints for password creation.

## Discussion and Limitations

**Memorability and Password Managers**: The proposed system is not intended to generate random, high-entropy passwords for a user to memorize; rather, it constrains a small subset of locally risky next-character actions while the user continues manually constructing their password. Password managers and vaults remain the strongest solution when available and adopted. Our mechanism specifically targets contexts where users still manually create passwords, aiming to intervene before a weak manual trajectory is completed. Furthermore, our offline simulation does not directly measure recall, creation time, frustration, or abandonment; we make no claims about preserved memorability without future human-subject validation.

**Deployment Privacy**: In this architecture, prefix scoring is highly sensitive. The 6GB FAISS index requires significant memory, meaning a local on-device deployment is ideal for preserving input privacy. If deployed server-side, raw partial passwords would be transmitted at every keystroke, introducing a new data-exposure surface. A production implementation must guarantee local processing to avoid logging or caching partial password trajectories.

**Limitations**: Neighborhood density inside our model is a proxy for structural commonness in leaked corpora, not direct evidence of cross-service account-level reuse. Second, our simulated typists do not capture human adaptation, confusion, or workarounds; perceived user agency and frustration remain open questions for future in-situ studies. Finally, our fixed threshold configuration ($T=80$) represents a single operating point; personalized or adaptive thresholds remain an avenue for future work.
