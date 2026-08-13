# Scaled & Complex Security Analysis Results (Camera-Ready)

Analysis generated on: 2026-08-13 09:07:48

**Core Simulation Scale**: $N=1000$ trajectories per condition.
**Dual Typist Complexity**: Evaluated across both Template (deterministic intent) and Markov (probabilistic fluid) simulated typists.

## Dual-Typist Security & Guessability Comparison
| Typist Model | Policy | Mean Final Risk | Max Risk | Mean $log_{10}$ Guesses | Avg Keystrokes (Efficiency) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Template | Baseline | 65.60 | 83.47 | 4.39 | 9.51 |
| Template | Posthoc | 64.19 | 79.98 | 4.40 | 10.31 |
| Template | Proactive | 47.20 | 78.73 | 7.33 | 9.51 |
| Markov | Baseline | 42.61 | 75.80 | 10.51 | 12.00 |
| Markov | Posthoc | 42.57 | 79.56 | 10.53 | 12.01 |
| Markov | Proactive | 39.30 | 78.17 | 10.59 | 12.00 |

*Takeaway*: Proactive intervention robustly improves both internal model risk and independent external guessability across both rigid (Template) and fluid (Markov) user behavioral models, while maintaining identical or superior keystroke efficiency compared to Post-hoc constraints.