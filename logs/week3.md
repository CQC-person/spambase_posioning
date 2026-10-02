# Research Log: Week 3 - Reproducible Pilot Experiment Runner

## What I Learned
During Week 3, I developed the unified multi-seed experiment runner (`src/experiments.py` and `scripts/run_attack_experiments.py`) to evaluate the three classical classifiers under both random and targeted data poisoning across seeds 0, 1, and 2. The pilot execution generated all 81 required result rows (9 clean baselines and 72 poisoned conditions).

Analyzing the mean and standard deviation curves revealed clear behavioral divergences across attack strategies and model families. Random label flipping acts as an availability threat, introducing symmetric label noise that causes steady, parallel degradation across accuracy, precision, and recall. In sharp contrast, targeted spam-to-ham flipping acts as a severe integrity threat: it selectively suppresses spam recall while maintaining deceptively high spam precision (~95%+). Under 20% targeted poisoning, Logistic Regression spam recall collapsed from ~90% down to under 45% (a loss exceeding 45 percentage points), proving that linear decision hyperplanes are easily manipulated when spam feature profiles are mislabeled as benign. Random Forest consistently demonstrated the highest noise robustness across all conditions, maintaining recall above 83% even under severe poisoning due to ensemble bagging and feature subspace sampling.

## What Failed
Enforcing the fair comparison invariant—guaranteeing that all three models receive identical train-test splits and identical poisoned training samples—required structuring `run_single_condition` to execute partitioning and poisoning prior to model induction. Standalone CLI testing encountered a joblib multiprocessing permission warning under macOS sandboxing, which was safely handled by operating in deterministic serial execution without altering any metric values. Zero failed conditions were logged across all 81 experimental runs.

## Next Steps
For Week 4, I will audit the complete pilot results, evaluate preliminary defense hypotheses, freeze the experimental protocol (`stage3_protocol.md`), and begin drafting the LaTeX research paper structure (`paper/main.tex` and `paper/references.bib`).
