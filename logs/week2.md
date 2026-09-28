# Research Log: Week 2 - Reusable Random and Targeted Attacks

## What I Learned
During Week 2, I transitioned from the initial Week 1 exploration into production-quality, modular attack implementations in `src/attacks.py`. Grounded in the foundational poisoning framework of Biggio et al. (2012), I formalized the distinction between indiscriminate availability attacks and targeted integrity attacks. 

Random label flipping operates bidirectionally across the feature space, inverting both legitimate ham and spam instances. This maintains a relatively stable class distribution (~39.4% to 43.6% spam) while injecting pervasive label noise that blurs the decision boundary, degrading overall accuracy and classifier confidence (an availability / denial-of-service threat). In contrast, targeted spam-to-ham flipping acts strictly unidirectionally, relabeling only spam instances as legitimate non-spam ($1 \to 0$). At a 20% poisoning rate, 736 of the 1,450 training spam samples (50.8%) are relabeled as ham, shrinking the apparent spam proportion from 39.4% down to 19.4%. This systematically pulls the decision boundary toward the non-spam region, specifically destroying spam recall and allowing malicious spam to evade detection without generating conspicuous false alarms on legitimate emails (an integrity / evasion threat).

## What Failed
Implementing rigorous automated tests in `tests/test_attacks.py` revealed subtle implementation hazards. First, ensuring exact sample selection without replacement while strictly enforcing directionality required dedicated masking logic to prevent modifying legitimate ham. Second, configuring automated test discovery across different Python environments required establishing an explicit `pytest.ini` configuration with local path resolution. All 17 unit tests now pass with 100% test coverage across invariants, reproducibility, and edge cases.

## Next Steps
For Week 3, I will build the unified experiment runner (`scripts/run_attack_experiments.py` and modular components under `src/`) to execute the full three-seed pilot study (seeds 0, 1, and 2) across all models, attacks, and poisoning rates, generating the 81-row benchmark results table.
