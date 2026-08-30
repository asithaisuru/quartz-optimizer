# Simulation-Derived Facet ML Evidence

Real reconstructed specimens were checked first. The available confirmed independent reconstructed specimen count is 1, so specimen-grouped real ML evaluation is not defensible.

This artifact therefore uses simulation-derived training labels. The target is a deterministic geometric/ray inclusion-visibility score, not expert grading, physically validated optical ray tracing, or a real-world beauty prediction.

- Dataset rows: 11016
- Scenario groups: 504
- Grouped split: {'train': 352, 'validation': 76, 'test': 76}
- Model: simulation_trained_randomized_tree_ensemble_regressor
- Model SHA-256: a27055e97c8c667898dea64eedcc5ede424f3be13354469d478ca3199f042646
- Test MAE/RMSE: 1.7499 / 2.4555
- Test Spearman: 0.9861
- Test top-orientation agreement: 0.6974
- Test angular error mean/median: 19.24 / 0.00 degrees
- Heuristic baseline MAE/RMSE: 13.8430 / 17.4637
- Heuristic baseline top-orientation agreement: 0.1579

Reproduce with:

```bash
python -B backend/research/facet_ml/generate_facet_ml_evidence.py
```
