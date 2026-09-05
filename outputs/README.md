# Generated outputs

Running the notebook from top to bottom regenerates the following artifacts:

| File | Contents |
|---|---|
| `submission.csv` | Final test-set probabilities in the required `Id,Drafted` format |
| `model_comparison.csv` | Aggregate OOF metrics for every individual model |
| `fold_results.csv` | ROC-AUC, PR-AUC, log loss, Brier score, and runtime for each fold |
| `oof_predictions.csv` | Row-level out-of-fold predictions for all models and the selected ensemble |
| `feature_importance.csv` | Held-out permutation importance for the best individual model |
| `run_summary.json` | Seed, CV setup, selected candidate, members, AUC interval, and data shapes |

The committed files document the verified reference run. Re-execution may produce small runtime differences, while deterministic predictions should remain stable under the pinned environment.
