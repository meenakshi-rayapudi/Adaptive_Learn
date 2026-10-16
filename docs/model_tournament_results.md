# Model tournament results

Data: ASSISTments benchmark, 336,866 training attempts (earliest 80%) and 84,217 test attempts (latest 20%). Features used: Q, A, T.

| Model | ROC-AUC | PR-AUC | Brier (lower is better) | Latency p95 (ms) |
|---|---|---|---|---|
| logistic_regression | 0.7934 | 0.6694 | 0.1524 | 1.1 |
| mlp | 0.8017 | 0.6770 | 0.1477 | 1.1 |
| random_forest | 0.8026 | 0.6771 | 0.1489 | 34.5 |
| lightgbm (winner) | 0.8052 | 0.6838 | 0.1462 | 2.0 |
| xgboost | 0.8042 | 0.6827 | 0.1465 | 2.7 |

The winner is the tree model with the best ROC-AUC, because TreeSHAP explanations need a tree model. Hyperparameters are fixed defaults with no tuning.
