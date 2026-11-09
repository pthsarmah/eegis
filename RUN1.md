# RUN1 — kNN Genetic-Algorithm Experiment

## Summary

| Item | Result |
|---|---:|
| Class mapping | class_6 = Sí, class_8 = No |
| Dataset | 924 instances |
| Train / validation / test | 591 / 148 / 185 |
| GA generations completed | 86 |
| GA early stopping | Yes, plateau patience = 10 |
| Best validation fitness | 0.9189 |
| Selected k | 5 |
| Selected electrodes | AF3, F7, F3, FC5, O1, O2, P8, FC6, F8, AF4 |
| Selected feature count | 108 |

## Metrics

| Evaluation split | Accuracy | Precision | Recall (Sí) | F1 (Sí) |
|---|---:|---:|---:|---:|
| Naive random final test | 0.5946 | 0.6296 | 0.5313 | 0.5763 |
| Corrected inferred trial-group split | 0.4323 | 0.4253 | 0.3854 | 0.4044 |

## Row-normalized confusion matrices

| Split | No → No | No → Sí | Sí → No | Sí → Sí |
|---|---:|---:|---:|---:|
| Naive random final test | 0.6629 | 0.3371 | 0.4688 | 0.5313 |
| Corrected inferred trial-group split | 0.4792 | 0.5208 | 0.6146 | 0.3854 |

## GA progression

| Generation | Best fitness | Average fitness |
|---:|---:|---:|
| 1 | 0.6757 | 0.6128 |
| 10 | 0.7365 | 0.6960 |
| 20 | 0.7905 | 0.7403 |
| 30 | 0.8041 | 0.7743 |
| 40 | 0.8311 | 0.7976 |
| 50 | 0.8649 | 0.8276 |
| 60 | 0.8784 | 0.8520 |
| 70 | 0.8919 | 0.8675 |
| 71 | 0.9054 | 0.8698 |
| 72 | 0.9122 | 0.8714 |
| 76 | 0.9189 | 0.8774 |
| 86 | 0.9189 | 0.8894 |

The reported 0.9189 is validation fitness used during GA optimization; it is not the final held-out test accuracy.
