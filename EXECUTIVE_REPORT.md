# Executive Report: Eight-Class Imagined-Speech EEG Ensemble

## Executive summary

The ten-model hard-voting ensemble achieved **27.30% accuracy** on the held-out test set across eight imagined-speech classes. This is above the **12.5% random baseline**, but it is not close to the previously targeted 90%+ accuracy.

The kNN cross-validation sweep selected **k = 1**. Its mean five-fold validation accuracy was **28.05%**, and the final ensemble accuracy was **27.30%**. The result indicates that the current feature representation and direct ensemble do not reliably separate the eight classes.

## Experiment configuration

| Item | Configuration |
|---|---|
| Classes | Sí, No, Baño, Hambre, Sed, Ayuda, Dolor, Gracias |
| Samples | 3,588 |
| Channels | 14 Emotiv EPOC+ electrodes |
| Sampling rate | 128 Hz |
| Input window | 1 second / 128 samples |
| Feature representation | 14 × 214 engineered features |
| Split | Stratified 80/20 random split |
| Training samples | 2,870 |
| Test samples | 718 |
| Ensemble | Hard voting across 10 classifiers |
| kNN selection | 5-fold stratified cross-validation |
| Selected k | 1 |

## Dataset distribution

| Class | Samples |
|---|---:|
| Sí | 444 |
| No | 480 |
| Baño | 372 |
| Hambre | 444 |
| Sed | 408 |
| Ayuda | 504 |
| Dolor | 492 |
| Gracias | 444 |

## Final test metrics

| Metric | Score |
|---|---:|
| Accuracy | **27.30%** |
| Weighted precision | 27.73% |
| Weighted recall | 27.30% |
| Weighted F1 | 26.87% |
| Macro precision | 27.76% |
| Macro recall | 27.00% |
| Macro F1 | 26.72% |

## Per-class recall

| True class | Recall |
|---|---:|
| Sí | 20.22% |
| No | 35.42% |
| Baño | 28.38% |
| Hambre | 33.71% |
| Sed | 19.51% |
| Ayuda | 36.63% |
| Dolor | 27.55% |
| Gracias | 14.61% |

The strongest class recalls were **Ayuda (36.63%)**, **No (35.42%)**, and **Hambre (33.71%)**. **Gracias (14.61%)**, **Sed (19.51%)**, and **Sí (20.22%)** were the most difficult classes.

## kNN cross-validation

| k | Mean CV accuracy |
|---:|---:|
| 1 | **28.05%** |
| 3 | 24.08% |
| 5 | 24.77% |
| 7 | 24.08% |
| 9 | 23.21% |
| 11 | 24.01% |
| 13 | 24.11% |
| 15 | 24.25% |
| 17 | 24.04% |
| 21 | 24.18% |
| 25 | 23.24% |
| 31 | 23.17% |
| 37 | 22.47% |

## Interpretation

The ensemble is only modestly better than chance. The similar macro and weighted scores show that class imbalance is not the primary cause of the low result. The confusion matrix shows broad cross-class confusion rather than one isolated class failure.

The most important score is the **held-out test accuracy: 27.30%**. The 28.05% cross-validation score is used for model selection and should not be treated as the final performance estimate.

## Limitations

1. The reported split is a random window-level split. Windows from the same underlying trials may therefore appear in both training and test sets, making this estimate potentially optimistic.
2. The supplied data does not expose subject identifiers, so a true subject-independent evaluation is not currently possible.
3. The ensemble uses default classifier parameters and hard voting; it does not optimize model weights or decision thresholds.
4. The result is not directly comparable to the paper’s reported 96% binary result, which used a different task, binary labels, and GA-selected features.

## Recommendation

Treat **27.30%** as the current baseline for the eight-class direct-ensemble approach. Before further model tuning, validate the data-to-label mapping and recover subject/trial metadata if possible; then compare group-aware performance against the random split.
