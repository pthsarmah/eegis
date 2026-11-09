# Executive Report: Eight-Class Imagined-Speech EEG Ensemble

## Executive summary

The improved ten-model soft ensemble achieved **37.74% accuracy** with the analytically selected k=1, and **30.08% accuracy** with the required production k=5 configuration. The production result is **2.78 percentage points above** the original 27.30% baseline and **17.58 points above** the 12.5% random baseline, but remains well below the previously targeted 90%+ accuracy.

The improved pipeline selected **k = 1** after training-only feature selection, but **k = 5 is retained for production use** because it is less sensitive to individual noisy or duplicated windows. The kNN five-fold validation accuracy was 34.32% for k=1 and 29.93% for k=5. The analytical k=1 test result was 37.74%; the production k=5 run scored 30.08% and is the result to use for deployment discussions.

## Experiment configuration

| Item | Configuration |
| - | - |
| Classes | Sí, No, Baño, Hambre, Sed, Ayuda, Dolor, Gracias |
| Samples | 3,588 |
| Channels | 14 Emotiv EPOC+ electrodes |
| Sampling rate | 128 Hz |
| Input window | 1 second / 128 samples |
| Feature representation | 14 × 214 engineered features, reduced to 512 features |
| Split | Stratified 80/20 random split |
| Training samples | 2,870 |
| Test samples | 718 |
| Ensemble | Validation-weighted soft voting across 10 classifiers |
| kNN selection | 5-fold stratified cross-validation |
| CV-selected k | 1 |
| Production kNN k | **5** |
| Feature selection | Training-only variance filtering + ANOVA |


## Dataset distribution

| Class | Samples |
| - | -: |
| Sí | 444 |
| No | 480 |
| Baño | 372 |
| Hambre | 444 |
| Sed | 408 |
| Ayuda | 504 |
| Dolor | 492 |
| Gracias | 444 |


## Improved final test metrics

| Metric | Score |
| - | -: |
| Accuracy | **37.74%** |
| Weighted precision | 38.02% |
| Weighted recall | 37.74% |
| Weighted F1 | 37.74% |
| Macro precision | 38.00% |
| Macro recall | 37.90% |
| Macro F1 | 37.80% |

### Baseline comparison

| Pipeline | Test accuracy |
|---|---:|
| Original hard-voting ensemble | 27.30% |
| Improved weighted soft ensemble, k=1 analysis run | 37.74% |
| Improved weighted soft ensemble, k=5 production run | **30.08%** |
| Production improvement over original | **+2.78 percentage points** |

## Production kNN run

The production run uses `--production-k 5`; the CV-selected k=1 is retained only as an analytical comparison. Both runs use the same split, feature cap, and ten-model weighted soft-voting architecture.

| Metric | k=5 production score |
|---|---:|
| Accuracy | **30.08%** |
| Weighted precision | 30.45% |
| Weighted recall | 30.08% |
| Weighted F1 | 30.02% |
| Macro precision | 30.45% |
| Macro recall | 30.01% |
| Macro F1 | 29.98% |

| Class | Production recall |
|---|---:|
| Sí | 24.72% |
| No | 33.33% |
| Baño | 33.78% |
| Hambre | 33.71% |
| Sed | 23.17% |
| Ayuda | 33.66% |
| Dolor | 29.59% |
| Gracias | 28.09% |


## Per-class recall

| True class | Recall |
| - | -: |
| Sí | 38.20% |
| No | 33.33% |
| Baño | 44.59% |
| Hambre | 38.20% |
| Sed | 35.37% |
| Ayuda | 37.62% |
| Dolor | 38.78% |
| Gracias | 37.08% |


The strongest class recalls were **Baño (44.59%)**, **Dolor (38.78%)**, and **Sí/Hambre (38.20%)**. Performance is now more balanced: the weakest class, **No (33.33%)**, is substantially above the prior weakest class, Gracias (14.61%).

## Validation performance by model

The table below belongs to the 37.74% analysis run, which used k=1. The
production k=5 run has a kNN CV score of 30.87% and a learned ensemble weight
of 22.25%; its full model weights are reported in the run output.

| Model | 5-fold CV accuracy | Ensemble weight |
|---|---:|---:|
| kNN (k=1 analysis) | 31.92% | 23.22% |
| kNN (k=5 production) | 30.87% | 22.25% |
| Random Forest | 27.70% | 18.18% |
| SVM | 24.29% | 14.10% |
| MLP | 20.52% | 9.60% |
| Gradient Boosting | 20.17% | 9.18% |
| LDA | 19.16% | 7.97% |
| Logistic Regression | 18.12% | 6.72% |
| Decision Tree | 17.35% | 5.80% |
| AdaBoost | 15.47% | 3.55% |
| Naive Bayes | 13.90% | 1.68% |

## kNN cross-validation

| k | Mean CV accuracy |
| -: | -: |
| 1 | **34.32%** |
| 3 | 29.44% |
| 5 | 29.93% |
| 7 | 28.92% |
| 9 | 29.34% |
| 11 | 28.50% |
| 13 | 27.32% |
| 15 | 26.79% |
| 17 | 26.79% |
| 21 | 26.90% |
| 25 | 26.34% |
| 31 | 25.99% |
| 37 | 25.09% |


## Interpretation

The improved ensemble is substantially better than the original baseline, but remains only moderately better than chance. The similar macro and weighted scores show that class imbalance is not the primary cause of the remaining error. The confusion matrix is more balanced across classes, although substantial cross-class confusion remains.

The most important score for exploratory comparison is the **37.74% held-out accuracy**. The most important score for the stated production configuration is **30.08% with k=5**. CV scores were used for selection and weighting; they are not final performance estimates. Both test scores use a random window-level split and should not be interpreted as subject-independent generalization.

## Limitations

1. **The headline score is a window-level estimate, not a subject-independent score.** The random stratified split can place highly correlated windows from the same trial—and potentially the same participant—in both training and test sets. This can make the 37.74% estimate optimistic and does not measure generalization to a new person.

2. **The supplied release lacks reliable grouping metadata.** It does not expose subject IDs or explicit trial IDs. The current inferred grouping from consecutive filenames is only a proxy and must not be presented as a validated subject- or trial-wise split. A definitive leakage-safe result requires the original metadata or a dataset release with those identifiers.

3. **The class mapping requires independent verification.** The implementation follows the requested mapping of class 6 to `/Sí/` and class 8 to `/No/`, while the repository README lists those two folders in the opposite order. This discrepancy can invalidate comparisons unless confirmed against the original dataset metadata.

4. **The result is not directly comparable to the paper’s 96% result.** The paper reports binary `/Sí/` versus `/No/` classification with GA-selected electrodes/features; this experiment uses eight classes, all engineered features, and a voting ensemble. The different label space, objective, and evaluation protocol make the scores non-equivalent.

5. **The feature definitions are partly implementation-dependent.** Several paper formulas are ambiguous or garbled, especially RQA, DFA-related details, Hurst exponent, and fractal features. The implementation uses documented library defaults and explicit numerical guards, so exact feature-level reproduction of the paper is not guaranteed.

6. **The improved model is still a bounded baseline, not a production model.** It applies training-only ANOVA feature selection, class balancing where supported, bounded model settings, and validation-weighted soft voting. It does not yet tune the complete hyperparameter space, calibrate probabilities, optimize class-specific costs, or learn weights through nested out-of-fold stacking.

7. **The kNN elbow selection has limited evidence.** k=1 was selected using five-fold cross-validation after training-only feature selection, which avoids test leakage, but it is sensitive to feature scaling, duplicated/correlated windows, and the candidate grid. The production configuration therefore fixes k=5; its grouped-validation performance still needs to be measured independently.

8. **There is no uncertainty estimate.** One fixed seed and one held-out split do not quantify variation caused by sampling, initialization, or class composition. Confidence intervals and repeated grouped splits are needed before drawing strong conclusions.

9. **The data are not guaranteed to be independent or balanced by participant.** The class counts differ, and the release-level organization does not show whether recording conditions, sessions, or participants are equally represented. Weighted metrics can conceal weak performance on individual classes; macro metrics should therefore be reported alongside accuracy.

10. **The reported metric is not a real-time or deployment metric.** The experiment classifies pre-cut, one-second windows from a curated dataset. It does not evaluate latency, overlapping-stream aggregation, calibration, missing channels, electrode variability, or robustness to new recording sessions.

## Recommendation

Treat **30.08% with k=5** as the current production-oriented random-split result and **37.74% with k=1** as an exploratory upper comparison, not deployment claims. First resolve the class-label discrepancy and obtain subject/trial metadata. Then use repeated subject-wise or trial-wise nested cross-validation, report per-class and macro metrics with confidence intervals, and compare both results against the grouped protocol. The next modeling step should be nested out-of-fold stacking or calibrated soft voting evaluated under that grouped protocol.

