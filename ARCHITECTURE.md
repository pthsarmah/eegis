# EEGIS Eight-Class Ensemble Architecture

## End-to-end pipeline

```mermaid
flowchart TD
    A[data/raw/class_1..class_8 CSV files] --> B[Load eight imagined-speech classes]
    B --> C[Raw tensor<br/>N x 14 x 128]
    C --> D[4th-order Butterworth bandpass<br/>sosfiltfilt, fs=128 Hz]
    D --> E[Layer tensor<br/>N x 14 x 128 x 6]
    C --> E

    E --> F[Feature extraction per channel/layer]
    F --> G[Raw: 34 features]
    F --> H[Each band: 36 features]
    G --> I[Feature map<br/>N x 14 x 214]
    H --> I
    I --> J[Cache to .npy]

    I --> K[Flatten per instance<br/>N x 2996]
    K --> L[Stratified 80/20 split]
    L --> M[Fit StandardScaler on train only]
    M --> N[Training matrix]
    M --> O[Test matrix]

    N --> P[Remove constant features]
    P --> Q[ANOVA SelectKBest<br/>512 features]
    Q --> R[Selected training features]
    O --> S[Apply fitted selector]
    S --> T[Selected test features]

    R --> U[kNN CV sweep]
    U --> V[Analytical k selection<br/>k=1]
    R --> W[Production ensemble<br/>force k=5]
    T --> X[Final production prediction]

    R --> W
    W --> Y[10 classifiers]
    Y --> Y1[kNN]
    Y --> Y2[Logistic Regression]
    Y --> Y3[Random Forest]
    Y --> Y4[Decision Tree]
    Y --> Y5[Gradient Boosting]
    Y --> Y6[AdaBoost]
    Y --> Y7[GaussianNB]
    Y --> Y8[LDA]
    Y --> Y9[MLP]
    Y --> Y10[SVM]

    Y --> Z[Training-only cross-validation scores]
    Z --> AA[Validation-derived soft-vote weights]
    Y --> AB[Fit models on complete training set]
    AB --> AC[Probability predictions]
    AA --> AD[Weighted probability aggregation]
    AC --> AD
    AD --> X
    X --> AE[Accuracy, precision, recall, F1]
    X --> AF[8 x 8 normalized confusion matrix]
```

## Feature extraction

```mermaid
flowchart LR
    A[One instance<br/>14 channels x 128 samples] --> B[Raw layer]
    A --> C[Delta<br/>0.5-4 Hz]
    A --> D[Theta<br/>4-8 Hz]
    A --> E[Alpha<br/>8-13 Hz]
    A --> F[Beta<br/>13-30 Hz]
    A --> G[Gamma<br/>30-40 Hz]

    B --> H[34 features]
    C --> I[36 features]
    D --> J[36 features]
    E --> K[36 features]
    F --> L[36 features]
    G --> M[36 features]

    H --> N[214 features/channel]
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
    N --> O[14 channels x 214 features]
```

Feature groups include statistical descriptors, DFA, Hjorth parameters,
entropy, fractal dimensions, Welch spectral power, spectral edge frequencies,
RQA measures, relative band power, and Hilbert-envelope amplitude. Degenerate
values are converted with `nan_to_num`.

## Model-selection and production boundary

```mermaid
flowchart TD
    A[Training split only] --> B[Fit scaler]
    B --> C[Remove zero-variance features]
    C --> D[Fit ANOVA selector]
    D --> E[Selected features]
    E --> F[5-fold stratified CV]
    F --> G[Evaluate k candidates]
    G --> H[Analytical choice: k=1]
    E --> I[Production choice: k=5]

    E --> J[Fit ten models]
    J --> K[Out-of-fold model accuracies]
    K --> L[Weights proportional to CV score above chance]
    I --> J
    L --> M[Weighted soft voting]
    M --> N[Held-out test prediction]
```

The k=1 result is retained for analysis because it maximized cross-validation
accuracy. Production inference uses k=5 to reduce sensitivity to individual
nearest-neighbor windows and noisy samples.

## Evaluation boundary

The final test set is not used for feature selection, k selection, model
weighting, or hyperparameter tuning. The current headline production result is
the eight-class, random window-level test accuracy with k=5. Subject- or
trial-independent evaluation requires reliable subject/trial identifiers,
which are not present in the supplied dataset release.
