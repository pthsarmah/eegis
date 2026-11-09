import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bands import make_layers
from data_loader import CLASS_NAMES, load_eight_class
from ensemble import make_ensemble, select_knn_k
from features import extract_features


def metrics(y_true, prediction):
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision_weighted": float(precision_score(
            y_true, prediction, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(
            y_true, prediction, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(
            y_true, prediction, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(
            y_true, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(
            y_true, prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(
            y_true, prediction, average="macro", zero_division=0)),
        "confusion_matrix_row_normalized": confusion_matrix(
            y_true, prediction, labels=np.arange(len(CLASS_NAMES)),
            normalize="true").tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--cache", default=".cache/features-8class.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    print(f"[run] loading eight-class data from {args.data}", flush=True)
    raw, labels, groups = load_eight_class(args.data, return_groups=True)
    print(f"[run] raw tensor={raw.shape}; counts={dict(zip(
        CLASS_NAMES, np.bincount(labels, minlength=len(CLASS_NAMES))))}", flush=True)
    layers = make_layers(raw)
    print(f"[run] layered tensor={layers.shape}", flush=True)
    features = extract_features(layers, args.cache)
    print(f"[run] feature tensor={features.shape}", flush=True)

    indices = np.arange(len(labels))
    train, test = train_test_split(
        indices, test_size=.2, stratify=labels, random_state=args.seed)
    x = features.reshape(len(features), -1)
    scaler = StandardScaler().fit(x[train])
    x_train, x_test = scaler.transform(x[train]), scaler.transform(x[test])

    print("[run] selecting kNN k with stratified cross-validation", flush=True)
    k, cv_curve = select_knn_k(
        x_train, labels[train], seed=args.seed, folds=args.cv_folds)
    print(f"[run] elbow k={k}; CV curve={cv_curve}", flush=True)

    ensemble, models = make_ensemble(k, args.seed)
    print("[run] fitting 10-model hard-voting ensemble", flush=True)
    ensemble.fit(x_train, labels[train])
    prediction = ensemble.predict(x_test)
    result = metrics(labels[test], prediction)
    result.update({
        "class_names": CLASS_NAMES,
        "model_names": [name for name, _ in models],
        "knn_k": k,
        "knn_cv_curve": cv_curve,
        "train_size": len(train),
        "test_size": len(test),
    })
    print("[run] final evaluation complete", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
