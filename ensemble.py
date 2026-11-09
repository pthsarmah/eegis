import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                               RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


K_CANDIDATES = (1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 37)


def select_knn_k(x, y, seed=42, candidates=K_CANDIDATES, folds=5):
    """Select the smallest k within one standard error of the best CV score."""
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    curve = []
    for k in candidates:
        if k >= len(y):
            continue
        scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=k), x, y, cv=split, scoring="accuracy",
            n_jobs=-1)
        curve.append({
            "k": k,
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
        })
    best = max(item["mean_accuracy"] for item in curve)
    best_item = next(item for item in curve if item["mean_accuracy"] == best)
    threshold = best_item["mean_accuracy"] - best_item["std_accuracy"] / np.sqrt(folds)
    elbow = min(item["k"] for item in curve if item["mean_accuracy"] >= threshold)
    return elbow, curve


def make_ensemble(k, seed=42):
    models = [
        ("knn", KNeighborsClassifier(n_neighbors=k)),
        ("logreg", LogisticRegression(random_state=seed)),
        ("rf", RandomForestClassifier(random_state=seed)),
        ("dt", DecisionTreeClassifier(random_state=seed)),
        ("gbm", GradientBoostingClassifier(random_state=seed)),
        ("adaboost", AdaBoostClassifier(random_state=seed)),
        ("nb", GaussianNB()),
        ("lda", LinearDiscriminantAnalysis()),
        ("mlp", MLPClassifier(random_state=seed)),
        ("svm", SVC(random_state=seed)),
    ]
    return VotingClassifier(estimators=models, voting="hard", n_jobs=-1), models
