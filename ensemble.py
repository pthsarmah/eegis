import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

K_CANDIDATES = (1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 37)


def select_knn_k(x, y, seed=42, candidates=K_CANDIDATES, folds=5):
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    curve = []
    for k in candidates:
        if k >= len(y):
            continue
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), x, y,
                                 cv=split, scoring="accuracy", n_jobs=-1)
        curve.append({"k": k, "mean_accuracy": float(scores.mean()),
                      "std_accuracy": float(scores.std())})
    best = max(item["mean_accuracy"] for item in curve)
    best_item = next(item for item in curve if item["mean_accuracy"] == best)
    threshold = best - best_item["std_accuracy"] / np.sqrt(folds)
    return min(item["k"] for item in curve if item["mean_accuracy"] >= threshold), curve


def select_features(x, y, max_features=512):
    """Fit feature selection on training data only."""
    variance = VarianceThreshold(threshold=0.0).fit(x)
    nonconstant = variance.transform(x)
    k = min(max_features, nonconstant.shape[1])
    selector = Pipeline([
        ("variance", variance),
        ("univariate", SelectKBest(f_classif, k=k)),
    ]).fit(x, y)
    return selector, selector.transform(x)


def make_models(k, seed=42):
    return [
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ("logreg", LogisticRegression(C=1., class_weight="balanced",
                                       max_iter=500, random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       min_samples_leaf=2, n_jobs=-1,
                                       random_state=seed)),
        ("dt", DecisionTreeClassifier(class_weight="balanced", max_depth=20,
                                      min_samples_leaf=2, random_state=seed)),
        ("gbm", GradientBoostingClassifier(n_estimators=150, learning_rate=.05,
                                           max_depth=2, random_state=seed)),
        ("adaboost", AdaBoostClassifier(n_estimators=200, learning_rate=.05,
                                        random_state=seed)),
        ("nb", GaussianNB(var_smoothing=1e-8)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128,), alpha=.001,
                              early_stopping=True, max_iter=300,
                              random_state=seed)),
        ("svm", SVC(C=2., gamma="scale", probability=True,
                    class_weight="balanced", random_state=seed)),
    ]


def fit_weighted_ensemble(models, x, y, cv=5, seed=42, cv_x=None,
                          selector_k=None):
    """Fit all models and weight soft votes by out-of-fold accuracy."""
    split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    fitted, scores = [], []
    cv_x = x if cv_x is None else cv_x
    for name, model in models:
        cv_model = model
        if selector_k is not None:
            cv_model = Pipeline([
                ("variance", VarianceThreshold(threshold=0.0)),
                ("select", SelectKBest(f_classif, k=selector_k)),
                ("model", model),
            ])
        score = cross_val_score(cv_model, cv_x, y, cv=split, scoring="accuracy",
                                n_jobs=1).mean()
        fitted.append((name, clone(model).fit(x, y)))
        scores.append(float(score))
        print(f"[ensemble] {name}: CV accuracy={score:.4f}", flush=True)
    # Keep every requested classifier, but prevent weak models from dominating.
    chance = 1. / len(np.unique(y))
    weights = np.maximum(np.asarray(scores) - chance, 0.) + 1e-6
    return fitted, (weights / weights.sum()).tolist(), scores


def predict_weighted_ensemble(fitted, weights, x):
    probabilities = np.stack(
        [model.predict_proba(x) for (_, model) in fitted], axis=0)
    return np.argmax(np.sum(probabilities * np.asarray(weights)[:, None, None],
                            axis=0), axis=1)
