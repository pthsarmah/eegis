import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

K_VALUES = (5, 7, 13, 15, 17, 21, 37)


def classifier(name, k=7):
    models = {
        "LogReg": LogisticRegression(),
        "RF": RandomForestClassifier(),
        "DT": DecisionTreeClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "GaussianNB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "MLP": MLPClassifier(),
        "SVC": SVC(),
    }
    return KNeighborsClassifier(k) if name == "kNN" else models[name]


def evaluate_genome(genome, features, labels, train, validation, name):
    electrodes = np.flatnonzero(genome[:14])
    selected = np.flatnonzero(genome[14:])
    if not len(electrodes) or not len(selected):
        return 0., None
    matrix = features[:, electrodes][:, :, selected].reshape(len(labels), -1)
    scaler = StandardScaler().fit(matrix[train])
    x_train, x_val = scaler.transform(matrix[train]), scaler.transform(matrix[validation])
    if name == "kNN":
        scores = []
        for k in (value for value in K_VALUES if value <= len(train)):
            model = classifier(name, k).fit(x_train, labels[train])
            scores.append((accuracy_score(labels[validation], model.predict(x_val)), k))
        return max(scores), (scaler, matrix)
    model = classifier(name).fit(x_train, labels[train])
    return accuracy_score(labels[validation], model.predict(x_val)), (scaler, matrix)
