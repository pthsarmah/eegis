import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from fitness import classifier


def evaluate_genome(genome, features, labels, train, test, name, k=7):
    electrodes = np.flatnonzero(genome[:14])
    selected = np.flatnonzero(genome[14:])
    matrix = features[:, electrodes][:, :, selected].reshape(len(labels), -1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(matrix[train])
    model = classifier(name, k).fit(scaler.transform(matrix[train]), labels[train])
    prediction = model.predict(scaler.transform(matrix[test]))
    return {
        "accuracy": accuracy_score(labels[test], prediction),
        "precision_weighted": precision_score(labels[test], prediction,
                                              average="weighted",
                                              zero_division=0),
        "recall_weighted": recall_score(labels[test], prediction,
                                        average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels[test], prediction,
                                average="weighted", zero_division=0),
        "precision_macro": precision_score(labels[test], prediction,
                                           average="macro", zero_division=0),
        "recall_macro": recall_score(labels[test], prediction,
                                     average="macro", zero_division=0),
        "f1_macro": f1_score(labels[test], prediction,
                             average="macro", zero_division=0),
        "confusion_matrix_row_normalized":
            (confusion_matrix(labels[test], prediction, normalize="true")).tolist(),
    }
