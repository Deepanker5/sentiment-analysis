# src/metrics.py
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def full_report(y_true, y_pred, labels):
    return classification_report(y_true, y_pred, target_names=labels, digits=4)

def confmat(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
