import numpy as np
from sklearn.metrics import recall_score, precision_score

def tune_threshold(y_true, y_proba):
    """
    Finds best probability threshold based on recall-precision tradeoff
    """

    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        results.append({
            "threshold": round(t, 2),
            "recall": round(recall, 3),
            "precision": round(precision, 3)
        })

    return results
