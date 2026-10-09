from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000, samples_out: bool = False
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC
    """
    alpha = 1 - conf
    sample_length = X.shape[0]

    # precomputed predictions
    preds = classifier.predict_proba(X)[:, 1]
    samples_roc_auc = []

    counter = len(samples_roc_auc)
    while counter < n_bootstraps:
        sample = np.random.randint(0, sample_length, (sample_length, 1))
        mask = y[sample].sum()
        if mask > 0 and mask < y.shape[0]:
            target = y[sample]
            pred = preds[sample]
            score = roc_auc_score(target, pred)
            samples_roc_auc.append(score)
            counter +=1

    bounds = np.quantile(np.array(samples_roc_auc), [alpha/2, 1-alpha/2], axis=0)
    if samples_out:
        res = (bounds, samples_roc_auc)
    else:
        res = tuple(bounds)
    return res


def tn_fn_tp_fp(y_true: np.ndarray,
                y_prob: np.ndarray,
                tn_flag: bool = True
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    """TN, FN, TP, FP calculation
    """
    sample_length = y_true.shape[0]
    y_sum = y_true.sum()
    TP = np.hstack([np.array([0]), y_true]).cumsum()
    FN  = np.hstack([np.array([y_sum]), - y_true]).cumsum()
    if tn_flag:
        TN  = np.hstack([np.array([sample_length - y_sum]),  y_true - 1]).cumsum()
    else:
        TN = None
    FP = np.hstack([np.array([0]), 1 - y_true]).cumsum()
    return TN, FN, TP, FP


def pr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float
) -> Tuple[float, float]:
    """Returns threshold and recall (from Precision-Recall Curve)
    """
    recall, precision, _, _ = pr_curve(y_true, y_prob, n_bootstrap=0)
    idx = np.arange(precision.shape[0])
    idx = idx[precision >= min_precision]
    if len(idx) < 2:
        threshold_proba = 1.0
        max_recall = 0.0
    elif len(idx) == len(precision):
        threshold_proba = 0.0
        max_recall = 1.0
    else:
        index = idx[-1]
        max_recall = recall[index]
        while recall[index-1] == max_recall:
            index -= 1
        threshold_proba = y_prob[index-1]
    return threshold_proba, max_recall

def sr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Specificity-Recall Curve)"""
    recall, specificity, _, _ = sr_curve(y_true, y_prob, n_bootstrap=0)

    low, upper = 0, len(specificity) - 1
    #cuted binary search for long arrays
    if len(specificity) > 1000:        
        steps = 0
        while steps < 5:
            mid = low + (upper - low) // 2
            if specificity[mid] < min_specificity:
                upper = mid - 1
            elif specificity[mid] > min_specificity:
                low = mid + 1
            steps += 1

    length = upper - low + 1
    idx = np.arange(length)
    index = idx[specificity[low:upper+1] >= min_specificity][-1]
    index += low
    threshold_proba = y_prob[index - 1]
    max_recall = recall[index]   
    return threshold_proba, max_recall


def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Precision-Recall curve and it's (LCB, UCB)
    """
    alpha = 1 - conf

    _, FN, TP, FP = tn_fn_tp_fp(y_true, y_prob, tn_flag=False)    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    precision = np.nan_to_num(precision, nan=1, posinf=1, neginf=1)
    recall = np.nan_to_num(recall, nan=0, posinf=0, neginf=0)

    if n_bootstrap == 0:
        return recall, precision, None, None

    # sample indices generation
    sample_length = y_true.shape[0]
    samples = np.random.randint(1, sample_length, (sample_length, n_bootstrap))

    precisions = []
    for k in range(n_bootstrap):
        sample = samples[:, k]
        target = y_true[sample]
        pred = y_prob[sample]

        idx = np.argsort(pred)[::-1]
        pred = pred[idx]
        target = target[idx]

        _, FN, TP, FP = tn_fn_tp_fp(target, pred, tn_flag=False)
        precision_ = TP / (TP + FP)
        precision_ = np.nan_to_num(precision_, nan=1, posinf=1, neginf=1)

        recall_ = TP / (TP + FN)
        recall_ = np.nan_to_num(recall_, nan=0, posinf=0, neginf=0)

        # interpolation re -> recall
        precision_ = np.interp(recall, recall_, precision_, left=0, right=1)
        precisions.append(precision_)

    precisions = np.vstack(precisions)
    bounds = np.quantile(precisions, [alpha/2, 1-alpha/2], axis=0)
    precision_lcb = bounds[0]
    precision_ucb = bounds[1]

    return (recall, precision, precision_lcb, precision_ucb)


def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Specificity-Recall curve and it's (LCB, UCB)
    """
    alpha = 1 - conf
    TN, FN, TP, FP = tn_fn_tp_fp(y_true, y_prob)
    specificity = TN / (FP + TN)
    recall = TP / (TP + FN)
    specificity = np.nan_to_num(specificity, nan=0, posinf=0, neginf=0)
    recall = np.nan_to_num(recall, nan=0, posinf=0, neginf=0)

    if n_bootstrap == 0:
        return recall, specificity, None, None

    # sample indices generation
    sample_length = y_true.shape[0]
    samples = np.random.randint(1, sample_length, (sample_length, n_bootstrap))

    specificities = []
    for k in range(n_bootstrap):
        sample = samples[:, k]
        target = y_true[sample]
        pred = y_prob[sample]
        idx = np.argsort(pred)[::-1]
        pred = pred[idx]
        target = target[idx]

        TN, FN, TP, FP = tn_fn_tp_fp(target, pred)
        specificity_ = TN / (FP + TN)
        recall_ = TP / (TP + FN)
        specificity_ = np.nan_to_num(specificity_, nan=0, posinf=0, neginf=0)
        recall_ = np.nan_to_num(recall_, nan=0, posinf=0, neginf=0)

        # interpolation re -> recall
        specificity_ = np.interp(recall, recall_, specificity_, left=0, right=1)
        specificities.append(specificity_)

    specificities = np.vstack(specificities)
    bounds = np.quantile(specificities, [alpha/2, 1-alpha/2], axis=0)
    specificity_lcb = np.clip(bounds[0], 0, 1)
    specificity_ucb = np.clip(bounds[1], 0, 1)

    return (recall, specificity, specificity_lcb, specificity_ucb)
