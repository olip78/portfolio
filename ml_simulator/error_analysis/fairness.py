from typing import List
from sklearn.metrics import pairwise_distances
from sklearn.metrics import log_loss

import numpy as np


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss terms"""
    if len(y_true) == len(y_pred):
        mask = (y_true == 0)*(y_pred == 0)
        mask += (y_true == 1)*(y_pred == 1)
        if sum(mask) == 0:
            loss = y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)
        else:
            raise ValueError("shape mismatch")
    else:
        raise ValueError("shape mismatch")
    return loss


def fairness(residuals: np.ndarray) -> float:
    """Compute Gini fairness of array of values"""
    _residuals = np.abs(residuals)
    _residuals /= _residuals.sum()
    gini = pairwise_distances(_residuals.reshape(-1, 1), metric='l1')
    gini = np.triu(gini).sum() / (_residuals.mean() * len(residuals)**2)
    return 1 - gini


def best_prediction(
    y_true: np.ndarray, y_preds: List[np.ndarray], fairness_drop: float = 0.05
) -> int:
    """Find index of best model"""
    models_residuals = [logloss(y_true, y_pred) for y_pred in y_preds]
    loglosses = [-residuals.mean() for residuals in models_residuals]
    fairnesses = [fairness(residuals) for residuals in models_residuals]

    loglosses = [ll - loglosses[0] for ll in loglosses]
    delta_fairness = [fairnesses[0] - f for f in fairnesses]
    mask = np.array(delta_fairness) <= fairness_drop
    indices = np.arange(len(loglosses))[mask]
    idx = np.argmin(np.array(loglosses)[indices])
    return int(indices[idx])
