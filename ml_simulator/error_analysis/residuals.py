import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Residuals"""
    return y_true - y_pred


def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Squared errors"""
    return (y_true - y_pred)**2


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


def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAPE terms"""
    loss = np.zeros(y_pred.shape)
    mask = (y_true == 0)*(y_pred != 0)
    if min(y_true) <= 0 or min(y_pred) < 0:
        raise ValueError
    mask = (y_true == 0) * (y_pred == 0)
    mask = [not m for m in mask]
    _loss = 0
    if sum(mask) > 0:
        _y_true = y_true[mask]
        _y_pred = y_pred[mask]
        _loss = (_y_true - _y_pred) / _y_true
    loss[mask] = _loss
    return loss


def quantile_loss(
    y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.01
) -> np.ndarray:
    """Quantile loss terms"""
    if len(y_true) == len(y_pred):
        zeros = np.zeros(np.array(y_true).shape)
        loss = (q*np.maximum(zeros,  np.array(y_true) - np.array(y_pred)) + 
                (1 - q)*np.maximum(zeros, np.array(y_pred) - np.array(y_true)))
    else:
        raise ValueError("shape mismatch")
    return loss
