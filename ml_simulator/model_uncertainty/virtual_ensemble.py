from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class PredictionDict:
    pred: np.ndarray = np.array([])
    uncertainty: np.ndarray = np.array([])
    pred_virt: np.ndarray = np.array([])
    lcb: np.ndarray = np.array([])
    ucb: np.ndarray = np.array([])
    final_pred: np.ndarray = np.array([])


def virtual_ensemble_iterations(
    model: GradientBoostingRegressor, k: int = 20
) -> List[int]:
    """ returns how many first trees to include in each virtual ensemble
    parameters:
      k: n of estimators step
    """
    n_estimators = model.n_estimators
    if k < n_estimators // 2:
        iterations = range(n_estimators // 2 - 1, n_estimators, k)
    else:
        raise IndexError("to little n_estimators for such k")
    return list(iterations)


def virtual_ensemble_predict(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> np.ndarray:
    """ gives the predictions for each object from X 
    for each model in the virtual ensemble as a matrix of size (N, K), where
    N is the number of rows in X and 
    K is the number of models in the virtual ensemble
    """
    iterations = virtual_ensemble_iterations(model, k)

    # predict using subsets
    stage_preds = []
    for stage, predictions in enumerate(model.staged_predict(X)):
        if stage in iterations:
            stage_preds.append(predictions)
    stage_preds = np.array(stage_preds).T 
    assert stage_preds.shape == (X.shape[0], len(iterations))
    return stage_preds


def predict_with_uncertainty(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> PredictionDict:
    stage_preds = virtual_ensemble_predict(model, X, k)
    final_pred = model.predict(X)
    uncertainty = np.var(stage_preds, axis=1)
    pred_virt = stage_preds.mean(axis=1)
    lcb = pred_virt - 3*np.sqrt(uncertainty)
    ucb = pred_virt + 3*np.sqrt(uncertainty)
    prediction_dict = PredictionDict(**{'pred': stage_preds, 'uncertainty': uncertainty, 
                       'pred_virt': pred_virt, 'lcb': lcb, 'ucb': ucb,
                        'final_pred': final_pred})
    return prediction_dict
