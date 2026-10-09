from typing import Optional
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import residuals


def worst_best_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    case: str = 'best',
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k best cases according to the given function"""
    if func is None:
        residual_function = getattr(residuals, "residuals")
    elif func in ['squared_errors', 'logloss', 'ape', 'residuals', 'quantile_loss']:
        residual_function = getattr(residuals, func)
    else:
        raise KeyError('wrong func name')
    resid = np.abs(residual_function(y_test, y_pred))
    resid = pd.Series(resid, index=y_test.index)
    if mask is None:
        _mask = [True]*len(resid)
    else:
        _mask = mask.values
    indices = [i for _, i in sorted(zip(resid[_mask], y_test.index[_mask]))]
    if case == 'best':
        indices = indices[:top_k]
    elif case == 'worst':
        indices = indices[-top_k:][::-1]
    else:
        raise ValueError('case variable can only take a "worst" or "best" value')
    result = {
        "X_test": X_test.loc[indices, :],
        "y_test": y_test[indices],
        "y_pred": y_pred[indices],
        "resid": resid[indices],
    }
    return result


def adversarial_validation(
    classifier: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    quantile: float = 0.1,
    func: Optional[str] = None,
) -> dict:
    """Adversarial validation residual analysis"""
    n_worst = int(quantile*len(y_test))
    worst_cases = worst_best_cases(
                                   X_test,
                                   y_test,
                                   y_pred,
                                   top_k=n_worst,
                                   case='worst',
                                   func=func
                                )
    data = X_test.copy()
    data['lables'] = 0
    data.loc[worst_cases['X_test'].index, 'lables'] = 1
    train_y = data['lables']
    train_x = data.drop('lables', axis=1)
    classifier.fit(train_x, train_y)
    roc_auc = roc_auc_score(train_y, classifier.predict_proba(train_x)[:, 1])
    if 'feature_importances_' in dir(classifier):
        feature_importances = pd.Series(np.abs(classifier.feature_importances_), index=train_x.columns)
    else:
        feature_importances = pd.Series(np.abs(classifier.coef_[0]), index=train_x.columns)
    result = {
        "ROC-AUC": roc_auc,
        "feature_importances": feature_importances,
    }

    return result
