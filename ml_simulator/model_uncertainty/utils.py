from typing import Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection._split import _BaseKFold

import numpy as np
import pandas as pd
import copy


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """mape metric
    """
    return 1 * (np.abs(y_true - y_pred) / np.abs(y_true)).mean()


def smape(y_true: np.array, y_pred: np.array) -> float:
    """smape metric
    """
    correction = np.abs(y_true) + np.abs(y_pred) == 0
    return np.mean(2 * np.abs(y_true - y_pred) /
                   (np.abs(y_true) + np.abs(y_pred) + correction))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """wape metric
    """
    return 1 * (np.abs(y_true - y_pred)).sum() / np.sum(np.abs(y_true))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """bias metric
    """
    return 1 * (y_pred - y_true).sum() /  np.sum(np.abs(y_true))


class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum groups for a single training set.
    test_size : int, default=None
        Number of groups in test
    gap : int, default=0
        Number of groups between train and test sets
    """

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        if isinstance(groups, pd.Series):
            _groups = groups.values
        else:
            _groups = copy.copy(groups)

        if isinstance(X, pd.DataFrame):
            main_indices = X.index
        else:
            main_indices = np.arange(X.shape[0])

        # sorting data according to group order
        keys = list(zip(_groups, main_indices))    
        keys = sorted(keys)
        _groups, main_indices = zip(*keys)
        _groups = np.array(_groups)
        main_indices = np.array(main_indices)
        
        # auxiliary variables
        _, start_index = np.unique(_groups, return_index=True)
        n_groups = len(start_index)
        if self.test_size is None:
            test_size = (n_groups - self.gap)  // (self.n_splits + 1)
        else:
            test_size = self.test_size

        # main loop
        for k in range(self.n_splits):
            # train
            train_end = (n_groups - self.gap -
                         (self.n_splits - k) * test_size)

            if self.max_train_size is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.max_train_size)

            train = main_indices[start_index[train_start]:
                                 start_index[train_end]]    

            # test
            test_start = train_end + self.gap
            test_end = test_start + test_size
            if n_groups - test_end == 0:
                test_end = None
            elif n_groups - test_end > 0:
                test_end = start_index[test_end]
            else:
                raise IndexError("not enough data for such setting")
            test_start = start_index[test_start]   
            test = main_indices[test_start:test_end]

            yield sorted(train), sorted(test)


def best_model() -> Any:
    """set model parameters
    """
    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error"
    }
    model = GradientBoostingRegressor(**params)

    return model
