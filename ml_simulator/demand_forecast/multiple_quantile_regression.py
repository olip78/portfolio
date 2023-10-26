from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm

def split_train_test(
    df: pd.DataFrame,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets

    The last `test_days` days are held out for testing

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data
        test_days (int): The number of days to include in the test set (default: 30)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        A tuple containing the train and test DataFrames
    """
    # preprocessing
    df['day'] = pd.to_datetime(df.day, format='%Y-%m-%d')
    mask = df.isna().sum(axis=1)==0
    df = df[mask]
    
    # splitting
    test_days = test_days + 1
    all_days = sorted(df.day.unique())
    threshold = all_days[-test_days:][0]
    df_train = df[df.day < threshold]
    df_test = df[df.day >= threshold]
    assert df_test.day.unique().shape[0] == test_days
    return df_train, df_test


class MultiTargetModel:
    def __init__(
        self,
        features: List[str],
        horizons: List[int] = [7, 14, 21],
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """
        Parameters
        ----------
        features : List[str]
            List of features columns.
        horizons : List[int]
            List of horizons.
        quantiles : List[float]
            List of quantiles.

        Attributes
        ----------
        fitted_models_ : dict
            Dictionary with fitted models for each sku_id.
            Example:
            {
                sku_id_1: {
                    (quantile_1, horizon_1): model_1,
                    (quantile_1, horizon_2): model_2,
                    ...
                },
            }

        """
        self.quantiles = quantiles
        self.horizons = horizons
        self.sku_col = "sku_id"
        self.date_col = "day"
        self.features = features
        self.targets = [f"next_{horizon}d" for horizon in self.horizons]
        self.fitted_models_ = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """Fit model on data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit on.
        verbose : bool, optional
        """
        if verbose:
            all_sku = tqdm(data[self.sku_col].unique())
        else:
            all_sku = data[self.sku_col].unique()

        for sku in all_sku:
            df_tmp_loop = data[data[self.sku_col]==sku]
            X_train = df_tmp_loop.loc[:, self.features]
            self.fitted_models_[sku] = {}
            for horizon in self.horizons:
                for quantile in self.quantiles:
                    y_train = df_tmp_loop.loc[:, f"next_{horizon}d"]
                    model = QuantileRegressor(quantile=quantile, solver="highs")
                    model.fit(X_train, y_train)
                    self.fitted_models_[sku][(quantile, horizon)] = model
                    

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on data.

        Predict 0 values for a new sku_id.

        Parameters
        ----------
        data : pd.DataFrame
            Data to predict on.

        Returns
        -------
        pd.DataFrame
            Predictions.
        """
        predictions = pd.DataFrame([])
        for sku in data[self.sku_col].unique():
            df_tmp_loop = data[data[self.sku_col]==sku]
            X = df_tmp_loop.loc[:, self.features]
            pred_df = df_tmp_loop.loc[:, [self.sku_col, self.date_col]]
            for horizon in self.horizons:
                for quantile in self.quantiles:
                    if sku in self.fitted_models_.keys():
                        pred = self.fitted_models_[sku][(quantile, horizon)].predict(X)
                    else:
                        pred = 0
                    pred_df.loc[:, 
                                f'pred_{horizon}d_q{int(quantile*100)}'] = pred
            predictions = pd.concat([predictions, pred_df], axis=0)
        return predictions


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile loss between the true and predicted values.

    The quantile loss measures the deviation between the true
        and predicted values at a specific quantile.

    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    quantile : float
        The quantile to calculate the loss for.

    Returns
    -------
    float
        The quantile loss.
    """
    zeros = np.zeros(y_true.shape)
    loss = (quantile * np.maximum(zeros, y_true - y_pred) + 
     (1 - quantile) * np.maximum(zeros, y_pred - y_true))
    return loss.mean()


def evaluate_model(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    horizons: List[int] = [7, 14, 21],
) -> pd.DataFrame:
    """Evaluate model on data.

    Parameters
    ----------
    df_true : pd.DataFrame
        True values.
    df_pred : pd.DataFrame
        Predicted values.
    quantiles : List[float], optional
        Quantiles to evaluate on, by default [0.1, 0.5, 0.9].
    horizons : List[int], optional
        Horizons to evaluate on, by default [7, 14, 21].

    Returns
    -------
    pd.DataFrame
        Evaluation results.
    """
    losses = {}

    for quantile in quantiles:
        for horizon in horizons:
            true = df_true[f"next_{horizon}d"].values
            pred = df_pred[f"pred_{horizon}d_q{int(quantile*100)}"].values
            loss = quantile_loss(true, pred, quantile)

            losses[(quantile, horizon)] = loss

    losses = pd.DataFrame(losses, index=["loss"]).T.reset_index()
    losses.columns = ["quantile", "horizon", "avg_quantile_loss"]  # type: ignore

    return losses



def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile loss between the true and predicted values.

    The quantile loss measures the deviation between the true
        and predicted values at a specific quantile.

    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    quantile : float
        The quantile to calculate the loss for.

    Returns
    -------
    float
        The quantile loss.
    """
    zeros = np.zeros(y_true.shape)
    loss = (quantile * np.maximum(zeros, y_true - y_pred) + 
     (1 - quantile) * np.maximum(zeros, y_pred - y_true))
    return loss.mean()


def evaluate_model(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    horizons: List[int] = [7, 14, 21],
) -> pd.DataFrame:
    """Evaluate model on data.

    Parameters
    ----------
    df_true : pd.DataFrame
        True values.
    df_pred : pd.DataFrame
        Predicted values.
    quantiles : List[float], optional
        Quantiles to evaluate on, by default [0.1, 0.5, 0.9].
    horizons : List[int], optional
        Horizons to evaluate on, by default [7, 14, 21].

    Returns
    -------
    pd.DataFrame
        Evaluation results.
    """
    losses = {}

    for quantile in quantiles:
        for horizon in horizons:
            true = df_true[f"next_{horizon}d"].values
            pred = df_pred[f"pred_{horizon}d_q{int(quantile*100)}"].values
            loss = quantile_loss(true, pred, quantile)
            losses[(quantile, horizon)] = loss

    losses = pd.DataFrame(losses, index=["loss"]).T.reset_index()
    losses.columns = ["quantile", "horizon", "avg_quantile_loss"]  # type: ignore

    return losses
