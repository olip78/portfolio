from typing import Tuple
import datetime
import numpy as np
import pandas as pd


def week_missed_profits(
    df: pd.DataFrame,
    sales_col: str,
    forecast_col: str,
    date_col: str = "day",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Calculates the missed profits every week for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed profits for.
        (Must contain columns "sku_id", "date", "price", "sales" and "forecast")
    sales_col : str
        The column with the actual sales.
    forecast_col : str
        The column with the forecasted sales.
    price_col : str, optional
        The column with the price, by default "price".

    Returns
    -------
    pd.DataFrame
        The DataFrame with the missed profits.
        (Contains columns "day", "revenue", "missed_profits")
    """
    def get_sunday(dates):
        """gets dates within a week, returns the sunday date
        """
        res = max(dates)
        if res.weekday != 6:
            res += datetime.timedelta(days=6-res.weekday())
        return res

    _df = df.copy()
    _df[date_col] = pd.to_datetime(_df[date_col], format='%Y-%m-%d')
    _df["revenue"] = _df[sales_col]*_df[price_col]
    mask = _df[forecast_col] > _df[sales_col]
    idx = _df[mask].index
    _df["missed_profits"] = 0
    _df.loc[idx, "missed_profits"] = (_df.loc[idx, forecast_col] -
                                _df.loc[idx, sales_col] ) * _df.loc[idx, price_col]
    columns = [date_col, "revenue", "missed_profits"]
    _df = _df.loc[:, columns]
    _df = _df.groupby(pd.Grouper(key='day', freq='W'), as_index=False).agg(
        day=(date_col, get_sunday),
        revenue=('revenue', 'sum'),
        missed_profits=('missed_profits', 'sum')
    )
    return _df


def missed_profits_ci(
    df: pd.DataFrame,
    missed_profits_col: str,
    confidence_level: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]:
    """
    Estimates the missed profits for the given DataFrame.
    Calculates average missed_profits per week and estimates
    the 95% confidence interval.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed_profits for.

    missed_profits_col : str
        The column with the missed_profits.

    confidence_level : float, optional
        The confidence level for the confidence interval, by default 0.95.

    n_bootstraps : int, optional
        The number of bootstrap samples to use for the confidence interval,
        by default 1000.

    Returns
    -------
    Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]
        Returns a tuple of tuples, where the first tuple is the absolute average
        missed profits with its CI, and the second is the relative average missed
        profits with its CI.

    Example:
    -------
    ((1200000, (1100000, 1300000)), (0.5, (0.4, 0.6)))
    """
    alpha = 1 - confidence_level
    n_weeks = df.shape[0]
    mean_revenue = df["revenue"].mean()
    
    _df = df.copy()
    samples = np.random.randint(0, n_weeks, (n_weeks, n_bootstraps))

    values = _df[missed_profits_col].values
    mid = values.mean()
    values = values[samples].mean(axis=0)
    bounds_abs = np.quantile(values, [alpha/2, 1-alpha/2], axis=0)

    res = ((mid, tuple(bounds_abs)), 
           (mid/mean_revenue, tuple(bounds_abs/mean_revenue)))
    return res
