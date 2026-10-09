from typing import Tuple, Optional
from scipy.stats import shapiro, ttest_1samp, bartlett, levene, fligner

import numpy as np


def test_normality(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    test: str = 'shapiro',
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """Normality test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the normality test

    is_rejected : bool
        True if the normality hypothesis is rejected, False otherwise

    """
    residuals = y_true - y_pred
    if test == 'shapiro':
        _, p_value = shapiro(residuals)
    elif test == 'tt':
        _, p_value = shapiro(residuals, popmean=0)
    is_rejected = p_value < alpha

    return p_value, is_rejected

def test_unbiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefer: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Unbiasedness test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    prefer : str, optional (default=None)
        If None or "two-sided", test whether the residuals are unbiased.
        If "positive", test whether the residuals are unbiased or positive.
        If "negative", test whether the residuals are unbiased or negative.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the unbiasedness hypothesis is rejected, False otherwise

    """
    residuals = y_true - y_pred

    if prefer is None or prefer == "two-sided":
        alternative = 'two-sided'
    elif prefer == 'positive':
        alternative = 'greater'
    elif prefer == 'negative':
        alternative = 'less'
    res = ttest_1samp(residuals, 0, alternative=alternative)
    p_value = res.pvalue
    is_rejected = p_value < alpha

    return p_value, is_rejected


def test_homoscedasticity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 10,
    test: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Homoscedasticity test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    bins : int, optional (default=10)
        Number of bins to use for the test.
        All bins are equal-width and have the same number of samples, except
        the last bin, which will include the remainder of the samples
        if n_samples is not divisible by bins parameter.

    test : str, optional (default=None)
        If None or "bartlett", perform Bartlett's test for equal variances.
        If "levene", perform Levene's test.
        If "fligner", perform Fligner-Killeen's test.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the homoscedasticity hypothesis is rejected, False otherwise

    """
    residuals = y_true - y_pred
    residuals = [r for _, r in sorted(zip(y_true, residuals))]

    correction = len(residuals) % bins
    length = len(residuals) // (bins - np.sign(correction))
    samples = [residuals[k*length:(k+1)*length] for k in range(bins - np.sign(correction))]
    if correction:
        samples += [residuals[-correction:]]

    if test is None or test == 'bartlett':
        _test = bartlett
    elif test == 'levene':
        _test = levene
    elif test == 'fligner':
        _test = fligner
    _, p_value = _test(*samples)
    is_rejected = p_value < alpha

    return p_value, is_rejected
