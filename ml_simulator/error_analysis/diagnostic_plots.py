import matplotlib.pyplot as plt
import scipy.stats as sps

import numpy as np


def xy_fitted_residuals(y_true, y_pred, plot=True):
    """Coordinates (x, y) for fitted residuals against true values."""
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals)
    plt.xlabel('fitted values')
    plt.ylabel('residuals')


def xy_normal_qq(y_true, y_pred, plot=True):
    """Coordinates (x, y) for normal Q-Q plot."""
    residuals = y_true - y_pred
    residuals_standardized = (residuals - residuals.mean()) / residuals.std()
    residuals_standardized = sorted(residuals_standardized)
    dist = sps.norm(loc=0, scale=1)
    theoretical_percentiles = np.linspace(0, 1, len(residuals), endpoint=False)
    theoretical_quantiles = dist.ppf(theoretical_percentiles)
    if plot:
        plt.plot(theoretical_quantiles, theoretical_quantiles, '--', color='r')
        plt.scatter(theoretical_quantiles, residuals_standardized)
        plt.xlabel('theoretical_quantiles')
        plt.ylabel('sample_quantiles')
    else:
        return theoretical_quantiles, np.array(residuals_standardized)
    

def xy_scale_location(y_true, y_pred, plot=True):
    """Coordinates (x, y) for scale-location plot."""
    residuals = y_true - y_pred
    residuals_standardized = (residuals - residuals.mean()) / residuals.std()
    if plot:
        plt.scatter(y_pred, np.sqrt(residuals_standardized))
        plt.xlabel('fitted values')
        plt.ylabel('sample_quantiles')
    else:
        return y_pred, np.sqrt(np.abs(residuals_standardized))
