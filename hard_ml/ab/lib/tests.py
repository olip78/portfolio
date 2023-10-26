import os
import json

import pandas as pd
import numpy as np

from scipy.stats import ttest_ind, t, norm

def calculate_theta(y, y_covariat) -> float:
    covariance = np.cov(y_covariat, y)[0, 1]
    variance = y_covariat.var()
    theta = covariance / variance
    return theta


def calc_strat_mean_var(df: pd.DataFrame, strat_column: str, target_name: str, strat_weights: pd.Series):
    """Compute stratified mean and variance.
    
    df - dataframe with target metric and data for stratification
    strat_column - names of the column for stratification
    target_name - name of the column with the target variable
    weights - dictionary - {stratum name: stratum weight}
    
    return: (float, float), mean_strat, var_strat
    """
    strat_mean = df.groupby(strat_column)[target_name].mean()
    mean = (strat_mean * strat_weights).sum()
    strat_var = df.groupby(strat_column)[target_name].var()
    var = (strat_var * strat_weights).sum()
    return mean, var

def run_strat_ttest(
    df_pilot: pd.DataFrame, df_control: pd.DataFrame,
    strat_column: str, target_name: str,
    strat_weights: pd.Series, alpha=0.95,
):
    """Tests the hypothesis of equality of means.
    
    Returns 1 if the mean of the metric in the pilot group
    is significantly greater than the control group, otherwise 0.
    """
    mean_strat_pilot, var_strat_pilot = calc_strat_mean_var(df_pilot, strat_column, 
                                                            target_name, strat_weights
                                                           )
    mean_strat_control, var_strat_control = calc_strat_mean_var(df_control, strat_column, 
                                                                target_name, strat_weights
                                                               )
    delta_mean_strat = mean_strat_pilot - mean_strat_control
    std_mean_strat = (var_strat_pilot / len(df_pilot) + var_strat_control / len(df_control)) ** 0.5
    
    pp = norm.ppf(alpha)
    left_bound = delta_mean_strat - pp * std_mean_strat
    right_bound = delta_mean_strat + pp * std_mean_strat
    
    return int(left_bound > 0)

def run_strat_ttest_two_sides(
    df_pilot: pd.DataFrame, df_control: pd.DataFrame,
    strat_column: str, target_name: str,
    strat_weights: pd.Series, alpha=0.975,
):
    """Tests the hypothesis of equality of means.
    
    Returns 1 if the mean of the metric in the pilot group
    is not equal significantly than the of the control group, otherwise 0.
    """
    mean_strat_pilot, var_strat_pilot = calc_strat_mean_var(df_pilot, strat_column, 
                                                            target_name, strat_weights
                                                           )
    mean_strat_control, var_strat_control = calc_strat_mean_var(df_control, strat_column, 
                                                                target_name, strat_weights
                                                               )

    delta_mean_strat = mean_strat_pilot - mean_strat_control
    std_mean_strat = (var_strat_pilot / len(df_pilot) + var_strat_control / len(df_control)) ** 0.5
    
    pp = norm.ppf(alpha)
    left_bound = delta_mean_strat - pp * std_mean_strat
    right_bound = delta_mean_strat + pp * std_mean_strat
    
    return int(left_bound > 0 or right_bound < 0)