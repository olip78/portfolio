import os
import json

import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import ttest_ind, t, mode, norm


def save_json(obj, path: str, convert_to_int=False):
    if convert_to_int:
        res = [int(b) for b in obj]
    else:
        res = obj
    with open(path, 'w') as file:
        json.dump(res, file)
        
def load_json(path: str):
    with open(path, 'r') as file:
        obj = json.load(file)
    return obj
        
def filter_outliers(ws, T, interval, new=False):
    if new:
        users_out = ws[(ws.week==T)&(ws.new==1)&
                       ((ws.sales>interval[1])|(ws.sales<interval[0]))].user_id.unique()
    else:
        users_out = ws[(ws.week==T)&
                       ((ws.sales>interval[1])|(ws.sales<interval[0]))].user_id.unique()
    return ws[~ws.user_id.isin(users_out)]

def get_week_sales(sales):
    """
    weekly data agregation
    """
    df = sales.copy()
    df['week'] = df['day'] // 7
    first_day = df.groupby('user_id', as_index=False).day.first().rename(columns={'day': 'first_day'})
    df = df.groupby(['user_id', 'week'], as_index=False).agg(sales=('sales', 'sum'),
                                                   n=('sales', 'count'),
                                                   last_sale=('day', 'last'))
    
    df = df.merge(first_day, on='user_id', how='left') 
    df['new'] = (df.first_day >= 7*df.week).astype(int)
    df['mean_sale'] = df['sales'] / df['n']
    df['last_sale'] = 7*(1 + df['week']) - df['last_sale']
    return df

def get_data_set(week_sales, users, weeks_in, target='sales', shifts=[1, 2, 3, 4]):
    """
    The functinon generates features for prediction model.
    """
    
    def pt(df, ws, target=target, shifts=shifts):
        """
        lags for a given target variable
        """
        df_pivot = ws.pivot_table(columns='week', index='user_id', values=target, aggfunc='first')
        for s in shifts:
            df = df.merge(df_pivot.shift(s, axis=1).stack().reset_index().rename(
                columns={0: target + '_' + str(s)}), 
                          on=keys, how='left')
        return df

    df = pd.DataFrame(product(week_sales.user_id.unique(), 
                              week_sales.week.unique()), columns=['user_id', 'week'])
    
    df = df[df.week.isin(weeks_in)]
    
    keys = ['user_id', 'week']
    df = df.merge(week_sales, on=keys, how='left')
    
    df = pt(df, week_sales, target='sales', shifts=shifts)
    df = pt(df, week_sales, target='n', shifts=shifts)
    df = pt(df, week_sales, target='last_sale', shifts=shifts)
    df = pt(df, week_sales, target='mean_sale', shifts=shifts)
    
    df = df.merge(users, on='user_id', how='left')
    
    df['week_type'] = df['week'] % 2
    
    return df.fillna(0)

def add_strata_columns(df, strata, strat_column):
    df[strat_column] = 0
    for i, s in enumerate(strata):
        indices = df.query(s).index
        df.loc[indices, strat_column] = i
    return df

# statistics

def get_sample_size_abs(epsilon, a_std, b_std, alpha=0.05, beta=0.2, two_sides=True):
    
    if two_sides:
        t_alpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    else:
        t_alpha = norm.ppf(1 - alpha, loc=0, scale=1)
        
    t_beta = norm.ppf(1 - beta, loc=0, scale=1)
    z_scores_sum_squared = (t_alpha + t_beta) ** 2
    sample_size = int(
        np.ceil(
            z_scores_sum_squared * (a_std ** 2 + b_std ** 2) / (epsilon ** 2)
        )
    )
    return sample_size

def get_sample_size_arb(mu, a_std, b_std, eff, alpha=0.05, beta=0.2, two_sides=True):
    epsilon = (eff - 1) * mu

    return get_sample_size_abs(epsilon, a_std, b_std, alpha=alpha, beta=beta, two_sides=two_sides)

def get_minimal_determinable_effect(std, sample_size, alpha=0.05, beta=0.2):
    t_alpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    t_beta = norm.ppf(1 - beta, loc=0, scale=1)
    disp_sum_sqrt = (2 * (std ** 2)) ** 0.5
    mde = (t_alpha + t_beta) * disp_sum_sqrt / np.sqrt(sample_size)
    return mde

def make_strata(ws, strata):
    ws['strata'] = 0
    for i, query in enumerate(strata):
        ii = ws.query(query).index
        ws.loc[ii, 'strata'] = i
    return ws 

def get_strat_mean_var(df, strata_weights, strat_column, target):
    ms = (df.groupby(strat_column)['cuped'].mean().sort_index().values)
    vs = (df.groupby(strat_column)['cuped'].var().sort_index().values)
    return np.dot(ms, strata_weights), np.dot(vs, strata_weights)