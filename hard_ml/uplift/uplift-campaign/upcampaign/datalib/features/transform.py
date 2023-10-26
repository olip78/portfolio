"""
contains all transformrer 
- functional (functional_transformer)
- ordinal (skbase.BaseEstimator, skbase.TransformerMixin)
"""

import pandas as pd
import numpy as np
import sklearn.base as skbase
import category_encoders as ce
import sklearn.preprocessing as skpreprocessing

from typing import List, Optional

from .base import functional_transformer

# functional transformers:

def divide_cols(data: pd.DataFrame, col_numerator: str, col_denominator: str, col_result: str = None):
    """division of two pandas df columns
    """
    col_result = col_result or f'ratio__{col_numerator}__{col_denominator}'
    data[col_result] = data[col_numerator] / data[col_denominator]
    return data

DivideColsTransformer = functional_transformer(divide_cols)

def do_binning(
    data: pd.DataFrame,
    col_value: str,
    col_result: str,
    bins: List[float],
    labels: Optional[List[str]] = None,
    use_tails: bool = True) -> pd.DataFrame:
    """df column value binarization 
    """

    if use_tails:
        if bins[0] != -np.inf:
            bins = [-np.inf] + bins
        if bins[-1] != np.inf:
            bins = bins + [np.inf]
    data[col_result] = pd.cut(data[col_value], bins=bins, labels=labels)
    return data

BinningTransformer = functional_transformer(do_binning)

def expression_transformer(data: pd.DataFrame, expression: str, col_result: str) -> pd.DataFrame:
    """evaluation of a given expression
    """

    data[col_result] = eval(expression.format(d='data'))
    return data

ExpressionTransformer = functional_transformer(expression_transformer)

def drop_cplumns(data: pd.DataFrame, columns_to_drop: List[str]):
    return data.drop(columns_to_drop, axis=1)

DropColumns = functional_transformer(drop_cplumns)

class OneHotEncoder(skbase.BaseEstimator, skbase.TransformerMixin):

    def __init__(self, cols: List[str], prefix: str = 'ohe', **ohe_params):
        self.cols = cols
        self.prefix = prefix
        self.encoder_ = skpreprocessing.OneHotEncoder(**(ohe_params or {}))

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        self.encoder_.fit(data[self.cols])
        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        result_column_names = []
        for col_idx, col in enumerate(self.cols):
            result_column_names += [
                f'{self.prefix}__{col}__{value}'
                for i, value in enumerate(self.encoder_.categories_[col_idx])
                if self.encoder_.drop_idx_ is None or i != self.encoder_.drop_idx_[col_idx]
            ]

        encoded = pd.DataFrame(
            self.encoder_.transform(data[self.cols]).todense(),
            columns=result_column_names
        )

        for col in encoded.columns:
            data[col] = encoded[col]
        return data

# ordinal transformers

class LocationRelativeConsumption(skbase.BaseEstimator, skbase.TransformerMixin):
    """the user's relative consumption to the average consumption in the location
    """
    def __init__(self, periods: List[int], drop_location=True):
        self.periods = periods
        self.drop_location = drop_location

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        return self

    def transform(self, df): 
        for d in self.periods:
            columns_to_drop = [c for c in df.columns if ('location_' in c) and (str(d)+'d' in c)] 
            """
            feature_name = f'location_relativ_purchase__{d}d'
            df[feature_name] = df[f'purchase_amt_mean__{d}d'] / df[f'location_mean__{d}d']
            feature_name = f'location_relativ_n__{d}d'
            df[feature_name] = df[f'purchase_amt_count__{d}d'] / df[f'location_count__{d}d']
            """
            feature_name = f'location_age_relativ_purchase__{d}d'
            df[feature_name] = df[f'purchase_amt_mean__{d}d'] / df[f'location_age_mean__{d}d']
            feature_name = f'location_age_relativ_n__{d}d'
            df[feature_name] = df[f'purchase_amt_count__{d}d'] / df[f'location_age_count__{d}d']
            
            if self.drop_location:
                df.drop(columns_to_drop, inplace=True, axis=1)
        return df
          
