import pandas as pd
import numpy as np
import pickle

import pylift

from typing import Any, List
from .utils.data import load_pickle

import matplotlib.pyplot as plt

from typing import List, Tuple


class ModelKeeper:
    """wrapper on model/features
    """
    def __init__(self, model: Any, column_set: List[str]):
        self.model = model
        self.column_set = column_set
 
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(data[self.column_set])[:, 1]
        else:
            return self.model.predict(data[self.column_set])

    def dump(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(
                obj={'model': self.model, 'column_set': self.column_set},
                file=file
            )


def load_model(path: str) -> ModelKeeper:
    """load ModelKeeper object
    """
    obj = load_pickle(path)
    return ModelKeeper(
        model=obj['model'],
        column_set=obj['column_set'],
    )


def plot_uplift_prediction(
    upeval: pylift.eval.UpliftEval, plot_type: str = 'uplift',
    n_bins: int = 20, do_plot: bool = True):
    """calculate qini curve for uplift (model) prediction values 
    """

    bin_range = np.linspace(0, len(upeval.treatment), n_bins+1).astype(int)
    
    def noncumulative_subset_func(i):
        return np.isin(list(range(len(upeval.treatment))), prob_index[bin_range[i]:bin_range[i+1]])
    
    def cumulative_subset_func(i):
        return np.isin(list(range(len(upeval.treatment))), prob_index[:bin_range[i+1]])

    subsetting_functions = {
        'cuplift': cumulative_subset_func,
        'uplift': noncumulative_subset_func,
    }
    
    # sorted customers
    prob_index = np.flip(np.argsort(upeval.prediction), 0)
    
    x = list()
    y = list()
    
    for i in range(n_bins):
        current_subset = subsetting_functions[plot_type](i)
        
        # Get the values of outcome in this subset for test and control.
        treated_subset = (upeval.treatment == 1) & current_subset
        untreated_subset = (upeval.treatment == 0) & current_subset
        
        # Get the policy for each of these as well.
        p_treated = upeval.p[treated_subset]
        p_untreated = upeval.p[untreated_subset]

        # Count the number of correct values (i.e. y==1) within each of these
        # sections as a fraction of total ads shown.
        nt1 = np.sum(0.5 / p_treated)
        nt0 = np.sum(0.5 / (1 - p_untreated))
        
        y.append(upeval.prediction[current_subset].mean())
        x.append(nt1 + nt0)
    
    
    x = np.cumsum(x)
    # Rescale x so it's between 0 and 1.
    percentile = x / np.amax(x)

    # percentile = np.insert(percentile, 0, 0)
    # y.insert(0,0)
    
    if do_plot:
        plt.plot(percentile, y)
    
    return percentile, y
