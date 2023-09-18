import pandas as pd
from dateutil import relativedelta
from itertools import product
from sklearn import linear_model
from sklearn.metrics import r2_score

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from lib.lib_data import df_read

def combine_X(include, features_comb, masks, config):
    X = pd.DataFrame([])
    for i in include:
        x = df_read(features_comb[i], config)
        print(features_comb[i], x.shape)
        if masks[i] == []:
            X = pd.concat([X, x], axis=1)
        else:
            X = pd.concat([X, x.iloc[:, masks[i]]], axis=1)
    return X    

def X_Y_shifting(X, Y, h_shift, config, features=True):
    K = config['feature_engineering']['K']
    Y_shifted = Y.copy()
    Y_shifted['t'] = Y['t'].map(lambda x: x - h_shift)
    t0 = X['t'].max()
    X = X[X['t'] <= t0 - h_shift]
    
    # add y, y_24, y_48, ...
    y_24_columns = ['y_' + str(24*k) for k in K['D']]
    XX = X.merge(Y_shifted, on = ['zone_id', 't'], how='left')
    YY = XX.loc[:, 'y'].values
    
    Y_df = XX.loc[:, ['zone_id', 't', 'y']]
    XX = XX.drop(['zone_id', 't', 'y', 'hours'] + y_24_columns, axis=1)
    if features:
        return XX, YY, Y_df, XX.columns
    else:
        return XX, YY, Y_df
    
def regression(X, Y, config): 
    reg = config['training']['reg']
    alpha = config['training']['alpha']
    if reg == 'ridge':
        model = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    model = model.fit(X, Y)
    return model
