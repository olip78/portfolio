import os
import pandas as pd
import numpy as np
import json

from flask import Flask, request, jsonify

import sys
sys.path.append('..')

from lib.tests import calc_strat_mean_var, run_strat_ttest, calculate_theta
from lib.utils import load_json, add_strata_columns, get_week_sales, filter_outliers, get_data_set

import mlflow


def check_strat_balans(df_a_one, df_a_two, df_b, strat_column, min_count=2):
    """Checks if there are strats in the groups.
    
    df_a_one, df_a_two, df_b - group dataframes
    min_count - minimum number of strat objects in each group
    
    Returns True if all groups have strats assigned, otherwise False."""
    
    list_strats = [df['strat_columns'].values for df in [df_a_one, df_a_two, df_b]]
    unique_strats = np.unique(np.hstack(list_strats))
    
    for unique_strat in unique_strats:
        for group_starts in list_strats:
            count = np.sum(group_starts == unique_strat)
            if count < min_count:
                return False
    return True

os.environ["MLFLOW_S3_ENDPOINT_URL"]="http://5.75.236.176:9000"
os.environ["MLFLOW_TRACKING_URI"]="http://5.75.236.176:5001"
os.environ["AWS_ACCESS_KEY_ID"]="IAM_ACCESS_KEY"
os.environ["AWS_SECRET_ACCESS_KEY"]="IAM_SECRET_KEY"

mlflow.set_tracking_uri("http://5.75.236.176:5001")
client = mlflow.tracking.MlflowClient()

MODEL_PATH = 'models:/sales_predict_prod/production'
model = mlflow.pyfunc.load_model(MODEL_PATH)
features = client.get_run('239c9f71dbaa4530b29976f304429487').data.params['features']
features = json.loads(features.replace("'", '"'))


PATH_DF_USERS = '../../data/df_users.csv'
PATH_DF_SALES = '../../data/df_sales.csv'

T = 7

users = pd.read_csv(PATH_DF_USERS)
sales = pd.read_csv(PATH_DF_SALES)
sales = sales[(sales.day >= 7*(T - 4))&(sales.day < 7*(T+1))]

ws = get_week_sales(sales)
for tau in [T-2, T-1, T]:
    ws = filter_outliers(ws, tau, interval=[50, 7500])

ws = filter_outliers(ws, T, interval=[50, 1200], new=True)

ws_test = get_data_set(ws, 
                       users,
                       weeks_in=[T])

df = ws_test[ws_test['sales'] > 0]
df = df.sort_values('user_id')
X_test = df.loc[:, features]
pred = model.predict(X_test)

del sales, X_test, df

ws = ws[ws.week==T] 
theta = calculate_theta(ws.sales.values, pred) 
ws['sales_cuped'] = ws.sales.values - theta*pred

ws = ws.merge(users, on='user_id', how='left')

strata = load_json('../artifacts/strata_age_gender_new.json')
strata_age_gender_new_weights = strata['strata_weights']
strata = strata['strata']

ws = add_strata_columns(ws, strata, 'strata_age_gender_new')

del pred, users

def aab_test(request, ws=ws, alpha=0.05):
    """runs one AAB test for request groups
    reques: {'test': {'group_a_one': [user_ids], 'group_a_two': [user_ids], 'group_b': [user_ids]}}
    """
    
    dfs_request = {} 
    for k in ['a_one', 'a_two', 'b']:
        request_df[k] = ws[ws.user_id.isin(request['test'][f'group_{k}'])]
    
    # checking for group balance
    if check_strat_balans(request_df['a_one'], request_df['a_two'], request_df['b'], 
                            'strata_age_gender'):
        strata_column = 'strata_age_gender_new'
    else:
        # groups are unbalanced
        return 0
    
    
    # aa test    
    res_aa = run_strat_ttest(request_df['a_one'], request_df['a_two'],
                             strat_column, 'sales_cuped',
                             strata_age_gender_new_weights
                            )
    
    if res_aa != 0:
        return 0
    
    # ab test  
    df = pd.concat([request_df['a_one'], request_df['a_two']])    
    res_ab = run_strat_ttest(df, request_df['b'],
                             strat_column, 'sales_cuped',
                             strata_age_gender_new_weights
                            )
    
    if res_ab == 1:
        return 1
    
    return 0



app = Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify(status = 'ok')

@app.route('/check_test', methods=['GET', 'POST'])
def check_test():
    has_effect = aab_test(json.loads(request.json))
    return jsonify(has_effect=int(has_effect))
