import pandas as pd
import numpy as np
import datetime
import json
from joblib import dump, load
from sklearn.metrics import r2_score

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from lib.lib_data import df_read, df_save, json_read, json_save, get_s3fs
from lib.lib_preprocessing import preprocessing, y_normalizer_inverse
from lib.lib_training import *

with open('../config.json') as json_file:
    config = json.load(json_file)

exchange = json_read('exchange', config)
if exchange['status'] == 'the model is being updated':
    print('the model is being updated, next run in one hour')
    sys.exit()

T_forecast = config['training']['T_forecast']  
K = config['feature_engineering']['K']
include = config['training']['include_groups']

year = datetime.datetime.strptime(config['training']['period'][1], config['format']).year
month = datetime.datetime.strptime(config['training']['period'][1], config['format']).month
day = datetime.datetime.now().day
hour = datetime.datetime.now().hour

period_prediction = [datetime.datetime(year, month, day, hour, 0) + datetime.timedelta(seconds=3600), 
                     datetime.datetime(year, month, day, hour, 0) + datetime.timedelta(days=1)]
preprocessing(period_prediction, mode='prediction')
exchange = json_read('exchange', config)

masks = {i: [] for i in include}
XX = combine_X(include, exchange['feature_combinations_prediction'], masks, config)
YY = df_read(exchange['Y'], config)

models_files = exchange['models']
df_output = pd.DataFrame([])

for h_shift in range(1, T_forecast + 1):
    X, Y, Y_df, features  =  X_Y_shifting(XX, YY, h_shift, config)

    if config['S3']:
        s3_file_system = get_s3fs()  
        with s3_file_system.open(config['S3_BUCKET'] + '/' + 
                                 models_files[str(h_shift)], 'rb') as f:
            model = load(f)
    else:
        model = load('../' + models_files[str(h_shift)]) 
    
    pred = model.predict(X)
    print('{}-h, r2_score: {}'.format(h_shift, r2_score(pred, Y)))

    Y = y_normalizer_inverse(Y_df, config)['y']  
    Y_df['y'] = pred
    pred = np.clip(y_normalizer_inverse(Y_df, config)['y'], 0, np.inf) 

    print('{}-h, real, r2_score: {}'.format(h_shift, r2_score(pred, Y)))

    df = XX[XX['t'] <= XX['t'].max() - h_shift].loc[:, ['t', 'zone_id']]
    df['fact'] = Y
    df['pred'] = pred
    df['h'] = h_shift
    df_output = pd.concat([df_output, df], axis=0)

df_output.pred = df_output.pred.fillna(0).astype(int)    
filename = os.path.join(config['path_data'], config['prediction']['results_day'])
df_save(df_output, filename, config)

exchange['status'] = 'prediction is completed'
exchange['timestamp'] = datetime.datetime.now().strftime(format=config['format'])
json_save(exchange, 'exchange', config)
