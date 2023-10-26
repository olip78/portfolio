import pandas as pd
import numpy as np
import datetime
import json
from joblib import dump

import os
import sys

# custom libs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from lib.lib_data import df_save, df_read, json_read, json_save, get_s3fs, cleaning
from lib.lib_preprocessing import preprocessing, y_normalizer_inverse
from lib.lib_training import *

with open('../config.json') as json_file:
    config = json.load(json_file)

# training period
period = [datetime.datetime.strptime(config['training']['period'][0], config['format']),
          datetime.datetime.strptime(config['training']['period'][1], config['format'])]
T_forecast = config['training']['T_forecast']
include = config['training']['include_groups']

# ------ training with holdout -------
test_begin = period[1] - datetime.timedelta(days=(config['training']['test_length'] - 1))
h0 = datetime.datetime.strptime(config['h0'], config['format'])

# training set
period_training = [period[0], test_begin]
preprocessing(period_training, mode='training')

exchange = json_read('exchange', config)
masks = {i: [] for i in include}
XX_train = combine_X(include, exchange['feature_combinations_training'], masks, config)
YY_train = df_read(exchange['Y_training'], config)
cleaning(exchange['feature_combinations_training'])

period_validation = [test_begin, period[1]]

# data preprocessing & feature engeniring
preprocessing(period_validation, mode='validation')

exchange = json_read('exchange', config)
masks = {i: [] for i in include}
XX_val = combine_X(include, exchange['feature_combinations_validation'], masks, config)
YY_val = df_read(exchange['Y_validation'], config)
cleaning(exchange['feature_combinations_validation'])

# training
# 1..T_forecast=6 loops. each for h-hour forcasting model 
r2 = 0
df_output = pd.DataFrame([])
for h_shift in range(1, T_forecast + 1):

    # correstipding shift of target variable
    X_train, Y_train, Y_train_df, features  =  X_Y_shifting(XX_train, YY_train, h_shift, config)
    X_val, Y_val, Y_val_df = X_Y_shifting(XX_val, YY_val, h_shift, config, features=False)
    
    # training
    model = regression(X_train, Y_train, config)
    
    # output results
    alpha = config['training']['alpha']
    print('h={}, C={}, train t2_score: {}'.format(h_shift, alpha, model.score(X_train, Y_train)))
    print('h={}, C={}, test r2_score: {}'.format(h_shift, alpha, model.score(X_val, Y_val)))
    
    # getting inverted target for output
    y_fact_inverted = y_normalizer_inverse(Y_train_df, config)['y']
    Y_train_df['y'] = model.predict(X_train)
    y_pred_inverted = np.clip(y_normalizer_inverse(Y_train_df, config)['y'], 0, np.inf)

    print('h={}, inverted, train r2_score: {}'.format(h_shift, r2_score(y_fact_inverted, y_pred_inverted)))
    
    y_fact_inverted = y_normalizer_inverse(Y_val_df, config)['y']  
    Y_val_df['y'] = model.predict(X_val)
    y_pred_inverted = np.clip(y_normalizer_inverse(Y_val_df, config)['y'], 0, np.inf) 

    r2_ = r2_score(y_fact_inverted, y_pred_inverted)
    print('h={}, inverted, val r2_score: {}'.format(h_shift, r2_))
    r2 += r2_

    df = XX_val[XX_val['t'] <= XX_val['t'].max() - h_shift].loc[:, ['t', 'zone_id']]
    df['fact'] = y_fact_inverted
    df['pred'] = y_pred_inverted
    df['h'] = h_shift
    df_output = pd.concat([df_output, df], axis=0)

print('average r2: {}'.format(r2/6))    
filename = os.path.join(config['path_data'], config['prediction']['results_month'])
df_save(df_output, filename, config)

# ------ training on the hole data ---------------

# data preprocessing & feature engeniring
preprocessing(period, mode='training')

exchange = json_read('exchange', config)
masks = {i: [] for i in include}
XX_train = combine_X(include, exchange['feature_combinations_training'], masks, config)
YY_train = df_read(exchange['Y'], config)
cleaning(exchange['feature_combinations_training'])

# for preventing conflict with prediction app
# if 'the model is being updated' - prediction is not possible
exchange['status'] = 'the model is being updated'
json_save(exchange, 'exchange', config)

# T_forecast models:
models_files = {}
for h_shift in range(1, T_forecast + 1):
    X_train, Y_train, Y_train_df, features  =  X_Y_shifting(XX_train, YY_train, h_shift, config)
    model = regression(X_train, Y_train, config)
    
    # model saving
    filename = config['training']['reg'] + '_model_' +  str(h_shift) + '.joblib'
    filename = os.path.join(config['path_models'], filename)
    if len(sys.argv) > 1:
        if sys.argv[1] == 'locally':
            s3_flag = False
    else:
        s3_flag = config['S3']
        
    if s3_flag:
        s3_file_system = get_s3fs()   
        with s3_file_system.open(config['S3_BUCKET'] + '/' + filename, 'wb') as f:
            dump(model, f)
    else:
        dump(model, '../' + filename)  
    models_files.update({h_shift: filename})
     
exchange['models'] = models_files
exchange['status'] = 'the model is updated: ' + datetime.datetime.now().strftime(format=config['format'])
print('The training is successfully finished.')
json_save(exchange, 'exchange', config)
