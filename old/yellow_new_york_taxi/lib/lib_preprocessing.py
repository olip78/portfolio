import pandas as pd
import numpy as np
import datetime
import json
import warnings
import itertools
import copy
from itertools import product
from sklearn.preprocessing import StandardScaler
from pickle import dump, load

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from lib.lib_data import *

def normalizer(df, not_to_normalize, mode, config):
    s3_bucket = config['S3_BUCKET']
    s3_file_system = get_s3fs()
    filename = 'models/scaler.pkl'
    
    to_normalize = [c for c in df.columns if c not in not_to_normalize]
    if mode == 'training':
        scaler = StandardScaler()
        scaler.fit(df.loc[:, to_normalize])
        with s3_file_system.open(s3_bucket + '/' + filename, 'wb') as f:
            dump(scaler, f)
    else:
        with s3_file_system.open(s3_bucket + '/' + filename, 'rb')  as f:
            scaler = load(f)
    df_sub_norn = pd.DataFrame(scaler.transform(df.loc[:, to_normalize]), 
                               columns=to_normalize)
    df_normalized = pd.concat([df.loc[:, not_to_normalize], 
                                   df_sub_norn], axis=1)
    return df_normalized

def y_normalizer(df, mode, config, column='y'):
    s3_bucket = config['S3_BUCKET']
    s3_file_system = get_s3fs()
    filename_y_max = 'models/scaler_y_max.csv'
    filename_y_min = 'models/scaler_y_min.csv'
    
    Y = df.loc[:, ['t', 'zone_id', column]].pivot_table(values=column, index='t', columns='zone_id').fillna(0)
    if mode == 'training':
        Y_min = Y.min(axis=0)
        Y_max = Y.max(axis=0)
        
        with s3_file_system.open(s3_bucket + '/' + filename_y_min, 'w') as f:
            Y_min.to_csv(f)
        with s3_file_system.open(s3_bucket + '/' + filename_y_max, 'w') as f:
            Y_max.to_csv(f)    
    else:
        with s3_file_system.open(s3_bucket + '/' + filename_y_min, 'r') as f:
            Y_min = pd.read_csv(f, index_col=0).squeeze()[Y.columns]
        with s3_file_system.open(s3_bucket + '/' + filename_y_max, 'r') as f:
            Y_max = pd.read_csv(f, index_col=0).squeeze()[Y.columns]
    y_norm = ((Y- Y_min.values)/
              (Y_max.values-Y_min.values)).unstack().reset_index().rename(columns={0: 'y_norm'})
    df_copy = df.merge(y_norm, how='left', on=['zone_id', 't'])
    df_copy[column] = df_copy['y_norm']
    return df_copy.drop('y_norm', axis=1)

def y_normalizer_inverse(df, config, column='y'):
    s3_bucket = config['S3_BUCKET']
    s3_file_system = get_s3fs()
    filename_y_max = 'models/scaler_y_max.csv'
    filename_y_min = 'models/scaler_y_min.csv'
    
    Y = df.loc[:, ['t', 'zone_id', column]].pivot_table(values=column, 
                                                     index='t', columns='zone_id').fillna(0)
    with s3_file_system.open(s3_bucket + '/' + filename_y_min, 'r') as f:
            Y_min = pd.read_csv(f, index_col=0).squeeze()[Y.columns]
    with s3_file_system.open(s3_bucket + '/' + filename_y_max, 'r') as f:
        Y_max = pd.read_csv(f, index_col=0).squeeze()[Y.columns]

    y_inverse = (Y*(Y_max.values-Y_min.values) + 
         Y_min.values).unstack().reset_index().rename(columns={0: 'y_inverse'})

    df_copy = df.merge(y_inverse, how='left', on=['zone_id', 't'])
    df_copy[column] = df_copy['y_inverse']
    return df_copy.drop('y_inverse', axis=1)

def harm_decomp_rf(x, K):
    T = x.shape[0]
    columns = []

    if K['year']:
        x_sin_week, x_cos_week = np.zeros((K['year'], T)), np.zeros((K['year'], T))
        for k in range(K['year']):
            x_sin_week[k] = np.vectorize(lambda x: np.sin(x*2*np.pi*(k + 1)/8766))(x)
            x_cos_week[k] = np.vectorize(lambda x: np.cos(x*2*np.pi*(k + 1)/8766))(x)
        columns += ['s_y' + str(k) for k in range(K['year'])] + ['c_y' + str(k) for k in 
                                                                range(K['year'])]
        X = np.concatenate((x_sin_week, x_cos_week), axis=0)

    if K['week']:
        x_sin_week, x_cos_week = np.zeros((K['week'], T)), np.zeros((K['week'], T))
        for k in range(K['week']):
            x_sin_week[k] = np.vectorize(lambda x: np.sin(x*2*np.pi*(k + 1)/168))(x)
            x_cos_week[k] = np.vectorize(lambda x: np.cos(x*2*np.pi*(k + 1)/168))(x)
        columns += ['s_w' + str(k) for k in range(K['week'])] + ['c_w' + str(k) for k in 
                                                                range(K['week'])]
        if K['year']:
            X = np.concatenate((X, x_sin_week, x_cos_week), axis=0)
        else:
            X = np.concatenate((x_sin_week, x_cos_week), axis=0)    

    if K['day']:
        x_sin_day, x_cos_day = np.zeros((K['day'], T)), np.zeros((K['day'], T))    
        for k in range(K['day']):
            x_sin_day[k] = np.vectorize(lambda x: np.sin(x*2*np.pi*(k + 1)/24))(x)
            x_cos_day[k] = np.vectorize(lambda x: np.cos(x*2*np.pi*(k + 1)/24))(x) 
        columns += ['s_d' + str(k) for k in range(K['day'])] + ['c_d' + str(k) for k in 
                                                                range(K['day'])]
        X = np.concatenate((X, x_sin_day, x_cos_day), axis=0)

    X = np.concatenate((x.reshape((1, -1)), X), axis=0)   
    return pd.DataFrame(X.T, columns=['t'] + columns)

def rolling_sum(df, sum_period):
    tmp = df.loc[:, ['t', 'zone_id', 'y']]
    tmp = tmp.pivot_table(values='y', columns='zone_id', index='t').sort_index().fillna(0)
    tmp = tmp.rolling(sum_period, axis=0).sum().stack()
    return tmp.reset_index().rename(columns={0: 'sum_' + str(sum_period)})

def cartesian(df1, df2):
    rows = itertools.product(df1.T.iterrows(), df2.T.iterrows())
    df = pd.DataFrame(left*right for (_, left), (_, right) in rows)
    return df.reset_index(drop=True).T

def one_hot(X, feature='zone_id'):
    res = pd.get_dummies(X.loc[:, [feature]], prefix=[feature], columns = [feature], drop_first=False)
    if feature=='weekday':
        for i in [1, 2, 3]:
            if 'weekday_' + str(float(i)) not in res.columns:
                res['weekday_' + str(float(i))] = 0      
    return res

def shifting(df, hours, no_shift=[]):
    df_ = df.copy().drop(no_shift, axis=1)
    df_.loc[:, 't'] = df_.loc[:, 't'].map(lambda t: t + hours)
    df_ = df_.rename(columns={name: name + '_' + str(hours) for name in df_.columns 
                              if name not in ['t', 'zone_id']})
    return df_

def add_base_features(all_data_file_name, period, config, mode):
    
    # parameters
    K = config['feature_engineering']['K']
    features = copy.deepcopy(config['feature_engineering']['features'])
    not_to_normalize = config['feature_engineering']['not_to_normalize']
    
    # period filter
    h0 = datetime.datetime.strptime(config['h0'], config['format'])
    
    data = df_read(all_data_file_name, config)
    data.loc[:, 'date']  = pd.to_datetime(data.loc[:, 'date'], format=config['format'])
    df = data[(data['date'] >= period[0])&(data['date'] < period[1])]
    df = df.sort_values('t')
    
    df = df.loc[:, features]

    # y normalization
    df = y_normalizer(df, mode, config)
    df = y_normalizer(df, 'preprocessing', config, column='dol')

    # normalization
    df = normalizer(df, not_to_normalize, mode, config)
    
    # sum and 24-h periodical features
    for s in K['sum']:
        df = df.merge(rolling_sum(df, sum_period=s), on = ['t', 'zone_id'], how='left').fillna(0)
    
    # autoregressive features

    df_d = df.loc[:, ['t', 'zone_id']]

    for h in range(0, K['d']):
        df_tmp = shifting(df.loc[:, ['t', 'zone_id', 'y']], hours=h, no_shift=[])
        df_d = df_d.merge(df_tmp, on = ['t', 'zone_id'], how='left').fillna(0)  
    df_d.drop(['t', 'zone_id'], axis=1, inplace=True)
    df = pd.concat([df, df_d], axis=1)
    
    # all dates 
    all_hours = range(24*(period[1].date() + 
                          datetime.timedelta(days=90) - h0.date()).days)
    
    combinations = product(all_hours, df.zone_id.unique())
    Y = pd.DataFrame(combinations, columns=['t', 'zone_id']) 
    Y = Y.merge(data.loc[:, ['y', 't', 'zone_id']], on = ['t', 'zone_id'], how='left').fillna(0)
    Y = y_normalizer(Y, 'prediction', config)
    
    for d in K['D']:
        df_tmp = shifting(Y.loc[:, ['y', 't', 'zone_id']], hours=d*24)
        Y = Y.merge(df_tmp, on = ['t', 'zone_id'], how='left').fillna(0) 
    t0 = df.t.min()
    df = df[df.t>=t0+max(K['sum'])-1]
    
    # harmonics
    H = harm_decomp_rf(np.array(all_hours), K['harm'])
    
    return df, Y, H

def feature_engineering(data, H, config, mode):

    df = data.copy()
    
    K = config['feature_engineering']['K']
    path =  config['path_features']
    
    def save_df(feature_group_name, df_features, config, path=path):
        file_name = os.path.join(path, mode, feature_group_name + '.csv')
        df_save(df_features, file_name, config)
        
        print('"{}" shape: {}'.format(feature_group_name, df_features.shape))
        return file_name
    
    df = df.merge(H, on='t', how='left')
    H_columns = H.drop('t', axis=1).columns
    
    df.weekday = df.weekday.replace({0: np.nan})

    # general feature groups
    #columns_d = ['y_'+str(k) for k in range(1, K['d']+1)] < -----
    columns_d = ['y_'+str(k) for k in range(0, K['d'])]
    columns_sum = ['sum_' + str(k) for k in K['sum']] 
    
    # feature groups for cartesian prod
    columns_plus_1 = ['distance', 'vendor', 'cost', 'tips', 'passengers'] 
    columns_plus_2 = ['duration', 'vendor', 'cost', 'tips', 'passengers']

    features_comb ={}
    
    group_name = 't'
    df_features = df.loc[:, ['zone_id', 't', 'hours']]
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'linear'
    df_features = df.loc[:, columns_d]
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'squared'
    df_features = df.loc[:, columns_d]*df.loc[:, columns_d]
    features_comb.update({group_name: save_df(group_name, df_features, config)})

    group_name = 'weekday_d'
    df_features = cartesian(df.loc[:, columns_d],
                            one_hot(df, feature='weekday'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})

    group_name = 'weekday_plus_2'
    df_features = cartesian(df.loc[:, columns_plus_2],
                            one_hot(df, feature='weekday'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})

    group_name = 'hours_plus_d'
    df_features = cartesian(df.loc[:, columns_d],
                            one_hot(df, feature='hours'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})

    group_name = 'hours_plus_d_squared'
    df_features = cartesian(df.loc[:, columns_d]*df.loc[:, columns_d], 
                         one_hot(df, feature='hours'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'hours_plus_sum'
    df_features = cartesian(df.loc[:, columns_sum],
                            one_hot(df, feature='hours'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'hours_plus_2'
    df_features = cartesian(df.loc[:, columns_plus_2],
                            one_hot(df, feature='hours'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'hours_plus_1'
    df_features = cartesian(df.loc[:, columns_plus_1],
                            one_hot(df, feature='hours'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'hours_plus_dol'
    df_features = cartesian(df.loc[:, ['dol']],
                            one_hot(df, feature='hours'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'id_plus_d'
    df_features = cartesian(df.loc[:, columns_d], 
                         one_hot(df, feature='zone_id'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})

    group_name = 'id_plus_d_squared'
    df_features = cartesian(df.loc[:, columns_d]*df.loc[:, columns_d], 
                         one_hot(df, feature='zone_id'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})

    group_name = 'id_plus_sum'
    df_features = cartesian(df.loc[:, columns_sum], 
                         one_hot(df, feature='zone_id'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'id_plus_2'
    df_features = cartesian(df.loc[:, columns_plus_2], 
                         one_hot(df, feature='zone_id'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'id_plus_1'
    df_features = cartesian(df.loc[:, columns_plus_1], 
                         one_hot(df, feature='zone_id'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'id_plus_dol'
    df_features = cartesian(df.loc[:, ['dol']], 
                            one_hot(df, feature='zone_id'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})

    group_name = 'harmonics_id'
    df_features = cartesian(df.loc[:, H_columns], one_hot(df, feature='zone_id'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    group_name = 'trend_id' 
    df_features = cartesian(df.loc[:, ['b', 't']], one_hot(df, feature='zone_id'))
    features_comb.update({group_name: save_df(group_name, df_features, config)})
    
    return features_comb

def preprocessing(period, mode):
    with open('../config.json') as json_file:
        config = json.load(json_file)
    
    # interval extension
    if mode != 'training':
        K = config['feature_engineering']['K']
        period[0] = period[0] - datetime.timedelta(days=1+int(max(K['sum'])/24))
        period[1] = period[1] + datetime.timedelta(days=1) 
    
    dp = DataTransformation(period, config)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dp.update()
        
    df, Y, H = add_base_features(dp.output_file_name, period, config, mode)
    features_comb = feature_engineering(df, H, config, mode)
    
    exchange = json_read('exchange', config) 
    path = os.path.join(config['path_features'], mode)
    file_name_Y = os.path.join(path, 'Y.csv')
    
    if mode=='training': 
        exchange['Y_training'] = file_name_Y
        exchange['feature_combinations_training'] = features_comb
    elif mode=='validation':
        exchange['Y_validation'] = file_name_Y
        exchange['feature_combinations_validation'] = features_comb
    elif mode=='prediction':
        exchange['Y_prediction'] = file_name_Y
        exchange['feature_combinations_prediction'] = features_comb    
    
    df_save(Y, file_name_Y, config)
   
    dp.clean()    
    
    exchange['Y'] = file_name_Y
    json_save(exchange, 'exchange', config)