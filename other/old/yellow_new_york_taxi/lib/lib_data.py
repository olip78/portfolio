import pandas as pd
import numpy as np
import datetime
from dateutil import relativedelta
import os
import urllib
import json
import fastparquet
import warnings
import itertools
import holidays
import copy
from itertools import product

import s3fs

def get_s3fs():
    S3_KEY = os.environ['S3_KEY']
    S3_SECRET = os.environ['S3_SECRET']
    s3_file_system = s3fs.S3FileSystem(key=S3_KEY,
                                   secret=S3_SECRET,
                                   client_kwargs=dict(endpoint_url="https://s3.amazonaws.com"),
                                   anon=False) 
    return s3_file_system

def df_save(df, filename, config):
    s3_bucket = config['S3_BUCKET']
    s3 = config['S3']
    if s3:
        s3_file_system = get_s3fs()
        with s3_file_system.open(s3_bucket + '/' + filename, 'w') as f:
            df.to_csv(f, header=True, index=False)
    else:
        df.to_csv('../'+filename, header=True, index=False)

def df_read(filename, config):
    s3_bucket = config['S3_BUCKET']
    s3 = config['S3']
    if s3:
        s3_file_system = get_s3fs()
        with s3_file_system.open(s3_bucket + '/' + filename, 'r') as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv('../'+filename)
    return df  

def json_read(name, config):
    s3_bucket = config['S3_BUCKET']
    s3 = config['S3']
    if s3:
        s3_file_system = get_s3fs()
        with s3_file_system.open(s3_bucket + '/' + name + '.json', 'r') as json_data:
            res = json.load(json_data)
    else:
        with open('../'+name + '.json', 'r') as json_data: 
            res = json.load(json_data)
    return res 

def json_save(ver, name, config):
    s3_bucket = config['S3_BUCKET']
    s3 = config['S3']
    if s3:
        s3_file_system = get_s3fs()
        with s3_file_system.open(s3_bucket + '/' + name + '.json', 'w') as json_data:
            json.dump(ver, json_data)
    else:
        with open('../'+name + '.json', 'w') as json_data: 
            json.dump(ver, json_data)      


def cleaning(files_dict):
    with open('../config.json') as json_file:
        config = json.load(json_file)
    if config['S3']:
        s3_file_system = get_s3fs()
        for f in files_dict:
            s3_file_system.rm('s3://' + config['S3_BUCKET'] + '/' + files_dict[f]) 
    else:
        for f in files_dict:
            os.remove('../' + files_dict[f])


def get_holidays(period_):
    """
    """
    def daytype(x):
        W = range(5)
        H = [5, 6, 7]
        if (x[1] in H) and (x[0] in W):  return 1   # holiday or weekend after wokring
        elif (x[1] in H) and (x[2] in W):  return 2 # wokring day after holiday or weekend
        elif (x[1] in W) and (x[2] in H):  return 3 # wokring day befor holiday or weekend 
        else: return np.nan
    
    period = [period_[0] - datetime.timedelta(days=1), 
              period_[1] + datetime.timedelta(days=1)]
    ny_holidays = holidays.country_holidays('US', subdiv='NY')[period[0].date().strftime(format='%Y-%m-%d'): 
                                                               period[1].date().strftime(format='%Y-%m-%d')]
    n = (period[1].date() - period[0].date()).days
    alldays = [(period[0] + datetime.timedelta(days=k)).date() for k in range(n)]
    alldays = pd.DataFrame(alldays, columns=['date'])
    alldays.loc[:, 'weekday'] = alldays.loc[:, 'date'].map(lambda x: x.weekday())

    indices = alldays[(alldays['weekday'].isin(range(5)))&(alldays['date'].isin(ny_holidays))].index
    alldays.loc[indices, 'weekday'] = 7*np.ones(len(indices))
    alldays.loc[:, '-1'] = alldays.loc[:, 'weekday'].shift(1)
    alldays.loc[:, '1'] = alldays.loc[:, 'weekday'].shift(-1)
    alldays.loc[:, 'weekday'] = alldays.apply(lambda x: daytype([x['-1'], x['weekday'], x['1']]), axis=1)
    
    return alldays.drop(['-1', '1'], axis=1).iloc[1:-1, :]


def get_t(hour, h0, config=None):
    if type(hour) == str:
        hour = datetime.datetime.strptime(hour, config['format'])
    if type(h0) == str:
        h0 = datetime.datetime.strptime(h0, config['format'])
    return int((hour - h0).total_seconds()//3600)


def download(suffix):
    with open('../config.json') as json_file:
        config = json.load(json_file)
    
    head = config['url_head']
    path = config['path_row']
    
    columns = set(['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
       'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
       'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra',
       'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
       'total_amount', 'congestion_surcharge'])

    filename = 'yellow_tripdata_' + suffix + '.parquet'
    try:
        df = pd.read_parquet(head + filename, engine = 'fastparquet')
        if not columns.issubset(set(df.columns)):
            raise ValueError('{}: defferent feature names!!!'.format(suffix))
        print(filename + ' is downloaded')
        filename = os.path.join(path, 'yellow_tripdata_' + suffix + '.csv')
        df_save(df, filename, config)
    except urllib.error.HTTPError or TypeError:
        print(filename + ' is not found')
        
class DataTransformation:
    """
    """
    def __init__(self, period, config):
        self.format = config['format']
        self.period = period
        self.h0 = datetime.datetime.strptime(config['h0'], self.format)
        self.path_row = config['path_row']
        self.path_preprocessed = config['path_preprocessed']      
        self.output_file_name = os.path.join(self.path_preprocessed, 
                                             'preprocessed_data_tmp.csv')
        self.zones_in_use = config['zones_in_use']
        self.config = config
    
    def clean(self):
        if self.config['S3']:
            s3_file_system = get_s3fs()
            s3_file_system.rm('s3://' + self.config['S3_BUCKET'] + '/' + self.output_file_name) 
        else:
            os.remove('../' + self.output_file_name)
            
    def update(self):
        """
        """
        def data_cleaning(df, suffix):
            """
            """
            df.columns = df.columns.map(lambda x: x.replace(' ', '')) 
            s = suffix.split('-')
            valid_preriod = [datetime.datetime(int(s[0]), int(s[1]), 1),
                 datetime.datetime(int(s[0]), int(s[1]), 1) + relativedelta.relativedelta(months=1),  
                 datetime.datetime(int(s[0]), int(s[1]), 1) + relativedelta.relativedelta(months=1) + datetime.timedelta(days=1)]
            df = df.query('(passenger_count > 0) & (trip_distance > 0)')
            df.loc[:, 'tpep_pickup_datetime']  = pd.to_datetime(df.loc[:, 'tpep_pickup_datetime'], 
                                                            format=self.format)
            df.loc[:, 'tpep_dropoff_datetime'] = pd.to_datetime(df.loc[:, 'tpep_dropoff_datetime'], 
                                                            format=self.format)
            df = df.query('tpep_pickup_datetime < tpep_dropoff_datetime') 
            df = df.query('@valid_preriod[0] <= tpep_pickup_datetime < @valid_preriod[1]')
            df = df.query('@valid_preriod[0] < tpep_dropoff_datetime < @valid_preriod[2]') 
            return df 
        
        def data_transformation(df):
            """
            """
            def pt(df, data, column, aggfunc='count', columns_in=[]):
                tmp = df.pivot_table(values='VendorID', index=['t', 'zone_id'], 
                                     columns=column, aggfunc=aggfunc)
                if len(columns_in)>0:
                    tmp = tmp[tmp.columns[tmp.columns.isin(columns_in)]] 
                else:
                    tmp = df.copy()
                tmp = tmp.rename(columns={c: column + '_' + str(c) for c in tmp.columns.values})
                return data.merge(tmp, how='left', left_on=['t', 'zone_id'], right_index=True)
            
            df['t'] = df.loc[:, 'tpep_pickup_datetime'].map(lambda x: x.replace(minute=0, second=0))
            df = df.rename(columns={'PULocationID': 'zone_id'}) 
            df['duration'] = df.apply(lambda x: (x['tpep_dropoff_datetime'] - 
                                                 x['tpep_pickup_datetime']).total_seconds()/60, axis=1)
            df = df.drop(['tpep_dropoff_datetime', 'tpep_pickup_datetime'], axis=1)
            data = df.groupby(['t', 'zone_id'], as_index=False).apply(lambda x: pd.Series({
                               'y' : x['zone_id'].count(),
                               'distance'   : x['trip_distance'].mean(),
                               'duration'   : x['duration'].mean(),
                               'passengers' : x['passenger_count'].mean(),
                               'cost'       : x['total_amount'].mean(),
                               'tips'       : x['tip_amount'].mean(),
                               'vendor'     : x['VendorID'].value_counts().to_dict().get(1)
            
                                }))
            
            dol = df.groupby(['t', 'DOLocationID'], as_index=False).apply(lambda x: pd.Series({
                              'dol' : x['zone_id'].count()}))
            data = data.merge(dol, how='left', left_on=['t', 'zone_id'], 
                              right_on=['t', 'DOLocationID']).drop('DOLocationID', axis=1)
            data.loc[:, 'zone_id'] = data.loc[:, 'zone_id'].astype(np.int32())
            data.loc[:, 'b'] = 1
            data = data[data.zone_id.isin(self.zones_in_use)]
            return data
        
        def preprocessing_time(df):
            df.loc[:, 't']  = pd.to_datetime(df.loc[:, 't'], format=self.format)
            df.loc[:, 'date'] = df.loc[:, 't'].map(lambda h: h.date())
            df.loc[:, 'hours_24'] = df.loc[:, 't']
            df = df.merge(get_holidays(self.period), 
                          on='date', how='left')
            df.loc[:, 't'] = df.loc[:, 't'].map(lambda x: get_t(x, self.h0))
            df.loc[:, 'hours'] = df.apply(lambda x: x['t']%24 , axis=1)
            df.loc[:, 'weekhours'] = df.apply(lambda x: x['t']%24 + 24*x['weekday'], axis=1)
            return df
        
        def get_suffix(m):
            y0 = self.period[0].year
            month = str(m%12 + 12*int(m%12==0))
            return str(y0 + (m-1)//12) + '-' + ('0' if len(month)==1 else '') + month
        
        if self.config['S3']:
            s3_file_system = get_s3fs()
            downloaded_files_in = s3_file_system.ls(self.config['S3_BUCKET'] + '/' + 
                                                    self.path_row)
            preprocessed_files_in = s3_file_system.ls(self.config['S3_BUCKET'] + '/' + 
                                                      self.path_preprocessed)
        else:
            downloaded_files_in = os.listdir('../' + self.path_row)
            preprocessed_files_in = os.listdir('../' + self.path_preprocessed)
            
        
        #check whether there are all needed files
        downloaded_files = set([f[-11:-4] for f in downloaded_files_in 
                                if f[:6] == 'yellow'])
        preprocessed_files = set([f[-11:-4] for f in preprocessed_files_in 
                                  if f[-24:-12] == 'preprocessed'])
        files_for_preprocessing = []
        files_to_concatenate = []
        
        y0, m0 = self.period[0].year, self.period[0].month
        y1, m1 = self.period[1].year, self.period[1].month
        
        for m in range(m0, 12*(y1 - y0) + m1 + 2):
            suffix = get_suffix(m)
            if suffix not in downloaded_files | preprocessed_files:
                download(suffix)
            if suffix not in preprocessed_files:
                file_name = 'yellow_tripdata_' + suffix + '.csv'
                files_for_preprocessing.append(os.path.join(self.path_row, file_name))
            else:
                file_name = 'preprocessed_' + suffix + '.csv'
                files_to_concatenate.append(os.path.join(self.path_preprocessed, file_name))
                print(suffix, 'is already preprocessed')

        # month data preprocessing
        self.holidays = get_holidays(self.period) 
        
        for file_name in files_for_preprocessing:
            df = df_read(file_name, self.config)
            suffix = file_name[-11:-4]
            df = data_cleaning(df, suffix)
            df = data_transformation(df) 
            file_name_out = os.path.join(self.path_preprocessed, 'preprocessed_' + suffix + '.csv')
            df_save(df, file_name_out, self.config)
            files_to_concatenate.append(file_name_out)
            print(suffix, 'preprocessed - ok', df.shape)
        
        output_df = pd.DataFrame([])
        for file_name in files_to_concatenate:
            df = df_read(file_name, self.config)
            df.loc[:, 't']  = pd.to_datetime(df.loc[:, 't'], format=self.format)
            output_df = pd.concat([output_df, df], axis=0)
        
        output_df.loc[:, 't']  = pd.to_datetime(output_df.loc[:, 't'], format=self.format)
        output_df = output_df[(output_df['t']>=self.period[0])&
                              (output_df['t']<self.period[1])]
        
        print(output_df.t.min(), output_df.t.max())
        
        output_df = preprocessing_time(output_df)
        output_df = output_df.sort_values('t')
        
        # all_data update
        df_save(output_df, self.output_file_name, self.config)
