import requests

import pandas as pd
import numpy as np

import time
import uuid

from collections import Counter

from lib.bandits import Bandits, get_user_features, Strategies

import warnings
warnings.filterwarnings("ignore")

PATH = '/Users/andreichekunov/andrei/hard_ml/dynamic pricing/final project dynamic pricing/data/'
PATH_ARTEFACTS = './artefacts/'

URL_BEGIN_DATA = 'https://lab.karpov.courses/hardml-api/project-1/task/{uuid}/begin'
URL_TASK_DATA_GET = 'https://lab.karpov.courses/hardml-api/project-1/task/{uuid}/data'
URL_TASK_RESULT_POST = 'https://lab.karpov.courses/hardml-api/project-1/task/{uuid}/result/'
URL_TASK_RESULT_GET = 'https://lab.karpov.courses/hardml-api/project-1/task/{uuid}/result'

UUID = uuid.uuid4().hex
print(UUID)

# (preprocessed) data
cost_prices = pd.read_csv(PATH + 'wholesale_trade_table.csv')
sales_plan = pd.read_csv(PATH + 'sales_plan.csv')
last_prices = pd.read_csv(PATH + 'last_price.csv', )
compet_prices = pd.read_csv(PATH + 'canc_df.csv')

transactions = pd.read_csv(PATH + 'transactions.csv')
transactions.dates = pd.to_datetime(transactions.dates)

sku_features = pd.read_csv(PATH_ARTEFACTS + 'sku_features.csv', index_col=0)

model_prices = pd.read_csv(PATH_ARTEFACTS + 'price_actions_optimization_30_plus.csv').loc[:, ]
model_prices.dates = pd.to_datetime(model_prices.dates)


# contextuala bandits
num_actions = 5
bandits = Bandits(model_prices, cost_prices, compet_prices, 
                  sales_plan,
                  transactions, sku_features,
                  num_actions=num_actions, nn=False, bb_factor=1.05) 

# connection
response = requests.post(URL_BEGIN_DATA.format(uuid=UUID))
status = response.json()['status']

# debugging
sales = bandits.results.loc[:, ['SKU', 'plan']] 

try:
    for i in range(29):
        # receiving batch with daily pairs of users / SKU
        response = requests.get(URL_TASK_DATA_GET.format(uuid=UUID))
        batch_data = pd.read_json(response.json())

        bandits.update_date(pd.to_datetime(batch_data.iloc[0, 0]))

        # actions -> prices
        actions = bandits.get_actions(batch_data)
        batch_data['actions'] = actions
        batch_data = bandits.strategies.set_prices(batch_data)
        
        # sending responses
        response = requests.post(URL_TASK_RESULT_POST.format(uuid=UUID),
                                 data=batch_data.drop('actions', axis=1).to_json(orient='records'))

        # debugging
        print(f'{batch_data.iloc[0, 0]}:') 
        current_actions = [0.00001]*num_actions
        for k, a in batch_data['actions'].value_counts().iteritems():
            current_actions[k] = a
        print(current_actions)

        # receiving feedback 
        response = requests.get(URL_TASK_RESULT_GET.format(uuid=UUID))
        batch_result = pd.read_json(response.json())
        batch_result.dates = pd.to_datetime(batch_result.dates)
        
        d = batch_result.dates.max()
        batch_result = batch_result[batch_result['dates']==d]    
        batch_result = batch_result.merge(batch_data.loc[:, ['SKU', 'user_id', 'actions']], 
                                          on=['SKU', 'user_id'], how='left')


        # model update
        bandits.update(batch_result)
        bandits.update_context(batch_result)

        actions_activ = batch_result[batch_result.bought==1].actions.value_counts().sort_index()                           

        actions_activ = [0]*num_actions
        for k, a in batch_result[batch_result.bought==1].actions.value_counts().iteritems():
            actions_activ[k] = a

        # debugging
        print(actions_activ)
        print(list(map(lambda x: round(x, 3), np.array(actions_activ)/np.array(current_actions))))
        sales[i] = bandits.results.loc[:, 'sold'].values
        sales.to_csv('sales.csv')
        bandits.sku_features.to_csv('bsf')

except Exception as e:
    response = requests.get(URL_TASK_DATA_GET.format(uuid=UUID))       
    print(response.json())  
    
print(UUID)