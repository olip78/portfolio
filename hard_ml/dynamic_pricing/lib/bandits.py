import pandas as pd
import numpy as np

import datetime

from collections import Counter
from sklearn.preprocessing import MinMaxScaler

from space_bandits import LinearBandits, NeuralBandits


def update_transaction(transactions, df_feedback)->pd.DataFrame:
    """update historical df transactions with transactions from requests
    """
    df_feedback = df_feedback[df_feedback.bought==1][['dates', 'SKU', 'user_id', 'price']]\
           .rename(columns={'user_id': 'user'})
    df_feedback.dates = pd.to_datetime(df_feedback.dates)
    return pd.concat([transactions, df_feedback])


def get_user_context(transactions):
    """extracts user features from request/historical transactions
    """
    
    result = pd.DataFrame(transactions.user.unique(), columns=['user'])
    
    df = transactions.groupby(['dates', 'user'], as_index=False).agg(purch_sum=('price', 'sum'))
    df = df.groupby('user', as_index=False).agg(cum_expenditure=('purch_sum', 'sum'),
                                                                  n_visits=('dates', 'count'))
    df['mean_bill'] = df['cum_expenditure']/df['n_visits']
    df['sum_expend_1'] = df['cum_expenditure'].map(lambda x: 1 if 190000 < x < 400000 else 0)
    df['sum_expend_2'] = df['cum_expenditure'].map(lambda x: 1 if  x >= 400000 else 0)
    df['sum_expend_0'] = df['cum_expenditure'].map(lambda x: 1 if  x <= 190000 else 0)
    
    df['visits_1'] = df['n_visits'].map(lambda x: 1 if 20 < x < 80 else 0)
    df['visits_2'] = df['n_visits'].map(lambda x: 1 if x > 80 else 0)
    df['visits_0'] = df['n_visits'].map(lambda x: 1 if x > 20 else 0)
    
    result = result.merge(df, on='user', how='left')

    last = transactions.groupby('user').price.rolling(1).sum()\
        .reset_index()\
        .groupby('user').last()\
        .reset_index().rename(columns={'price':'last_purchase'})[['user', 'last_purchase']]
    result = result.merge(last, on='user', how='left').fillna(0)
    
    last3 = transactions.groupby('user').price.rolling(3).sum()\
        .reset_index()\
        .groupby('user').last()\
        .reset_index().rename(columns={'price':'last_3_purchases'})[['user', 'last_3_purchases']]
    result = result.merge(last3, on='user', how='left').fillna(0)

    periods = transactions.sort_values('dates').groupby('user').agg(last_date=('dates', 'last'),
                                                                    first_date=('dates', 'first'),
                                                                    last_purchase_sum=('price', 'last'))

    
    result = result.merge(periods, on='user', how='left')
    result['days_in_service'] = (result['last_date'] - result['first_date']).dt.days
    
    result['old'] = 1
    columns = ['user', 'mean_bill', 'n_visits', 'last_3_purchases',
           'last_purchase_sum', 'days_in_service', 'sum_expend_1',
           'sum_expend_2', 'sum_expend_0', 'visits_1', 'visits_2', 'visits_0', 'old']
    return result.loc[:, columns]

def get_user_features(user_features, user):
    """selects context for a given user with history / takes zero vector for a new one
    """
    columns = user_features.columns[1:]
    if user in user_features.user.unique():
        res = user_features[user_features.user==user].loc[:, columns].values
        return res
    else:
        res[:] = 0
        return np.array([res])


class Strategies:
    """price strategies for comtextual bandits
    """
    def __init__(self, model_prices, cost_prices, compet_prices):
        self.model_prices = model_prices.loc[:, ['dates', 'SKU', 'base_strategy', 
                                                 'last_two_week_price', 'last_price']]
        self.model_prices['week_num'] = model_prices['dates'].dt.week
        
        cost_prices_ = cost_prices[(cost_prices.year==2019)&(cost_prices.month==12)].drop(['year', 'month'], 
                                                                                         axis=1)

        cost_prices_['fixed_margin_price_18'] = 1.18*cost_prices_['cost_price'] 
        cost_prices_['fixed_margin_price_20'] = 1.20*cost_prices_['cost_price']
        cost_prices_['fixed_margin_price_21'] = 1.21*cost_prices_['cost_price']
       
        self.model_prices = self.model_prices.merge(cost_prices_, on=['SKU', 'week_num'], how='left')
        
        compet_prices['avg_comp_price'] = compet_prices.iloc[:, -3:].mean(axis=1)
        compet_prices['min_comp_price'] = compet_prices.iloc[:, -3:].min(axis=1)
        compet_prices_ = compet_prices.iloc[:, [1, 2, -2, -1]]
        
        self.model_prices = self.model_prices.merge(compet_prices_, on=['SKU', 'week_num'], how='left')
        ii = self.model_prices['avg_comp_price'].isna()
        
        to_fill = 'last_price'
        self.model_prices.loc[ii, 'avg_comp_price'] = self.model_prices.loc[ii, to_fill].values
        self.model_prices.loc[ii, 'min_comp_price'] = self.model_prices.loc[ii, to_fill].values*0.98
        
        self.strategies  = {0: 'fixed_margin_price_18', 
                            1: 'fixed_margin_price_20',  
                            2: 'fixed_margin_price_21',  
                            3: 'min_comp_price',  
                            4: 'base_strategy,
                           }
        
    def set_prices(self, batch):
        """takes prices for selected actions (strategies)
        """
        d = batch.iloc[0, 0]
        model_prices = self.model_prices[self.model_prices.dates==d]
        batch['price'] = 0
        for a in self.strategies.keys():
            ii = batch[batch.actions==a].index
            df = batch.loc[ii, ['SKU']]
            columns = ['SKU', self.strategies[a]]
            df = df.merge(model_prices.loc[:, columns], on = ['SKU'], how='left')
            batch.loc[ii, 'price'] = df.loc[:, self.strategies[a]].values
        return batch


class Bandits:
    """class for contextual bandits
    """
    def __init__(self, model_prices, cost_prices, compet_prices, sales_plan, 
                 transactions, sku_features,
                 num_actions=5, 
                 nn=False
                ):
        
        self.strategies = Strategies(model_prices, cost_prices, compet_prices)
        self.current_date = {'date': 0, 'week': 0, 'day': 0}
        
        self.transactions = transactions
        self.cost_prices = cost_prices[(cost_prices.year==2019)&
                                       (cost_prices.month==12)]
        
        # sales dynamic
        self.results = sales_plan[(sales_plan.year==2019)&(sales_plan.month==12)]
        self.results.index = self.results.SKU
        self.results['sold'] = 0
        self.results['plan_day'] = self.results['plan'] / 29
        
        # scaler for user features
        self.user_features = get_user_context(transactions)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.user_features.iloc[:, 1:-7])
        
        self.sku_features = sku_features.loc[self.results.SKU, :]
        
        self.num_actions = num_actions 
        self.num_features = self.user_features.shape[1] - 1 + sku_features.shape[1]

        print(f'num_actions: {self.num_actions}, num_features: {self.num_features}')

        self.nn = nn
        
        self.low_bound = 0
        
        self.reset_model()
         
    def reset_model(self):
        """resets the model 
        """
        if self.nn:
            memory_size = 10000
            self.model = NeuralBandits(self.num_actions, self.num_features, initial_pulls=100, 
                                       memory_size=memory_size, layer_sizes=[50, 12], verbose=False)
            assert self.model.data_h.memory_size == memory_size
        else:
            self.model = LinearBandits(self.num_actions, self.num_features)
    
    def update_date(self, date):
        """updates current date
        """
        self.current_date['date'] = date
        self.current_date['week'] = date.week
        self.current_date['day'] = date.day
        
        
    def get_actions(self, batch): 
        """optimal actions (strategies) selection
        """
        flag = self.current_date['date'] in [pd.to_datetime('12-01-2019 00:00:00'),
                                             pd.to_datetime('12-17-2019 00:00:00')]
        actions = []
        for i in batch.index:
            date, sku, user_id = batch.loc[i, :].values
            context = self.get_context(sku, user_id)
           
            if not flag:
                action = self.model.action(context)
            else:
                # forced random setting
                action = np.random.choice(self.num_actions, 1)[0]
            actions.append(action)
        return actions    
        
    def update_context(self, df_feedback):
        self.transactions = update_transaction(self.transactions, df_feedback)
        self.user_features = get_user_context(self.transactions)
        
    def get_context(self, sku, user_id):
        user_features = get_user_features(self.user_features, user_id)
        user_features[0][:-7] = self.scaler.transform(user_features[0][:-7].reshape(1, -1))[0]
        context = np.hstack([user_features, self.sku_features.loc[sku, :].values.reshape(1, -1)])
        return context[0]
    
    def get_reward(self, sku, price, bought):
        cost_price = self.cost_prices[(self.cost_prices.week_num==self.current_date['week'])&
                                      (self.cost_prices.SKU==sku)].cost_price.values[0]     
        
        reward = price - cost_price

        # taking back bonuses into account
        if self.current_date['day'] > 5:
            # adding back bonuse summation for SKU with bad but not too bad dynamic
            r =  self.sku_features.loc[sku, 'dynamic']
            not_completed = self.results.loc[sku, 'sold'] < self.results.loc[sku, 'plan']
            if (r < 1.01 and r > self.low_bound) and not_completed:
                if self.results.loc[sku, 'plan'] > 0:
                    # growthing importans of back bonuse summation
                    bb_specific = (self.results.loc[sku, 'back_bonus'] /
                                  (self.results.loc[sku, 'plan'] - self.results.loc[sku, 'sold']))
                else:
                    bb_specific = 0
                reward += bb_specific

        return bought*reward
    
    def update(self, new_transactions):
        for i in new_transactions.index:
            date, sku, user_id, price, bought, action = new_transactions.loc[i, :].values
            context = self.get_context(sku, user_id)
            reward = self.get_reward(sku, price, bought)
            self.model.update(context, action, reward)
            
        day_res = pd.DataFrame(self.results.index, columns=['SKU'])
        day_res = day_res.merge(new_transactions.groupby('SKU', as_index=False).agg(sales=('bought', 'sum')), 
                                on='SKU', how='left').fillna(0)
        self.results['sold'] = self.results['sold'] + day_res.sales.values
        d = self.current_date['day']
        self.sku_features['dynamic'] = self.results['sold'] / (d*self.results['plan_day'])
        self.sku_features['completed'] = self.results['sold'] >= (d*self.results['plan'])
        self.low_bound = 0.94*(np.exp(d/20)/(1 - 2*np.exp(d/20)) + 1.6)
        print(new_transactions.bought.sum()/new_transactions.shape[0]) 
        