import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import datetime
import holidays
from calendar import monthrange

from itertools import product
from collections import Counter

PATH = '/Users/andreichekunov/andrei/hard_ml/dynamic pricing/final project dynamic pricing/data/'

def get_transactions(sample_1000, wholesale_trade_table, promo_df):
    """prepares transactions df, joins some features from other data sources
    """
    transactions = pd.read_csv(PATH + 'transactions.csv')
    transactions.dates = pd.to_datetime(transactions.dates)
    
    transactions['year'] = transactions.dates.dt.year
    transactions['week_num'] = transactions.dates.dt.isocalendar().week
    transactions['month'] = transactions.dates.dt.month
    
    transactions = transactions.merge(promo_df, on=['SKU', 'week_num', 'year'], how='left')
    transactions.discount = transactions.discount.fillna(0)
    
    columns = ['year', 'week_num', 'SKU']
    transactions = transactions.merge(wholesale_trade_table.loc[:, ['cost_price', 'year', 'week_num', 'SKU']], 
                                  on=columns, how='left')
    
    f = lambda x: x['price']*(1 - x['discount']) - x['cost_price']
    transactions['margin'] = transactions.apply(f, axis=1)
    
    columns = ['SKU', 'actual']
    transactions = transactions.merge(sample_1000.loc[:, columns], on='SKU', how='left') 
    
    return transactions

def get_sales_plan(sample_1000):
    """prepares sales plan / back bonuses df
    """

    sales_plan = pd.read_csv(PATH + 'sales_plan.csv')
    columns = ['SKU', 'ui1_code', 'ui2_code', 'ui3_code', 'creation_date', 'expiration_date']
    sales_plan = sales_plan.merge(sample_1000.loc[:, columns], on='SKU', how='left') 

    for k in range(1, 4):
        code = 'ui' + str(k) + '_code'
        plan = 'ui' + str(k) + '_plan'
        sales_plan_ui = sales_plan.groupby([code, 'year', 'month'], as_index=False).agg(ui_plan=('plan', 'sum'))
        sales_plan_ui = sales_plan_ui.rename(columns={'ui_plan': plan})
        columns = ['year', 'month', code, plan]
        sales_plan = sales_plan.merge(sales_plan_ui.loc[:, columns], on=['year', 'month', code], how='left')    
        sales_plan[plan] = sales_plan[plan]/sales_plan.plan

    sales_plan['ui1_plan_delta'] = 0
    sales_plan['ui2_plan_delta'] = 0
    sales_plan['ui3_plan_delta'] = 0

    for sku, g in sales_plan.groupby('SKU'):
        ii = sales_plan[sales_plan['SKU']==sku].index
        delta = g.ui2_plan - g.ui2_plan.shift(1)
        sales_plan.loc[ii, 'ui2_plan_delta'] = delta.values
        delta = g.ui3_plan - g.ui3_plan.shift(1)
        sales_plan.loc[ii, 'ui3_plan_delta'] = delta.values
        delta = g.ui1_plan - g.ui1_plan.shift(1)
        sales_plan.loc[ii, 'ui1_plan_delta'] = delta.values
        
    sales_plan[(sales_plan.year==2019)&(sales_plan.month==12)].loc[:, ['SKU', 'plan', 'back_bonus', 'back_bonus_spec']]
        
    return sales_plan

def get_price_diff(daily):
    """calculates price differentials
    """
    weekly = daily.groupby(['SKU', 'week_num'], as_index=False).agg(price_week=('price_avg', 'mean'))
    
    daily['prev_price'] = 0
    for sku, g in daily.groupby('SKU'):
        g = g.sort_values('dates').reset_index(drop=True)
        ii = daily[daily.SKU==sku].index
        daily.loc[ii, 'prev_price'] = g.price_avg.shift(1).values
        daily['price_diff'] = (daily['price_avg'] - daily['prev_price'])/daily_data['prev_price']
    
    weekly['price_prev_week'] = 0
    for sku, g in weekly.groupby('SKU'):
        g = g.sort_values('week_num').reset_index(drop=True)
        ii = weekly[weekly.SKU==sku].index
        weekly.loc[ii, 'price_prev_week'] = g.price_week.shift(1).values
        weekly['price_week_diff'] = (weekly['price_prev_week'] - weekly['prev_price'])/weekly['price_prev_week']
        

def make_daily(transactions, sample_1000, promo_df, competitors, wholesale_trade_table, sales_plan):
    """daily aggregation. data preparation + some basic features
    """
    sku_pred = transactions[transactions.actual].SKU.unique()
    g = lambda d: '' if d > 9 else '0'
    dates_pred = [np.datetime64('2019-12-' + g(d) + str(d)) for d in range(1, 30)]

    df_pred = pd.DataFrame(product(sku_pred, dates_pred), columns=['SKU', 'dates'])
    df_pred['price_avg'] = 0
    df_pred['sales'] = 0
    
    daily = transactions.groupby(['SKU', 'dates'], as_index=False).agg(price_avg=('price', 'mean'),
                                                                   sales=('price', 'count'))

    daily = pd.concat([daily, df_pred], axis=0)

    daily['year'] = daily.dates.dt.year
    daily['month'] = daily.dates.dt.month
    daily['weekday'] = daily.dates.dt.weekday
    daily['day'] = daily.dates.dt.day
    daily['week_num'] = daily.dates.dt.isocalendar().week
    daily['week'] = daily.dates.dt.week
    
    mask = daily['dates'] > np.datetime64('2019-12-29')
    daily.loc[daily.index[mask], 'week'] = 52
      
    daily = daily.merge(promo_df, on=['SKU', 'week_num', 'year'], how='left')
    columns = ['year', 'week_num', 'SKU', 'month']
    daily = daily.merge(wholesale_trade_table.loc[:, ['cost_price', 'year', 'month', 'week_num', 'SKU']], 
                        on=columns, how='left')
    columns = ['year', 'week_num', 'SKU']
    daily = daily.merge(competitors.loc[:, columns + ['min_comp_price']], on=columns, how='left')

    columns = ['SKU', 'ui2_code', 'ui3_code', 'brand_code', 'actual']
    daily = daily.merge(sample_1000.loc[:, columns], on='SKU', how='left') 
    
    daily['min_comp_price'] = daily['min_comp_price'].fillna(-999) 
    daily['discount'] = daily['discount'].fillna(-999)  
    
    # fill nan
    for sku in [79558, 94018, 16915, 44626, 98008, 13197, 68494, 67286, 69661, 52736, 55566, 44124]:
        ii = daily[daily.SKU==sku].index
        ss = daily.loc[ii, 'cost_price'].fillna(method='ffill')
        daily.loc[ii, 'cost_price'] = ss.values
    
    daily_ui2_code = daily.groupby(['ui2_code', 'dates'], as_index=False).agg(ui2_sales=('sales', 'sum'))
    daily_ui3_code = daily.groupby(['ui3_code', 'dates'], as_index=False).agg(ui3_sales=('sales', 'sum'))
    daily_ui3_price_min = daily.groupby(['ui3_code', 'dates'], as_index=False).agg(ui3_price_min=('price_avg', 'min'))

    columns = [ 'dates', 'ui2_code', 'ui2_sales']
    daily = daily.merge(daily_ui2_code.loc[:, columns], on=['dates', 'ui2_code'], how='left') 
    columns = [ 'dates', 'ui3_code', 'ui3_sales']
    daily = daily.merge(daily_ui3_code.loc[:, columns], on=['dates', 'ui3_code'], how='left') 
    columns = [ 'dates', 'ui3_code', 'ui3_price_min']
    daily = daily.merge(daily_ui3_price_min.loc[:, columns], on=['dates', 'ui3_code'], how='left')

    columns = ['SKU', 'year', 'month', 'plan', 'ui1_plan', 'ui2_plan', 'ui3_plan', 
               'ui1_plan_delta', 'ui2_plan_delta', 'ui3_plan_delta']
    daily = daily.merge(sales_plan.loc[:, columns], on=['SKU', 'year', 'month'], how='left')
    daily = daily.sort_values('dates')
    
    # additional features
    def summer(m):
        if 4 < m < 10:
            return True
        else:
            return False

    def end_of_year(week):
        if week > 48:
            return True
        else:
            return False

    russia_holidays = holidays.Russia()

    daily['holidays'] = daily.dates.map(lambda x: x in russia_holidays)
    daily['summer'] = daily.month.map(summer)
    daily['end_of_year'] = daily.week.map(end_of_year)
    daily['monthlength'] = daily.apply(lambda x: monthrange(x.year, x.month)[0], axis=1) 
    daily['plan_per_day'] = daily['plan'] / daily['monthlength']

    daily = daily.sort_values('dates')
    daily['mov_avg_price'] = 0
    
    for sku, g in daily.groupby('SKU'):
        g = g.sort_values('dates').reset_index(drop=True)
        g['mov_avg_price'] = g.price_avg.cumsum() / (g.index + 1)
        ii = daily[daily.SKU==sku].index

        daily.loc[ii, 'mov_avg_price'] = g['mov_avg_price'].values
    return daily
 
def get_data(mode='daily'):
    """data preparation 'pipeline'
    """
    sample_1000 = pd.read_csv(PATH + 'sample_1000.csv')
    sample_1000.expiration_date = pd.to_datetime(sample_1000.expiration_date)
    sample_1000.creation_date = pd.to_datetime(sample_1000.creation_date)

    d = pd.to_datetime('12-01-2019 00:00:00')
    sample_1000['actual'] = sample_1000.expiration_date.map(lambda x: x > d)
    sample_1000 = sample_1000.rename(columns={'sku_id': 'SKU'})
    
    promo_df = pd.read_csv(PATH + 'promo_df.csv')
    wholesale_trade_table = pd.read_csv(PATH + 'wholesale_trade_table.csv')
    competitors = pd.read_csv(PATH + 'canc_df.csv')
    competitors['min_comp_price'] = competitors.iloc[:, -3:].min(axis=1)
    
    sales_plan = get_sales_plan(sample_1000)
    transactions = get_transactions(sample_1000, wholesale_trade_table, promo_df)
   
    daily = make_daily(transactions, sample_1000, promo_df, competitors, 
                       wholesale_trade_table, sales_plan)
    return transactions, daily

def get_price_diff(daily):
    """calculate daily and weekly price diffs
    """
    
    daily = daily.sort_values('dates')
    daily['prev_price'] = 0
    for sku, g in daily.groupby('SKU'):
        g = g.sort_values('dates').reset_index(drop=True)
        ii = daily[daily.SKU==sku].index
        daily.loc[ii, 'prev_price'] = g.price_avg.shift(1).values
    daily['price_diff'] = (daily['price_avg'] - daily['prev_price'])/daily['prev_price']
    
    daily['week_num'] = daily['week_num'] + 52*(daily['year']-2018)
    weekly = daily.groupby(['SKU', 'week_num'], as_index=False).agg(price_week=('price_avg', 'mean'))
    
    weekly['price_prev_week'] = 0
    for sku, g in weekly.groupby('SKU'):
        g = g.sort_values('week_num').reset_index(drop=True)
        ii = weekly[weekly.SKU==sku].index
        weekly.loc[ii, 'price_prev_week'] = g.price_week.shift(1).values
    weekly['price_week_diff'] = (weekly['price_week'] - weekly['price_prev_week'])/weekly['price_prev_week']
    
    daily = daily.merge(weekly.loc[:, ['SKU', 'week_num', 'price_week_diff']], on=['SKU', 'week_num'], how='left')
    return daily


def get_ts_periods(daily, sku_in):
    """returns sales periods of products
    """
    df = daily[daily.SKU.isin(sku_in)].loc[:, ['SKU', 'dates', 'sales']]
    ts_dates = pd.DataFrame([], index=df.SKU.unique())
    ts_dates['begin'] = 0
    ts_dates['end'] = 0

    for sku in sku_in:
        g = df[df.SKU==sku]
        ts_dates.loc[sku, :] = (g.dates.min(), g.dates.max())

    ts_dates['length'] = (ts_dates['end'] - ts_dates['begin']).map(lambda x: x.days)
    
    return ts_dates
