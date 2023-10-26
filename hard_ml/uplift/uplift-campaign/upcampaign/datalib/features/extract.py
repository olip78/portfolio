"""
contains all culcers (data extractors)
"""

import dask.dataframe as dd
import pandas as pd
import datetime
import sklearn.base as skbase
import sklearn.preprocessing as skpreprocessing
import copy

from typing import List, Dict, Union

from .base import DateFeatureCalcer, FeatureCalcer

def dask_groupby(
    data: dd.DataFrame,
    by: List[str],
    config: Dict[str, Union[str, List[str]]]
) -> dd.DataFrame:
    """wrapper on dask groupby method
    """
    data_ = data.copy()
    dask_agg_config = dict()

    for col, aggs in config.items():
        aggs = aggs if isinstance(aggs, list) else [aggs]
        for agg in aggs:
            fictious_col = f'{col}_{agg}'
            data_ = data_.assign(**{fictious_col: lambda d: d[col]})
            dask_agg_config[fictious_col] = agg

    result = data_.groupby(by=by).agg(dask_agg_config)
    return result

def get_location_age(customers):
    """builds a location/age based customer feature
    """
    
    customers['age_group'] = 0
    ii = customers[(customers.age>30)&(customers.age<51)].index
    customers.loc[ii, 'age_group'] = 1
    ii = customers[(customers.age>50)].index
    customers.loc[ii, 'age_group'] = 2
    customers['location_age'] = customers.apply(lambda x: x['location'] + '_' + 
                                                str(x['age_group']), axis=1)
    return customers

class AgeLocationCalcer(FeatureCalcer):
    """extracts location, age of customers
    """
    name = 'age_location'
    keys = ['customer_id']
    
    def compute(self) -> dd.DataFrame:
        customers = self.engine.get_table('customers')
        customers = get_location_age(customers)
        return customers[self.keys + ['age', 'location', 'location_age']]


class ReceiptsBasicFeatureCalcer(DateFeatureCalcer):
    """extracts raw receipts for a given period and 
    calculates basic statistics per customer
    """
    name = 'receipts_basic'
    keys = ['customer_id']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')

        date_from = self.date_to - self.delta
        date_mask = (receipts['date'] > date_from) & (receipts['date'] <= self.date_to)

        features = (
            receipts
            .loc[date_mask]
        )
        features = dask_groupby(
            features,
            by=['customer_id'],
            config={
                "purchase_amt": ["count", "sum", "max", "min", "mean"],
                "date": ["min", "max"],
            }
        )
        features = (
            features
            .assign(
                mean_time_interval=lambda d: (
                    (d['date_max'] - d['date_min'])
                    / d['purchase_amt_count']
                )
            )
            .assign(
                time_since_last=lambda d: (
                    self.date_to - d['date_max']
                )
            )
        )

        features = features.drop(['date_min', 'date_max'], axis=1)

        features = features.reset_index()
        features = features.rename(columns={
            col: col + f'__{self.delta}d' for col in features.columns if col not in self.keys
        })

        return features.compute()


class LocationSalesCalcer(DateFeatureCalcer):
    """calculates mean/number of sales statistics per location for a given time period
    """
    
    name = 'location_sales'
    keys = ['location']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')
        customers = self.engine.get_table('customers')
        
        date_from = self.date_to - self.delta
        date_mask = (receipts['date'] > date_from) & (receipts['date'] <= self.date_to)

        features = (
            receipts
            .loc[date_mask]
            .merge(
                customers[['location', 'customer_id']],
                on=['customer_id'],
                how='left'
            )
        )

        location_sales = dask_groupby(
            features,
            by=['location'],
            config={
                "purchase_amt": ["mean", "count"],
            }
        )
        
        location_sales = location_sales.compute()

        s = customers.location.value_counts()
        s.name = 'citizens'
        location_sales = location_sales.merge(s, left_on='location', right_index=True, how='left')

        location_sales['purchase_amt_count'] = location_sales['purchase_amt_count'] / location_sales['citizens']
        location_sales = location_sales.drop('citizens', axis=1)

        location_sales = location_sales.rename(columns={col: col.replace("purchase_amt", "location") + 
                    f'__{self.delta}d' 
                    for col in location_sales.columns if col not in self.keys})
        location_sales.reset_index()

        return location_sales

class LocationAgeSalesCalcer(DateFeatureCalcer):
    """calculates mean/number of sales statistics per location/age for a given time period
    """

    name = 'location_age_sales'
    keys = ['location_age']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')
        customers = self.engine.get_table('customers')
        
        customers = get_location_age(customers)

        date_from = self.date_to - self.delta
        date_mask = (receipts['date'] > date_from) & (receipts['date'] <= self.date_to)

        features = (
            receipts
            .loc[date_mask]
            .merge(
                customers[['location_age', 'customer_id']],
                on=['customer_id'],
                how='left'
            )
        )

        location_age_sales = dask_groupby(
            features,
            by=['location_age'],
            config={
                "purchase_amt": ["mean", "count"],
            }
        )
        
        location_age_sales = location_age_sales.compute()

        s = customers.location_age.value_counts()
        s.name = 'citizens'
        location_age_sales = location_age_sales.merge(s, left_on='location_age', right_index=True, how='left')

        location_age_sales['purchase_amt_count'] = (location_age_sales['purchase_amt_count'] / 
                                                    location_age_sales['citizens'])
        location_age_sales = location_age_sales.drop('citizens', axis=1)

        location_age_sales = location_age_sales.rename(columns={col: col.replace("purchase_amt", "location_age") + 
                                             f'__{self.delta}d' 
                                             for col in location_age_sales.columns if col not in self.keys})
        location_age_sales.reset_index()

        return location_age_sales

class TargetFromCampaignCalcer(FeatureCalcer):
    """extracts targerts = purchases information from a historical marketing campaign
    """
    
    name = 'target_from_campaign'
    keys = ['customer_id']
    
    def compute(self) -> dd.DataFrame:
        
        customers = self.engine.get_table('customers').loc[:, ['customer_id']]
        campaigns = self.engine.get_table('campaigns')
        receipts = self.engine.get_table('receipts')

        customers['treatment_flg'] = 0
        treatment_mask = customers.customer_id.isin(campaigns.customer_id)
        customers.loc[customers[treatment_mask].index, 'treatment_flg'] = 1

        date_campaign_start = campaigns.date.min()
        date_mask = (receipts['date'] >= date_campaign_start) & (receipts['date'] < date_campaign_start + 30)
        discount_mask =  ((receipts['date'] >= date_campaign_start) & 
                          (receipts['date'] < date_campaign_start + 7)).astype(int)
        receipts['discount_flag'] = discount_mask
        receipts = dask_groupby(
            receipts[date_mask],
            by=['customer_id'],
            config={
                "purchase_sum": ["sum"],
                "purchase_amt": ["sum"],
                "discount_flag": ["max"],
            }
        ).compute()
        
       
        result = customers.merge(receipts, on='customer_id', how='left').fillna(0)
        result['purchase_treatment_flg'] = result.purchase_sum_sum.astype(bool)*result.treatment_flg
        
        return result
