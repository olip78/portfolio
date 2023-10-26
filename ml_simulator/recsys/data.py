import os
import sys
import pickle
from typing import List

import pandas as pd
import numpy as np
from pydantic import BaseModel

import implicit
from scipy.sparse import csr_matrix


class User(BaseModel):
    """Class of json output"""
    user_id: int
    personal: List

        
def df_import(path_from):
    """data import
    """
    data = pd.read_csv(path_from, header = None)
    data.columns = ['uid', 'session_id', 'streamer_name', 'time_start', 'time_end']
    data["uid"] = data["uid"].astype("category")
    data["streamer_name"] = data["streamer_name"].astype("category")
    data["user_id"] = data["uid"].cat.codes
    data["streamer_id"] = data["streamer_name"].cat.codes
    data["duration"] = data["time_end"] - data["time_start"]
    return data


def process_data_popularity(path_from: str):
    """Function process

    Parameters
    ----------
    path_from : str
        path to read data

    Returns
    -------
    data: pandas.DataFrame
        dataframe after proccessing
    sparse_item_user: scipy.sparse.csc_matrix
        sparce item user csc matrix
    """
    data = df_import(path_from)

    views_data = data.groupby(['user_id', 'streamer_id'], as_index=False).duration.sum()
    views_data = views_data.sort_values(['user_id', 'streamer_id'])
    sparse_item_user = csr_matrix((views_data.duration.values, 
                              (views_data.loc[:, 'user_id'], 
                               views_data.loc[:, 'streamer_id'])))
    return data, sparse_item_user


def process_data_factorisation(path_from: str, time_now: int = 6147):
    """Function process

    Parameters
    ----------
    path_from : str
        path to read data
    time_now : int
        time to filter data

    Returns
    -------
    data: pandas.DataFrame
        dataframe after proccessing
    """
    data = df_import(path_from)
    data = data[data.time_start < time_now]
    data['online'] = (data.time_start.values <= time_now) * (time_now < data.time_end.values)

    strimers_online = data.groupby('streamer_name', as_index=False).online.max()
    strimers_online = strimers_online.query(f'online').streamer_name.values
    data = data[data.streamer_name.isin(strimers_online)]
    data.loc[:, 'time_end'] = np.minimum(data.time_end.values, time_now)
    data['time_intervals'] = data.apply(lambda x: np.arange(x.time_start, x.time_end+1), axis=1)

    return data
