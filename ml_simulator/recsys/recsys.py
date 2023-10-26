import os
import sys
import pickle
from typing import List

import pandas as pd
import numpy as np
from pydantic import BaseModel

import implicit
from scipy.sparse import csr_matrix


def recomend_popularity(data: pd.DataFrame):
    """Recomend Popularity

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    popular_streamers: List
    """
    def total_agg(x): 
        return len(set(np.concatenate(x.values)))
    def sum_agg(x): 
        return np.concatenate(x.values).shape[0]
    
    streamers = data.groupby('streamer_name', as_index=False).agg(
                            sum_duration=('time_intervals', sum_agg),
                            total_duration=('time_intervals', total_agg),
                            online=('online', 'sum')
                           )
    #streamers['popularity'] = streamers.sum_duration / streamers.total_duration
    streamers['popularity'] = streamers.online
    
    streamers = streamers.sort_values('popularity', ascending=False)
    popular_streamers = list(streamers.streamer_name.values)
    return popular_streamers


def fit_model(
    sparse_item_user,
    model_path: str,
    iterations: int = 12,
    factors: int = 500,
    regularization: float = 0.2,
    alpha: float = 100,
    random_state: int = 42,
):
    """function fit ALS

    Parameters
    ----------
    sparse_item_user : csr_matrix
        Сompressed Sparse Row matrix
    model_path: str
        Path to save model as pickle format
    iterations : int, optional
        Number of iterations, by default 12
    factors : int, optional
        Number of factors, by default 500
    regularization : float, optional
        Regularization, by default 0.2
    alpha : int, optional
        Alpha increments matrix values, by default 100
    random_state : int, optional
        Random state, by default 42

    Returns
    -------
    model: AlternatingLeastSquares
        trained model
    """
    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 regularization=regularization, 
                                                 iterations=iterations, 
                                                 random_state=random_state
                                                )

    data = (sparse_item_user.copy() * alpha).astype('double')
    model.fit(data)

    with open('model.pkl', "wb") as file:
        pickle.dump(model, file)
    return model


def personal_recomendations(
    user_id: int,
    n_similar: int,
    model: implicit.als.AlternatingLeastSquares,
    data: pd.DataFrame,
) -> List:
    """Give similar items from model

    Parameters
    ----------
    user_id : int
        User to whom we will recommend similar items
    n_similar : int
        Number of similar items
    model : als.AlternatingLeastSquares
        ALS model
    data : pd.DataFrame
        DataFrame containing streamer names & their ids

    Returns
    -------
    similar_items: List
        list of similar item to recomed user
    """

    streamer_mapper = data.loc[:, ['streamer_id', 'streamer_name']].drop_duplicates()
    streamer_mapper = streamer_mapper.sort_values('streamer_id').streamer_name.values
    
    ids, _ = model.similar_items(user_id)
    similar_items = streamer_mapper[ids]

    return similar_items[:n_similar]


def load_model(
    model_path: str,
):
    """Function that load model from path

    Parameters
    ----------
    path : str
        Path to read model as pickle format

    Returns
    -------
    model: AlternatingLeastSquares
        Trained model
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model
