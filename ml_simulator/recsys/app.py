import os
import sys
from typing import List

import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from data import User, process_data_popularity, process_data_factorisation
from recsys import load_model, personal_recomendations

app = FastAPI()


@app.get("/popular/user/{user_id}")
async def get_popularity(user_id: int, time: int = 6147):
    """Fast Api Web Application

    Parameters
    ----------
    user_id : int
        user id
    time : int, optional
        time, by default 6147

    Returns
    -------
    user: json
        user informations
    """
    path = os.path.join(sys.path[0], os.environ["data_path"])
    data = process_data_popularity(path_from=path, time_now=time)
    popular_streamers = recomend_popularity(data)

    user = User(user_id=user_id, time=time, popular_streamers=popular_streamers)
    return user


@app.get("/recomendations/user/{user_id}")
async def get_recomendation(user_id: int):
    """Fast Api Web Application

    Parameters
    ----------
    user_id : int
        user to whom we will recommend streamers

    Returns
    -------
    user: json
        user informations
    """
    data_path = os.path.join(sys.path[0], os.environ["data_path"])
    model_path = os.path.join(sys.path[0], os.environ["model_path"])

    data, _ = process_data_factorisation(data_path)
    model = load_model(model_path)

    personal = personal_recomendations(user_id, 100, model, data)
    user = User(user_id=user_id, personal=list(personal))
    return user.__dict__


def main() -> None:
    """Run application"""
    uvicorn.run("solution:app", host="localhost")


if __name__ == "__main__":
    main()
