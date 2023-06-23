"""feature extraction / zransformation piplines

needs:
- registration of calcers from extract.py
- registration of transformes from transform.py
"""


import sklearn.base as skbase
import sklearn.pipeline as skpipe
import functools
import dask.dataframe as dd
import datetime

from typing import List

from . import extract
from . import transform
from .base import FeatureCalcer
from ..connection import Engine


# calcer registration

def register_calcer(calcer_class, calcer_reference) -> None:
    calcer_reference[calcer_class.name] = calcer_class


CALCER_REFERENCE = {}
register_calcer(extract.AgeLocationCalcer, CALCER_REFERENCE)
register_calcer(extract.ReceiptsBasicFeatureCalcer, CALCER_REFERENCE)
register_calcer(extract.LocationSalesCalcer, CALCER_REFERENCE)
register_calcer(extract.LocationAgeSalesCalcer, CALCER_REFERENCE)
register_calcer(extract.TargetFromCampaignCalcer, CALCER_REFERENCE)


# combined feature extractor

def create_calcer(name: str, calcer_reference=CALCER_REFERENCE, **kwargs) -> FeatureCalcer:
    return calcer_reference[name](**kwargs)

def extract_features(engine: Engine, config: dict, calcer_reference=CALCER_REFERENCE) -> dd.DataFrame:
    calcers = list()
    """extract featured, defined in config
    """
    
    for feature_config in config:
        calcer_args = feature_config["args"]
        calcer_args["engine"] = engine

        calcer = create_calcer(feature_config["name"], calcer_reference=calcer_reference, **calcer_args)
        calcers.append(calcer)

    result = calcers[0].compute()
    for calcer in calcers[1:]:
        df = calcer.compute()
        result = result.merge(df, on=calcer.keys, how='outer')

    return result

# registration of transformers

def register_transformer(transformer_class, name: str, transformer_reference) -> None:
    transformer_reference[name] = transformer_class

TRANSFORMER_REFERENCE = {}
register_transformer(transform.ExpressionTransformer, 'expression', TRANSFORMER_REFERENCE)
register_transformer(transform.LocationRelativeConsumption, 'location_relative_consumption', TRANSFORMER_REFERENCE)
register_transformer(transform.OneHotEncoder, 'one_hot_encode', TRANSFORMER_REFERENCE)
register_transformer(transform.DropColumns, 'drop_columns', TRANSFORMER_REFERENCE)
register_transformer(transform.BinningTransformer, 'binning', TRANSFORMER_REFERENCE)


# transformers pipeline

def create_transformer(name: str, transformer_reference=TRANSFORMER_REFERENCE, **kwargs) -> skbase.BaseEstimator:
    return transformer_reference[name](**kwargs)

def create_pipeline(transform_config: dict, transformer_reference=TRANSFORMER_REFERENCE) -> skpipe.Pipeline:
    """build feature transformation pipeline
    """
    
    transformers = list()

    for i, transformer_config in enumerate(transform_config):
        transformer_args = transformer_config["args"]
        transformer = create_transformer(transformer_config["name"], transformer_reference=transformer_reference, **transformer_args)
        uname = transformer_config.get("uname", f'stage_{i}')

        transformers.append((uname, transformer))

    pipeline = skpipe.Pipeline(transformers)
    return pipeline
