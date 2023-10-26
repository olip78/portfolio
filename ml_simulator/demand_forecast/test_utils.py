import pandas as pd


def plausibility_test(losses: pd.DataFrame):
    """model plausibility test
    """

    # quantile
    for horizon in losses.horizon.unique():
        df = losses[losses.horizon==horizon]
        indices_base = df.sort_values('quantile')
        indices_check = df.sort_values('avg_quantile_loss')
        assert (indices_base.values - indices_check.values).sum() == 0, f"""
        model is not plausible: y(quantile1) =>  y(quantile2), quantile1 < quantile2"""

    # horizon
    for quantile in losses['quantile'].unique():
        df = losses[losses['quantile']==quantile]
        indices_base = df.sort_values('horizon')
        indices_check = df.sort_values('avg_quantile_loss')
        assert (indices_base.values - indices_check.values).sum() == 0, f"""
        model is not plausible: y(horizon1) =>  y(horizon2), horizon1 < horizon2"""
