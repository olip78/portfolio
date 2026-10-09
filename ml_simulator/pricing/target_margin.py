from typing import List
from dataclasses import dataclass
from scipy.optimize import milp
from scipy.optimize import LinearConstraint, Bounds
from scipy.sparse import csr_array

import pandas as pd
import numpy as np


@dataclass
class PriceRecommender:
    """revenue optimization under constrained on overall weighted margin
    """
    def recommend_price(self, df: pd.DataFrame, target: float) -> List[bool]:
        """revenue under weighted general margin constrain optimizer 
        """
        n_sku = df.sku.unique().shape[0]
        n_alternatives = df.shape[0] // n_sku
        _df = df.copy()
        _df['revenue'] = _df['qty']*_df['price']
        _df['weighted_margin'] = (1 - target)*_df['revenue'] - _df['qty']*_df['cost']

        revenue = _df.revenue.values
        row_ind = np.repeat(range(n_sku), n_alternatives).astype(int)
        col_ind = np.arange(n_sku*n_alternatives, dtype='int').astype(int)

        equality_constraints = csr_array((np.ones(revenue.shape),
                                         (row_ind, col_ind)),
                                         shape=(n_sku, n_sku*n_alternatives))
        inequality_constraints = csr_array((_df['weighted_margin'].values,
                                           (row_ind, col_ind)),
                                           shape=(n_sku, n_sku*n_alternatives))
        equality_constraints = LinearConstraint(equality_constraints, 1, 1)
        inequality_constraints = LinearConstraint(inequality_constraints, 0, np.inf)
        bounds = Bounds(0, 1)
        integrality = np.full_like(revenue, True)

        res = milp(c=-revenue,
                   constraints=[equality_constraints,
                                inequality_constraints],
                   bounds=bounds,
                   integrality=integrality)
        if res.x is not None:
            return [bool(x) for x in res.x]
        else:
            #raise ValueError('cannot be solved under a such constraint')
            def argmax(price):
                return price == price.max()
            mask = df.groupby("sku")["price"].apply(argmax)
            mask = mask.values.tolist()
            return mask

    @staticmethod
    def gmv(df: pd.DataFrame) -> pd.Series:
        """returns revenue
        """
        return df['price'] * df['qty']

    @staticmethod
    def margin(df: pd.DataFrame) -> pd.Series:
        """returns margin
        """
        return (df['price'] - df['cost']) / df['price']

    @staticmethod
    def weighted_margin(df: pd.DataFrame) -> float:
        """returns avg weighted general margin
        """
        n_sku = df.sku.unique().shape[0]
        n_alternatives = df.shape[0] // n_sku
        gvm = PriceRecommender.gmv(df)
        margin = PriceRecommender.margin(df)
        return (margin * gvm / gvm.sum()).sum() / n_alternatives
