from __future__ import annotations
from sklearn.base import TransformerMixin
from scipy.optimize import milp
from scipy.optimize import LinearConstraint

import pandas as pd
import numpy as np


class CocaCola(TransformerMixin):
    """Post-processing of the given optimal prices according to 
    the ranked volume constraints: p(v1) < p(v2) < ... < p(vn) if v1 < v2 < ... < vn 
    for products in one vertical line and 
    p1 < p2 < ... < pn for products in one horizontal line. 
    The class finds the minimum vertical and horizontal adjustments
    so that the resulting values satisfy these conditions.
    """
    def _v_adj(self, prices: pd.Series, rank: pd.Series) -> np.array:
        """vertical adjustment
        """
        length = prices.shape[0]

        if length > 1:
            norm_volume = rank / rank.min()
            # constraints
            # rank price: p1 < p2 if v1 < v2
            b_l_rank = np.sign(rank.diff(1).dropna())
            b_u_rank = (b_l_rank*np.inf).fillna(0)
            row = np.zeros(length)
            row[0:2] = [-1, 1]
            m_rank_price = np.array([np.roll(row, s) for s in range(length - 1)])

            # rank price / volume: p1/v1 > p2/v2
            m_rank_price_volume = np.zeros((length - 1, length))
            for i in range(length - 1):
                m_rank_price_volume[i, i] = norm_volume.values[i+1]
                m_rank_price_volume[i, i+1] = -norm_volume.values[i]

            # abs
            row = np.zeros(2*length)
            row[[0, length]] = [1, 1]
            m_abs_plus = np.array([np.roll(row, s) for s in range(length)])
            row = np.zeros(2*length)
            row[[0, length]] = [-1, 1]
            m_abs_minus = np.array([np.roll(row, s) for s in range(length)])

            # optimization
            zero_block_rank = np.zeros(m_rank_price.shape)
            constraints = np.vstack([np.hstack([m_rank_price, zero_block_rank]),
                                     np.hstack([m_rank_price_volume, zero_block_rank]),
                                     m_abs_plus,
                                     m_abs_minus
                                    ])
            costs = np.hstack([np.zeros(length), np.ones(length)])

            b_l = np.hstack([b_l_rank.values, b_l_rank.values, prices.values, -prices.values])
            b_u = np.hstack([b_u_rank.values, b_u_rank.values, [np.inf]*2*length])

            constraints = LinearConstraint(constraints, b_l, b_u)
            integrality = np.full_like(costs, True)
            res = milp(c=costs,
                       constraints=[constraints],
                       integrality=integrality
                      )
            res = res.x[:length].astype(int)
        else:
            res = prices
        return res

    def fix_vlines(self, df: pd.DataFrame) -> pd.DataFrame:
        """applys vertical adjustment to all vlines groups
        """
        _df = df.copy()
        _df.vline = _df.vline.replace('', np.nan)
        for _, g in _df.groupby('vline'):
            group = g.sort_values(['netto'])
            _df.loc[group.index, 'price'] = self._v_adj(group.price, group.netto)
        _df.vline = df.vline
        return _df

    def _h_adj(self, prices):
        """horisontal adjustment
        """
        length = prices.shape[0]

        if length > 1:
            # constraints
            # abs
            m_abs_plus = np.hstack([np.ones((length, 1)), np.eye(length)])
            m_abs_minus = np.hstack([-np.ones((length, 1)), np.eye(length)])

            # optimization
            constraints = np.vstack([m_abs_plus, m_abs_minus])
            costs = np.hstack([np.zeros(1), np.ones(length)])
            b_l = np.hstack([prices.values, -prices.values])
            b_u = np.hstack([[np.inf]*2*length])
            constraints = LinearConstraint(constraints, b_l, b_u)
            integrality = np.full_like(costs, True)
            res = milp(c=costs,
                       constraints=[constraints],
                       integrality=integrality
                      )
            res = [res.x[0].astype(int)]*length
        else:
            res = prices
        return res

    def fix_hlines(self, df: pd.DataFrame) -> pd.DataFrame:
        """applys horisontal adjustment to all vlines groups
        """
        _df = df.copy()
        _df.hline = _df.hline.replace('', np.nan)
        for _, g in _df.groupby('hline'):
            _df.loc[g.index, 'price'] = self._h_adj(g.price)

        _df.hline = df.hline
        return _df

    def fit(self, df: pd.DataFrame, y: pd.Series = None) -> CocaCola:
        """dummy fit
        """
        return self

    def transform(self, df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """applys vertical and hirisontal adjustments 
        """
        _df = self.fix_vlines(df)
        _df = self.fix_hlines(_df)
        return _df
