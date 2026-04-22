from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import ASSET_NAMES, JPM_LTCMA, RISK_FREE_RATE


class ExpectedReturns:
    """
    Expected return vector for MVO optimizer.

    Source: JPMorgan 2026 Long-Term Capital Market Assumptions.
    All returns annualized, USD, 10-15 year horizon.

    Public assets: JPM LTCMA directly.
    Private assets: JPM LTCMA alternatives section.
    Hedge funds: JPM LTCMA average of 6 strategies.

    Converts annual to quarterly for consistency
    with quarterly return data used in covariance.
    """

    def __init__(self, assets: list[str] | None = None) -> None:
        self.assets = assets or ASSET_NAMES

    def annual(self) -> pd.Series:
        """Annual expected returns from JPM LTCMA."""
        return pd.Series(
            {a: JPM_LTCMA[a] for a in self.assets},
            name="expected_return_annual",
        )

    def quarterly(self) -> pd.Series:
        """
        Quarterly expected returns.
        Converts: (1 + r_annual)^(1/4) - 1
        """
        ann = self.annual()
        qtr = (1 + ann) ** (1 / 4) - 1
        qtr.name = "expected_return_quarterly"
        return qtr

    def excess_quarterly(self) -> pd.Series:
        """
        Quarterly excess returns over risk-free rate.
        Used as MVO objective input.
        rf_quarterly = (1 + rf_annual)^(1/4) - 1
        """
        rf_q = (1 + RISK_FREE_RATE) ** (1 / 4) - 1
        return (self.quarterly() - rf_q).rename(
            "excess_return_quarterly"
        )

    def summary(self) -> pd.DataFrame:
        """Print summary of expected returns."""
        df = pd.DataFrame({
            "Annual %":    self.annual() * 100,
            "Quarterly %": self.quarterly() * 100,
            "Excess Q %":  self.excess_quarterly() * 100,
        }).round(3)
        return df