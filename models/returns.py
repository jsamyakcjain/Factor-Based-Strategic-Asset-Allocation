from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import ASSET_NAMES, JPM_LTCMA, RISK_FREE_RATE

logger = logging.getLogger(__name__)


class ExpectedReturns:
    """
    Expected return vector for MVO optimizer.

    Source: JPMorgan 2026 Long-Term Capital Market Assumptions.
    All returns annualized, USD, 10-15 year horizon.
    """

    def __init__(self, assets: list[str] | None = None) -> None:
        # FIXED: Explicit None check prevents resurrecting filtered assets if an empty list is passed
        self.assets = ASSET_NAMES if assets is None else assets

    def annual(self) -> pd.Series:
        """Annual expected returns from JPM LTCMA."""
        returns = {}
        for a in self.assets:
            if a not in JPM_LTCMA:
                logger.error(f"Asset '{a}' missing from JPM_LTCMA. Dropping from expected returns.")
                continue
            returns[a] = JPM_LTCMA[a]
            
        if not returns:
            raise ValueError("No valid assets found in JPM_LTCMA mapping.")
            
        return pd.Series(returns, name="expected_return_annual")

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
        """
        # Geometric conversion for the risk-free rate
        rf_q = (1 + RISK_FREE_RATE) ** (1 / 4) - 1
        
        excess = self.quarterly() - rf_q
        excess.name = "excess_return_quarterly"
        return excess

    def summary(self) -> pd.DataFrame:
        """Print summary of expected returns."""
        df = pd.DataFrame({
            "Annual %":    self.annual() * 100,
            "Quarterly %": self.quarterly() * 100,
            "Excess Q %":  self.excess_quarterly() * 100,
        }).round(3)
        return df