from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from config.settings import (
    ALPHA_MAX,
    ALPHA_MIN,
    CA_FILES,
    DATA_CACHE_DIR,
    PRIVATE_ASSETS,
)

logger = logging.getLogger(__name__)


class CALoader:
    """
    Loads Cambridge Associates private market benchmarks.

    Applies Geltner (1993) unsmoothing to correct for
    appraisal-smoothed NAV valuations before factor analysis.

    Expected files in data_cache/:
        ca_private_equity.csv
        ca_private_credit.csv
        ca_private_real_estate.csv
        ca_infrastructure.csv

    Each CSV must have two columns:
        date         — quarter end date (YYYY-MM-DD)
        return       — quarterly net return (decimal or percent)

    Status: STUB — awaiting data from professor.
    Returns empty DataFrame until files are provided.
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self.alphas: dict[str, float] = {}

    # ── Private ────────────────────────────────────────────────────

    def _cache(self) -> Path:
        return DATA_CACHE_DIR / "ca_private_markets.parquet"

    def _files_exist(self) -> bool:
        """Check if all CA data files are present."""
        return all(
            (DATA_CACHE_DIR / fname).exists()
            for fname in CA_FILES.values()
        )

    def _load_raw(self, asset: str) -> pd.Series:
        """Load one CA CSV file."""
        path = DATA_CACHE_DIR / CA_FILES[asset]
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        s = df.iloc[:, 0].rename(asset)

        # Convert percent to decimal if needed
        if s.abs().max() > 1.0:
            logger.info(f"Converting {asset} from percent to decimal")
            s = s / 100.0

        return s

    def _estimate_alpha(self, returns: pd.Series) -> float:
        """
        Estimate smoothing parameter via OLS.
        r_t = alpha * r_{t-1} + epsilon
        """
        y = returns.iloc[1:].values
        X = add_constant(returns.iloc[:-1].values)
        model = OLS(y, X).fit()
        alpha = float(np.clip(model.params[1], ALPHA_MIN, ALPHA_MAX))
        logger.info(
            f"Smoothing alpha for {returns.name}: "
            f"{alpha:.3f} (t={model.tvalues[1]:.2f})"
        )
        return alpha

    def _geltner_unsmooth(
        self,
        returns: pd.Series,
        alpha: float,
    ) -> pd.Series:
        """
        Geltner (1993) unsmoothing filter.

        Recovers true economic returns from smoothed NAVs:
            r_true(t) = [r_rep(t) - alpha * r_rep(t-1)] / (1 - alpha)

        After unsmoothing expect:
        - Higher volatility (doubles for PE typically)
        - Lower autocorrelation (drops from 0.5+ to <0.15)
        - Higher correlation with public market factors
        - More negative tail observations
        """
        r = returns.values
        r_true = np.array([
            (r[t] - alpha * r[t - 1]) / (1 - alpha)
            for t in range(1, len(r))
        ])
        return pd.Series(
            r_true,
            index=returns.index[1:],
            name=returns.name,
        )

    def _print_diagnostics(
        self,
        raw: pd.Series,
        unsmoothed: pd.Series,
    ) -> None:
        """Print before/after comparison for validation."""
        common = raw.index.intersection(unsmoothed.index)
        r, u = raw.loc[common], unsmoothed.loc[common]

        print(f"\n{'='*52}")
        print(f"  {raw.name.upper().replace('_', ' ')}")
        print(f"{'='*52}")
        print(f"  {'Metric':<28} {'Raw':>8} {'Unsmoothed':>12}")
        print(f"  {'-'*50}")
        print(f"  {'AC1 (smoothing)':<28} "
              f"{r.autocorr(1):>8.3f} {u.autocorr(1):>12.3f}")
        print(f"  {'Ann. Volatility':<28} "
              f"{r.std()*2:>8.1%} {u.std()*2:>12.1%}")
        print(f"  {'Ann. Return':<28} "
              f"{(1+r.mean())**4-1:>8.1%} "
              f"{(1+u.mean())**4-1:>12.1%}")
        print(f"  {'Min Quarter':<28} "
              f"{r.min():>8.1%} {u.min():>12.1%}")
        print(f"{'='*52}")

    # ── Public ─────────────────────────────────────────────────────

    def get_private_market_returns(
        self,
        diagnostics: bool = True,
    ) -> pd.DataFrame:
        """
        Load, unsmooth, and return all CA private market series.

        Returns empty DataFrame if data files not yet available.
        Full pipeline activates once professor provides CSV files.

        Parameters
        ----------
        diagnostics : bool
            Print before/after unsmoothing comparison.

        Returns
        -------
        DataFrame: columns = PRIVATE_ASSETS
                   index   = quarter-end DatetimeIndex
                   values  = decimal quarterly returns (unsmoothed)
        """
        # ── Stub: return empty until data arrives ──────────────────
        if not self._files_exist():
            logger.warning(
                "CA data files not found in data_cache/. "
                "Returning empty DataFrame. "
                "Add CSV files when professor provides data."
            )
            return pd.DataFrame(columns=PRIVATE_ASSETS)

        # ── Cache check ────────────────────────────────────────────
        cache_path = self._cache()
        if self.use_cache and cache_path.exists():
            logger.info("Cache hit: ca_private_markets")
            return pd.read_parquet(cache_path)

        # ── Full pipeline ──────────────────────────────────────────
        series = {}

        for asset in PRIVATE_ASSETS:
            logger.info(f"Processing {asset}...")
            raw = self._load_raw(asset)
            alpha = self._estimate_alpha(raw)
            self.alphas[asset] = alpha
            unsmoothed = self._geltner_unsmooth(raw, alpha)

            if diagnostics:
                self._print_diagnostics(raw, unsmoothed)

            series[asset] = unsmoothed

        df = pd.concat(series, axis=1).dropna()
        df.to_parquet(cache_path)
        logger.info(
            f"CA private markets: {df.shape[0]} quarters, "
            f"{df.index[0].date()} to {df.index[-1].date()}"
        )
        return df