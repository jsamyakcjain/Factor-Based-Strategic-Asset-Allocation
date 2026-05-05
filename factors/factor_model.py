from __future__ import annotations
import warnings
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from config.settings import ASSET_NAMES, FACTOR_NAMES

logger = logging.getLogger(__name__)

EQUITY_LIQUIDITY_ASSETS = [
    "us_large_cap", "us_mid_cap", "us_small_cap", "em_equity",
    "reits", "commodities",
]

CREDIT_LIQUIDITY_ASSETS = [
    "long_treasury", "tips", "ig_credit", "hy_credit",
    "private_equity_unsmthd", "real_estate_unsmthd", "hedge_funds",
    "private_credit", "private_real_estate", "infrastructure",
]

@dataclass
class BetaResult:
    betas:     pd.DataFrame
    t_stats:   pd.DataFrame
    p_values:  pd.DataFrame
    r_squared: pd.Series
    alphas:    pd.Series
    method:    str

    def summary(self) -> None:
        print(f"\n{'='*65}")
        print(f"FACTOR BETAS — {self.method.upper()}")
        print(f"{'='*65}")
        print(self.betas.round(3).to_string())
        print(f"\nAdjusted R-squared:")
        print(self.r_squared.round(3).to_string())
        print(f"{'='*65}\n")


class _BaseFactorModel:
    """Helper class to enforce identical data prep across all models."""
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
        credit_liquidity: pd.Series | None = None,
    ) -> None:
        # ── 1. Create data attributes FIRST ──────────────────────────
        common = factor_returns.index.intersection(asset_returns.index)
        self.F = factor_returns.loc[common].astype(float)
        self.R = asset_returns.loc[common].astype(float)
        self.CL = None
        if credit_liquidity is not None:
            self.CL = credit_liquidity.reindex(common).astype(float)
        
        # ── 2. Compute global standardization parameters AFTER ───────
        self.F_mean = self.F.mean()
        self.F_std = self.F.std().where(self.F.std() > 1e-10, 1.0)

    def _get_asset_matrices(self, asset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Returns (X_raw, X_std, y) perfectly matched for the specific asset."""
        y = self.R[asset].dropna()
        F_asset = self.F.loc[y.index].copy()

        use_credit_liq = asset in CREDIT_LIQUIDITY_ASSETS and self.CL is not None

        if use_credit_liq:
            F_asset["liquidity"] = self.CL.loc[y.index]

        # Raw X — used for unstandardized betas and alpha
        X_raw = sm.add_constant(F_asset)

        # Standardization: use per-column mean/std that matches the proxy actually used.
        # If we swapped liquidity to quality spread, we must use quality spread's own
        # mean/std — not PS liquidity's — otherwise t-statistics for the liquidity
        # factor are on an incomparable scale for credit assets.
        F_mean = self.F_mean.copy()
        F_std  = self.F_std.copy()
        if use_credit_liq:
            cl_series = self.CL.reindex(y.index)
            F_mean["liquidity"] = float(cl_series.mean())
            cl_std = float(cl_series.std())
            F_std["liquidity"]  = cl_std if cl_std > 1e-10 else 1.0

        F_standardized = (F_asset - F_mean) / F_std
        X_std = sm.add_constant(F_standardized)

        return X_raw, X_std, y


class OLSFactorModel(_BaseFactorModel):
    def __init__(self, *args, hac_lags: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hac_lags = hac_lags
        self.result: BetaResult | None = None
        self.result_std: BetaResult | None = None

    def fit(self) -> BetaResult:
        betas_raw, betas_std = {}, {}
        t_stats, p_values, r2, alphas = {}, {}, {}, {}

        for asset in self.R.columns:
            X_raw, X_std, y = self._get_asset_matrices(asset)

            # 1. Run on RAW data to get true Alpha and Unstandardized Betas
            model_raw = sm.OLS(y, X_raw).fit(cov_type="HAC", cov_kwds={"maxlags": self.hac_lags})
            
            betas_raw[asset] = model_raw.params.reindex(FACTOR_NAMES)
            alphas[asset]    = model_raw.params.get("const", np.nan)
            r2[asset]        = model_raw.rsquared_adj
            
            # 2. Run on STD data to get comparable Betas and valid T-Stats
            model_std = sm.OLS(y, X_std).fit(cov_type="HAC", cov_kwds={"maxlags": self.hac_lags})
            
            betas_std[asset] = model_std.params.reindex(FACTOR_NAMES)
            t_stats[asset]   = model_std.tvalues.reindex(FACTOR_NAMES)
            p_values[asset]  = model_std.pvalues.reindex(FACTOR_NAMES)

            logger.info(f"OLS {asset:<22} R²={model_raw.rsquared_adj:.3f} α={alphas[asset]:.4f}")

        self.result = BetaResult(
            betas     = pd.DataFrame.from_dict(betas_raw, orient="index"),
            t_stats   = pd.DataFrame.from_dict(t_stats, orient="index"),
            p_values  = pd.DataFrame.from_dict(p_values, orient="index"),
            r_squared = pd.Series(r2),
            alphas    = pd.Series(alphas),
            method    = "OLS-HAC"
        )
        
        self.result_std = BetaResult(
            betas     = pd.DataFrame.from_dict(betas_std, orient="index"),
            t_stats   = pd.DataFrame.from_dict(t_stats, orient="index"),
            p_values  = pd.DataFrame.from_dict(p_values, orient="index"),
            r_squared = pd.Series(r2),
            alphas    = pd.Series(alphas),
            method    = "OLS-HAC-STANDARDIZED"
        )
        
        return self.result


class RollingFactorModel(_BaseFactorModel):
    def __init__(self, *args, window: int = 20, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window = window
        self.rolling_betas: dict[str, pd.DataFrame] = {}

    def fit(self) -> dict[str, pd.DataFrame]:
        for asset in self.R.columns:
            X_raw, _, y = self._get_asset_matrices(asset)
            dates = y.index

            if len(dates) < self.window:
                continue

            beta_rows = []
            for i in range(self.window, len(dates) + 1):
                window_idx = dates[i - self.window:i]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = sm.OLS(y.loc[window_idx], X_raw.loc[window_idx]).fit()
                    
                    row = model.params.reindex(FACTOR_NAMES).to_dict()
                    row["date"] = dates[i - 1]
                    row["r_squared"] = model.rsquared_adj
                    beta_rows.append(row)
                except Exception:
                    continue

            if beta_rows:
                self.rolling_betas[asset] = pd.DataFrame(beta_rows).set_index("date")
        return self.rolling_betas


class QuantileFactorModel(_BaseFactorModel):
    def __init__(self, *args, quantiles: list[float] = [0.10, 0.25, 0.50, 0.75, 0.90], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quantiles = quantiles
        self.results: dict[float, BetaResult] = {}

    def fit(self) -> dict[float, BetaResult]:
        for q in self.quantiles:
            betas, t_stats, p_values, r2, alphas = {}, {}, {}, {}, {}

            for asset in self.R.columns:
                X_raw, _, y = self._get_asset_matrices(asset)
                try:
                    model = QuantReg(y, X_raw).fit(q=q, vcov="iid", max_iter=20000)
                    
                    betas[asset]    = model.params.reindex(FACTOR_NAMES)
                    t_stats[asset]  = model.tvalues.reindex(FACTOR_NAMES)
                    p_values[asset] = model.pvalues.reindex(FACTOR_NAMES)
                    alphas[asset]   = model.params.get("const", np.nan)
                    r2[asset]       = 0.0 
                except Exception:
                    continue

            if betas:
                self.results[q] = BetaResult(
                    betas     = pd.DataFrame.from_dict(betas, orient="index"),
                    t_stats   = pd.DataFrame.from_dict(t_stats, orient="index"),
                    p_values  = pd.DataFrame.from_dict(p_values, orient="index"),
                    r_squared = pd.Series(r2),
                    alphas    = pd.Series(alphas),
                    method    = f"QUANTILE-q{int(q*100)}"
                )
        return self.results


class FactorModel:
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
        credit_liquidity: pd.Series | None = None,
    ) -> None:
        self.ols      = OLSFactorModel(factor_returns, asset_returns, credit_liquidity=credit_liquidity)
        self.rolling  = RollingFactorModel(factor_returns, asset_returns, credit_liquidity=credit_liquidity)
        self.quantile = QuantileFactorModel(factor_returns, asset_returns, credit_liquidity=credit_liquidity, quantiles=[0.10, 0.50, 0.90])

    def fit_all(self) -> None:
        logger.info("Running OLS factor model...")
        self.ols.fit()
        self.ols.result.summary()

        logger.info("Running rolling window factor model...")
        self.rolling.fit()

        logger.info("Running quantile factor model...")
        self.quantile.fit()

        logger.info("All factor models complete.")

    def comparison_table(self) -> pd.DataFrame:
        ols_betas = self.ols.result.betas["equity_premium"]
        q10_betas = self.quantile.results[0.10].betas["equity_premium"]
        q90_betas = self.quantile.results[0.90].betas["equity_premium"]

        df = pd.DataFrame({
            "OLS (mean)":     ols_betas,
            "Q10 (stress)":   q10_betas,
            "Q90 (rally)":    q90_betas,
            "Stress uplift":  q10_betas - ols_betas,
        }).round(3)

        return df.sort_values("OLS (mean)", ascending=False)