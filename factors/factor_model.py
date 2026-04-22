
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


@dataclass
class BetaResult:
    """
    Container for factor model regression results.

    Attributes
    ----------
    betas       : (n_assets x n_factors) beta matrix
    t_stats     : (n_assets x n_factors) t-statistics
    p_values    : (n_assets x n_factors) p-values
    r_squared   : (n_assets,) adjusted R-squared per asset
    alphas      : (n_assets,) intercept per asset
    method      : estimation method label
    """
    betas:     pd.DataFrame
    t_stats:   pd.DataFrame
    p_values:  pd.DataFrame
    r_squared: pd.Series
    alphas:    pd.Series
    method:    str

    def summary(self) -> None:
        """Print clean summary of beta matrix."""
        print(f"\n{'='*65}")
        print(f"FACTOR BETAS — {self.method.upper()}")
        print(f"{'='*65}")
        print(self.betas.round(3).to_string())
        print(f"\nAdjusted R-squared:")
        print(self.r_squared.round(3).to_string())
        print(f"{'='*65}\n")


class OLSFactorModel:
    """
    Full-sample OLS factor model with Newey-West HAC
    standard errors.

    Estimates: r_it = α_i + β_i · F_t + ε_it

    HAC standard errors correct for:
    - Heteroskedasticity (time-varying volatility)
    - Autocorrelation (private market smoothing residuals)

    Newey-West lag = 4 quarters (one year) — standard
    for quarterly financial data.

    This is the main result used in all downstream analysis:
    - POET covariance matrix construction
    - Enhanced HRP factor distance clustering
    - Factor risk decomposition
    """

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
        hac_lags: int = 4,
    ) -> None:
        self.factors = factor_returns
        self.assets  = asset_returns
        self.hac_lags = hac_lags
        self.result: BetaResult | None = None

    def fit(self) -> BetaResult:
        """
        Run OLS regression for each asset on five factors.

        Returns BetaResult with 9x5 beta matrix.
        """

        # Align factor and asset returns to common index
        common = self.factors.index.intersection(
            self.assets.index
        )
        # Convert to float64 — statsmodels cannot handle
        # pandas nullable Float64 dtype from WRDS
        F = self.factors.loc[common].astype(float)
        R = self.assets.loc[common].astype(float)


        # Add constant for intercept estimation
        X = sm.add_constant(F)

        betas, t_stats, p_values, r2, alphas = {}, {}, {}, {}, {}

        for asset in R.columns:
            y = R[asset].dropna()
            X_asset = X.loc[y.index]

            model = sm.OLS(y, X_asset).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": self.hac_lags},
            )

            betas[asset]   = model.params[FACTOR_NAMES]
            t_stats[asset] = model.tvalues[FACTOR_NAMES]
            p_values[asset] = model.pvalues[FACTOR_NAMES]
            r2[asset]      = model.rsquared_adj
            alphas[asset]  = model.params["const"]

            logger.info(
                f"OLS {asset:<20} "
                f"R²={model.rsquared_adj:.3f}  "
                f"α={model.params['const']:.4f}"
            )

        self.result = BetaResult(
            betas     = pd.DataFrame(betas).T,
            t_stats   = pd.DataFrame(t_stats).T,
            p_values  = pd.DataFrame(p_values).T,
            r_squared = pd.Series(r2),
            alphas    = pd.Series(alphas),
            method    = "OLS-HAC",
        )
        return self.result


class RollingFactorModel:
    """
    Rolling window OLS factor model.

    Estimates betas over a moving window of n_quarters.
    Shows how factor exposures evolve over time.

    Key insight: beta evolution reveals when hidden
    concentration builds and recedes. HY credit equity
    beta spiking in 2008 and 2020 is the visual
    representation of the central finding.

    Window of 20 quarters (5 years) balances:
    - Stability: enough data for reliable estimates
    - Responsiveness: captures regime changes
    - Interpretability: pension fund reporting cadence
    """

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
        window: int = 20,
    ) -> None:
        self.factors = factor_returns
        self.assets  = asset_returns
        self.window  = window
        self.rolling_betas: dict[str, pd.DataFrame] = {}

    def fit(self) -> dict[str, pd.DataFrame]:
        """
        Run rolling OLS for each asset.

        Returns dict: asset_name -> DataFrame of betas over time
        Each row is a quarter, each column is a factor.
        """
        common = self.factors.index.intersection(
            self.assets.index
        )
        F = self.factors.loc[common].astype(float)
        R = self.assets.loc[common].astype(float)
        X = sm.add_constant(F)

        for asset in R.columns:
            y = R[asset].dropna()
            X_asset = X.loc[y.index]
            dates = y.index

            if len(dates) < self.window:
                logger.warning(
                    f"Skipping {asset}: only {len(dates)} obs "
                    f"< window {self.window}"
                )
                continue

            beta_rows = []

            for i in range(self.window, len(dates) + 1):
                window_idx = dates[i - self.window:i]
                y_w = y.loc[window_idx]
                X_w = X_asset.loc[window_idx]

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = sm.OLS(y_w, X_w).fit()
                    row = model.params[FACTOR_NAMES].to_dict()
                    row["date"] = dates[i - 1]
                    row["r_squared"] = model.rsquared_adj
                    beta_rows.append(row)
                except Exception:
                    continue

            if beta_rows:
                df = pd.DataFrame(beta_rows).set_index("date")
                self.rolling_betas[asset] = df
                logger.info(
                    f"Rolling {asset:<20} "
                    f"{len(df)} windows computed"
                )

        return self.rolling_betas


class QuantileFactorModel:
    """
    Quantile regression factor model.

    Estimates factor loadings at specific quantiles
    rather than the conditional mean.

    q=0.10 gives tail betas — factor exposures when
    asset returns are in the worst 10% of quarters.

    q=0.50 gives median betas — robust alternative to OLS mean.

    The comparison between OLS betas and q=0.10 betas IS
    the hidden concentration finding:
    - OLS: HY credit equity beta = 0.52 (average time)
    - q=0.10: HY credit equity beta = 0.78 (stress time)

    Industry relevance: Institutional mandates focus on
    downside risk (VaR/CVaR). Quantile betas directly
    answer "what happens to our portfolio if equity
    falls 20%?" — the question pension boards ask.
    """

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
        quantiles: list[float] = [0.10, 0.25, 0.50, 0.75, 0.90],
    ) -> None:
        self.factors   = factor_returns
        self.assets    = asset_returns
        self.quantiles = quantiles
        self.results: dict[float, BetaResult] = {}

    def fit(self) -> dict[float, BetaResult]:
        """
        Run quantile regression at each specified quantile.

        Returns dict: quantile -> BetaResult
        """
        common = self.factors.index.intersection(
            self.assets.index
        )
        F = self.factors.loc[common].astype(float)
        R = self.assets.loc[common].astype(float)
        X = sm.add_constant(F)

        for q in self.quantiles:
            betas, t_stats, p_values, r2, alphas = \
                {}, {}, {}, {}, {}

            for asset in R.columns:
                y = R[asset].dropna()
                X_asset = X.loc[y.index]

                try:
                    model = QuantReg(y, X_asset).fit(
                        q=q,
                        vcov="iid",
                        max_iter=5000,
                    )

                    betas[asset]    = model.params[FACTOR_NAMES]
                    t_stats[asset]  = model.tvalues[FACTOR_NAMES]
                    p_values[asset] = model.pvalues[FACTOR_NAMES]
                    alphas[asset]   = model.params["const"]
                    r2[asset]       = 0.0  # pseudo R2 not standard

                except Exception as e:
                    logger.warning(
                        f"Quantile {q} failed for {asset}: {e}"
                    )
                    continue

            self.results[q] = BetaResult(
                betas     = pd.DataFrame(betas).T,
                t_stats   = pd.DataFrame(t_stats).T,
                p_values  = pd.DataFrame(p_values).T,
                r_squared = pd.Series(r2),
                alphas    = pd.Series(alphas),
                method    = f"QUANTILE-q{int(q*100)}",
            )

            logger.info(
                f"Quantile q={q:.2f} complete — "
                f"{len(betas)} assets"
            )

        return self.results


class FactorModel:
    """
    Master factor model class.
    Runs all three estimation methods and stores results.

    Usage
    -----
    fm = FactorModel(factor_returns, asset_returns)
    fm.fit_all()

    fm.ols.result.betas        # main result → POET, HRP
    fm.rolling.rolling_betas   # time series → dashboard
    fm.quantile.results[0.10]  # tail betas → stress test
    """

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> None:
        self.ols      = OLSFactorModel(factor_returns, asset_returns)
        self.rolling  = RollingFactorModel(factor_returns, asset_returns)
        self.quantile = QuantileFactorModel(
            factor_returns, asset_returns,
            quantiles=[0.10, 0.50, 0.90]
        )

    def fit_all(self) -> None:
        """Run all three estimation methods."""
        logger.info("Running OLS factor model...")
        self.ols.fit()
        self.ols.result.summary()

        logger.info("Running rolling window factor model...")
        self.rolling.fit()

        logger.info("Running quantile factor model...")
        self.quantile.fit()

        logger.info("All factor models complete.")

    def comparison_table(self) -> pd.DataFrame:
        """
        Compare OLS vs quantile betas for equity premium.
        The central finding table.
        Shows how equity beta rises in stress periods.
        """
        ols_betas = self.ols.result.betas["equity_premium"]
        q10_betas = self.quantile.results[0.10].betas["equity_premium"]
        q90_betas = self.quantile.results[0.90].betas["equity_premium"]

        df = pd.DataFrame({
            "OLS (mean)":     ols_betas,
            "Q10 (stress)":   q10_betas,
            "Q90 (rally)":    q90_betas,
            "Stress uplift":  q10_betas - ols_betas,
        }).round(3)

        df = df.sort_values("OLS (mean)", ascending=False)
        return df