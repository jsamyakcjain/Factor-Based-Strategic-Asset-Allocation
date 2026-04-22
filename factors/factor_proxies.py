from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from config.settings import FACTOR_NAMES

logger = logging.getLogger(__name__)


class FactorProxies:
    """
    Validates and summarizes the five factor return series
    before running regressions.

    Responsibilities:
    1. Stationarity tests (ADF) on each factor
    2. Summary statistics table
    3. Factor correlation matrix
    4. Flag any issues before regression

    This is a quality control layer — not data construction.
    Data construction happens in DataManager.
    """

    def __init__(self, factor_returns: pd.DataFrame) -> None:
        self.factors = factor_returns.copy()
        self.adf_results: dict = {}
        self.is_validated: bool = False

    # ── Stationarity ───────────────────────────────────────────────

    def run_adf_tests(self) -> pd.DataFrame:
        """
        Augmented Dickey-Fuller stationarity test for each factor.

        H0: Series has a unit root (non-stationary)
        H1: Series is stationary

        We reject H0 (confirm stationarity) when p-value < 0.05.
        Non-stationary factors in OLS produce spurious results.

        Returns
        -------
        DataFrame with ADF statistic, p-value, and
        stationarity verdict for each factor.
        """
        results = []

        for factor in FACTOR_NAMES:
            series = self.factors[factor].dropna()

            # ADF test with automatic lag selection (AIC criterion)
            adf_stat, p_value, n_lags, n_obs, critical, _ = adfuller(
                series,
                autolag="AIC",
                regression="c",   # include constant
            )

            stationary = p_value < 0.05
            verdict = "STATIONARY" if stationary else "NON-STATIONARY"

            self.adf_results[factor] = {
                "adf_stat":  adf_stat,
                "p_value":   p_value,
                "n_lags":    n_lags,
                "n_obs":     n_obs,
                "cv_1pct":   critical["1%"],
                "cv_5pct":   critical["5%"],
                "stationary": stationary,
            }

            results.append({
                "Factor":      factor,
                "ADF Stat":    round(adf_stat, 3),
                "P-Value":     round(p_value, 4),
                "Lags":        n_lags,
                "Stationary":  verdict,
                "CV 1%":       round(critical["1%"], 3),
                "CV 5%":       round(critical["5%"], 3),
            })

            status = "✓" if stationary else "✗ WARNING"
            logger.info(
                f"ADF {factor:<20} "
                f"stat={adf_stat:>7.3f}  "
                f"p={p_value:.4f}  "
                f"{status}"
            )

        df = pd.DataFrame(results).set_index("Factor")

        # Flag non-stationary factors
        non_stationary = [
            f for f, r in self.adf_results.items()
            if not r["stationary"]
        ]
        if non_stationary:
            logger.warning(
                f"Non-stationary factors detected: {non_stationary}. "
                "Consider differencing before regression."
            )
        else:
            logger.info("All factors passed stationarity tests.")

        return df

    # ── Summary statistics ─────────────────────────────────────────

    def summary_stats(self) -> pd.DataFrame:
        """
        Summary statistics for all five factors.
        Quarterly frequency, decimal units.
        """
        stats = pd.DataFrame({
            "Mean (%)":    self.factors.mean() * 100,
            "Std (%)":     self.factors.std() * 100,
            "Ann Mean (%)": self.factors.mean() * 4 * 100,
            "Ann Std (%)":  self.factors.std() * np.sqrt(4) * 100,
            "Sharpe":      (
                self.factors.mean() /
                self.factors.std() *
                np.sqrt(4)
            ),
            "Min (%)":     self.factors.min() * 100,
            "Max (%)":     self.factors.max() * 100,
            "Skew":        self.factors.skew(),
            "Kurt":        self.factors.kurt(),
            "Obs":         self.factors.count(),
        }).round(3)

        return stats

    # ── Correlation ────────────────────────────────────────────────

    def correlation_matrix(self) -> pd.DataFrame:
        """Factor correlation matrix with VIF diagnostics."""
        corr = self.factors.corr().round(3)
        return corr

    def vif_check(self) -> pd.DataFrame:
        """
        Variance Inflation Factors for each factor.
        VIF > 5 = moderate concern.
        VIF > 10 = serious multicollinearity.
        """
        from sklearn.linear_model import LinearRegression

        vifs = []
        X = self.factors.dropna()

        for i, col in enumerate(X.columns):
            y = X[col].values
            X_other = X.drop(columns=[col]).values

            r2 = LinearRegression().fit(X_other, y).score(X_other, y)
            vif = 1 / (1 - r2) if r2 < 1 else np.inf

            status = (
                "OK" if vif < 5
                else "MODERATE" if vif < 10
                else "HIGH"
            )
            vifs.append({
                "Factor": col,
                "VIF":    round(vif, 3),
                "Status": status,
            })
            logger.info(f"VIF {col:<20} = {vif:.3f}  {status}")

        return pd.DataFrame(vifs).set_index("Factor")

    # ── Full validation ────────────────────────────────────────────

    def validate(self) -> bool:
        """
        Run all validation checks.
        Returns True if all checks pass.
        Logs warnings for any issues found.
        """
        print("\n" + "=" * 60)
        print("FACTOR VALIDATION REPORT")
        print("=" * 60)

        print("\n1. Summary Statistics")
        print(self.summary_stats().to_string())

        print("\n2. ADF Stationarity Tests")
        adf = self.run_adf_tests()
        print(adf[["ADF Stat", "P-Value", "Stationary"]].to_string())

        print("\n3. Correlation Matrix")
        print(self.correlation_matrix().to_string())

        print("\n4. VIF Check")
        print(self.vif_check().to_string())

        # Overall pass/fail
        non_stationary = [
            f for f, r in self.adf_results.items()
            if not r["stationary"]
        ]
        high_vif = []

        print("\n" + "=" * 60)
        if not non_stationary and not high_vif:
            print("VALIDATION PASSED — All checks OK")
            self.is_validated = True
        else:
            if non_stationary:
                print(f"WARNING: Non-stationary: {non_stationary}")
            if high_vif:
                print(f"WARNING: High VIF: {high_vif}")
            self.is_validated = False
        print("=" * 60 + "\n")

        return self.is_validated