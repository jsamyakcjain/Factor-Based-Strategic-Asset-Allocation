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
    """

    def __init__(self, factor_returns: pd.DataFrame) -> None:
        self.factors = factor_returns.copy()
        self.adf_results: dict = {}
        self.high_vif_factors: list[str] = []
        self.is_validated: bool = False

    def run_adf_tests(self) -> pd.DataFrame:
        results = []

        for factor in FACTOR_NAMES:
            series = self.factors[factor].dropna()
            
            if len(series) < 10:
                logger.error(f"Not enough data to run ADF for {factor}")
                continue

            # ADF test with automatic lag selection (AIC criterion)
            adf_stat, p_value, n_lags, n_obs, critical, _ = adfuller(
                series, autolag="AIC", regression="c"
            )

            stationary = p_value < 0.05
            self.adf_results[factor] = {
                "adf_stat":  adf_stat, "p_value": p_value,
                "n_lags": n_lags, "n_obs": n_obs,
                "cv_1pct": critical["1%"], "cv_5pct": critical["5%"],
                "stationary": stationary,
            }

            results.append({
                "Factor":      factor,
                "ADF Stat":    round(adf_stat, 3),
                "P-Value":     round(p_value, 4),
                "Lags":        n_lags,
                "Stationary":  "STATIONARY" if stationary else "NON-STATIONARY",
            })

            status = "✓" if stationary else "✗ WARNING"
            logger.info(f"ADF {factor:<20} stat={adf_stat:>7.3f} p={p_value:.4f} {status}")

        df = pd.DataFrame(results).set_index("Factor")

        non_stationary = [f for f, r in self.adf_results.items() if not r["stationary"]]
        if non_stationary:
            logger.warning(f"Non-stationary factors detected: {non_stationary}.")
        else:
            logger.info("All factors passed stationarity tests.")

        return df


    def summary_stats(self) -> pd.DataFrame:
        stats = pd.DataFrame({
            "Mean (%)":    self.factors.mean() * 100,
            "Std (%)":     self.factors.std() * 100,
            "Ann Mean (%)": self.factors.mean() * 4 * 100,
            "Ann Std (%)":  self.factors.std() * np.sqrt(4) * 100,
            "Sharpe":      (self.factors.mean() / self.factors.std() * np.sqrt(4)),
            "Min (%)":     self.factors.min() * 100,
            "Max (%)":     self.factors.max() * 100,
            "Skew":        self.factors.skew(),
            "Kurt":        self.factors.kurt(),
            "Obs":         self.factors.count(),
        }).round(3)
        return stats


    def correlation_matrix(self) -> pd.DataFrame:
        return self.factors.corr().round(3)


    def vif_check(self) -> pd.DataFrame:
        from sklearn.linear_model import LinearRegression

        vifs = []
        self.high_vif_factors = []
        
        # We must drop NAs to run multiple regression, but log the surviving window
        X = self.factors[FACTOR_NAMES].dropna()
        logger.info(f"VIF Check running on overlapping window of {len(X)} observations.")

        for i, col in enumerate(X.columns):
            y = X[col].values
            X_other = X.drop(columns=[col]).values

            r2 = LinearRegression().fit(X_other, y).score(X_other, y)
            vif = 1 / (1 - r2) if r2 < 1 else np.inf

            if vif >= 5:
                self.high_vif_factors.append(col)

            status = "OK" if vif < 5 else "MODERATE" if vif < 10 else "HIGH"
            vifs.append({
                "Factor": col,
                "VIF":    round(vif, 3),
                "Status": status,
            })
            logger.info(f"VIF {col:<20} = {vif:.3f}  {status}")

        return pd.DataFrame(vifs).set_index("Factor")


    def validate(self) -> bool:
        print("\n" + "=" * 60)
        print("FACTOR VALIDATION REPORT")
        print("=" * 60)

        print("\n1. Summary Statistics")
        print(self.summary_stats().to_string())

        print("\n2. ADF Stationarity Tests")
        adf = self.run_adf_tests()
        if not adf.empty:
            print(adf.to_string())

        print("\n3. Correlation Matrix")
        print(self.correlation_matrix().to_string())

        print("\n4. VIF Check")
        # FIXED: Actually running the check and populating self.high_vif_factors
        print(self.vif_check().to_string())

        # FIXED: Extract actual results for pass/fail logic
        non_stationary = [f for f, r in self.adf_results.items() if not r["stationary"]]
        
        print("\n" + "=" * 60)
        if not non_stationary and not self.high_vif_factors:
            print("VALIDATION PASSED — All checks OK")
            self.is_validated = True
        else:
            if non_stationary:
                print(f"WARNING: Non-stationary factors: {non_stationary}")
            if self.high_vif_factors:
                print(f"WARNING: High VIF factors (>5): {self.high_vif_factors}")
            self.is_validated = False
            
        print("=" * 60 + "\n")
        return self.is_validated