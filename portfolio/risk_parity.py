from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config.settings import FACTOR_NAMES

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class RiskParity:
    """
    Equal Risk Contribution (ERC) portfolio.
    """

    def __init__(
        self,
        covariance: pd.DataFrame,
    ) -> None:
        self.sigma   = covariance
        self.weights: pd.Series | None = None

    def _risk_contributions(
        self,
        w: np.ndarray,
        Sigma: np.ndarray,
    ) -> np.ndarray:
        """Compute marginal risk contributions safely."""
        port_var = float(w @ Sigma @ w)
        # FIXED: Prevent negative variance square root crashes
        safe_port_var = max(port_var, 1e-10)
        
        marginal = Sigma @ w
        return w * marginal / np.sqrt(safe_port_var)

    def _objective(
        self,
        w: np.ndarray,
        Sigma: np.ndarray,
    ) -> float:
        """
        Minimize sum of squared differences between
        risk contributions. Scaled for solver gradient stability.
        """
        rc = self._risk_contributions(w, Sigma)
        rc_mean = rc.mean()
        
        # FIXED: Multiply by 1e6 to give SLSQP a workable gradient scale
        return float(np.sum((rc - rc_mean) ** 2)) * 1e6

    def fit(self) -> pd.Series:
        """Solve for equal risk contribution weights."""
        # FIXED: Safe subsetting to align with the rest of our robust pipeline
        original_assets = self.sigma.index.copy()
        
        valid_assets = [
            a for a in self.sigma.index 
            if pd.notna(self.sigma.loc[a, a])
        ]
        
        if len(valid_assets) < 2:
            raise ValueError("Not enough valid assets to run Risk Parity.")
            
        Sigma_df = self.sigma.loc[valid_assets, valid_assets]
        assets = list(Sigma_df.index)
        
        # Enforce exact symmetry for Scipy math
        Sigma_raw = Sigma_df.values.astype(float)
        Sigma = (Sigma_raw + Sigma_raw.T) / 2.0
        n = len(assets)

        # FIXED: Prevent ZeroDivisionError on initial guess
        vols = np.maximum(np.sqrt(np.diag(Sigma)), 1e-10)
        w0   = (1 / vols) / (1 / vols).sum()

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(1e-6, 1.0)] * n

        result = minimize(
            self._objective,
            w0,
            args=(Sigma,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning(f"Risk Parity convergence warning: {result.message}")

        weights_raw = pd.Series(
            np.maximum(result.x, 0),
            index=assets,
        )
        weights_raw = weights_raw / weights_raw.sum()
        
        # FIXED: Pad missing assets to maintain structural consistency
        self.weights = pd.Series(0.0, index=original_assets, name="risk_parity")
        self.weights.update(weights_raw)

        # Verify equal risk contributions
        rc = self._risk_contributions(weights_raw.values, Sigma)
        rc_std = rc.std() / rc.mean() if rc.mean() > 1e-10 else 0
        logger.info(
            f"Risk Parity complete — "
            f"RC coefficient of variation: {rc_std:.4f} "
            f"(0=perfect) n_active={int((weights_raw > 0.01).sum())}"
        )
        self._log_weights()
        return self.weights

    def _log_weights(self) -> None:
        if self.weights is None:
            return
        print("\n=== Risk Parity Weights ===")
        for asset, w in self.weights.sort_values(ascending=False).items():
            if w > 0.001:
                bar = "█" * int(w * 40)
                print(f"  {asset:<25} {w:>6.1%}  {bar}")
        print()


class FactorRiskParity:
    """
    Factor-aware Equal Risk Contribution (ERC) portfolio.

    Uses standard asset-level ERC on the full POET covariance matrix.
    Factor structure is implicitly enforced: highly correlated equity
    assets share covariance and collectively receive less budget than
    an equal-weight allocation would give them.

    Attempting to equalize raw factor-variance contributions directly
    (the naive approach) degenerates because equity premium variance is
    10–100× larger than inflation/liquidity variance, making the 20%-each
    target mathematically unreachable and collapsing the optimizer to
    2 assets.  Standard ERC on POET covariance is the industry-standard
    implementation of factor-aware risk parity (cf. Bridgewater All Weather).
    """

    def __init__(
        self,
        covariance: pd.DataFrame,
        beta_matrix: pd.DataFrame,
        factor_cov: pd.DataFrame,
    ) -> None:
        self.sigma      = covariance
        self.betas      = beta_matrix
        self.factor_cov = factor_cov
        self.weights: pd.Series | None = None

    @staticmethod
    def _risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        port_var = max(float(w @ Sigma @ w), 1e-14)
        return w * (Sigma @ w) / np.sqrt(port_var)

    def _objective(self, w: np.ndarray, Sigma: np.ndarray) -> float:
        rc = self._risk_contributions(w, Sigma)
        return float(np.sum((rc - rc.mean()) ** 2)) * 1e6

    def fit(self) -> pd.Series:
        original_assets = self.sigma.index.copy()

        valid_assets = [
            a for a in self.sigma.index
            if a in self.betas.index
            and not self.betas.loc[a, FACTOR_NAMES].isna().any()
        ]

        if len(valid_assets) < 2:
            raise ValueError("Not enough valid assets for Factor Risk Parity.")

        assets  = valid_assets
        n       = len(assets)
        Sigma_raw = self.sigma.loc[assets, assets].values.astype(float)
        Sigma     = (Sigma_raw + Sigma_raw.T) / 2.0      # enforce symmetry

        vols = np.sqrt(np.maximum(np.diag(Sigma), 1e-14))
        w0   = (1.0 / vols) / (1.0 / vols).sum()

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        bounds      = [(1e-6, 1.0)] * n

        result = minimize(
            self._objective,
            w0,
            args=(Sigma,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning("Factor Risk Parity convergence warning: %s", result.message)

        weights_raw = pd.Series(np.maximum(result.x, 0.0), index=assets)
        weights_raw = weights_raw / weights_raw.sum()

        self.weights = pd.Series(0.0, index=original_assets, name="risk_parity")
        self.weights.update(weights_raw)

        rc = self._risk_contributions(weights_raw.values, Sigma)
        rc_cv = rc.std() / rc.mean() if rc.mean() > 1e-10 else 0.0
        logger.info(
            "Factor Risk Parity (ERC) complete — RC CV=%.4f  n_active=%d",
            rc_cv, int((weights_raw > 0.01).sum()),
        )
        self._log_weights()
        return self.weights

    def _log_weights(self) -> None:
        if self.weights is None:
            return
        print("\n=== Factor Risk Parity Weights ===")
        for asset, w in self.weights.sort_values(ascending=False).items():
            if w > 0.001:
                bar = "█" * int(w * 40)
                print(f"  {asset:<25} {w:>6.1%}  {bar}")