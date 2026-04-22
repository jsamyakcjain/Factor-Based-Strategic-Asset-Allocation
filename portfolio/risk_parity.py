from __future__ import annotations

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class RiskParity:
    """
    Equal Risk Contribution (ERC) portfolio.

    Finds weights such that each asset contributes
    equally to total portfolio volatility:

        w_i × (Σw)_i / σ_p = σ_p / n   for all i

    No expected return inputs required.
    Uses only the covariance matrix.

    Why Risk Parity:
    - Equal risk contribution does not mean equal weights
    - Low-volatility assets (bonds) get higher weights
    - Reduces equity concentration vs equal weight
    - But does not achieve factor-level diversification
      (shown by comparing with Enhanced HRP)

    Optimization: minimize Σ_i Σ_j (RC_i - RC_j)²
    where RC_i = w_i × (Σw)_i is the risk contribution of asset i.
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
        """Compute marginal risk contributions."""
        port_var = w @ Sigma @ w
        marginal = Sigma @ w
        return w * marginal / np.sqrt(port_var)

    def _objective(
        self,
        w: np.ndarray,
        Sigma: np.ndarray,
    ) -> float:
        """
        Minimize sum of squared differences between
        risk contributions. Zero when all RC are equal.
        """
        rc = self._risk_contributions(w, Sigma)
        rc_mean = rc.mean()
        return float(np.sum((rc - rc_mean) ** 2))

    def fit(self) -> pd.Series:
        """
        Solve for equal risk contribution weights.

        Returns portfolio weights as labeled Series.
        """
        assets = list(self.sigma.index)
        Sigma  = self.sigma.values.astype(float)
        n      = len(assets)

        # Initial guess: inverse volatility weights
        vols = np.sqrt(np.diag(Sigma))
        w0   = (1 / vols) / (1 / vols).sum()

        # Constraints and bounds
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
            logger.warning(
                f"Risk Parity convergence warning: {result.message}"
            )

        weights = pd.Series(
            np.maximum(result.x, 0),
            index=assets,
            name="risk_parity",
        )
        weights = weights / weights.sum()
        self.weights = weights

        # Verify equal risk contributions
        rc = self._risk_contributions(weights.values, Sigma)
        rc_std = rc.std() / rc.mean()
        logger.info(
            f"Risk Parity complete — "
            f"RC coefficient of variation: {rc_std:.4f} "
            f"(0=perfect)"
        )
        self._log_weights()
        return weights

    def _log_weights(self) -> None:
        if self.weights is None:
            return
        print("\n=== Risk Parity Weights ===")
        for asset, w in self.weights.sort_values(
            ascending=False
        ).items():
            bar = "█" * int(w * 40)
            print(f"  {asset:<25} {w:>6.1%}  {bar}")
        print()