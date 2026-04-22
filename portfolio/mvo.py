from __future__ import annotations

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cvxpy as cp

from config.settings import (
    MAX_PRIVATE_TOTAL,
    MAX_WEIGHT,
    MIN_WEIGHT,
    PRIVATE_ASSETS,
    RISK_AVERSION,
)

logger = logging.getLogger(__name__)


class MVO:
    """
    Mean-Variance Optimizer using JPM LTCMA expected returns
    and POET covariance matrix.

    Maximizes:   w'μ - (λ/2) w'Σw
    Subject to:  Σw = 1
                 w ≥ 0            (long only)
                 w ≤ max_weight   (concentration limit)
                 Σw_private ≤ max_private  (illiquidity limit)

    Risk aversion λ=3.0 is standard for institutional SAA.
    Long-only constraint reflects realistic pension fund mandate.
    Private markets capped at 30% — typical institutional limit.

    Uses cvxpy for reliable convex optimization.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance:       pd.DataFrame,
        risk_aversion:    float = RISK_AVERSION,
        max_weight:       float = MAX_WEIGHT,
        max_private:      float = MAX_PRIVATE_TOTAL,
    ) -> None:
        self.mu           = expected_returns
        self.sigma        = covariance
        self.risk_aversion = risk_aversion
        self.max_weight   = max_weight
        self.max_private  = max_private
        self.weights:     pd.Series | None = None
        self.status:      str = "not_run"

    def fit(self) -> pd.Series:
        """
        Run mean-variance optimization.

        Returns portfolio weights as labeled Series.
        """
        # Align assets
        assets = list(self.sigma.index)
        mu = self.mu.reindex(assets).values
        Sigma = self.sigma.values
        n = len(assets)

        # Decision variable
        w = cp.Variable(n)

        # Objective: maximize utility = μ'w - (λ/2) w'Σw
        ret  = mu @ w
        risk = cp.quad_form(w, Sigma)
        obj  = cp.Maximize(ret - (self.risk_aversion / 2) * risk)

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= MIN_WEIGHT,
            w <= self.max_weight,
        ]

        # Private markets constraint if applicable
        private_idx = [
            i for i, a in enumerate(assets)
            if a in PRIVATE_ASSETS
        ]
        if private_idx:
            constraints.append(
                cp.sum(w[private_idx]) <= self.max_private
            )

        # Solve
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        self.status = prob.status
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"MVO solver status: {prob.status}")
            return pd.Series(
                np.ones(n) / n,
                index=assets,
                name="mvo"
            )

        weights = pd.Series(
            np.maximum(w.value, 0),
            index=assets,
            name="mvo",
        )
        weights = weights / weights.sum()
        self.weights = weights

        logger.info(
            f"MVO complete — status={prob.status}  "
            f"n_active={int((weights > 0.01).sum())}"
        )
        self._log_weights()
        return weights

    def _log_weights(self) -> None:
        if self.weights is None:
            return
        print("\n=== MVO Weights ===")
        for asset, w in self.weights.sort_values(
            ascending=False
        ).items():
            bar = "█" * int(w * 40)
            print(f"  {asset:<25} {w:>6.1%}  {bar}")
        print()