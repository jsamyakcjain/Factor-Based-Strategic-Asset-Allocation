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
        # ── 1. Safe Alignment ──────────────────────────────────────────────
        # Intersect assets to guarantee no NaNs enter the solver
        valid_assets = [
            a for a in self.sigma.index 
            if a in self.mu.index and pd.notna(self.mu[a])
        ]
        
        if len(valid_assets) < 2:
            raise ValueError("Not enough valid assets to run MVO.")

        assets = valid_assets
        mu = self.mu[assets].values
        
        # Extract and enforce absolute symmetry to survive CVXPY's DCP checks
        Sigma_raw = self.sigma.loc[assets, assets].values
        Sigma = (Sigma_raw + Sigma_raw.T) / 2.0
        n = len(assets)

        # ── 2. Check Feasibility of Constraints ────────────────────────────
        if MIN_WEIGHT * n > 1.0:
            raise ValueError(f"Infeasible constraints: MIN_WEIGHT ({MIN_WEIGHT}) * n_assets ({n}) > 100%.")

        # ── 3. Optimization Setup ──────────────────────────────────────────
        w = cp.Variable(n)

        # Use psd_wrap to prevent micro-float asymmetry from crashing DCP
        ret  = mu @ w
        risk = cp.quad_form(w, cp.psd_wrap(Sigma))
        obj  = cp.Maximize(ret - (self.risk_aversion / 2) * risk)

        constraints = [
            cp.sum(w) == 1,
            w >= 0 ,
            w <= self.max_weight,
        ]

        private_idx = [i for i, a in enumerate(assets) if a in PRIVATE_ASSETS]
        if private_idx:
            constraints.append(cp.sum(w[private_idx]) <= self.max_private)

        # ── 4. Solve ───────────────────────────────────────────────────────
        prob = cp.Problem(obj, constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except cp.error.SolverError as e:
            logger.error(f"Solver crashed: {e}")
            raise

        self.status = prob.status
        
        # ── 5. Status Validation ───────────────────────────────────────────
        if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
            # FIXED: Do not silently return an illegal equal-weight portfolio. Fail loudly.
            err_msg = f"MVO solver failed with status: {prob.status}. Inputs or constraints are infeasible."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Clean microscopic negative weights from float imprecision
        weights_array = np.maximum(w.value, 0.0)
        weights_array[weights_array < 0.001] = 0.0
        weights_array = weights_array / weights_array.sum()

        # Pad dropped assets with zero to maintain system-wide vector shapes
        self.weights = pd.Series(0.0, index=self.sigma.index)
        self.weights[assets] = weights_array

        logger.info(f"MVO complete — status={prob.status} n_active={int((self.weights > 0.01).sum())}")
        self._log_weights()
        
        return self.weights

    def _log_weights(self) -> None:
        if self.weights is None:
            return
        print("\n=== MVO Weights ===")
        for asset, w in self.weights.sort_values(ascending=False).items():
            if w > 0.001:  # Only print meaningful weights
                bar = "█" * int(w * 40)
                print(f"  {asset:<25} {w:>6.1%}  {bar}")
        print()