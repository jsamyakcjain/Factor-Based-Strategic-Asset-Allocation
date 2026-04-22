from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

from config.settings import FACTOR_NAMES

logger = logging.getLogger(__name__)


class POETCovariance:
    """
    Principal Orthogonal complEment Thresholding (POET).
    Fan, Liao, Mincheva (2013) — Journal of the Royal Statistical Society.

    Decomposes covariance into:
        Σ = B · Σ_f · B' + Σ_u (thresholded)

    Where:
        B    = factor loading matrix (n_assets x n_factors)
        Σ_f  = factor covariance matrix (n_factors x n_factors)
        Σ_u  = thresholded residual covariance (sparse)

    Why POET over Ledoit-Wolf:
    - Designed specifically for factor-structured data
    - Thresholds residual correlations to remove noise
    - Produces sparser, more interpretable covariance
    - Better condition number for portfolio optimization
    - Used in MSCI Barra methodology

    WLS decay=0.94 option:
    - Exponentially weights recent observations more
    - RiskMetrics standard for institutional covariance
    - More relevant for current portfolio construction
    - Used ONLY for MVO inputs, not for regression betas
    """

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        asset_returns:  pd.DataFrame,
        beta_matrix:    pd.DataFrame,
        decay:          float = 1.0,
    ) -> None:
        self.factors     = factor_returns.astype(float)
        self.assets      = asset_returns.astype(float)
        self.betas       = beta_matrix
        self.decay       = decay
        self.sigma:      np.ndarray | None = None
        self.sigma_lw:   np.ndarray | None = None
        self.residuals:  pd.DataFrame | None = None

    # ── Weights ───────────────────────────────────────────────────

    def _exp_weights(self, T: int) -> np.ndarray:
        """
        Exponentially decaying weights.
        w_t = decay^(T-t), normalized to sum to 1.
        decay=1.0 gives equal weights (standard OLS).
        decay=0.94 gives RiskMetrics weighting.
        """
        w = np.array([self.decay ** i for i in range(T)])
        w = w[::-1]
        return w / w.sum()

    # ── Weighted covariance ───────────────────────────────────────

    def _weighted_cov(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted covariance matrix."""
        T = X.shape[0]
        w = self._exp_weights(T)
        mu = np.average(X, weights=w, axis=0)
        X_c = X - mu
        return (X_c.T * w) @ X_c

    # ── Threshold selection ───────────────────────────────────────

    def _universal_threshold(
        self,
        residuals: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Universal threshold from Fan et al. (2013).
        tau = C * sqrt(log(p) / T)
        where C is chosen by cross-validation.
        We use C=0.5 following the paper's recommendation.
        """
        T, p = residuals.shape
        return 0.5 * np.sqrt(np.log(p) / T)

    def _soft_threshold(
        self,
        matrix: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        """
        Soft thresholding of off-diagonal elements.
        Shrinks small correlations to zero.
        Preserves diagonal (idiosyncratic variances).
        """
        result = matrix.copy()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j:
                    if abs(matrix[i, j]) <= tau:
                        result[i, j] = 0.0
                    elif matrix[i, j] > tau:
                        result[i, j] = matrix[i, j] - tau
                    else:
                        result[i, j] = matrix[i, j] + tau
        return result

    # ── Main estimation ───────────────────────────────────────────

    def fit(self) -> POETCovariance:
        """
        Estimate POET covariance matrix.

        Steps:
        1. Align factor and asset returns
        2. Compute residuals from factor model
        3. Estimate factor covariance Σ_f
        4. Compute systematic covariance B·Σ_f·B'
        5. Estimate residual covariance Σ_u
        6. Apply soft thresholding to Σ_u
        7. Combine: Σ_POET = B·Σ_f·B' + Σ_u_thresholded
        8. Ensure positive definiteness

        Returns self for method chaining.
        """
        # ── 1. Align ───────────────────────────────────────────────
        common = self.factors.index.intersection(self.assets.index)
        F = self.factors.loc[common].values
        R = self.assets.loc[common].values
        T, p = R.shape
        k = F.shape[1]

        # Align beta matrix to asset order
        assets_in_order = list(self.assets.columns)
        B = self.betas.loc[assets_in_order, FACTOR_NAMES].values
        # B is (p x k)

        logger.info(
            f"POET: T={T} quarters, p={p} assets, k={k} factors"
        )

        # ── 2. Compute residuals ───────────────────────────────────
        # residuals = R - F·B'  (T x p)
        residuals = R - F @ B.T
        self.residuals = pd.DataFrame(
            residuals,
            index=self.assets.loc[common].index,
            columns=self.assets.columns,
        )

        # ── 3. Factor covariance ───────────────────────────────────
        w = self._exp_weights(T)
        Sigma_f = self._weighted_cov(F)   # (k x k)

        # ── 4. Systematic covariance ───────────────────────────────
        Sigma_systematic = B @ Sigma_f @ B.T   # (p x p)

        # ── 5. Residual covariance ─────────────────────────────────
        Sigma_u_raw = self._weighted_cov(residuals)   # (p x p)

        # ── 6. Threshold residual covariance ──────────────────────
        # Convert to correlation for thresholding
        std_u = np.sqrt(np.diag(Sigma_u_raw))
        std_u = np.where(std_u < 1e-10, 1e-10, std_u)
        D_inv = np.diag(1.0 / std_u)
        Corr_u = D_inv @ Sigma_u_raw @ D_inv

        # Apply threshold
        tau = self._universal_threshold(residuals, w)
        Corr_u_thresh = self._soft_threshold(Corr_u, tau)

        # Convert back to covariance
        D = np.diag(std_u)
        Sigma_u_thresh = D @ Corr_u_thresh @ D

        logger.info(
            f"POET threshold tau={tau:.4f}  "
            f"sparsity={np.mean(Corr_u_thresh == 0):.1%}"
        )

        # ── 7. POET covariance ─────────────────────────────────────
        Sigma_poet = Sigma_systematic + Sigma_u_thresh

        # ── 8. Positive definiteness ──────────────────────────────
        Sigma_poet = self._ensure_pd(Sigma_poet)

        self.sigma = Sigma_poet

        # Diagnostics
        self._log_diagnostics(Sigma_systematic, Sigma_u_thresh)

        return self

    def _ensure_pd(self, matrix: np.ndarray) -> np.ndarray:
        """
        Ensure matrix is positive definite by adding
        small diagonal perturbation if needed.
        """
        min_eig = np.linalg.eigvalsh(matrix).min()
        if min_eig < 1e-8:
            delta = abs(min_eig) + 1e-6
            matrix = matrix + delta * np.eye(matrix.shape[0])
            logger.info(
                f"Added {delta:.2e} to diagonal for PD"
            )
        return matrix

    def _log_diagnostics(
        self,
        systematic: np.ndarray,
        residual:   np.ndarray,
    ) -> None:
        """Log diagnostics for the POET decomposition."""
        p = self.sigma.shape[0]
        total_var  = np.trace(self.sigma)
        syst_var   = np.trace(systematic)
        resid_var  = np.trace(residual)
        cond_num   = np.linalg.cond(self.sigma)
        min_eig    = np.linalg.eigvalsh(self.sigma).min()

        logger.info(f"POET diagnostics:")
        logger.info(
            f"  Systematic variance share : "
            f"{syst_var/total_var:.1%}"
        )
        logger.info(
            f"  Idiosyncratic variance share: "
            f"{resid_var/total_var:.1%}"
        )
        logger.info(f"  Condition number : {cond_num:.1f}")
        logger.info(f"  Min eigenvalue   : {min_eig:.6f}")
        logger.info(
            f"  Matrix is PD     : {min_eig > 0}"
        )

    # ── Ledoit-Wolf comparison ────────────────────────────────────

    def fit_ledoit_wolf(self) -> POETCovariance:
        """
        Ledoit-Wolf shrinkage estimator for comparison.
        Standard alternative to POET.
        Shows POET is more stable for factor-structured data.
        """
        from sklearn.covariance import LedoitWolf

        common = self.factors.index.intersection(self.assets.index)
        R = self.assets.loc[common].values.astype(float)

        lw = LedoitWolf()
        lw.fit(R)
        self.sigma_lw = lw.covariance_

        cond_lw   = np.linalg.cond(self.sigma_lw)
        min_eig_lw = np.linalg.eigvalsh(self.sigma_lw).min()

        logger.info(
            f"Ledoit-Wolf: cond={cond_lw:.1f}  "
            f"min_eig={min_eig_lw:.6f}"
        )
        return self

    # ── Output helpers ────────────────────────────────────────────

    def as_dataframe(self, ew: bool = False) -> pd.DataFrame:
        """Return POET covariance as labeled DataFrame."""
        mat = self.sigma
        return pd.DataFrame(
            mat,
            index=self.assets.columns,
            columns=self.assets.columns,
        )

    def correlation_matrix(self) -> pd.DataFrame:
        """Return correlation matrix derived from POET."""
        cov = self.as_dataframe()
        std = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(std, std)
        return pd.DataFrame(
            corr,
            index=cov.index,
            columns=cov.columns,
        )

    def compare_with_lw(self) -> pd.DataFrame:
        """
        Compare POET vs Ledoit-Wolf diagnostics.
        Shows POET produces better conditioned matrix
        for factor-structured asset returns.
        """
        if self.sigma_lw is None:
            self.fit_ledoit_wolf()

        metrics = {
            "Condition Number": {
                "POET":        np.linalg.cond(self.sigma),
                "Ledoit-Wolf": np.linalg.cond(self.sigma_lw),
            },
            "Min Eigenvalue": {
                "POET":        np.linalg.eigvalsh(self.sigma).min(),
                "Ledoit-Wolf": np.linalg.eigvalsh(self.sigma_lw).min(),
            },
            "Max Eigenvalue": {
                "POET":        np.linalg.eigvalsh(self.sigma).max(),
                "Ledoit-Wolf": np.linalg.eigvalsh(self.sigma_lw).max(),
            },
            "Trace": {
                "POET":        np.trace(self.sigma),
                "Ledoit-Wolf": np.trace(self.sigma_lw),
            },
        }
        return pd.DataFrame(metrics).T.round(6)