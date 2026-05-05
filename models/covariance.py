from __future__ import annotations

import warnings
import logging

import numpy as np
import pandas as pd

from config.settings import FACTOR_NAMES

logger = logging.getLogger(__name__)


class POETCovariance:
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
        beta_matrix: pd.DataFrame,
        decay: float = 1.0,
        start_date: str | None = None,
        alphas: pd.Series | None = None,
    ) -> None:
        self.factors = factor_returns.astype(float)
        self.assets = asset_returns.astype(float)
        self.betas = beta_matrix
        self.decay = decay
        self.start_date = start_date
        self.alphas = alphas
        self.sigma: np.ndarray | None = None
        self.sigma_lw: np.ndarray | None = None
        self.factor_cov: np.ndarray | None = None
        self.residuals: pd.DataFrame | None = None
        self._valid_assets: list[str] = []

    def _exp_weights(self, T: int) -> np.ndarray:
        w = np.array([self.decay ** i for i in range(T)])
        w = w[::-1]
        return w / w.sum()

    def _weighted_cov(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        w = self._exp_weights(T)
        mu = np.average(X, weights=w, axis=0)
        X_c = X - mu
        return (X_c.T * w) @ X_c

    def _universal_threshold(self, residuals: np.ndarray, weights: np.ndarray) -> float:
        T, p = residuals.shape
        T_eff = 1.0 / np.sum(weights ** 2) if self.decay < 1.0 else T
        # Adaptive C following Fan & Liao (2013): tau = C * sqrt(log(p)/T).
        # C >= 2 is the theoretical minimum; scale up for fat-tailed residuals.
        std_u = np.sqrt(np.average(residuals ** 2, weights=weights, axis=0).clip(1e-20))
        r_std = residuals / std_u
        excess_kurt = max(
            float(np.mean(np.average(r_std ** 4, weights=weights, axis=0))) - 3.0, 0.0
        )
        C = float(np.clip(2.0 + 0.5 * np.sqrt(excess_kurt), 1.5, 4.0))
        return C * np.sqrt(np.log(p) / T_eff)

    def _soft_threshold(self, matrix: np.ndarray, tau: float) -> np.ndarray:
        result = np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0.0)
        np.fill_diagonal(result, np.diag(matrix))
        return result

    def fit(self) -> "POETCovariance":
        # ── 1. Safe Alignment ──────────────────────────────────────
        valid_assets = [a for a in self.assets.columns if a in self.betas.index]

        combined = pd.concat([
            self.factors[FACTOR_NAMES],
            self.assets[valid_assets]
        ], axis=1)

        # ── 2. Apply start_date filter if provided ─────────────────
        if self.start_date is not None:
            start_dt = pd.Timestamp(self.start_date)
            combined = combined[combined.index >= start_dt]
            logger.info(f"POET: Enforcing start date {self.start_date}")

        # Drop incomplete rows
        combined = combined.dropna()

        if combined.empty:
            raise ValueError("No overlapping dates found across factors and valid assets.")

        # Log date range
        logger.info(f"POET: Using data from {combined.index[0].date()} to {combined.index[-1].date()}")
        logger.info(f"POET: {len(combined)} observations")

        common = combined.index
        F = combined[FACTOR_NAMES].values
        R = combined[valid_assets].values
        T, p = R.shape
        k = F.shape[1]

        # Extract Betas safely matching the valid assets
        B = self.betas.loc[valid_assets, FACTOR_NAMES].values

        logger.info(f"POET: T={T} quarters, p={p} assets, k={k} factors")

        # ── 3. Compute residuals ───────────────────────────────────
        w = self._exp_weights(T)

        if self.alphas is not None:
            # Use proper alphas from OLS for residual construction
            alphas_arr = self.alphas.reindex(valid_assets).fillna(0).values
            residuals = R - alphas_arr - F @ B.T
        else:
            # Fallback: estimate alpha as weighted mean of raw residuals
            raw_residuals = R - F @ B.T
            est_alphas = np.average(raw_residuals, weights=w, axis=0)
            residuals = raw_residuals - est_alphas

        self.residuals = pd.DataFrame(residuals, index=common, columns=valid_assets)

        # ── 4. Factor covariance ───────────────────────────────────
        Sigma_f = self._weighted_cov(F)
        self.factor_cov = Sigma_f

        # ── 5. Systematic covariance ───────────────────────────────
        Sigma_systematic = B @ Sigma_f @ B.T

        # ── 6. Residual covariance ─────────────────────────────────
        Sigma_u_raw = self._weighted_cov(residuals)

        # ── 7. Threshold residual covariance ───────────────────────
        std_u = np.sqrt(np.diag(Sigma_u_raw))
        std_u = np.where(std_u < 1e-10, 1e-10, std_u)
        D_inv = np.diag(1.0 / std_u)
        Corr_u = D_inv @ Sigma_u_raw @ D_inv

        tau = self._universal_threshold(residuals, w)
        Corr_u_thresh = self._soft_threshold(Corr_u, tau)

        D = np.diag(std_u)
        Sigma_u_thresh = D @ Corr_u_thresh @ D

        logger.info(
            f"POET threshold tau={tau:.4f}  "
            f"sparsity={np.mean(Corr_u_thresh == 0):.1%}"
        )

        # ── 8. POET covariance ─────────────────────────────────────
        Sigma_poet = Sigma_systematic + Sigma_u_thresh

        # ── 9. Positive definiteness ───────────────────────────────
        self.sigma = self._ensure_pd(Sigma_poet)

        # Storing valid_assets for downstream labeling
        self._valid_assets = valid_assets

        self._log_diagnostics(Sigma_systematic, Sigma_u_thresh)
        return self

    def _ensure_pd(self, matrix: np.ndarray) -> np.ndarray:
        matrix = (matrix + matrix.T) / 2
        try:
            min_eig = np.linalg.eigvalsh(matrix).min()
        except np.linalg.LinAlgError:
            matrix = matrix + 1e-6 * np.eye(matrix.shape[0])
            min_eig = np.linalg.eigvalsh(matrix).min()

        if min_eig < 1e-8:
            delta = abs(min_eig) + 1e-6
            matrix = matrix + delta * np.eye(matrix.shape[0])
        return matrix

    def _log_diagnostics(self, systematic: np.ndarray, residual: np.ndarray) -> None:
        total_var = np.trace(self.sigma)
        syst_var = np.trace(systematic)
        resid_var = np.trace(residual)
        cond_num = np.linalg.cond(self.sigma)
        min_eig = np.linalg.eigvalsh(self.sigma).min()

        logger.info("POET diagnostics:")
        logger.info(f"  Systematic variance share : {syst_var/total_var:.1%}")
        logger.info(f"  Idiosyncratic variance share: {resid_var/total_var:.1%}")
        logger.info(f"  Condition number : {cond_num:.1f}")
        logger.info(f"  Min eigenvalue   : {min_eig:.6f}")
        logger.info(f"  Matrix is PD     : {min_eig > 0}")

    def fit_ledoit_wolf(self) -> "POETCovariance":
        from sklearn.covariance import LedoitWolf

        combined = pd.concat([
            self.factors[FACTOR_NAMES],
            self.assets[self._valid_assets]
        ], axis=1)

        if self.start_date is not None:
            start_dt = pd.Timestamp(self.start_date)
            combined = combined[combined.index >= start_dt]

        combined = combined.dropna()
        R = combined[self._valid_assets].values

        lw = LedoitWolf()
        lw.fit(R)
        self.sigma_lw = lw.covariance_

        cond_lw = np.linalg.cond(self.sigma_lw)
        min_eig_lw = np.linalg.eigvalsh(self.sigma_lw).min()

        logger.info(f"Ledoit-Wolf: cond={cond_lw:.1f}  min_eig={min_eig_lw:.6f}")
        return self

    def as_dataframe(self) -> pd.DataFrame:
        if self.sigma is None:
            raise ValueError("Must call fit() before accessing covariance dataframe.")

        return pd.DataFrame(
            self.sigma,
            index=self._valid_assets,
            columns=self._valid_assets,
        )

    def correlation_matrix(self) -> pd.DataFrame:
        cov = self.as_dataframe()
        std = np.sqrt(np.diag(cov.values))
        std = np.where(std < 1e-10, 1e-10, std)
        corr = cov.values / np.outer(std, std)

        return pd.DataFrame(
            corr,
            index=cov.index,
            columns=cov.columns,
        )

    def compare_with_lw(self) -> pd.DataFrame:
        if self.sigma_lw is None:
            self.fit_ledoit_wolf()

        metrics = {
            "Condition Number": {
                "POET": np.linalg.cond(self.sigma),
                "Ledoit-Wolf": np.linalg.cond(self.sigma_lw),
            },
            "Min Eigenvalue": {
                "POET": np.linalg.eigvalsh(self.sigma).min(),
                "Ledoit-Wolf": np.linalg.eigvalsh(self.sigma_lw).min(),
            },
            "Max Eigenvalue": {
                "POET": np.linalg.eigvalsh(self.sigma).max(),
                "Ledoit-Wolf": np.linalg.eigvalsh(self.sigma_lw).max(),
            },
            "Trace": {
                "POET": np.trace(self.sigma),
                "Ledoit-Wolf": np.trace(self.sigma_lw),
            },
        }
        return pd.DataFrame(metrics).T.round(6)