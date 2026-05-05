from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from config.settings import FACTOR_NAMES

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class EnhancedHRP:
    """
    Enhanced Hierarchical Risk Parity (Lopez de Prado, 2016).

    Enhancement over vanilla HRP: the clustering distance matrix is built from
    factor-residual correlation (Sigma_idio = Sigma - B Sigma_f B') so the
    dendrogram reflects idiosyncratic co-movement rather than shared factor
    exposure.  Recursive bisection then uses the full POET covariance so that
    systematic variance is properly priced into each cluster's risk budget.

    Steps
    -----
    1. Residual covariance  →  factor-adjusted correlation  →  distance matrix
    2. Ward-linkage hierarchical clustering
    3. Quasi-diagonalization  (leaf ordering via scipy ``leaves_list``)
    4. Recursive bisection   (weight ∝ inverse cluster variance)
    """

    def __init__(
        self,
        covariance: pd.DataFrame,
        beta_matrix: pd.DataFrame,
        factor_cov: pd.DataFrame | None = None,
        max_weight: float = 0.20,
    ) -> None:
        self.sigma      = covariance
        self.betas      = beta_matrix
        self.factor_cov = factor_cov
        self.max_weight = max_weight

        self.weights:      pd.Series | None  = None
        self.sorted_items: list[str]         = []
        self.link_matrix:  np.ndarray | None = None

    # ── Correlation / Distance helpers ────────────────────────────

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(np.diag(cov), 1e-14))
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    @staticmethod
    def _corr_to_dist(corr: np.ndarray) -> np.ndarray:
        """Lopez de Prado distance: d(i,j) = sqrt(0.5 * (1 − ρ))."""
        return np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))

    def _factor_adj_corr(self, assets: list[str]) -> np.ndarray:
        """
        Idiosyncratic correlation for clustering.

        Remove the factor-driven covariance so assets group by idiosyncratic
        similarity, not by shared beta exposure.
        """
        Sigma = self.sigma.loc[assets, assets].values.astype(float)

        if self.factor_cov is not None:
            B       = self.betas.loc[assets, FACTOR_NAMES].values.astype(float)
            Sigma_f = self.factor_cov.values.astype(float)
            Sigma_sys  = B @ Sigma_f @ B.T
            Sigma_idio = Sigma - Sigma_sys

            # Preserve positive diagonal  (numerical safety)
            diag_full = np.diag(Sigma)
            np.fill_diagonal(
                Sigma_idio,
                np.maximum(np.diag(Sigma_idio), diag_full * 0.01),
            )

            # Project onto PSD cone
            eigvals, eigvecs = np.linalg.eigh(Sigma_idio)
            eigvals = np.maximum(eigvals, 0.0)
            Sigma_idio = eigvecs @ np.diag(eigvals) @ eigvecs.T

            return self._cov_to_corr(Sigma_idio)

        return self._cov_to_corr(Sigma)

    # ── Hierarchical Clustering ────────────────────────────────────

    def _build_linkage(self, assets: list[str]) -> np.ndarray:
        """Ward linkage on the factor-adjusted distance matrix."""
        corr      = self._factor_adj_corr(assets)
        dist      = self._corr_to_dist(corr)
        condensed = squareform(dist, checks=False)
        return linkage(condensed, method="ward")

    # ── Recursive Bisection ────────────────────────────────────────

    def _ivp(self, sub_assets: list[str]) -> np.ndarray:
        """Inverse-variance portfolio weights for a sub-cluster."""
        var = np.maximum(np.diag(self.sigma.loc[sub_assets, sub_assets].values), 1e-14)
        w   = 1.0 / var
        return w / w.sum()

    def _cluster_var(self, sub_assets: list[str]) -> float:
        """Variance of the IVP portfolio for a sub-cluster."""
        cov_sub = self.sigma.loc[sub_assets, sub_assets].values.astype(float)
        w       = self._ivp(sub_assets)
        return float(w @ cov_sub @ w)

    def _recursive_bisect(self, sorted_assets: list[str]) -> pd.Series:
        """
        Allocate weights by walking up the dendrogram.

        At each bisection the left cluster receives fraction
            α = var_right / (var_left + var_right)
        so the lower-variance cluster captures proportionally more weight.
        """
        weights  = pd.Series(1.0, index=sorted_assets)
        clusters = [list(sorted_assets)]

        while clusters:
            next_clusters: list[list[str]] = []
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                mid   = len(cluster) // 2
                left  = cluster[:mid]
                right = cluster[mid:]

                var_l = self._cluster_var(left)
                var_r = self._cluster_var(right)
                total = var_l + var_r

                # α = fraction going to left; 0.5 fallback if both zero
                alpha = var_r / total if total > 1e-20 else 0.5

                weights[left]  *= alpha
                weights[right] *= (1.0 - alpha)

                if len(left)  > 1: next_clusters.append(left)
                if len(right) > 1: next_clusters.append(right)

            clusters = next_clusters

        return weights

    # ── Weight capping ─────────────────────────────────────────────

    def _apply_cap(self, weights: pd.Series) -> pd.Series:
        """
        Iteratively clip weights to [0, max_weight] and redistribute
        any excess proportionally to uncapped assets.  Converges in
        O(n) iterations; usually done in < 10 passes.
        """
        if self.max_weight >= 1.0:
            return weights / weights.sum()
        w = weights.copy()
        for _ in range(100):
            over = w > self.max_weight
            if not over.any():
                break
            excess       = (w[over] - self.max_weight).sum()
            w[over]      = self.max_weight
            below        = w[~over]
            if below.sum() > 1e-12:
                w[~over] = below + excess * below / below.sum()
        return w / w.sum()

    # ── Main fit ──────────────────────────────────────────────────

    def fit(self) -> pd.Series:
        original_assets = self.sigma.index.tolist()

        valid_assets = [
            a for a in self.sigma.index
            if a in self.betas.index
            and not self.betas.loc[a, FACTOR_NAMES].isna().any()
        ]

        if len(valid_assets) < 2:
            raise ValueError("Not enough valid assets to run Enhanced HRP.")

        self.sigma = self.sigma.loc[valid_assets, valid_assets]

        # Step 1–2: Factor-adjusted clustering
        self.link_matrix = self._build_linkage(valid_assets)

        # Step 3: Quasi-diagonalization  (leaf order from dendrogram)
        leaf_order         = list(leaves_list(self.link_matrix))
        self.sorted_items  = [valid_assets[i] for i in leaf_order]

        logger.info("Enhanced HRP — quasi-diagonal order: %s", self.sorted_items)

        # Step 4: Recursive bisection
        weights = self._recursive_bisect(self.sorted_items)
        weights = weights / weights.sum()

        # Step 5: Cap any single asset at max_weight (default 20%)
        weights = self._apply_cap(weights)
        weights.name = "hrp"

        self.weights = pd.Series(0.0, index=original_assets)
        self.weights.update(weights)

        logger.info(
            "Enhanced HRP complete — n_active=%d",
            int((self.weights > 0.01).sum()),
        )
        self._log_weights()
        return self.weights

    def _log_weights(self) -> None:
        if self.weights is None:
            return
        print("\n=== Enhanced HRP Weights (Lopez de Prado, Factor-Adjusted) ===")
        print(f"  Dendrogram order: {' → '.join(self.sorted_items)}")
        print()
        sorted_w = self.weights[self.sorted_items]
        for asset in self.sorted_items:
            w = float(self.weights.get(asset, 0))
            if w > 0.001:
                bar = "█" * int(w * 50)
                print(f"  {asset:<28} {w:>6.1%}  {bar}")
        print()
