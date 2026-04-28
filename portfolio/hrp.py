from __future__ import annotations

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from config.settings import FACTOR_NAMES

logger = logging.getLogger(__name__)


class EnhancedHRP:
    """
    Enhanced Hierarchical Risk Parity using factor loading
    distance matrix for clustering.

    Standard HRP (Lopez de Prado 2016) clusters assets by
    price correlation. Enhanced HRP clusters by factor loading
    profiles — the economic risk DNA of each asset.

    Distance metric:
        d(i,j) = sqrt(Σ_k (β_ik - β_jk)²)

    Assets with similar factor loadings cluster together.
    The algorithm then allocates across clusters first,
    then within clusters — ensuring genuine factor
    diversification rather than just asset count diversification.

    This is the key innovation of the paper:
    - MVO: return-driven, concentrates in high-return assets
    - Risk Parity: equalizes asset-level risk, not factor-level
    - Enhanced HRP: explicitly targets factor diversification
      through the clustering step

    The comparison of factor risk decomposition across these
    three methods is the central empirical finding.
    """

    def __init__(
        self,
        covariance:   pd.DataFrame,
        beta_matrix:  pd.DataFrame,
    ) -> None:
        self.sigma   = covariance
        self.betas   = beta_matrix
        self.weights: pd.Series | None = None
        self.clusters: list | None = None

    # ── Distance matrix ───────────────────────────────────────────

    def _factor_distance_matrix(self) -> pd.DataFrame:
        """
        Build distance matrix from standardised factor loading profiles.
        Z-score per factor before Euclidean distance so each factor
        contributes equally regardless of numerical scale.
        Without standardisation, equity premium (range 0.1-1.2) would
        dominate inflation (range 0.01-0.08) purely due to scale.
        """
        from scipy.spatial.distance import pdist, squareform
        assets = list(self.sigma.index)
        B = self.betas.reindex(assets)[FACTOR_NAMES].values.astype(float)
        mu  = B.mean(axis=0)
        sig = B.std(axis=0)
        sig[sig == 0] = 1.0
        B_z = (B - mu) / sig
        dist_mtx = squareform(pdist(B_z, metric="euclidean"))
        return pd.DataFrame(dist_mtx, index=assets, columns=assets)
        return pd.DataFrame(dist, index=assets, columns=assets)

    # ── Clustering ────────────────────────────────────────────────

    def _get_quasi_diagonal(
        self, link: np.ndarray, n: int
    ) -> list[int]:
        """
        Sort assets by hierarchical clustering linkage.
        Produces quasi-diagonal covariance matrix ordering.
        Similar assets placed adjacent — reduces off-diagonal
        covariance for bisection step.
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

        num_items = link[-1, 3]
        while sort_ix.max() >= n:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= n]
            i = df0.index
            j = df0.values - n
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    # ── Recursive bisection ───────────────────────────────────────

    def _get_cluster_var(
        self,
        cov: pd.DataFrame,
        c_items: list,
    ) -> float:
        """
        Compute minimum variance portfolio variance
        for a cluster of assets.
        Uses inverse-variance weighting within cluster.
        """
        cov_slice = cov.loc[c_items, c_items]
        w = self._get_ivp(cov_slice)
        c_var = float(w @ cov_slice.values @ w)
        return c_var

    def _get_ivp(self, cov: pd.DataFrame) -> np.ndarray:
        """Inverse variance portfolio weights."""
        ivp = 1.0 / np.diag(cov.values)
        return ivp / ivp.sum()

    def _get_hrp_weights(
        self,
        cov: pd.DataFrame,
        sort_ix: list,
    ) -> pd.Series:
        """
        Recursive bisection allocation.

        Splits sorted asset list into two halves.
        Allocates between halves proportional to
        inverse of their cluster variance.
        Recurses within each half.
        """
        w = pd.Series(1.0, index=cov.index)
        c_items = [sort_ix]

        while len(c_items) > 0:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in (
                    (0, len(i) // 2),
                    (len(i) // 2, len(i)),
                )
                if len(i) > 1
            ]

            for i in range(0, len(c_items), 2):
                if i + 1 >= len(c_items):
                    break
                c_left  = c_items[i]
                c_right = c_items[i + 1]

                var_left  = self._get_cluster_var(cov, c_left)
                var_right = self._get_cluster_var(cov, c_right)

                alpha = 1 - var_left / (var_left + var_right)
                w[c_left]  *= alpha
                w[c_right] *= (1 - alpha)

        return w

    # ── Main fit ──────────────────────────────────────────────────

    def fit(self) -> pd.Series:
        """
        Run Enhanced HRP.

        Steps:
        1. Build factor loading distance matrix
        2. Hierarchical clustering (Ward linkage)
        3. Sort assets by cluster (quasi-diagonal)
        4. Recursive bisection allocation
        5. Return normalized weights

        Returns portfolio weights as labeled Series.
        """
        assets = list(self.sigma.index)
        n      = len(assets)

        # ── 1. Factor distance matrix ──────────────────────────────
        dist_df = self._factor_distance_matrix()
        logger.info(
            f"Factor distance matrix: {dist_df.shape}"
        )

        # ── 2. Hierarchical clustering ─────────────────────────────
        # Ward linkage minimizes total within-cluster variance
        dist_condensed = squareform(dist_df.values)
        link = linkage(dist_condensed, method="ward")
        self.clusters = link

        # ── 3. Quasi-diagonal sort ─────────────────────────────────
        sort_ix = self._get_quasi_diagonal(link, n)
        sorted_assets = [assets[i] for i in sort_ix]

        logger.info(
            f"Cluster order: {sorted_assets}"
        )

        # ── 4. Recursive bisection ─────────────────────────────────
        cov_sorted = self.sigma.loc[sorted_assets, sorted_assets]
        weights_raw = self._get_hrp_weights(cov_sorted, sorted_assets)

        # ── 5. Normalize ───────────────────────────────────────────
        weights = weights_raw / weights_raw.sum()
        weights.name = "hrp"
        self.weights = weights.reindex(assets)

        logger.info(
            f"Enhanced HRP complete — "
            f"n_active={int((weights > 0.01).sum())}"
        )
        self._log_weights()
        return self.weights

    def _log_weights(self) -> None:
        if self.weights is None:
            return
        print("\n=== Enhanced HRP Weights ===")
        for asset, w in self.weights.sort_values(
            ascending=False
        ).items():
            bar = "█" * int(w * 40)
            print(f"  {asset:<25} {w:>6.1%}  {bar}")
        print()