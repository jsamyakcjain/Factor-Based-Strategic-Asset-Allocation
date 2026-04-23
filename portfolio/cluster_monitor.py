from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

from config.settings import FACTOR_NAMES

logger = logging.getLogger(__name__)


class ClusterMonitor:
    """
    Stochastic Rebalancing Monitor for Enhanced HRP portfolios.

    Detects when factor distance between assets has contracted
    sufficiently to indicate cluster structure has broken down.

    Two key design decisions:

    1. Trailing 8-quarter baseline (not static)
       Distinguishes acute shocks from slow structural shifts.
       A static baseline generates perma-signals during regime
       changes. The trailing baseline adapts, ensuring only
       genuine shocks trigger alerts.

    2. Two-quarter persistence filter
       Rolling betas are noisy. A single-quarter threshold
       crossing does not warrant rebalancing $500M+.
       Signal must persist for 2 consecutive quarters
       before escalating to REBALANCE.

    Signal hierarchy:
        HOLD      — P(rebalance) < 0.50
        MONITOR   — P(rebalance) ≥ 0.50 for 1+ quarters
        REBALANCE — P(rebalance) ≥ 0.85 for 2+ consecutive quarters

    Economic interpretation:
        When assets in different clusters converge in factor space
        (e.g., hedge funds loading on equity premium like large cap),
        the diversification benefit of the current allocation is
        compromised. The monitor detects this convergence and
        signals when rebalancing is warranted.
    """

    def __init__(
        self,
        rolling_betas:       dict[str, pd.DataFrame],
        cluster_assignments: dict[str, list[str]],
        baseline_window:     int   = 8,
        monitor_threshold:   float = 0.50,
        rebalance_threshold: float = 0.85,
        persistence:         int   = 2,
    ) -> None:
        """
        Parameters
        ----------
        rolling_betas : dict
            Output of RollingFactorModel.rolling_betas.
            Keys are asset names, values are DataFrames
            with columns = FACTOR_NAMES, index = dates.

        cluster_assignments : dict
            Keys are cluster labels, values are lists of assets.
            Derived from Enhanced HRP clustering step.
            Example: {'growth': ['us_large_cap', 'em_equity'],
                      'defensive': ['long_treasury', 'tips']}

        baseline_window : int
            Number of quarters for trailing baseline.
            Default 8 = 2 years.

        monitor_threshold : float
            Probability threshold for MONITOR signal.

        rebalance_threshold : float
            Probability threshold for REBALANCE signal.

        persistence : int
            Consecutive quarters above rebalance_threshold
            required before REBALANCE signal fires.
        """
        self.rolling_betas       = rolling_betas
        self.clusters            = cluster_assignments
        self.baseline_window     = baseline_window
        self.monitor_threshold   = monitor_threshold
        self.rebalance_threshold = rebalance_threshold
        self.persistence         = persistence

        # Results stored after compute()
        self.pairwise_distances: pd.DataFrame | None = None
        self.cluster_stability:  pd.DataFrame | None = None
        self.probabilities:      pd.DataFrame | None = None
        self.signals:            pd.DataFrame | None = None

    # ── Step 1: Pairwise factor distances over time ───────────────

    def compute_rolling_distances(self) -> pd.DataFrame:
        """
        Compute pairwise factor loading distances at each quarter.

        d(i,j,t) = √Σ_k (β_ik(t) - β_jk(t))²

        Uses rolling betas so distances evolve as factor
        exposures change over time.

        Returns
        -------
        pd.DataFrame with MultiIndex columns (asset_i, asset_j)
        and DatetimeIndex rows.
        """
        # Get common dates across all assets
        all_dates = None
        for asset, df in self.rolling_betas.items():
            dates = df.dropna().index
            if all_dates is None:
                all_dates = dates
            else:
                all_dates = all_dates.intersection(dates)

        if all_dates is None or len(all_dates) == 0:
            logger.warning("No common dates in rolling betas")
            return pd.DataFrame()

        assets = list(self.rolling_betas.keys())
        n = len(assets)
        records = []

        for date in all_dates:
            row = {"date": date}
            for i in range(n):
                for j in range(i + 1, n):
                    ai = assets[i]
                    aj = assets[j]
                    try:
                        bi = self.rolling_betas[ai].loc[date, FACTOR_NAMES].values.astype(float)
                        bj = self.rolling_betas[aj].loc[date, FACTOR_NAMES].values.astype(float)
                        if np.any(np.isnan(bi)) or np.any(np.isnan(bj)):
                            row[f"{ai}|{aj}"] = np.nan
                        else:
                            row[f"{ai}|{aj}"] = float(
                                np.sqrt(np.sum((bi - bj) ** 2))
                            )
                    except Exception:
                        row[f"{ai}|{aj}"] = np.nan
            records.append(row)

        df = pd.DataFrame(records).set_index("date")
        self.pairwise_distances = df

        logger.info(
            f"Rolling distances: {len(df)} quarters x "
            f"{len(df.columns)} pairs"
        )
        return df

    # ── Step 2: Within-cluster stability over time ────────────────

    def compute_cluster_stability(self) -> pd.DataFrame:
        """
        Compute average within-cluster factor distance over time.

        stability(cluster, t) = mean pairwise distance
                                 between assets in cluster at t

        Low stability = assets diverging in factor space
        High stability = assets converging (cluster coherent)

        A rising within-cluster distance means assets that
        were similar are becoming more different — the cluster
        is breaking apart. A falling between-cluster distance
        means assets in different clusters are converging —
        the diversification benefit is eroding.
        """
        if self.pairwise_distances is None:
            self.compute_rolling_distances()

        stability_records = []

        for date in self.pairwise_distances.index:
            row = {"date": date}
            for cluster_name, members in self.clusters.items():
                if len(members) < 2:
                    row[cluster_name] = np.nan
                    continue

                pair_distances = []
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        ai = members[i]
                        aj = members[j]
                        key = f"{ai}|{aj}"
                        rev_key = f"{aj}|{ai}"
                        val = self.pairwise_distances.loc[
                            date, key
                        ] if key in self.pairwise_distances.columns \
                            else self.pairwise_distances.loc[
                                date, rev_key
                            ] if rev_key in self.pairwise_distances.columns \
                            else np.nan
                        if not np.isnan(val):
                            pair_distances.append(val)

                row[cluster_name] = (
                    np.mean(pair_distances)
                    if pair_distances else np.nan
                )
            stability_records.append(row)

        self.cluster_stability = pd.DataFrame(
            stability_records
        ).set_index("date")

        logger.info(
            f"Cluster stability: {self.cluster_stability.shape}"
        )
        return self.cluster_stability

    # ── Step 3: Rebalancing probability ──────────────────────────

    def compute_rebalancing_probability(self) -> pd.DataFrame:
        """
        Compute rebalancing probability for each cluster.

        Uses trailing 8-quarter baseline to distinguish
        acute shocks from structural regime shifts.

        P(rebalance, t) = Φ((d_baseline(t) - d_current(t)) / σ(t))

        Where:
            d_baseline(t) = mean of d over past 8 quarters
            σ(t)          = std of d over past 8 quarters
            Φ             = standard normal CDF

        Positive shock = current distance BELOW baseline
        = assets converging = diversification eroding
        = higher rebalancing probability
        """
        if self.cluster_stability is None:
            self.compute_cluster_stability()

        prob_records = []

        for i, date in enumerate(self.cluster_stability.index):
            row = {"date": date}

            for cluster in self.cluster_stability.columns:
                series = self.cluster_stability[cluster].dropna()

                # Need at least baseline_window + 1 observations
                loc = series.index.get_loc(date) \
                    if date in series.index else -1

                if loc < self.baseline_window:
                    row[cluster] = 0.0
                    continue

                # Trailing baseline — excludes current observation
                baseline_vals = series.iloc[
                    loc - self.baseline_window: loc
                ]
                d_baseline = baseline_vals.mean()
                d_sigma    = baseline_vals.std()
                d_current  = series.iloc[loc]

                if d_sigma < 1e-8:
                    row[cluster] = 0.0
                    continue

                # Positive z = current distance BELOW baseline
                # = assets converging = higher rebalance probability
                z_score = (d_baseline - d_current) / d_sigma
                prob    = float(norm.cdf(z_score))
                row[cluster] = round(prob, 4)

            prob_records.append(row)

        self.probabilities = pd.DataFrame(
            prob_records
        ).set_index("date")

        logger.info(
            f"Rebalancing probabilities computed: "
            f"{self.probabilities.shape}"
        )
        return self.probabilities

    # ── Step 4: Persistence filter ────────────────────────────────

    def _apply_persistence_filter(
        self,
        prob_series: pd.Series,
    ) -> pd.Series:
        """
        Apply two-quarter persistence filter.

        Prevents signal flicker from noisy rolling betas.
        REBALANCE only fires if probability exceeds threshold
        for self.persistence consecutive quarters.
        MONITOR fires immediately when probability ≥ 0.50.

        This is the key practical improvement over naive
        threshold crossing — prevents trading $500M+ on
        a single noisy observation.
        """
        n = len(prob_series)
        signals = pd.Series("HOLD", index=prob_series.index)

        for i in range(n):
            p = prob_series.iloc[i]

            # Check persistence window for REBALANCE
            if i >= self.persistence - 1:
                window = prob_series.iloc[
                    i - self.persistence + 1: i + 1
                ]
                if (window >= self.rebalance_threshold).all():
                    signals.iloc[i] = "REBALANCE"
                    continue

            # MONITOR — immediate, no persistence needed
            if p >= self.monitor_threshold:
                signals.iloc[i] = "MONITOR"

        return signals

    # ── Step 5: Full signal history ───────────────────────────────

    def compute_signals(self) -> pd.DataFrame:
        """
        Compute rebalancing signals for all clusters.

        Returns DataFrame with HOLD/MONITOR/REBALANCE
        for each cluster at each quarter.
        """
        if self.probabilities is None:
            self.compute_rebalancing_probability()

        signal_df = pd.DataFrame(index=self.probabilities.index)

        for cluster in self.probabilities.columns:
            signal_df[cluster] = self._apply_persistence_filter(
                self.probabilities[cluster]
            )

        self.signals = signal_df

        # Log rebalancing events
        for cluster in signal_df.columns:
            rebalance_dates = signal_df.index[
                signal_df[cluster] == "REBALANCE"
            ]
            if len(rebalance_dates) > 0:
                logger.info(
                    f"Cluster '{cluster}': "
                    f"{len(rebalance_dates)} REBALANCE signals — "
                    f"first: {rebalance_dates[0].date()}, "
                    f"last: {rebalance_dates[-1].date()}"
                )

        return signal_df

    # ── Full pipeline ─────────────────────────────────────────────

    def compute(self) -> ClusterMonitor:
        """
        Run full monitoring pipeline.
        1. Rolling pairwise distances
        2. Within-cluster stability
        3. Rebalancing probabilities
        4. Persistence-filtered signals
        """
        self.compute_rolling_distances()
        self.compute_cluster_stability()
        self.compute_rebalancing_probability()
        self.compute_signals()
        logger.info("ClusterMonitor pipeline complete.")
        return self

    # ── Current status ────────────────────────────────────────────

    def current_status(self) -> dict:
        """
        Return current rebalancing status for each cluster.
        Used by dashboard for real-time display.
        """
        if self.signals is None:
            self.compute()

        latest = self.signals.iloc[-1]
        probs  = self.probabilities.iloc[-1]

        status = {}
        for cluster in latest.index:
            status[cluster] = {
                "signal":      latest[cluster],
                "probability": float(probs[cluster]),
                "date":        str(self.signals.index[-1].date()),
            }
        return status

    # ── Cluster assignment from HRP ───────────────────────────────

    @staticmethod
    def clusters_from_hrp(
        hrp_cluster_order: list[str],
        n_clusters:        int = 3,
    ) -> dict[str, list[str]]:
        """
        Convert HRP cluster order to named cluster assignments.

        Splits the sorted asset list into n_clusters groups
        based on the HRP dendrogram structure.

        Economic naming:
            First third  → Growth (equity-like)
            Middle third → Diversifier (mixed)
            Last third   → Defensive (bond-like)
        """
        n = len(hrp_cluster_order)
        size = n // n_clusters
        names = ["growth", "diversifier", "defensive"]

        clusters = {}
        for i in range(n_clusters):
            start = i * size
            end   = (i + 1) * size if i < n_clusters - 1 else n
            clusters[names[i]] = hrp_cluster_order[start:end]

        return clusters