from __future__ import annotations

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config.settings import FACTOR_NAMES

logger = logging.getLogger(__name__)


class FactorRiskDecomposition:
    """
    Decomposes portfolio risk into five systematic factors
    plus idiosyncratic component.

    This is the central analytical result of the paper.

    Method:
        Portfolio factor exposure: β_port = w' · B
        where B is (n_assets × n_factors) beta matrix

        Factor variance contribution:
        σ²_factor_k = (β_port_k)² × σ²_factor_k

        Total systematic variance:
        σ²_systematic = β_port' · Σ_f · β_port

        Total portfolio variance:
        σ²_port = w' · Σ · w

        Idiosyncratic variance:
        σ²_idio = σ²_port - σ²_systematic

        Factor share:
        share_k = σ²_factor_k / σ²_port

    The equity premium share is the key diagnostic.
    A well-diversified portfolio should have equity premium
    share well below 50%. Most institutional portfolios
    have 60-75% — hidden concentration.

    Five portfolios compared:
    1. Equal weight (naive baseline)
    2. 60/40 (institutional benchmark)
    3. MVO (return-driven)
    4. Risk Parity (risk-driven)
    5. Enhanced HRP (factor-driven)
    """

    def __init__(
        self,
        beta_matrix:    pd.DataFrame,
        factor_cov:     pd.DataFrame,
        asset_cov:      pd.DataFrame,
    ) -> None:
        self.betas      = beta_matrix
        self.factor_cov = factor_cov
        self.asset_cov  = asset_cov

    # ── Core decomposition ────────────────────────────────────────

    def decompose(
        self,
        weights: pd.Series,
        label:   str = "portfolio",
    ) -> pd.Series:
        """
        Decompose portfolio risk into factor contributions.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights summing to 1.
        label : str
            Label for the portfolio.

        Returns
        -------
        pd.Series with factor risk shares (sum = 1).
        Index: factor names + 'idiosyncratic'
        """
        # Align to common assets
        assets = list(weights.index)
        w = weights.values
        B = self.betas.reindex(assets)[FACTOR_NAMES].values
        Sigma   = self.asset_cov.loc[assets, assets].values
        Sigma_f = self.factor_cov.values

        # Portfolio factor exposures
        # beta_port = w' · B  →  shape (n_factors,)
        beta_port = w @ B

        # Total portfolio variance
        port_var = float(w @ Sigma @ w)

        # Systematic variance per factor
        # For factor k: contribution = beta_port_k² × Sigma_f[k,k]
        # Full systematic: beta_port' · Sigma_f · beta_port
        systematic_var = float(beta_port @ Sigma_f @ beta_port)

        # Individual factor contributions
        # Marginal contribution of factor k:
        # beta_port_k × (Sigma_f · beta_port)_k
        factor_contributions = beta_port * (Sigma_f @ beta_port)

        # Idiosyncratic variance
        idio_var = max(port_var - systematic_var, 0)

        # Build result series
        result = {}
        for i, factor in enumerate(FACTOR_NAMES):
            result[factor] = factor_contributions[i]
        result["idiosyncratic"] = idio_var

        # Normalize to shares
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        s = pd.Series(result, name=label)

        logger.info(
            f"{label:<20} "
            f"ERP={s['equity_premium']:.1%}  "
            f"TERM={s['term_premium']:.1%}  "
            f"IDIO={s['idiosyncratic']:.1%}"
        )
        return s

    # ── Benchmark portfolios ──────────────────────────────────────

    def equal_weight(self, assets: list[str]) -> pd.Series:
        """Naive equal weight across all assets."""
        n = len(assets)
        return pd.Series(
            np.ones(n) / n,
            index=assets,
            name="equal_weight",
        )

    def sixty_forty(
        self,
        equity_assets: list[str],
        bond_assets:   list[str],
    ) -> pd.Series:
        """
        60/40 benchmark portfolio.
        60% equally split across equity assets.
        40% equally split across bond assets.
        Universal institutional reference point.
        """
        weights = {}
        n_eq = len(equity_assets)
        n_bd = len(bond_assets)

        for a in equity_assets:
            weights[a] = 0.60 / n_eq
        for a in bond_assets:
            weights[a] = 0.40 / n_bd

        return pd.Series(weights, name="sixty_forty")

    # ── Compare all portfolios ────────────────────────────────────

    def compare(
        self,
        portfolios: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Run decomposition for all portfolios.
        Returns DataFrame — rows are portfolios,
        columns are factor risk shares.

        This is the central result table of the paper.
        """
        results = []
        for name, weights in portfolios.items():
            # Align weights to assets in covariance matrix
            aligned = weights.reindex(
                self.asset_cov.index
            ).fillna(0)
            aligned = aligned / aligned.sum()
            s = self.decompose(aligned, label=name)
            results.append(s)

        df = pd.DataFrame(results) * 100  # convert to %
        df.columns = [
            c.replace("_", " ").title()
            for c in df.columns
        ]
        return df.round(1)

    # ── Summary print ─────────────────────────────────────────────

    def print_summary(
        self,
        result_df: pd.DataFrame,
    ) -> None:
        """
        Print the central finding table.
        Highlights equity premium concentration per portfolio.
        """
        print("\n" + "=" * 75)
        print("FACTOR RISK DECOMPOSITION — % of Total Portfolio Risk")
        print("=" * 75)
        print(result_df.to_string())
        print("=" * 75)

        # Highlight equity concentration
        erp_col = "Equity Premium"
        if erp_col in result_df.columns:
            print("\nEquity Premium Concentration:")
            for port, erp in result_df[erp_col].items():
                flag = ""
                if erp > 60:
                    flag = "  ← HIGH concentration"
                elif erp < 40:
                    flag = "  ← WELL diversified"
                print(f"  {port:<20} {erp:.1f}%{flag}")
        print()