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
    plus idiosyncratic component using Euler risk decomposition.
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
        # FIXED: Safe intersection of assets across all three data structures
        valid_assets = [
            a for a in weights.index
            if a in self.asset_cov.index 
            and a in self.betas.index 
            and not self.betas.loc[a, FACTOR_NAMES].isna().any()
        ]

        if not valid_assets:
            logger.warning(f"No valid assets found for decomposition of {label}.")
            return pd.Series(0.0, index=FACTOR_NAMES + ["idiosyncratic"], name=label)

        # Subset and rigorously re-normalize weights
        w_raw = weights[valid_assets]
        w_sum = w_raw.sum()
        if w_sum == 0:
            return pd.Series(0.0, index=FACTOR_NAMES + ["idiosyncratic"], name=label)
            
        w = (w_raw / w_sum).values
        
        B = self.betas.loc[valid_assets, FACTOR_NAMES].values
        Sigma   = self.asset_cov.loc[valid_assets, valid_assets].values
        Sigma_f = self.factor_cov.values

        # Portfolio factor exposures
        beta_port = w @ B

        # Total portfolio variance
        port_var = float(w @ Sigma @ w)

        # Systematic variance via Euler decomposition
        systematic_var = float(beta_port @ Sigma_f @ beta_port)

        # Marginal contribution of each factor to variance
        factor_contributions = beta_port * (Sigma_f @ beta_port)

        # Idiosyncratic variance is the residual
        idio_var = max(port_var - systematic_var, 0.0)

        # Build result mapping
        result = {
            factor: factor_contributions[i] 
            for i, factor in enumerate(FACTOR_NAMES)
        }
        result["idiosyncratic"] = idio_var

        # Normalize by sum of parts so components always sum to 100%.
        # When systematic_var > port_var (factor model inconsistency with asset cov),
        # idio_var is clamped to 0 and sum(parts) = systematic_var != port_var.
        total = sum(result.values())
        if total > 1e-12:
            result = {k: v / total for k, v in result.items()}
        else:
            result = {k: 0.0 for k in result.keys()}

        s = pd.Series(result, name=label)

        logger.info(
            f"{label:<20} "
            f"ERP={s.get('equity_premium', 0):.1%}  "
            f"TERM={s.get('term_premium', 0):.1%}  "
            f"IDIO={s.get('idiosyncratic', 0):.1%}"
        )
        return s

    # ── Benchmark portfolios ──────────────────────────────────────

    def equal_weight(self, assets: list[str]) -> pd.Series:
        """Naive equal weight across all assets."""
        n = len(assets)
        if n == 0:
            return pd.Series(dtype=float, name="equal_weight")
        return pd.Series(1.0 / n, index=assets, name="equal_weight")

    def sixty_forty(
        self,
        equity_assets: list[str],
        bond_assets:   list[str],
    ) -> pd.Series:
        """60/40 benchmark portfolio."""
        # FIXED: Guard against ZeroDivisionError
        n_eq = len(equity_assets)
        n_bd = len(bond_assets)
        
        if n_eq == 0 or n_bd == 0:
            raise ValueError("60/40 benchmark requires at least 1 equity and 1 bond asset.")

        weights = {a: 0.60 / n_eq for a in equity_assets}
        weights.update({a: 0.40 / n_bd for a in bond_assets})

        return pd.Series(weights, name="sixty_forty")

    # ── Compare all portfolios ────────────────────────────────────

    def compare(self, portfolios: dict[str, pd.Series]) -> pd.DataFrame:
        """Run decomposition for all portfolios."""
        results = []
        for name, weights in portfolios.items():
            s = self.decompose(weights, label=name)
            results.append(s)

        df = pd.DataFrame(results) * 100  # convert to %
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        return df.round(1)

    # ── Summary print ─────────────────────────────────────────────

    def print_summary(self, result_df: pd.DataFrame) -> None:
        """Print the central finding table."""
        print("\n" + "=" * 75)
        print("FACTOR RISK DECOMPOSITION — % of Total Portfolio Risk")
        print("=" * 75)
        print(result_df.to_string())
        print("=" * 75)

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