from __future__ import annotations

import logging

import pandas as pd
from data.professor_loader import ProfessorLoader
from config.settings import (
    DATA_CACHE_DIR,
    PUBLIC_ASSETS,
    START_DATE_TIER1,
    START_DATE_TIER2,
    FACTOR_NAMES
)
from data.ca_loader import CALoader
from data.fred_loader import FREDLoader
from data.market_loader import MarketLoader
from data.wrds_loader import WRDSLoader

logger = logging.getLogger(__name__)


class DataManager:
    """
    Single orchestration layer for all data.

    Builds two analysis-ready matrices at quarterly frequency:

    Tier 1 — Public assets only, 1980-2024
        factor_returns_t1 : (T1, 5)  — five factor proxies
        asset_returns_t1  : (T1, 9)  — nine public assets

    Tier 2 — Full universe, 2004-2024
        factor_returns_t2 : (T2, 5)  — five factor proxies
        asset_returns_t2  : (T2, 13) — nine public + four private

    All returns quarterly, decimal.
    Private markets Geltner-unsmoothed before joining.

    Usage
    -----
    dm = DataManager()
    dm.build()          # first run — hits APIs
    dm.load_cached()    # subsequent runs — loads from parquet
    """

    def __init__(self, use_cache: bool = True) -> None:
        self._wrds   = WRDSLoader(use_cache=use_cache)
        self._fred   = FREDLoader(use_cache=use_cache)
        self._market = MarketLoader(use_cache=use_cache)
        self._ca     = CALoader(use_cache=use_cache)

        # Tier 1 — public markets only
        self.factor_returns_t1: pd.DataFrame | None = None
        self.asset_returns_t1:  pd.DataFrame | None = None

        # Tier 2 — full universe including private markets
        self.factor_returns_t2: pd.DataFrame | None = None
        self.asset_returns_t2:  pd.DataFrame | None = None
        self._professor = ProfessorLoader(use_cache=use_cache)
        self.asset_returns_t1_complete:  pd.DataFrame | None = None
        # Shared
        self.recession: pd.Series | None = None
        self.rf:        pd.Series | None = None
        self.credit_liquidity:  pd.Series | None = None

    # ── Factor construction ────────────────────────────────────────

    def _build_equity_premium(
        self, ff: pd.DataFrame
    ) -> pd.Series:
        """
        Equity premium = FF Mkt-RF.
        Authoritative academic series for market excess return.
        Already in decimal from WRDSLoader.
        """
        return ff["mktrf"].rename("equity_premium")

    def _build_term_premium(
        self, treasury: pd.DataFrame
    ) -> pd.Series:
        """
        Term premium = 20Y Treasury return - 30D T-bill return.
        Realized compensation for bearing duration risk.
        Using 20Y (not 10Y) to match long_treasury asset proxy.
        """
        return (
            treasury["treasury_20y"] - treasury["tbill_30d"]
        ).rename("term_premium")

    def _build_credit_spread(
        self, credit: pd.DataFrame
    ) -> pd.Series:
        """
        Credit spread = negative monthly change in HY OAS.
        Negated: spread tightening = positive = good for credit.
        Consistent direction with other factors (positive = good).
        """
        return (-credit["hy_oas"].diff()).rename("credit_spread")

    def _build_inflation(
        self,
        tips: pd.Series,
        cpi: pd.Series,
    ) -> pd.Series:
        """
        Inflation = monthly change in TIPS breakeven.
        Captures market-implied inflation expectation changes.
        Backfilled with CPI monthly change pre-2003 when TIPS
        market was illiquid and breakeven data unreliable.
        """
        infl = tips.diff().rename("inflation")
        backfill_end = "2003-03-31"
        infl.loc[:backfill_end] = cpi.loc[:backfill_end]
        logger.info(
            "Inflation: TIPS breakeven from 2003-04, "
            "CPI backfill pre-2003"
        )
        return infl
    
    def _build_liquidity(
        self,
        ps_liq: pd.Series,
    ) -> pd.Series:
        """
        Liquidity factor = Pastor-Stambaugh (2003) innovation.
        Monthly unexpected change in equity market liquidity.
        Positive = liquidity improved unexpectedly.
        Stationary by construction. Covers 1980-2024.
        Source: ff.liq_ps on WRDS.
        """
        return ps_liq.rename("liquidity")
    
    def _build_credit_liquidity(
        self, credit: pd.DataFrame
    ) -> pd.Series:
        """
        Credit market liquidity = negative change in BAA-GS10 spread.
        More relevant for fixed income assets than PS innovation.
        Wide spread = tight credit liquidity conditions.
        """
        return (-credit["hy_oas"].diff()).rename("credit_liquidity")

    # ── Asset construction ─────────────────────────────────────────

    def _build_public_assets(
        self,
        crsp_market: pd.DataFrame,
        etf_returns: pd.DataFrame,
        ) -> pd.DataFrame:
        """
    Build 9-asset public market return matrix.

    Asset          Source
    ─────────────────────────────────────────────
    us_large_cap   CRSP vwretd
    us_mid_cap     iShares IJH ETF
    us_small_cap   iShares IWM ETF
    em_equity      iShares EEM ETF
    long_treasury  iShares TLT ETF
    tips           iShares TIP ETF
    ig_credit      iShares LQD ETF
    reits          Vanguard VNQ ETF
    commodities    ^SPGSCI spliced with GSG ETF
    """
        assets = {}

    # CRSP for us_large_cap — cleaner than SPY, goes to 1926
        assets["us_large_cap"] = crsp_market["vwretd"].rename(
        "us_large_cap"
    )

        for asset in PUBLIC_ASSETS:
            if asset == "us_large_cap":
                continue
            if asset in etf_returns.columns:
                assets[asset] = etf_returns[asset].rename(asset)

        return pd.DataFrame(assets)

    # ── Monthly to quarterly ───────────────────────────────────────

    @staticmethod
    def _to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compound monthly returns to quarterly.
        Returns NaN for quarters with no data.
        """
        def compound(x):
            valid = x.dropna()
            if len(valid) == 0:
                return float("nan")
            return (1 + valid).prod() - 1
        return df.resample("QE").apply(compound)

    @staticmethod
    def _to_quarterly_series(s: pd.Series) -> pd.Series:
        def compound(x):
            valid = x.dropna()
            if len(valid) == 0:
                return float("nan")
            return (1 + valid).prod() - 1
        return s.resample("QE").apply(compound)

    # ── Main build ─────────────────────────────────────────────────

    def build(self) -> DataManager:
        """
        Fetch all data, construct factors and assets,
        convert to quarterly, align indices.

        Returns self for method chaining:
            dm = DataManager().build()
        """
        logger.info("=" * 60)
        logger.info("Building data matrices...")
        logger.info("=" * 60)

        # ── 1. Fetch raw data ──────────────────────────────────────
        ff           = self._wrds.get_ff_factors()
        treasury     = self._wrds.get_treasury_returns()
        crsp_market  = self._wrds.get_market_returns()
        etf_returns  = self._market.get_etf_returns()
        credit       = self._fred.get_credit_spreads()
        tips         = self._fred.get_tips_breakeven()
        cpi          = self._fred.get_cpi()
        recession    = self._fred.get_recession_indicator()
        rf           = self._fred.get_risk_free_rate()
        ps_liq       = self._wrds.get_ps_liquidity()
        self.credit_liquidity = self._to_quarterly_series(
            (-credit["hy_oas"].diff()).rename("credit_liquidity")
        )
        # Credit liquidity for fixed income assets
        # Negative change in BAA-GS10 spread
        # Different from credit_spread factor which uses compounded change


        # ── 2. Build monthly factor series ────────────────────────

        factors_monthly = pd.concat([
            self._build_equity_premium(ff),
            self._build_term_premium(treasury),
            self._build_credit_spread(credit),
            self._build_inflation(tips, cpi),
            self._build_liquidity(ps_liq),
        ], axis=1)


        # ── 3. Build monthly public asset returns ──────────────────
        public_monthly = self._build_public_assets(
            crsp_market, etf_returns
        )

        # ── 3b. Load professor private assets ─────────────────────
        professor_assets = self._professor.get_private_assets()
        if not professor_assets.empty:
            public_monthly = pd.concat(
                [public_monthly, professor_assets], axis=1
            )
            logger.info(
                f"Added professor assets: "
                f"{list(professor_assets.columns)}"
            )

        # ── 4. Convert to quarterly ────────────────────────────────
        # Return-based factors: compound monthly returns
        factors_q   = self._to_quarterly(factors_monthly[FACTOR_NAMES])
        public_q    = self._to_quarterly(public_monthly)
        recession_q = recession.resample("QE").last().astype(int)
        rf_q        = self._to_quarterly_series(rf)

        # ── 5. Tier 1 — public markets, 1980 to 2024 ──────────────
        # ── 5. Tier 1 — 2004 anchor, complete cases for POET ──────
        t1_start = pd.Timestamp(START_DATE_TIER1)
        t1_end   = pd.Timestamp("2024-12-31")

        # Full factor index — no NaN in factors
        t1_idx = factors_q.dropna().index
        t1_idx = t1_idx[
            (t1_idx >= t1_start) & (t1_idx <= t1_end)
        ]

        # Asset returns — reindex to full factor grid
        # NaNs preserved for pre-inception ETFs
        self.factor_returns_t1 = factors_q.loc[t1_idx].copy()
        self.asset_returns_t1  = public_q.reindex(t1_idx).copy()

        # Complete cases only — for POET covariance estimation
        # POET requires balanced panel — no NaN allowed
        # Rolling betas use asset_returns_t1 (per-asset dropna)
        complete_idx = self.asset_returns_t1.dropna(
            how="any"
        ).index
        self.asset_returns_t1_complete = (
            self.asset_returns_t1.loc[complete_idx].copy()
        )

        logger.info(
            f"Tier 1: {len(t1_idx)} quarters total, "
            f"{len(complete_idx)} complete cases for POET"
        )

        # ── 6. Tier 2 — full universe, 2004 to 2024 ───────────────
        private_q = self._ca.get_private_market_returns()

        if private_q.empty:
            logger.warning(
                "Private market data not available. "
                "Tier 2 uses public assets only."
            )
            t2_idx = t1_idx[t1_idx >= START_DATE_TIER2]
            self.factor_returns_t2 = factors_q.loc[t2_idx].copy()
            self.asset_returns_t2  = public_q.loc[t2_idx].copy()
        else:
            all_assets_q = pd.concat(
                [public_q, private_q], axis=1
            )
            t2_idx = factors_q.dropna().index\
                .intersection(all_assets_q.dropna(how="all").index)
            t2_idx = t2_idx[t2_idx >= START_DATE_TIER2]

            self.factor_returns_t2 = factors_q.loc[t2_idx].copy()
            self.asset_returns_t2  = all_assets_q.loc[t2_idx].copy()

        # ── 7. Shared series ───────────────────────────────────────
        self.recession = recession_q.reindex(t1_idx).fillna(0)
        self.rf        = rf_q.reindex(t1_idx).ffill()

        # ── 8. Cache ───────────────────────────────────────────────
        self._save_cache()

        # ── 9. Summary ─────────────────────────────────────────────
        self._print_summary()
        return self

    def load_cached(self) -> DataManager:
        """
        Load pre-built matrices from parquet.
        Use after build() has been run once.
        Loads in milliseconds vs minutes for full rebuild.
        """
        self.factor_returns_t1 = pd.read_parquet(
            DATA_CACHE_DIR / "factor_returns_t1.parquet"
        ) 
        self.asset_returns_t1 = pd.read_parquet(
            DATA_CACHE_DIR / "asset_returns_t1.parquet"
        )
        self.factor_returns_t2 = pd.read_parquet(
            DATA_CACHE_DIR / "factor_returns_t2.parquet"
        )
        self.asset_returns_t2 = pd.read_parquet(
            DATA_CACHE_DIR / "asset_returns_t2.parquet"
        )
        self.recession = pd.read_parquet(
            DATA_CACHE_DIR / "recession.parquet"
        ).squeeze()
        self.asset_returns_t1_complete = pd.read_parquet(
            DATA_CACHE_DIR / "asset_returns_t1_complete.parquet"
        )
        self.rf = pd.read_parquet(
            DATA_CACHE_DIR / "rf.parquet"
        ).squeeze()
        credit = self._fred.get_credit_spreads()
        self.credit_liquidity = self._to_quarterly_series(
            self._build_credit_liquidity(credit)
        )
        logger.info("Loaded all matrices from cache.")
        self._print_summary()
        return self

    # ── Cache helpers ──────────────────────────────────────────────

    def _save_cache(self) -> None:
        self.factor_returns_t1.to_parquet(
            DATA_CACHE_DIR / "factor_returns_t1.parquet"
        )
        self.asset_returns_t1.to_parquet(
            DATA_CACHE_DIR / "asset_returns_t1.parquet"
        )
        self.asset_returns_t1_complete.to_parquet(
            DATA_CACHE_DIR / "asset_returns_t1_complete.parquet"
        )
        self.factor_returns_t2.to_parquet(
            DATA_CACHE_DIR / "factor_returns_t2.parquet"
        )
        self.asset_returns_t2.to_parquet(
            DATA_CACHE_DIR / "asset_returns_t2.parquet"
        )
        self.recession.to_frame().to_parquet(
            DATA_CACHE_DIR / "recession.parquet"
        )
        self.rf.to_frame().to_parquet(
            DATA_CACHE_DIR / "rf.parquet"
        )
        logger.info("All matrices cached to parquet.")

    # ── Summary ────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)

        if self.factor_returns_t1 is not None:
            t1 = self.factor_returns_t1
            a1 = self.asset_returns_t1
            print("Tier 1 — Public Markets Only")
            print(
                f"  Date range : {t1.index[0].date()} "
                f"to {t1.index[-1].date()}"
            )
            print(f"  Quarters   : {len(t1)}")
            print(f"  Assets     : {a1.shape[1]}")
            print()

        if self.factor_returns_t2 is not None:
            t2 = self.factor_returns_t2
            a2 = self.asset_returns_t2
            print("Tier 2 — Full Universe")
            print(
                f"  Date range : {t2.index[0].date()} "
                f"to {t2.index[-1].date()}"
            )
            print(f"  Quarters   : {len(t2)}")
            print(f"  Assets     : {a2.shape[1]}")
            print()

        if self.factor_returns_t1 is not None:
            print("Factor Returns (quarterly %):")
            stats = (
            self.factor_returns_t1 * 100
            ).describe().loc[["mean", "std"]].T
            stats.columns = ["Qtr.Mean%", "Qtr.Vol%"]
            print(stats.round(3).to_string())

        print("=" * 60 + "\n")