from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    DATA_CACHE_DIR,
    PUBLIC_ASSETS,
    START_DATE_TIER1,
    START_DATE_TIER2,
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

        # Shared
        self.recession: pd.Series | None = None
        self.rf:        pd.Series | None = None

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
        self, credit: pd.DataFrame
    ) -> pd.Series:
        """
        Liquidity = HY OAS minus IG OAS level.
        Spread-of-spreads captures systematic illiquidity premium.
        Wide spread = tight liquidity = high illiquidity premium.
        Used as level (not change) — slow-moving structural signal.
        """
        return (
            credit["hy_oas"] - credit["ig_oas"]
        ).rename("liquidity")

    # ── Asset construction ─────────────────────────────────────────

    def _build_public_assets(
        self,
        crsp_market: pd.DataFrame,
        etf_returns: pd.DataFrame,
        hy_total_return: pd.Series,
    ) -> pd.DataFrame:
        """
        Build 9-asset public market return matrix.

        Asset          Source
        ─────────────────────────────────────────────
        us_large_cap   CRSP vwretd
        us_small_cap   yfinance IWM (placeholder for CRSP deciles)
        em_equity      yfinance EEM
        long_treasury  yfinance TLT
        tips           yfinance TIP
        ig_credit      yfinance LQD
        hy_credit      ICE BofA HY Total Return (FRED)
        reits          yfinance VNQ
        commodities    yfinance GSG
        """
        assets = {}

        # CRSP for us_large_cap — cleaner than SPY
        assets["us_large_cap"] = crsp_market["vwretd"].rename(
            "us_large_cap"
        )

        for asset in PUBLIC_ASSETS:
            if asset == "us_large_cap":
                continue  # already added above
            if asset == "hy_credit":
                assets["hy_credit"] = hy_total_return
            elif asset in etf_returns.columns:
                assets[asset] = etf_returns[asset].rename(asset)

        return pd.DataFrame(assets)

    # ── Monthly to quarterly ───────────────────────────────────────

    @staticmethod
    def _to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compound monthly returns to quarterly.

        Formula: (1+r1)(1+r2)(1+r3) - 1
        Mathematically exact — no approximation.
        """
        return (1 + df).resample("QE").prod() - 1

    @staticmethod
    def _to_quarterly_series(s: pd.Series) -> pd.Series:
        return (1 + s).resample("QE").prod() - 1

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

        # HY total return from FRED for hy_credit asset
        # Using ICE BofA HY Total Return index — longer history than HYG
        hy_total = self._fred._fetch(
            "BAMLHYH0A0HYM2EY", "hy_credit", to_decimal=False
        ).pct_change()

        # ── 2. Build monthly factor series ────────────────────────
        factors_monthly = pd.concat([
            self._build_equity_premium(ff),
            self._build_term_premium(treasury),
            self._build_credit_spread(credit),
            self._build_inflation(tips, cpi),
            self._build_liquidity(credit),
        ], axis=1)

        # ── 3. Build monthly public asset returns ──────────────────
        public_monthly = self._build_public_assets(
            crsp_market, etf_returns, hy_total
        )

        # ── 4. Convert to quarterly ────────────────────────────────
        factors_q  = self._to_quarterly(factors_monthly)
        public_q   = self._to_quarterly(public_monthly)
        recession_q = recession.resample("QE").last().astype(int)
        rf_q       = self._to_quarterly_series(rf)

        # ── 5. Tier 1 — public markets, 1980 to 2024 ──────────────
        t1_idx = factors_q.dropna().index.intersection(
            public_q.dropna(how="all").index
        )
        t1_idx = t1_idx[t1_idx >= START_DATE_TIER1]

        self.factor_returns_t1 = factors_q.loc[t1_idx].copy()
        self.asset_returns_t1  = public_q.loc[t1_idx].copy()

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
        self.rf = pd.read_parquet(
            DATA_CACHE_DIR / "rf.parquet"
        ).squeeze()
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
            print("Factor Returns (annualized %):")
            stats = (
                self.factor_returns_t1 * 4 * 100
            ).describe().loc[["mean", "std"]].T
            stats.columns = ["Ann.Mean%", "Ann.Vol%"]
            print(stats.round(2).to_string())

        print("=" * 60 + "\n")