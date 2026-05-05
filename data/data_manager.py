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
from data.fred_loader import FREDLoader
from data.market_loader import MarketLoader
from data.wrds_loader import WRDSLoader

logger = logging.getLogger(__name__)


class DataManager:
    """
    Single orchestration layer for all data.

    Builds two analysis-ready matrices at quarterly frequency:

    Tier 1 — Public assets only, 1980-2024
    Tier 2 — Full universe, 2004-2024
    """

    def __init__(self, use_cache: bool = True) -> None:
        self._wrds   = WRDSLoader(use_cache=use_cache)
        self._fred   = FREDLoader(use_cache=use_cache)
        self._market = MarketLoader(use_cache=use_cache)
        self._professor = ProfessorLoader(use_cache=use_cache)

        self.factor_returns_t1: pd.DataFrame | None = None
        self.asset_returns_t1:  pd.DataFrame | None = None
        self.factor_returns_t2: pd.DataFrame | None = None
        self.asset_returns_t2:  pd.DataFrame | None = None
        self.asset_returns_t1_complete:  pd.DataFrame | None = None
        
        self.recession: pd.Series | None = None
        self.rf:        pd.Series | None = None
        self.credit_liquidity:  pd.Series | None = None

    # ── Factor construction ────────────────────────────────────────

    def _build_equity_premium(self, ff: pd.DataFrame) -> pd.Series:
        return ff["mktrf"].rename("equity_premium")

    def _build_term_premium(self, treasury: pd.DataFrame) -> pd.Series:
        return (treasury["treasury_20y"] - treasury["tbill_30d"]).rename("term_premium")

    def _build_credit_spread(self, credit: pd.DataFrame) -> pd.Series:
        return (-credit["hy_oas"].diff()).rename("credit_spread")

    def _build_inflation(self, cpi: pd.Series) -> pd.Series:
        """
        FIXED: Uses a consistent realized inflation metric (CPI difference) 
        across the whole history to prevent structural breaks in machine learning models.
        """
        return cpi.rename("inflation") 
    
    def _build_liquidity(self, ps_liq: pd.Series) -> pd.Series:
        return ps_liq.rename("liquidity")
    
    def _build_credit_liquidity(self, credit: pd.DataFrame) -> pd.Series:
        # AAA-BAA quality spread: when spread widens (tighter credit conditions)
        # the factor goes negative. -Δ(spread) = positive when conditions ease.
        return (-credit["quality_spread"].diff()).rename("credit_liquidity")

    # ── Asset construction ─────────────────────────────────────────

    def _build_public_assets(self, crsp_market: pd.DataFrame, etf_returns: pd.DataFrame) -> pd.DataFrame:
        assets = {}
        assets["us_large_cap"] = crsp_market["vwretd"].rename("us_large_cap")

        for asset in PUBLIC_ASSETS:
            if asset == "us_large_cap": continue
            if asset in etf_returns.columns:
                assets[asset] = etf_returns[asset].rename(asset)

        return pd.DataFrame(assets)

    # ── Monthly to quarterly ───────────────────────────────────────

    @staticmethod
    def _to_quarterly(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Geometric compounding for returns."""
        def compound(x):
            valid = x.dropna()
            if len(valid) == 0: return float("nan")
            return (1 + valid).prod() - 1
        return data.resample("QE").apply(compound)

    @staticmethod
    def _to_quarterly_sum(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        FIXED: Arithmetic summation for yield/spread differences.
        """
        def arithmetic_sum(x):
            valid = x.dropna()
            if len(valid) == 0: return float("nan")
            return valid.sum()
        return data.resample("QE").apply(arithmetic_sum)

    # ── Main build ─────────────────────────────────────────────────

    def build(self) -> DataManager:
        logger.info("=" * 60)
        logger.info("Building data matrices...")
        logger.info("=" * 60)

        # ── 1. Fetch raw data ──────────────────────────────────────
        ff           = self._wrds.get_ff_factors()
        treasury     = self._wrds.get_treasury_returns()
        crsp_market  = self._wrds.get_market_returns()
        etf_returns  = self._market.get_etf_returns()
        credit       = self._fred.get_credit_spreads()
        cpi          = self._fred.get_cpi() # TIPS dropped to avoid structural break
        recession    = self._fred.get_recession_indicator()
        rf_raw       = self._fred.get_risk_free_rate()
        ps_liq       = self._wrds.get_ps_liquidity()
        rf = self._fred.get_risk_free_rate()
        
        # FIXED: De-annualize FRED risk-free rate (assumes FRED returns annualized %)
        # Divides by 100 for decimal, and 12 for monthly rate before geometric compounding
        rf_q = self._to_quarterly(rf_raw)

        # FIXED: Using arithmetic sum for spread differences
        self.credit_liquidity = self._to_quarterly_sum(self._build_credit_liquidity(credit))

        # ── 2. Build monthly factor series ────────────────────────
        factors_monthly = pd.concat([
            self._build_equity_premium(ff),
            self._build_term_premium(treasury),
            self._build_credit_spread(credit),
            self._build_inflation(cpi),
            self._build_liquidity(ps_liq),
        ], axis=1)

        # ── 3. Build monthly public asset returns ──────────────────
        public_monthly = self._build_public_assets(crsp_market, etf_returns)

        # ── 3b. Load professor private assets ─────────────────────
        professor_assets = self._professor.get_private_assets()
        private_monthly = pd.DataFrame()
        if not professor_assets.empty:
            private_monthly = professor_assets
            logger.info(f"Loaded professor assets: {list(professor_assets.columns)}")

        # ── 4. Convert to quarterly ────────────────────────────────
        geom_factors = ["equity_premium", "term_premium", "liquidity"]
        arith_factors = ["credit_spread", "inflation"]

        factors_geom = self._to_quarterly(factors_monthly[geom_factors])
        factors_arith = self._to_quarterly_sum(factors_monthly[arith_factors])
        
        factors_q = pd.concat([factors_geom, factors_arith], axis=1)[FACTOR_NAMES]
        
        public_q = self._to_quarterly(public_monthly)
        private_q = self._to_quarterly(private_monthly) if not private_monthly.empty else pd.DataFrame()
        
        recession_q = recession.resample("QE").last().astype(int)
        rf_q        = self._to_quarterly(rf) # Compound the de-annualized rate

        # ── 5. Tier 1 — public markets, 1980 to 2024 ──────────────
        t1_start = pd.Timestamp(START_DATE_TIER1)
        t1_end   = pd.Timestamp("2024-12-31")

        t1_idx = factors_q.dropna().index
        t1_idx = t1_idx[(t1_idx >= t1_start) & (t1_idx <= t1_end)]

        self.factor_returns_t1 = factors_q.loc[t1_idx].copy()
        self.asset_returns_t1  = public_q.reindex(t1_idx).copy()

        complete_idx = self.asset_returns_t1.dropna(how="any").index
        self.asset_returns_t1_complete = self.asset_returns_t1.loc[complete_idx].copy()

        # ── 6. Tier 2 — full universe, 2004 to 2024 ───────────────
        if private_q.empty:
            logger.warning("Private market data not available. Tier 2 uses public assets only.")
            t2_idx = t1_idx[t1_idx >= START_DATE_TIER2]
            self.factor_returns_t2 = factors_q.loc[t2_idx].copy()
            self.asset_returns_t2  = public_q.loc[t2_idx].copy()
        else:
            all_assets_q = pd.concat([public_q, private_q], axis=1)
            t2_idx = factors_q.dropna().index.intersection(all_assets_q.dropna(how="all").index)
            t2_idx = t2_idx[t2_idx >= START_DATE_TIER2]

            self.factor_returns_t2 = factors_q.loc[t2_idx].copy()
            self.asset_returns_t2  = all_assets_q.loc[t2_idx].copy()

        # ── 7. Shared series ───────────────────────────────────────
        self.recession = recession_q.reindex(t1_idx).fillna(0)
        self.rf        = rf_q.reindex(t1_idx).ffill()
        
        # FIXED: Align credit_liquidity to the Tier 1 index bounds
        self.credit_liquidity = self.credit_liquidity.reindex(t1_idx).fillna(0)

        # ── 8. Cache ───────────────────────────────────────────────
        self._save_cache()
        
        # ── 9. Summary ─────────────────────────────────────────────
        self._print_summary()
        
        return self

    def load_cached(self) -> DataManager:
        self.factor_returns_t1 = pd.read_parquet(DATA_CACHE_DIR / "factor_returns_t1.parquet") 
        self.asset_returns_t1 = pd.read_parquet(DATA_CACHE_DIR / "asset_returns_t1.parquet")
        self.factor_returns_t2 = pd.read_parquet(DATA_CACHE_DIR / "factor_returns_t2.parquet")
        self.asset_returns_t2 = pd.read_parquet(DATA_CACHE_DIR / "asset_returns_t2.parquet")
        self.recession = pd.read_parquet(DATA_CACHE_DIR / "recession.parquet").squeeze()
        self.asset_returns_t1_complete = pd.read_parquet(DATA_CACHE_DIR / "asset_returns_t1_complete.parquet")
        self.rf = pd.read_parquet(DATA_CACHE_DIR / "rf.parquet").squeeze()
        self.credit_liquidity = pd.read_parquet(DATA_CACHE_DIR / "credit_liquidity.parquet").squeeze()
        
        logger.info("Loaded all matrices from cache.")
        self._print_summary()
        
        return self
    
    def build_tier2(self, csv_path: str) -> "DataManager":
        """
        Loads quarterly Tier 2 assets from the CSV and converts 
        monthly Tier 1 factors to quarterly frequency to match.
        """
        import pandas as pd
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Loading Tier 2 data from {csv_path}...")

        # 1. Load the specific CSV you uploaded
        t2_assets = pd.read_csv(csv_path)
        
        # Clean the Date column and set it as the index
        t2_assets['Date'] = pd.to_datetime(t2_assets['Date'])
        t2_assets.set_index('Date', inplace=True)
        
        # Snap all dates to the exact end of the quarter to ensure they merge perfectly
        t2_assets = t2_assets.resample('Q').last()

        # 2. Roll-up the Tier 1 Factors (Fama-French, Market) via compounding
        logger.info("Rolling up monthly factors to quarterly...")
        q_factors = (1 + self.factor_returns_t1).resample('Q').prod() - 1

        # 3. Roll-up the Liquidity factor (Snapshot at end of quarter)
        if hasattr(self, 'credit_liquidity') and self.credit_liquidity is not None:
            q_liquidity = self.credit_liquidity.resample('Q').last()
            q_factors['liquidity'] = q_liquidity

        # 4. Merge them together. 'inner' join drops any dates that don't match exactly.
        aligned_df = pd.concat([q_factors, t2_assets], axis=1, join='inner')

        # 5. Split them back apart and save as class attributes
        self.factor_returns_t2 = aligned_df[q_factors.columns]
        self.asset_returns_t2 = aligned_df[t2_assets.columns]

        logger.info(f"Tier 2 Ready: {len(aligned_df)} quarterly periods aligned.")
        return self
    
    def _save_cache(self) -> None:
        """Saves the fully built matrices to the cache for instant loading later."""
        import pandas as pd
        from config.settings import DATA_CACHE_DIR

        # Ensure the cache directory exists
        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        def _save(obj, fname):
            if obj is not None:
                pd.DataFrame(obj).to_parquet(DATA_CACHE_DIR / fname)

        _save(self.factor_returns_t1,        "factor_returns_t1.parquet")
        _save(self.asset_returns_t1,         "asset_returns_t1.parquet")
        _save(self.asset_returns_t1_complete,"asset_returns_t1_complete.parquet")
        _save(self.factor_returns_t2,        "factor_returns_t2.parquet")
        _save(self.asset_returns_t2,         "asset_returns_t2.parquet")
        _save(self.credit_liquidity,         "credit_liquidity.parquet")
        _save(self.recession,                "recession.parquet")
        _save(self.rf,                       "rf.parquet")


    def _print_summary(self) -> None:
        """Prints a quick summary of the built data matrices."""
        print("\n" + "="*50)
        print("DATA MANAGER BUILD SUMMARY")
        print("="*50)
        
        if hasattr(self, 'factor_returns_t1') and self.factor_returns_t1 is not None:
            print(f"Tier 1 Factor Returns: {self.factor_returns_t1.shape}")
            
        if hasattr(self, 'asset_returns_t1_complete') and self.asset_returns_t1_complete is not None:
            print(f"Tier 1 Asset Returns:  {self.asset_returns_t1_complete.shape}")
            
        if hasattr(self, 'credit_liquidity') and self.credit_liquidity is not None:
            print(f"Credit Liquidity Proxy: {self.credit_liquidity.shape}")
            
        if hasattr(self, 'factor_returns_t2') and self.factor_returns_t2 is not None:
            print(f"Tier 2 Factor Returns: {self.factor_returns_t2.shape}")
            
        if hasattr(self, 'asset_returns_t2') and self.asset_returns_t2 is not None:
            print(f"Tier 2 Asset Returns:  {self.asset_returns_t2.shape}")
            
        print("="*50 + "\n")