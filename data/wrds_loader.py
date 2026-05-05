from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import wrds

from config.settings import (
    DATA_CACHE_DIR,
    END_DATE,
    START_DATE_TIER1,
    WRDS_USERNAME,
)

logger = logging.getLogger(__name__)


class WRDSLoader:
    """
    Fetches data from WRDS.

    Provides four datasets:
    1. CRSP market returns    — us_large_cap proxy
    2. CRSP treasury returns  — term premium construction
    3. Fama-French factors    — equity premium construction
    4. Pastor-Stambaugh       — liquidity factor proxy

    All returns in decimal. Date index at month-end, tz-naive.
    Parquet caching to avoid repeated API calls.
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self._conn: wrds.Connection | None = None

    def _connect(self) -> None:
        if self._conn is None:
            logger.info("Connecting to WRDS...")
            self._conn = wrds.Connection(wrds_username=WRDS_USERNAME)
            logger.info("WRDS connected.")

    def _cache(self, name: str) -> Path:
        return DATA_CACHE_DIR / f"wrds_{name}.parquet"

    def _load(self, name: str) -> pd.DataFrame | None:
        p = self._cache(name)
        if self.use_cache and p.exists():
            logger.info(f"Cache hit: wrds_{name}")
            return pd.read_parquet(p)
        return None

    def _save(self, df: pd.DataFrame, name: str) -> None:
        df.to_parquet(self._cache(name))
        logger.info(f"Cached: wrds_{name}")

    def _month_end(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Standardize date index to month-end, tz-naive."""
        s = pd.to_datetime(df[col])
        if s.dt.tz is not None:
            s = s.dt.tz_localize(None)
            
        df[col] = s + pd.offsets.MonthEnd(0)
        return df.set_index(col).sort_index()

    def get_market_returns(self) -> pd.DataFrame:
        cached = self._load("market")
        if cached is not None:
            return cached

        self._connect()
        logger.info("Fetching CRSP market returns...")
        
        # Pull Value-Weighted Return from CRSP Monthly Stock Indices
        df = self._conn.raw_sql(f"""
            SELECT date, vwretd
            FROM crsp.msi
            WHERE date >= '{START_DATE_TIER1}'
              AND date <= '{END_DATE}'
            ORDER BY date
        """)
        
        df = self._month_end(df, "date")
        
        # FIXED: Increased ceiling to 1.5 to survive the August 1932 +83.2% market return
        assert df["vwretd"].abs().max() < 1.5, "Market returns are likely still in percent"
        
        self._save(df, "market")
        return df

    def get_treasury_returns(self) -> pd.DataFrame:
        cached = self._load("treasury")
        if cached is not None:
            return cached

        self._connect()

        # 20-year Treasury total return
        df20 = self._conn.raw_sql(f"""
            SELECT qtdate AS date, totret AS treasury_20y
            FROM crsp.cs20yr
            WHERE qtdate >= '{START_DATE_TIER1}'
              AND qtdate <= '{END_DATE}'
            ORDER BY qtdate
        """)
        df20 = self._month_end(df20, "date")
        df20["treasury_20y"] = df20["treasury_20y"] / 100.0

        # 90-day T-bill yield converted to monthly return
        df90 = self._conn.raw_sql(f"""
            SELECT qtdate AS date, yldmat AS tbill_yield
            FROM crsp.cs90d
            WHERE qtdate >= '{START_DATE_TIER1}'
              AND qtdate <= '{END_DATE}'
            ORDER BY qtdate
        """)
        df90 = self._month_end(df90, "date")
        df90["tbill_30d"] = df90["tbill_yield"] / 100.0 / 12.0
        df90 = df90[["tbill_30d"]]

        df = df20.join(df90, how="inner")
        self._save(df, "treasury")
        logger.info(f"CRSP treasury: {len(df)} months")
        return df

    def get_ps_liquidity(self) -> pd.Series:
        cached = self._load("ps_liquidity")
        if cached is not None:
            return cached["liquidity"]

        self._connect()
        df = self._conn.raw_sql(f"""
            SELECT date, ps_innov AS liquidity
            FROM ff.liq_ps
            WHERE date >= '{START_DATE_TIER1}'
              AND date <= '{END_DATE}'
            ORDER BY date
        """)
        df = self._month_end(df, "date")
        self._save(df, "ps_liquidity")
        logger.info(f"PS liquidity: {len(df)} months")
        return df["liquidity"]

    def get_ff_factors(self) -> pd.DataFrame:
        cached = self._load("ff")
        if cached is not None:
            return cached

        self._connect()
        logger.info("Fetching Fama-French factors...")
        
        # Pull the core factors from the Fama-French monthly tables
        df = self._conn.raw_sql(f"""
            SELECT date, mktrf, smb, hml, rf
            FROM ff.factors_monthly
            WHERE date >= '{START_DATE_TIER1}'
              AND date <= '{END_DATE}'
            ORDER BY date
        """)
        
        df = self._month_end(df, "date")

        # WRDS ff.factors_monthly stores values in decimal form (0.05 = 5%), NOT percent.
        # No division needed. Assert values are in plausible decimal range.
        assert df["mktrf"].abs().max() < 1.0, "FF mktrf looks too large — check WRDS format"
        assert df["mktrf"].abs().max() > 0.005, "FF mktrf looks too small — may have been pre-divided"
        
        self._save(df, "ff")
        return df

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> WRDSLoader:
        return self

    def __exit__(self, *args) -> None:
        self.close()