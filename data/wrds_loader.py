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
    2. CRSP small cap returns — us_small_cap proxy
    3. CRSP treasury returns  — term premium construction
    4. Fama-French factors    — equity premium construction

    All returns in decimal. Date index at month-end.
    Parquet caching to avoid repeated API calls.
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self._conn: wrds.Connection | None = None

    # ── Private ────────────────────────────────────────────────────

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

    def _month_end(
        self, df: pd.DataFrame, col: str
    ) -> pd.DataFrame:
        """Standardize date index to month-end."""
        df[col] = pd.to_datetime(df[col]) + pd.offsets.MonthEnd(0)
        return df.set_index(col).sort_index()

    # ── Public ─────────────────────────────────────────────────────

    def get_market_returns(self) -> pd.DataFrame:
        """
        CRSP value-weighted and equal-weighted market returns.

        Why CRSP not SPY:
        - Covers all NYSE/AMEX/NASDAQ stocks
        - No survivorship bias
        - History from 1926
        - No ETF fees or tracking error

        Returns decimal monthly returns.
        """
        cached = self._load("market")
        if cached is not None:
            return cached

        self._connect()
        df = self._conn.raw_sql(f"""
            SELECT date, vwretd, ewretd
            FROM crsp.msi
            WHERE date >= '{START_DATE_TIER1}'
              AND date <= '{END_DATE}'
            ORDER BY date
        """)
        df = self._month_end(df, "date")
        assert df["vwretd"].abs().max() < 0.5, "Check units"
        self._save(df, "market")
        logger.info(f"CRSP market: {len(df)} months")
        return df

    def get_treasury_returns(self) -> pd.DataFrame:
        """
        CRSP monthly treasury returns by maturity.

        Used to construct term premium factor:
            term_premium = treasury_20y_return - tbill_30d_return

        Using return difference not yield difference — captures
        realized compensation for duration risk each period.
        """
        cached = self._load("treasury")
        if cached is not None:
            return cached

        self._connect()
        df = self._conn.raw_sql(f"""
            SELECT caldt   AS date,
                   t30ret  AS tbill_30d,
                   t90ret  AS tbill_90d,
                   t1yret  AS treasury_1y,
                   t5yret  AS treasury_5y,
                   t10yret AS treasury_10y,
                   t20yret AS treasury_20y,
                   t30yret AS treasury_30y
            FROM crsp.mth_treasury
            WHERE caldt >= '{START_DATE_TIER1}'
              AND caldt <= '{END_DATE}'
            ORDER BY caldt
        """)
        df = self._month_end(df, "date")
        self._save(df, "treasury")
        logger.info(f"CRSP treasury: {len(df)} months")
        return df

    def get_ff_factors(self) -> pd.DataFrame:
        """
        Fama-French monthly factors.

        mktrf = market excess return over risk-free rate.
        This is our equity premium factor proxy.
        rf    = risk-free rate (1M T-bill).

        Note: FF factors come in PERCENT — converted to decimal.
        """
        cached = self._load("ff")
        if cached is not None:
            return cached

        self._connect()
        df = self._conn.raw_sql(f"""
            SELECT date, mktrf, smb, hml, rf, umd
            FROM ff.factors_monthly
            WHERE date >= '{START_DATE_TIER1}'
              AND date <= '{END_DATE}'
            ORDER BY date
        """)
        df = self._month_end(df, "date")
        df = df / 100.0
        assert df["mktrf"].abs().max() < 0.5, "Check FF units"
        self._save(df, "ff")
        logger.info(f"FF factors: {len(df)} months")
        return df

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> WRDSLoader:
        return self

    def __exit__(self, *args) -> None:
        self.close()