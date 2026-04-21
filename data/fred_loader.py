from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fredapi import Fred

from config.settings import (
    DATA_CACHE_DIR,
    END_DATE,
    FRED_API_KEY,
    FRED_SERIES,
    START_DATE_TIER1,
)

logger = logging.getLogger(__name__)


class FREDLoader:
    """
    Fetches macro data from FRED.

    Provides five series:
    1. Credit spreads     — IG and HY OAS levels
    2. TIPS breakeven     — inflation factor construction
    3. CPI                — pre-2003 inflation backfill
    4. Risk free rate     — 3M T-bill monthly return
    5. Recession indicator — NBER 0/1 flag

    All series resampled to month-end.
    Rates converted from percent to decimal.
    Parquet caching throughout.
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.fred = Fred(api_key=FRED_API_KEY)
        self.use_cache = use_cache

    # ── Private ────────────────────────────────────────────────────

    def _cache(self, name: str) -> Path:
        return DATA_CACHE_DIR / f"fred_{name}.parquet"

    def _load(self, name: str) -> pd.DataFrame | None:
        p = self._cache(name)
        if self.use_cache and p.exists():
            logger.info(f"Cache hit: fred_{name}")
            return pd.read_parquet(p)
        return None

    def _save(self, df: pd.DataFrame, name: str) -> None:
        df.to_parquet(self._cache(name))
        logger.info(f"Cached: fred_{name}")

    def _fetch(
        self,
        series_id: str,
        name: str,
        to_decimal: bool = True,
    ) -> pd.Series:
        """Fetch one FRED series and resample to month-end."""
        raw = self.fred.get_series(
            series_id,
            observation_start=START_DATE_TIER1,
            observation_end=END_DATE,
        )
        monthly = raw.resample("ME").last()
        monthly.name = name
        if to_decimal:
            monthly = monthly / 100.0
        return monthly

    # ── Public ─────────────────────────────────────────────────────

    def get_credit_spreads(self) -> pd.DataFrame:
        """
        ICE BofA IG and HY option-adjusted spreads.

        Levels in decimal. Used to construct:
        - credit_spread factor: negative quarterly change in HY OAS
        - liquidity factor: HY OAS minus IG OAS level
        """
        cached = self._load("credit_spreads")
        if cached is not None:
            return cached

        logger.info("Fetching credit spreads from FRED...")
        ig = self._fetch(FRED_SERIES["ig_oas"], "ig_oas")
        hy = self._fetch(FRED_SERIES["hy_oas"], "hy_oas")
        df = pd.concat([ig, hy], axis=1).dropna()

        # Sanity check: IG spread always below HY
        invalid = (df["ig_oas"] >= df["hy_oas"]).sum()
        if invalid > 0:
            logger.warning(f"{invalid} months where IG >= HY OAS")

        self._save(df, "credit_spreads")
        logger.info(f"Credit spreads: {len(df)} months")
        return df

    def get_tips_breakeven(self) -> pd.Series:
        """
        10-year TIPS breakeven inflation rate.

        Monthly change used as inflation factor.
        Starts April 2003 — pre-2003 backfilled with CPI
        in DataManager.
        """
        cached = self._load("tips_breakeven")
        if cached is not None:
            return cached["tips_breakeven"]

        logger.info("Fetching TIPS breakeven from FRED...")
        s = self._fetch(FRED_SERIES["tips_breakeven"], "tips_breakeven")
        self._save(s.to_frame(), "tips_breakeven")
        return s

    def get_cpi(self) -> pd.Series:
        """
        CPI all-items monthly percent change.

        Used to backfill inflation factor pre-2003
        before TIPS breakeven data is available.
        """
        cached = self._load("cpi")
        if cached is not None:
            return cached["cpi_change"]

        logger.info("Fetching CPI from FRED...")
        raw = self._fetch(FRED_SERIES["cpi"], "cpi", to_decimal=False)
        cpi_change = raw.pct_change().rename("cpi_change")
        self._save(cpi_change.to_frame(), "cpi")
        return cpi_change

    def get_risk_free_rate(self) -> pd.Series:
        """
        3M T-bill rate as monthly return.

        Converts annualized percent to monthly decimal:
        monthly_rf = (annual_rate / 100) / 12
        """
        cached = self._load("rf")
        if cached is not None:
            return cached["rf"]

        logger.info("Fetching T-bill rate from FRED...")
        s = self._fetch(FRED_SERIES["tbill_3m"], "rf") / 12.0
        self._save(s.to_frame(), "rf")
        return s

    def get_recession_indicator(self) -> pd.Series:
        """
        NBER recession indicator.
        0 = expansion, 1 = contraction.
        Used for regime analysis and validation.
        """
        cached = self._load("recession")
        if cached is not None:
            return cached["recession"]

        logger.info("Fetching NBER recession indicator from FRED...")
        s = self._fetch(
            FRED_SERIES["nber_recession"],
            "recession",
            to_decimal=False,
        ).astype(int)

        self._save(s.to_frame(), "recession")
        n_rec = s.sum()
        logger.info(
            f"Recession indicator: {n_rec}/{len(s)} months "
            f"({100*s.mean():.1f}% contraction)"
        )
        return s