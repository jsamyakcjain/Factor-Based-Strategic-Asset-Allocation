from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import (
    DATA_CACHE_DIR,
    END_DATE,
    START_DATE_TIER1,
)

logger = logging.getLogger(__name__)

# ETF proxies for asset classes not cleanly available on WRDS
# Adjusted close prices capture total return including dividends
ETF_MAP: dict[str, str] = {
    "em_equity":   "EEM",   # MSCI Emerging Markets — starts 2003
    "tips":        "TIP",   # Bloomberg US TIPS — starts 2003
    "ig_credit":   "LQD",   # Bloomberg Corp IG — starts 2002
    "hy_credit":   "HYG",   # ICE BofA HY — starts 2007
    "reits":       "VNQ",   # FTSE NAREIT — starts 2004
    "commodities": "GSG",   # S&P GSCI — starts 2006
}

# Longer history sources for Tier 1 analysis (1980+)
# These replace ETFs when available
# Loaded separately in DataManager
LONG_HISTORY_NOTES: dict[str, str] = {
    "em_equity":   "MSCI EM index from Bloomberg 1988+",
    "tips":        "Bloomberg TIPS index 1997+, CPI backfill pre-1997",
    "ig_credit":   "Ibbotson SBBI LT Corporate 1926+",
    "hy_credit":   "ICE BofA HY Total Return from FRED 1986+",
    "reits":       "FTSE NAREIT All Equity from nareit.com 1972+",
    "commodities": "S&P GSCI Total Return from Bloomberg 1970+",
}


class MarketLoader:
    """
    Downloads ETF total return data from yfinance.

    Used for asset classes not available on WRDS:
    EM equity, TIPS, IG credit, HY credit, REITs, commodities.

    Note on history:
    ETFs have limited history (earliest 2002). For Tier 1
    analysis (1980+), longer series from Bloomberg/FRED
    are substituted in DataManager. These ETFs provide
    the Tier 2 (2004+) baseline.

    Adjusted close prices account for:
    - Dividend and distribution income
    - Capital gains distributions
    - Stock splits
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache

    # ── Private ────────────────────────────────────────────────────

    def _cache(self) -> Path:
        return DATA_CACHE_DIR / "market_etf_returns.parquet"

    # ── Public ─────────────────────────────────────────────────────

    def get_etf_returns(self) -> pd.DataFrame:
        """
        Monthly total returns for all ETF proxies.

        Process:
        1. Download daily adjusted close prices
        2. Resample to month-end
        3. Compute simple percentage returns
        4. Rename tickers to internal asset names

        Returns decimal monthly returns with month-end index.
        """
        cache_path = self._cache()
        if self.use_cache and cache_path.exists():
            logger.info("Cache hit: market_etf_returns")
            return pd.read_parquet(cache_path)

        logger.info("Downloading ETF data from yfinance...")
        tickers = list(ETF_MAP.values())

        raw = yf.download(
            tickers,
            start=START_DATE_TIER1,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
        )["Close"]

        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])

        # Month-end prices then simple returns
        monthly = raw.resample("ME").last().pct_change()

        # Rename tickers to internal names
        ticker_to_name = {v: k for k, v in ETF_MAP.items()}
        monthly = monthly.rename(columns=ticker_to_name)

        # Log coverage per asset
        for asset, ticker in ETF_MAP.items():
            if asset in monthly.columns:
                first = monthly[asset].first_valid_index()
                n = monthly[asset].notna().sum()
                logger.info(
                    f"  {asset:<20} ({ticker}): "
                    f"starts {first.date()}, n={n}"
                )

        monthly.to_parquet(cache_path)
        logger.info(
            f"ETF returns: {monthly.shape[0]} months "
            f"x {monthly.shape[1]} assets"
        )
        return monthly