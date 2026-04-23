from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import DATA_CACHE_DIR

logger = logging.getLogger(__name__)

ETF_MAP: dict[str, str] = {
    "us_mid_cap":    "IJH",
    "us_small_cap":  "IWM",
    "em_equity":     "EEM",
    "long_treasury": "TLT",
    "tips":          "TIP",
    "ig_credit":     "LQD",
    "reits":         "VNQ",
}

COMMODITIES_SPLICE_DATE = "2006-08-31"
COMMODITIES_PRIMARY     = "GSG"
COMMODITIES_BACKFILL    = "^SPGSCI"


class MarketLoader:
    """
    Downloads ETF and index return data via yfinance.

    Asset mapping:
        us_mid_cap    IJH  — iShares S&P MidCap 400 (2004+)
        us_small_cap  IWM  — iShares Russell 2000
        em_equity     EEM  — iShares MSCI EM
        long_treasury TLT  — iShares 20+ Year Treasury
        tips          TIP  — iShares TIPS Bond
        ig_credit     LQD  — iShares IG Corporate Bond
        reits         VNQ  — Vanguard Real Estate
        commodities   GSG spliced with ^SPGSCI

    Commodities splice:
        Pre-2006-08-31:  ^SPGSCI spot index (price return only)
        Post-2006-08-31: GSG ETF total return
        Limitation: pre-splice excludes roll yield.
        Acceptable for beta estimation. Documented in paper.
    """

    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: Path = DATA_CACHE_DIR,
    ) -> None:
        self.use_cache = use_cache
        self.cache_dir = cache_dir

    def get_etf_returns(self) -> pd.DataFrame:
        """Download monthly total returns for all ETF assets."""
        cache_path = self.cache_dir / "market_etf_returns.parquet"
        if self.use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.info(
                f"Cache hit: market_etf_returns — {df.shape[1]} assets"
            )
            return df

        logger.info("Downloading ETF data from yfinance...")
        returns = {}

        for asset, ticker in ETF_MAP.items():
            try:
                raw = yf.download(
                    ticker,
                    start="1990-01-01",
                    auto_adjust=True,
                    progress=False,
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    raw = raw["Close"]
                else:
                    raw = raw["Close"]

                monthly = raw.resample("ME").last().squeeze()
                ret = monthly.pct_change().dropna()
                if isinstance(ret, pd.DataFrame):
                    ret = ret.iloc[:, 0]
                ret = ret.squeeze()
                ret.name = asset
                returns[asset] = ret

                logger.info(
                    f"  {asset:<20} ({ticker}): "
                    f"starts {ret.index[0].date()}, n={len(ret)}"
                )
            except Exception as e:
                logger.warning(f"  {asset} ({ticker}): {e}")

        returns["commodities"] = self._get_commodities_spliced()

        df = pd.DataFrame(returns)
        df.index = pd.to_datetime(df.index)
        df.to_parquet(cache_path)
        logger.info(
            f"ETF returns: {df.shape[0]} months x {df.shape[1]} assets"
        )
        return df

    def _get_commodities_spliced(self) -> pd.Series:
        """Commodities splice: ^SPGSCI pre-2006, GSG post-2006."""
        cache_path = self.cache_dir / "commodities_spliced.parquet"
        if self.use_cache and cache_path.exists():
            logger.info("Cache hit: commodities_spliced")
            return pd.read_parquet(cache_path).squeeze()

        logger.info("Building commodities splice (^SPGSCI + GSG)...")

        try:
            spgsci_raw = yf.download(
                COMMODITIES_BACKFILL, start="1990-01-01",
                auto_adjust=True, progress=False,
            )
            if isinstance(spgsci_raw.columns, pd.MultiIndex):
                spgsci_raw = spgsci_raw["Close"]
            else:
                spgsci_raw = spgsci_raw["Close"]
            spgsci_ret = spgsci_raw.resample("ME").last().pct_change().dropna()
        except Exception as e:
            logger.warning(f"^SPGSCI failed: {e}")
            spgsci_ret = pd.Series(dtype=float)

        try:
            gsg_raw = yf.download(
                COMMODITIES_PRIMARY, start="2006-01-01",
                auto_adjust=True, progress=False,
            )
            if isinstance(gsg_raw.columns, pd.MultiIndex):
                gsg_raw = gsg_raw["Close"]
            else:
                gsg_raw = gsg_raw["Close"]
            gsg_ret = gsg_raw.resample("ME").last().squeeze()
            if isinstance(gsg_ret, pd.DataFrame):
                gsg_ret = gsg_ret.iloc[:, 0]
            gsg_ret = gsg_ret.pct_change().dropna()
        except Exception as e:
            logger.warning(f"GSG failed: {e}")
            gsg_ret = pd.Series(dtype=float)

        # Also ensure spgsci_ret is a Series
        if isinstance(spgsci_ret, pd.DataFrame):
            spgsci_ret = spgsci_ret.iloc[:, 0]
        spgsci_ret = spgsci_ret.squeeze()

        splice_date = pd.Timestamp(COMMODITIES_SPLICE_DATE)

        if len(spgsci_ret) > 0 and len(gsg_ret) > 0:
            pre  = spgsci_ret[spgsci_ret.index <  splice_date]
            post = gsg_ret[gsg_ret.index       >= splice_date]
            combined = pd.concat([pre, post]).sort_index()
        elif len(gsg_ret) > 0:
            combined = gsg_ret
        else:
            combined = spgsci_ret

        if isinstance(combined, pd.DataFrame):
            combined = combined.iloc[:, 0]
        combined = combined.squeeze()
        combined.name = "commodities"
        combined.to_frame().to_parquet(cache_path)
        logger.info(
            f"Commodities spliced: {len(combined)} months, "
            f"splice at {COMMODITIES_SPLICE_DATE}"
        )
        return combined
