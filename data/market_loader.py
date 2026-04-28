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
    def __init__(self, use_cache: bool = True, cache_dir: Path = DATA_CACHE_DIR) -> None:
        self.use_cache = use_cache
        self.cache_dir = cache_dir

    def get_etf_returns(self) -> pd.DataFrame:
        cache_path = self.cache_dir / "market_etf_returns.parquet"
        if self.use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.info(f"Cache hit: market_etf_returns — {df.shape[1]} assets")
            return df

        logger.info("Downloading ETF data from yfinance...")
        returns = {}

        for asset, ticker in ETF_MAP.items():
            try:
                raw = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw = raw["Close"]
                else:
                    raw = raw["Close"]
                if isinstance(raw, pd.DataFrame):
                    raw = raw.iloc[:, 0]
                raw = raw.squeeze()
                monthly = raw.resample("ME").last().squeeze()
                ret = monthly.pct_change().dropna().squeeze()
                ret.name = asset
                returns[asset] = ret
                logger.info(f"  {asset:<20} ({ticker}): starts {ret.index[0].date()}, n={len(ret)}")
            except Exception as e:
                logger.warning(f"  {asset} ({ticker}): {e}")

        returns["commodities"] = self._get_commodities_spliced()

        df = pd.DataFrame(returns)
        df.index = pd.to_datetime(df.index)
        df.to_parquet(cache_path)
        logger.info(f"ETF returns: {df.shape[0]} months x {df.shape[1]} assets")
        return df

    def _get_commodities_spliced(self) -> pd.Series:
        cache_path = self.cache_dir / "commodities_spliced.parquet"
        if self.use_cache and cache_path.exists():
            logger.info("Cache hit: commodities_spliced")
            return pd.read_parquet(cache_path).squeeze()

        logger.info("Building commodities splice (^SPGSCI + GSG)...")

        try:
            s = yf.download(COMMODITIES_BACKFILL, start="1990-01-01", auto_adjust=True, progress=False)
            if isinstance(s.columns, pd.MultiIndex):
                s = s["Close"]
            else:
                s = s["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            spgsci_ret = s.squeeze().resample("ME").last().squeeze().pct_change().dropna()
        except Exception as e:
            logger.warning(f"^SPGSCI failed: {e}")
            spgsci_ret = pd.Series(dtype=float)

        try:
            g = yf.download(COMMODITIES_PRIMARY, start="2006-01-01", auto_adjust=True, progress=False)
            if isinstance(g.columns, pd.MultiIndex):
                g = g["Close"]
            else:
                g = g["Close"]
            if isinstance(g, pd.DataFrame):
                g = g.iloc[:, 0]
            gsg_ret = g.squeeze().resample("ME").last().squeeze().pct_change().dropna()
        except Exception as e:
            logger.warning(f"GSG failed: {e}")
            gsg_ret = pd.Series(dtype=float)

        splice_date = pd.Timestamp(COMMODITIES_SPLICE_DATE)
        if len(spgsci_ret) > 0 and len(gsg_ret) > 0:
            combined = pd.concat([
                spgsci_ret[spgsci_ret.index < splice_date],
                gsg_ret[gsg_ret.index >= splice_date],
            ]).sort_index()
        elif len(gsg_ret) > 0:
            combined = gsg_ret
        else:
            combined = spgsci_ret

        if isinstance(combined, pd.DataFrame):
            combined = combined.iloc[:, 0]
        combined = combined.squeeze()
        combined.name = "commodities"
        combined.to_frame().to_parquet(cache_path)
        logger.info(f"Commodities spliced: {len(combined)} months, splice at {COMMODITIES_SPLICE_DATE}")
        return combined
