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

COMMODITIES_SPLICE_DATE = "2002-09-30"   # PCRIX inception Jun 2002; first full quarter end
COMMODITIES_PRIMARY     = "PCRIX"        # PIMCO Bloomberg Commodity Total Return (from 2002)
COMMODITIES_BACKFILL    = "^BCOM"        # Bloomberg Commodity Index (from 1991)
COMMODITIES_BACKFILL_ALT = "GSG"         # iShares GSCI ETF (last-resort fallback, from 2006)


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

        logger.info("Building commodities splice (^BCOM backfill + PCRIX primary)...")

        # Primary: PCRIX — PIMCO Bloomberg Commodity Total Return Fund (inception Jun 2002).
        # Tracks Bloomberg Commodity Index (BCOM), more diversified than GSCI (less energy-heavy).
        pcrix_ret = pd.Series(dtype=float)
        try:
            p = yf.download(COMMODITIES_PRIMARY, start="2002-01-01", auto_adjust=True, progress=False)
            if isinstance(p.columns, pd.MultiIndex):
                p = p["Close"]
            else:
                p = p["Close"]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[:, 0]
            pcrix_ret = p.squeeze().resample("ME").last().squeeze().pct_change().dropna()
            if len(pcrix_ret) > 0:
                logger.info(f"PCRIX loaded: {len(pcrix_ret)} months, "
                            f"{pcrix_ret.index[0].date()} to {pcrix_ret.index[-1].date()}")
        except Exception as e:
            logger.warning(f"PCRIX failed: {e}")

        # Backfill: ^BCOM Bloomberg Commodity Index (1991+), then GSG as last resort.
        bcom_ret = pd.Series(dtype=float)
        for ticker in [COMMODITIES_BACKFILL, COMMODITIES_BACKFILL_ALT]:
            try:
                s = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
                if isinstance(s.columns, pd.MultiIndex):
                    s = s["Close"]
                else:
                    s = s["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                bcom_ret = s.squeeze().resample("ME").last().squeeze().pct_change().dropna()
                if len(bcom_ret) > 0:
                    if ticker == COMMODITIES_BACKFILL_ALT:
                        logger.warning(
                            f"^BCOM unavailable — using {ticker} as backfill. "
                            f"Index differs (GSCI vs BCOM); structural break possible at splice date."
                        )
                    else:
                        logger.info(f"^BCOM backfill: {len(bcom_ret)} months")
                    break
            except Exception as e:
                logger.warning(f"{ticker} failed: {e}")

        splice_date = pd.Timestamp(COMMODITIES_SPLICE_DATE)
        if len(pcrix_ret) > 0 and len(bcom_ret) > 0:
            combined = pd.concat([
                bcom_ret[bcom_ret.index < splice_date],
                pcrix_ret[pcrix_ret.index >= splice_date],
            ]).sort_index()
            logger.info(f"Commodities: ^BCOM pre-{COMMODITIES_SPLICE_DATE}, PCRIX from {COMMODITIES_SPLICE_DATE}")
        elif len(pcrix_ret) > 0:
            combined = pcrix_ret
        else:
            combined = bcom_ret

        if isinstance(combined, pd.DataFrame):
            combined = combined.iloc[:, 0]
        combined = combined.squeeze()
        combined.name = "commodities"
        combined.to_frame().to_parquet(cache_path)
        logger.info(f"Commodities spliced: {len(combined)} months")
        return combined
