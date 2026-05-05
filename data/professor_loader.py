from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import DATA_CACHE_DIR

logger = logging.getLogger(__name__)

DATASET_PATH = DATA_CACHE_DIR / "professor_dataset.xlsx"

# Keyword sets for flexible string-based column search (searched case-insensitively
# across the first 5 header rows). Falls back to integer indices if no match found.
ASSET_SEARCH: dict[str, list[str]] = {
    "hy_credit":              ["hy credit", "high yield", "us hy", "corp hy"],
    "private_equity_unsmthd": ["pe unsmooth", "priv eq unsmooth", "unsmooth pe", "pe_unsmth"],
    "real_estate_unsmthd":    ["re unsmooth", "real estate unsmooth", "unsmooth re", "re_unsmth"],
    "hedge_funds":            ["hedge fund", "hf index", "eureka"],
    "private_equity":         ["private equity", "pe return", "priv equity"],
    "private_real_estate":    ["private real estate", "priv real estate"],
}

# Integer-index fallback (0-indexed) — used only when string search fails
ASSET_COLS_FALLBACK: dict[int, str] = {
    21: "hy_credit",
    29: "private_equity",
    30: "private_real_estate",
    31: "hedge_funds",
    33: "private_equity_unsmthd",
    34: "real_estate_unsmthd",
}


class ProfessorLoader:
    """
    Loads private market and alternative asset returns from
    professor-provided dataset.xlsx.

    Assets loaded:
        hy_credit              — US Corp Credit HY (smoothed)
        private_equity_unsmthd — Geltner-unsmoothed PE returns
        real_estate_unsmthd    — Geltner-unsmoothed RE returns
        hedge_funds            — Global Hedge Funds index

    We use unsmoothed PE and RE because:
    - Smoothed series have artificially low volatility
    - Smoothed betas are biased toward zero
    - Unsmoothed reflects true economic risk exposure
    - Standard practice in academic PE/RE factor models

    Date range: 2000-Q1 to 2025-Q3 (103-105 quarters)
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self._data:     pd.DataFrame | None = None

    def _load_raw(self) -> pd.DataFrame:
        """Load and parse the raw Excel dataset."""
        if self._data is not None:
            return self._data

        if not DATASET_PATH.exists():
            logger.warning(
                f"Professor dataset not found at {DATASET_PATH}. "
                f"Copy dataset.xlsx to data_cache/ as professor_dataset.xlsx"
            )
            return pd.DataFrame()

        raw = pd.read_excel(DATASET_PATH, sheet_name="data", header=None)
        data = raw.iloc[5:].copy().reset_index(drop=True)

        # Parse dates from column 16
        dates = pd.to_datetime(data[16], errors="coerce")
        valid = dates.notna().values
        data = data[valid].copy()
        
        # FIXED: Ensure timezone-naive DatetimeIndex for safe concatenation downstream
        idx = pd.DatetimeIndex(dates[valid])
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        data.index = idx

        self._data = data
        return data

    def _resolve_columns(self, data: pd.DataFrame) -> dict[int, str]:
        """
        Map asset names to column indices.
        Searches the raw Excel header rows by keyword first; falls back to
        ASSET_COLS_FALLBACK integer indices if no keyword match is found.
        """
        col_map: dict[int, str] = {}

        for asset_name, keywords in ASSET_SEARCH.items():
            found = None
            # Search first 5 rows of the original raw file for header keywords
            for row_idx in range(5):
                try:
                    row = pd.read_excel(
                        DATASET_PATH, sheet_name="data",
                        header=None, nrows=row_idx + 1
                    ).iloc[row_idx].astype(str).str.lower()
                except Exception:
                    break
                for col_idx, val in enumerate(row):
                    if any(kw.lower() in val for kw in keywords):
                        found = col_idx
                        break
                if found is not None:
                    break

            if found is not None:
                col_map[found] = asset_name
            else:
                # Fall back to known integer index
                fallback = {v: k for k, v in ASSET_COLS_FALLBACK.items()}.get(asset_name)
                if fallback is not None:
                    col_map[fallback] = asset_name
                    logger.debug(f"Using integer fallback col {fallback} for {asset_name}")

        if not col_map:
            logger.warning("String search found no columns — using integer fallback map entirely.")
            col_map = ASSET_COLS_FALLBACK.copy()

        logger.info(f"Professor column map: {col_map}")
        return col_map

    def get_private_assets(self) -> pd.DataFrame:
        """
        Returns quarterly return series for private market assets.
        Uses unsmoothed PE and RE.
        """
        cache_path = DATA_CACHE_DIR / "professor_assets.parquet"
        if self.use_cache and cache_path.exists():
            logger.info("Cache hit: professor_assets")
            return pd.read_parquet(cache_path)

        data = self._load_raw()
        if data.empty:
            return pd.DataFrame()

        # Build column map: try string-based header search first, fall back to integers.
        col_map = self._resolve_columns(data)

        assets = {}
        for col, name in col_map.items():
            s = pd.to_numeric(data[col], errors="coerce")
            s.name = name
            assets[name] = s

        df = pd.DataFrame(assets)
        df.index.name = "date"
        df = df.sort_index()

        # Drop smoothed versions — keep only unsmoothed PE and RE
        df = df.drop(
            columns=["private_equity", "private_real_estate"],
            errors="ignore"
        )

        df.to_parquet(cache_path)
        logger.info(
            f"Professor assets loaded: {df.shape[1]} assets, "
            f"{len(df)} quarters, "
            f"{df.index[0].date()} to {df.index[-1].date()}"
        )
        return df
