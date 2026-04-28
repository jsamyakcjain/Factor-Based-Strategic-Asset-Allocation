from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import DATA_CACHE_DIR

logger = logging.getLogger(__name__)

DATASET_PATH = DATA_CACHE_DIR / "professor_dataset.xlsx"

# Column indices in the dataset (0-indexed from raw header row)
ASSET_COLS = {
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
        data.index = dates[valid].values

        self._data = data
        return data

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

        assets = {}
        for col, name in ASSET_COLS.items():
            s = data[col].astype(float)
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
