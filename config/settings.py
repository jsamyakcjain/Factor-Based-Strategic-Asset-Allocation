from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
ROOT_DIR:       Path = Path(__file__).parent.parent
DATA_CACHE_DIR: Path = ROOT_DIR / "data_cache"
OUTPUTS_DIR:    Path = ROOT_DIR / "outputs"

DATA_CACHE_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Credentials ────────────────────────────────────────────────────
FRED_API_KEY:  str | None = os.getenv("FRED_API_KEY")
WRDS_USERNAME: str | None = os.getenv("WRDS_USERNAME")

# ── Sample Period ───────────────────────────────────────────────────
START_DATE_TIER1: str = "1980-01-01"
START_DATE_TIER2: str = "2004-01-01"
END_DATE:         str = "2024-12-31"

# ── Asset Universe ──────────────────────────────────────────────────
PUBLIC_ASSETS: list[str] = [
    "us_large_cap",
    "us_small_cap",
    "em_equity",
    "long_treasury",
    "tips",
    "ig_credit",
    "hy_credit",
    "reits",
    "commodities",
]

PRIVATE_ASSETS: list[str] = [
    "private_equity",
    "private_credit",
    "private_real_estate",
    "infrastructure",
]

ASSET_NAMES: list[str] = PUBLIC_ASSETS + PRIVATE_ASSETS
N_ASSETS:    int       = len(ASSET_NAMES)

# ── Factor Definitions ──────────────────────────────────────────────
FACTOR_NAMES: list[str] = [
    "equity_premium",
    "term_premium",
    "credit_spread",
    "inflation",
    "liquidity",
]
N_FACTORS: int = len(FACTOR_NAMES)

# ── FRED Series ─────────────────────────────────────────────────────
FRED_SERIES: dict[str, str] = {
    "tbill_3m":       "TB3MS",
    "treasury_10y":   "GS10",
    "tips_breakeven": "T10YIE",
    "baa_yield":      "BAA",        # Moody's BAA corporate yield
    "aaa_yield":      "AAA",        # Moody's AAA corporate yield
    "cpi":            "CPIAUCSL",
    "nber_recession": "USREC",
}

# ── Benchmarks ──────────────────────────────────────────────────────
BENCHMARKS: dict[str, str] = {
    "us_large_cap":        "S&P 500 Total Return",
    "us_small_cap":        "Russell 2000 Total Return",
    "em_equity":           "MSCI EM Total Return",
    "long_treasury":       "Bloomberg Treasury 20+ Total Return",
    "tips":                "Bloomberg US TIPS Total Return",
    "ig_credit":           "Bloomberg US Corp IG Total Return",
    "hy_credit":           "ICE BofA US HY Total Return",
    "reits":               "FTSE NAREIT All Equity Total Return",
    "commodities":         "S&P GSCI Total Return",
    "private_equity":      "Cambridge Associates US PE Index",
    "private_credit":      "Cambridge Associates US Private Credit",
    "private_real_estate": "Cambridge Associates US Real Estate",
    "infrastructure":      "Cambridge Associates Global Infrastructure",
}

# ── CA Files ────────────────────────────────────────────────────────
CA_FILES: dict[str, str] = {
    "private_equity":       "ca_private_equity.csv",
    "private_credit":       "ca_private_credit.csv",
    "private_real_estate":  "ca_private_real_estate.csv",
    "infrastructure":       "ca_infrastructure.csv",
}

# ── JPM 2026 LTCMA ──────────────────────────────────────────────────
JPM_LTCMA: dict[str, float] = {
    "us_large_cap":        0.067,
    "us_small_cap":        0.069,
    "em_equity":           0.078,
    "long_treasury":       0.052,
    "tips":                0.043,
    "ig_credit":           0.052,
    "hy_credit":           0.061,
    "reits":               0.088,
    "commodities":         0.046,
    "private_equity":      0.109,
    "private_credit":      0.082,
    "private_real_estate": 0.073,
    "infrastructure":      0.068,
}

# ── Portfolio Parameters ────────────────────────────────────────────
RISK_FREE_RATE:    float = 0.031
RISK_AVERSION:     float = 3.0
MIN_WEIGHT:        float = 0.0
MAX_WEIGHT:        float = 0.35
MAX_PRIVATE_TOTAL: float = 0.30
FREQUENCY:         str   = "QE"

# ── Geltner Bounds ──────────────────────────────────────────────────
ALPHA_MIN: float = 0.05
ALPHA_MAX: float = 0.95

# ── Colors ──────────────────────────────────────────────────────────
FACTOR_COLORS: dict[str, str] = {
    "equity_premium": "#C0392B",
    "term_premium":   "#1E3A8A",
    "credit_spread":  "#D97706",
    "inflation":      "#1A7A4A",
    "liquidity":      "#6D28D9",
    "idiosyncratic":  "#94A3B8",
}

ASSET_COLORS: dict[str, str] = {
    "us_large_cap":        "#1E3A8A",
    "us_small_cap":        "#2563EB",
    "em_equity":           "#F97316",
    "long_treasury":       "#059669",
    "tips":                "#10B981",
    "ig_credit":           "#F59E0B",
    "hy_credit":           "#EF4444",
    "reits":               "#8B5CF6",
    "commodities":         "#EC4899",
    "private_equity":      "#0F172A",
    "private_credit":      "#374151",
    "private_real_estate": "#6B7280",
    "infrastructure":      "#9CA3AF",
}

METHOD_COLORS: dict[str, str] = {
    "mvo":          "#1E3A8A",
    "risk_parity":  "#D97706",
    "hrp":          "#1A7A4A",
    "equal_weight": "#94A3B8",
    "sixty_forty":  "#6B7280",
}
