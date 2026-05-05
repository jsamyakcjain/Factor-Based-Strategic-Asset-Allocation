"""
Factor-Based SAA Analytics Report
Run from project root: python generate_analytics.py
Outputs: analytics_report.xlsx
"""
from __future__ import annotations

import contextlib
import logging
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from openpyxl import Workbook
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side, numbers as xl_numbers)
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage

# ══════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════
from data.data_manager import DataManager
from factors.factor_model import OLSFactorModel, QuantileFactorModel
from models.covariance import POETCovariance
from models.returns import ExpectedReturns
from portfolio.mvo import MVO
from portfolio.risk_parity import FactorRiskParity
from portfolio.hrp import EnhancedHRP
from portfolio.risk_decomp import FactorRiskDecomposition
from config.settings import (
    ASSET_DISPLAY_NAMES,
    FACTOR_NAMES,
    RISK_AVERSION,
    RISK_FREE_RATE,
    START_DATE_TIER2,
    DATA_CACHE_DIR
)

def dn(a): 
    return ASSET_DISPLAY_NAMES.get(a, a.replace("_", " ").title())

# ══════════════════════════════════════════════════════════════════
# LOAD PIPELINE & BUILD ALIGNED DATASET
# ══════════════════════════════════════════════════════════════════
print("Loading data and running models...")

# ── 1. Load or Build Data ──────────────────────────────────────────
dm = DataManager(use_cache=True)

# Check if ALL required cache files exist
required_cache_files = [
    "factor_returns_t1.parquet",
    "asset_returns_t1.parquet",
    "asset_returns_t1_complete.parquet",
    "factor_returns_t2.parquet",
    "asset_returns_t2.parquet",
    "recession.parquet",
    "rf.parquet",
    "credit_liquidity.parquet"
]

cache_exists = all((DATA_CACHE_DIR / f).exists() for f in required_cache_files)

if cache_exists:
    print("Loading from cache...")
    dm.load_cached()
else:
    print("Cache not found or incomplete. Building data from scratch...")
    print("This will take a few minutes on first run...")
    dm.build()

# ... rest continues

# ── 2. Data Alignment Block ────────────────────────────────────────
print("Aligning data to quarterly frequency...")

# dm.factor_returns_t1 is already quarterly (aggregated in DataManager).
# equity_premium / term_premium were compounded geometrically (they are returns).
# credit_spread / inflation / liquidity (PS) were summed arithmetically (level changes).
q_factors_t1 = dm.factor_returns_t1.copy()
q_assets_t1 = (1 + dm.asset_returns_t1_complete).resample('QE').prod() - 1

# Use credit_liquidity (quality spread change) as the single liquidity factor for all
# assets so that the POET factor covariance is internally consistent: every beta in B
# is estimated against the same series, so B Sigma_f B' is well-defined.
# (Using PS for equities and credit_liquidity for bonds would mix scales and break POET.)
# OLSFactorModel still receives credit_liquidity separately to handle standardisation.
q_factors_t1['liquidity'] = dm.credit_liquidity.reindex(q_factors_t1.index).fillna(0)

# Tier 2 is already quarterly
t2_assets = dm.asset_returns_t2

# FIXED: Identify unique assets (Tier 2 takes priority for overlapping assets)
t1_only_assets = [a for a in q_assets_t1.columns if a not in t2_assets.columns]
print(f"Tier 1 only assets: {t1_only_assets}")
print(f"Tier 2 assets: {list(t2_assets.columns)}")

# Combine without duplicates
combined_assets = pd.concat([
    q_assets_t1[t1_only_assets],  # Only T1-unique assets
    t2_assets                       # All T2 assets (includes overlapping ones)
], axis=1)

# Create master aligned dataframe
master_df = pd.concat([q_factors_t1, combined_assets], axis=1)
master_df = master_df.dropna()  # Complete-case only

# Extract perfectly aligned components
final_q_factors = master_df[q_factors_t1.columns]
final_q_assets = master_df[combined_assets.columns]

# Validation
print(f"✓ Factor data: {final_q_factors.shape}, {final_q_factors.index[0].date()} to {final_q_factors.index[-1].date()}")
print(f"✓ Asset data:  {final_q_assets.shape}, {final_q_assets.index[0].date()} to {final_q_assets.index[-1].date()}")
print(f"✓ Asset names: {list(final_q_assets.columns)}")

# Verify no duplicates
assert len(final_q_assets.columns) == len(set(final_q_assets.columns)), "Duplicate columns detected!"
assert final_q_factors.index.equals(final_q_assets.index), "Date index mismatch!"
# ── 3. Factor Model Estimation ─────────────────────────────────────
print("Running OLS factor model...")
ols = OLSFactorModel(final_q_factors, final_q_assets, credit_liquidity=dm.credit_liquidity)
result = ols.fit()

print("Running quantile factor model...")
qm = QuantileFactorModel(final_q_factors, final_q_assets, credit_liquidity=dm.credit_liquidity, quantiles=[0.10, 0.50, 0.90])
qm.fit()

print("\n=== SCALE DIAGNOSTICS ===")
print("Factor magnitudes (should be ~0.01-0.05):")
print(final_q_factors.describe().loc[['mean', 'std']].round(6))
print("\nAsset magnitudes (should be ~0.01-0.05):")
print(final_q_assets.describe().loc[['mean', 'std']].round(6))
print()

# If means/stds differ by ~100x, scale mismatch confirmed
factor_scale = final_q_factors['equity_premium'].std()
asset_scale = final_q_assets['us_large_cap'].std()
ratio = asset_scale / factor_scale
print(f"Scale ratio (asset/factor): {ratio:.2f}")
print(f"If ratio is ~100, factors need to be multiplied by 100")
print(f"If ratio is ~0.01, assets need to be multiplied by 100")

# ── 4. Covariance Estimation ───────────────────────────────────────
# ── 4. Covariance Estimation ───────────────────────────────────────
print("Building POET covariance matrix...")

# Filter data to start from Tier 2 start date (2004-Q4)
start_dt = pd.Timestamp(START_DATE_TIER2)
filtered_factors = final_q_factors[final_q_factors.index >= start_dt]
filtered_assets = final_q_assets[final_q_assets.index >= start_dt]

print(f"POET using data from {filtered_factors.index[0].date()} to {filtered_factors.index[-1].date()}")
print(f"POET observations: {len(filtered_factors)}")

poet = POETCovariance(
    factor_returns=filtered_factors,
    asset_returns=filtered_assets,
    beta_matrix=result.betas,
    decay=0.94,
    start_date=START_DATE_TIER2,
    alphas=result.alphas
)
poet.fit()
cov = poet.as_dataframe()

# Factor covariance — use POET's own EWM-weighted estimate (decay=0.94) so that
# syst_i = β_i' Σ_f β_i is consistent with cov[i,i] = syst_i + resid_i.
# Must index by FACTOR_NAMES (not F.columns) since poet.factor_cov is built
# on FACTOR_NAMES only, regardless of any extra columns in filtered_factors.
F = filtered_factors.astype(float)
factor_cov = pd.DataFrame(
    poet.factor_cov,
    index=FACTOR_NAMES,
    columns=FACTOR_NAMES,
)

# ── 5. Expected Returns ────────────────────────────────────────────
print("Loading expected returns (JPM LTCMA)...")
er = ExpectedReturns(assets=list(cov.index))
mu = er.excess_quarterly()

# ── 6. Diagnostic: Check MVO Inputs ────────────────────────────────
print("\n=== MVO Input Diagnostics ===")
print(f"Expected returns (mu): {len(mu)} assets")
print(f"  Range: {mu.min()*100:.2f}% to {mu.max()*100:.2f}% quarterly")
print(f"Covariance matrix: {cov.shape}")
print(f"  Min eigenvalue: {np.linalg.eigvalsh(cov.values).min():.6f}")
print(f"  Condition number: {np.linalg.cond(cov.values):.1f}")
print(f"Risk aversion: {RISK_AVERSION}")

# ── 7. Portfolio Optimization ──────────────────────────────────────
print("\nRunning portfolio optimizers...")

# MVO
mvo = MVO(mu, cov, risk_aversion=RISK_AVERSION)
mvo_w = mvo.fit()
print(f"MVO status: {mvo.status}, sum of weights: {mvo_w.sum():.4f}")

# Risk Parity — factor-level ERC: equalises each factor's contribution to
# systematic variance (target 20% per factor) instead of asset-level ERC.
rp = FactorRiskParity(cov, result.betas, factor_cov)
rp_w = rp.fit()

# Enhanced HRP — pass factor_cov so allocation uses BΣ_fB' (systematic only)
hrp = EnhancedHRP(cov, result.betas, factor_cov=factor_cov)
hrp_w = hrp.fit()

# ── 8. Risk Decomposition Setup ────────────────────────────────────
rd = FactorRiskDecomposition(
    beta_matrix=result.betas,
    factor_cov=factor_cov,
    asset_cov=cov
)

# ── 9. Asset Classification ────────────────────────────────────────
assets = list(cov.index)
eq_assets = [a for a in assets if a in ["us_large_cap", "us_mid_cap", "us_small_cap", "em_equity", "reits"]]
bnd_assets = [a for a in assets if a in ["long_treasury", "tips", "ig_credit"]]

# ── 10. Portfolio Dictionary ───────────────────────────────────────
portfolios = {
    "Equal Weight": rd.equal_weight(assets),
    "60/40":        rd.sixty_forty(eq_assets, bnd_assets),
    "MVO":          mvo_w,
    "Risk Parity":  rp_w,
    "Enhanced HRP": hrp_w,
}

# ── 11. Risk Decomposition ─────────────────────────────────────────
print("\nComputing factor risk decomposition...")
decomp = rd.compare(portfolios)

# ── 12. Full-sample returns — used for charts and stress testing ────
asset_rets = final_q_assets.dropna()
w_6040_aligned = portfolios["60/40"].reindex(asset_rets.columns).fillna(0)
if w_6040_aligned.sum() != 0:
    w_6040_aligned = w_6040_aligned / w_6040_aligned.sum()

def _full_port_ret(w: pd.Series) -> pd.Series:
    w_al = w.reindex(asset_rets.columns).fillna(0)
    if w_al.sum() != 0:
        w_al = w_al / w_al.sum()
    return asset_rets @ w_al

full_returns  = {name: _full_port_ret(w) for name, w in portfolios.items()}
benchmark_ret = full_returns["60/40"]
print(f"✓ Full-sample benchmark: {len(benchmark_ret)} quarters, {benchmark_ret.mean()*4*100:.2f}% ann.")

# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST  (expanding window, 20-quarter burn-in)
# Eliminates look-ahead bias from performance statistics.
# Sharpe/Sortino/Vol/TE are computed on these out-of-sample returns.
# Charts and stress testing still use full_returns (final weights on
# entire history) so the GFC 2008 crash is visible in the report.
# ══════════════════════════════════════════════════════════════════

def walk_forward_backtest(
    factor_data: pd.DataFrame,
    asset_data:  pd.DataFrame,
    credit_liq:  pd.Series,
    burn_in:     int = 20,
) -> dict[str, pd.Series]:
    n_periods = len(asset_data)
    all_cols  = list(asset_data.columns)
    eq_cols   = [a for a in all_cols if a in ["us_large_cap","us_mid_cap","us_small_cap","em_equity","reits"]]
    bnd_cols  = [a for a in all_cols if a in ["long_treasury","tips","ig_credit"]]

    oos: dict[str, list[float]] = {k: [] for k in ["Equal Weight","60/40","MVO","Risk Parity","Enhanced HRP"]}
    idx: list = []
    n_oos = n_periods - burn_in
    print(f"  {n_oos} out-of-sample quarters (burn-in = {burn_in}Q)...")

    for t in range(burn_in, n_periods):
        if (t - burn_in) % 10 == 0:
            print(f"    step {t-burn_in+1}/{n_oos}  ({asset_data.index[t].date()})")

        F_t  = factor_data.iloc[:t]
        R_t  = asset_data.iloc[:t]
        cl_t = credit_liq.reindex(F_t.index).fillna(0)
        next_ret = asset_data.iloc[t]
        idx.append(asset_data.index[t])

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ols_t  = OLSFactorModel(F_t, R_t, credit_liquidity=cl_t)
                res_t  = ols_t.fit()
                poet_t = POETCovariance(
                    factor_returns=F_t, asset_returns=R_t,
                    beta_matrix=res_t.betas, decay=0.94, alphas=res_t.alphas,
                )
                poet_t.fit()
                cov_t    = poet_t.as_dataframe()
                mu_t     = ExpectedReturns(assets=list(cov_t.index)).excess_quarterly()
                assets_t = list(cov_t.index)
                eq_t     = [a for a in assets_t if a in eq_cols]
                bnd_t    = [a for a in assets_t if a in bnd_cols]

                w_ew = pd.Series(1.0 / len(assets_t), index=assets_t)

                if eq_t and bnd_t:
                    w64 = {a: 0.60 / len(eq_t) for a in eq_t}
                    w64.update({a: 0.40 / len(bnd_t) for a in bnd_t})
                    w_6040_t = pd.Series(w64)
                else:
                    w_6040_t = w_ew.copy()

                try:
                    w_mvo = MVO(mu_t, cov_t, risk_aversion=RISK_AVERSION).fit()
                except Exception:
                    w_mvo = w_ew.copy()

                try:
                    fcov_t = pd.DataFrame(poet_t.factor_cov, index=FACTOR_NAMES, columns=FACTOR_NAMES)
                    w_rp = FactorRiskParity(cov_t, res_t.betas, fcov_t).fit()
                except Exception:
                    w_rp = w_ew.copy()

                try:
                    fcov_t = pd.DataFrame(poet_t.factor_cov, index=FACTOR_NAMES, columns=FACTOR_NAMES)
                    w_hrp = EnhancedHRP(cov_t, res_t.betas, factor_cov=fcov_t).fit()
                except Exception:
                    w_hrp = w_ew.copy()

            wts = {"Equal Weight": w_ew, "60/40": w_6040_t,
                   "MVO": w_mvo, "Risk Parity": w_rp, "Enhanced HRP": w_hrp}

        except Exception as e:
            print(f"    Warning: step {t} failed ({e}) — using equal weight")
            w_fb = pd.Series(1.0 / len(all_cols), index=all_cols)
            wts  = {k: w_fb for k in oos}

        for pname, wt in wts.items():
            w_al = wt.reindex(all_cols).fillna(0)
            if w_al.sum() > 0:
                w_al /= w_al.sum()
            oos[pname].append(float(next_ret @ w_al))

    return {k: pd.Series(v, index=idx, name=k) for k, v in oos.items()}


print("\nRunning walk-forward backtest...")
logging.disable(logging.WARNING)   # suppress RiskParity convergence warnings inside loop
wf_rets      = walk_forward_backtest(final_q_factors, final_q_assets, dm.credit_liquidity, burn_in=20)
logging.disable(logging.NOTSET)    # restore normal logging
wf_benchmark = wf_rets["60/40"]
print(f"✓ OOS period: {wf_rets['60/40'].index[0].date()} to {wf_rets['60/40'].index[-1].date()}")

# ══════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS  (OOS returns for Sharpe/Vol/TE,
#                       full-sample returns for charts & stress test)
# ══════════════════════════════════════════════════════════════════

def port_metrics(name: str) -> tuple:
    pr_oos  = wf_rets[name].dropna()        # out-of-sample — honest stats
    pr_full = full_returns[name]            # full history   — charts & stress test
    rf_ann  = RISK_FREE_RATE * 100

    ann_ret = float(pr_oos.mean()) * 4 * 100
    ann_vol = float(pr_oos.std())  * np.sqrt(4) * 100
    sharpe  = (ann_ret - rf_ann) / ann_vol if ann_vol > 0 else 0

    downside_returns = np.minimum(pr_oos.values, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(4) * 100)
    sortino = (ann_ret - rf_ann) / downside_dev if downside_dev > 0 else 0

    cum      = (1 + pr_oos).cumprod()
    roll_max = cum.cummax()
    max_dd   = float(((cum - roll_max) / roll_max).min()) * 100
    calmar   = ann_ret / abs(max_dd) if max_dd < 0 else 0

    diff           = pr_oos - wf_benchmark.reindex(pr_oos.index).fillna(0)
    tracking_error = float(diff.std()) * np.sqrt(4) * 100

    # pr_full returned so downstream charts and stress testing cover full 20-year history
    return ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, tracking_error, pr_full


print("Computing portfolio metrics...")
metrics = {name: port_metrics(name) for name in portfolios}

# ══════════════════════════════════════════════════════════════════
# STRESS TESTING
# ══════════════════════════════════════════════════════════════════

stress_periods = {
    "GFC (2008 Q3 - 2009 Q1)": ("2008-07-01", "2009-03-31"),
    "COVID Crash (2020 Q1)":   ("2020-01-01", "2020-06-30"),
    "Rate Shock (2022)":       ("2022-01-01", "2022-12-31"),
}

def period_return(port_ret: pd.Series, start: str, end: str) -> float:
    """Calculate cumulative return over a specific period."""
    mask = (port_ret.index >= pd.Timestamp(start)) & (port_ret.index <= pd.Timestamp(end))
    sub = port_ret[mask]
    if len(sub) == 0:
        return float("nan")
    return float((1 + sub).prod() - 1) * 100


# ══════════════════════════════════════════════════════════════════
# EXCEL STYLING HELPERS
# ══════════════════════════════════════════════════════════════════

NAVY   = "0A1628"
BLUE   = "1A3A5C"
PANEL  = "16202E"
MUTED  = "7A90A8"
RED    = "B03A2E"
AMBER  = "B7770D"
GREEN  = "1E7B45"

PORT_HEX = {
    "Equal Weight": "7F8C8D",
    "60/40": "B03A2E",
    "MVO": "2980B9",
    "Risk Parity": "E67E22",
    "Enhanced HRP": "1E7B45",
}

FACTOR_HEX = {
    "Equity Premium": "C0392B",
    "Term Premium": "2980B9",
    "Credit Spread": "E67E22",
    "Inflation": "27AE60",
    "Liquidity": "8E44AD",
    "Idiosyncratic": "566573",
}

def fill(hex_): 
    return PatternFill("solid", start_color=hex_, end_color=hex_)

def font(hex_="F0F4F8", sz=10, bold=False): 
    return Font(name="Arial", color=hex_, size=sz, bold=bold)

def aln(h="left", v="center", wrap=False): 
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

def border_thin(): 
    s = Side(style="thin", color="1E2D3D")
    return Border(left=s, right=s, top=s, bottom=s)

def header_row(ws, row, labels, widths=None, bg=NAVY):
    for c, label in enumerate(labels, 1):
        cell = ws.cell(row=row, column=c, value=label)
        cell.fill = fill(bg)
        cell.font = font("F0F4F8", 10, True)
        cell.alignment = aln("center")
        cell.border = border_thin()
        if widths:
            ws.column_dimensions[get_column_letter(c)].width = widths[c-1]

def data_row(ws, row, values, bg=PANEL, fmt_map=None):
    for c, val in enumerate(values, 1):
        cell = ws.cell(row=row, column=c, value=val)
        cell.fill = fill(bg)
        cell.font = font("D0D8E4", 10)
        cell.alignment = aln("center")
        cell.border = border_thin()
        if fmt_map and c in fmt_map:
            cell.number_format = fmt_map[c]

def section_title(ws, row, col, text, span=1):
    cell = ws.cell(row=row, column=col, value=text)
    cell.fill = fill(BLUE)
    cell.font = font("F0F4F8", 11, True)
    cell.alignment = aln("left")
    cell.border = border_thin()
    if span > 1:
        ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col+span-1)

def img_to_xl(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0e1117", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return XLImage(buf)

def fig_style():
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#16202E",
        "axes.edgecolor": "#1E2D3D",
        "axes.labelcolor": "#D0D8E4",
        "xtick.color": "#D0D8E4",
        "ytick.color": "#D0D8E4",
        "text.color": "#D0D8E4",
        "grid.color": "#1E2D3D",
        "grid.linewidth": 0.6,
        "font.family": "Arial",
        "font.size": 9,
    })

fig_style()

# Alternating background colors
bg_alt = [PANEL, "111A27"]

# Factor column keys
factor_col_keys = list(factor_cov.index)

# ══════════════════════════════════════════════════════════════════
# BUILD WORKBOOK
# ══════════════════════════════════════════════════════════════════

wb = Workbook()
wb.remove(wb.active)

print("Building Excel workbook...")

# ═══════════════════════════════════════════════════════════════════
# SHEET 1: EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════

ws1 = wb.create_sheet("Executive Summary")
ws1.sheet_view.showGridLines = False
ws1.column_dimensions["A"].width = 32
for col in ["B", "C", "D", "E", "F"]:
    ws1.column_dimensions[col].width = 18

ws1.merge_cells("A1:F1")
c = ws1["A1"]
c.value = "FACTOR-BASED STRATEGIC ASSET ALLOCATION — ANALYTICS REPORT"
c.fill = fill(NAVY)
c.font = font("F0F4F8", 14, True)
c.alignment = aln("center")

ws1.merge_cells("A2:F2")
c = ws1["A2"]
c.value = "POET Covariance  |  JPMorgan 2026 LTCMA  |  Enhanced HRP  |  13 Asset Classes  |  Full History: 2004 Q4 – 2024 Q4  |  OOS Backtest: 2009 Q4 – 2024 Q4"
c.fill = fill(BLUE)
c.font = font(MUTED, 10)
c.alignment = aln("center")

ws1.row_dimensions[3].height = 8

# Key Finding
erp_60 = float(decomp.loc["60/40", "Equity Premium"]) if "60/40" in decomp.index else 0
erp_hrp = float(decomp.loc["Enhanced HRP", "Equity Premium"]) if "Enhanced HRP" in decomp.index else 0
ws1.merge_cells("A4:F4")
c = ws1["A4"]
c.value = (
    f"Key Finding: A conventional 60/40 portfolio allocates {erp_60:.1f}% of total risk to Equity Premium. "
    f"Enhanced HRP reduces this to {erp_hrp:.1f}% — a {erp_60-erp_hrp:.1f} pp reduction — "
    f"while maintaining balanced systematic risk exposure across all five factors."
)
c.fill = fill(BLUE)
c.font = font("F0F4F8", 10)
c.alignment = aln("left", wrap=True)
ws1.row_dimensions[4].height = 36
ws1.row_dimensions[5].height = 8

# Performance Metrics Table
section_title(ws1, 6, 1, "Portfolio Performance Metrics (Walk-Forward Out-of-Sample, 2009 Q4 – 2024 Q4)", 8)
header_row(
    ws1, 7,
    ["Portfolio", "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio", "Tracking Error"],
    [32, 18, 18, 18, 18, 18, 18, 18]
)

for i, (name, w) in enumerate(portfolios.items()):
    ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, tracking_error, _ = metrics[name]
    bg = bg_alt[i % 2]
    data_row(
        ws1, 8+i,
        [name, ann_ret/100, ann_vol/100, sharpe, sortino, max_dd/100, calmar, tracking_error/100],
        bg=bg,
        fmt_map={2: "0.0%", 3: "0.0%", 4: "0.00", 5: "0.00", 6: "0.0%", 7: "0.00", 8: "0.0%"}
    )
    ws1.cell(8+i, 1).font = font("F0F4F8", 10, True)
    ws1.cell(8+i, 1).alignment = aln("left")

ws1.row_dimensions[13].height = 8

# Factor Risk Attribution
section_title(ws1, 14, 1, "Factor Risk Attribution (% of Total Portfolio Risk)", 6)
factor_cols = [c for c in ["Equity Premium", "Term Premium", "Credit Spread", "Inflation", "Liquidity", "Idiosyncratic"] if c in decomp.columns]
header_row(ws1, 15, ["Portfolio"] + factor_cols, [32] + [18]*len(factor_cols))

for i, (name, _) in enumerate(portfolios.items()):
    if name not in decomp.index:
        continue
    row_data = [name] + [decomp.loc[name, c]/100 for c in factor_cols]
    bg = bg_alt[i % 2]
    data_row(ws1, 16+i, row_data, bg=bg, fmt_map={j+2: "0.0%" for j in range(len(factor_cols))})
    ws1.cell(16+i, 1).font = font("F0F4F8", 10, True)
    ws1.cell(16+i, 1).alignment = aln("left")
    
    # Color-code Equity Premium concentration
    erp_val = decomp.loc[name, "Equity Premium"] if "Equity Premium" in decomp.columns else 0
    erp_cell = ws1.cell(16+i, 2)
    if erp_val > 70:
        erp_cell.font = Font(name="Arial", color=RED, size=10, bold=True)
    elif erp_val > 55:
        erp_cell.font = Font(name="Arial", color=AMBER, size=10, bold=True)
    else:
        erp_cell.font = Font(name="Arial", color=GREEN, size=10, bold=True)

ws1.row_dimensions[21].height = 8

# POET Diagnostics
B_factors = result.betas.reindex(columns=factor_col_keys).values.astype(float)
total_var = np.trace(cov.values)
factor_var = np.trace(B_factors @ factor_cov.values @ B_factors.T)
sys_share = factor_var / total_var
idio_share = 1 - sys_share

section_title(ws1, 22, 1, "POET Covariance Diagnostics", 4)
header_row(ws1, 23, ["Parameter", "Value", "Description"], [32, 18, 40])

diag_rows = [
    ("Sample Observations (T)", len(F), "Complete-case balanced panel"),
    ("Asset Classes (p)", cov.shape[0], "Number of assets"),
    ("Systematic Factors (k)", F.shape[1], "Factor model dimension"),
    ("Systematic Variance Share", f"{sys_share:.1%}", "Variance explained by factors"),
    ("Idiosyncratic Share", f"{idio_share:.1%}", "Residual variance"),
]

for i, (param, val, desc) in enumerate(diag_rows):
    bg = bg_alt[i % 2]
    ws1.cell(24+i, 1, param).fill = fill(bg)
    ws1.cell(24+i, 1).font = font("F0F4F8", 10)
    ws1.cell(24+i, 1).border = border_thin()
    ws1.cell(24+i, 1).alignment = aln("left")
    
    ws1.cell(24+i, 2, val).fill = fill(bg)
    ws1.cell(24+i, 2).font = font("D0D8E4", 10)
    ws1.cell(24+i, 2).border = border_thin()
    ws1.cell(24+i, 2).alignment = aln("center")
    
    ws1.cell(24+i, 3, desc).fill = fill(bg)
    ws1.cell(24+i, 3).font = font(MUTED, 9)
    ws1.cell(24+i, 3).border = border_thin()
    ws1.cell(24+i, 3).alignment = aln("left")

# ═══════════════════════════════════════════════════════════════════
# SHEET 2: BACKTEST SUMMARY
# ═══════════════════════════════════════════════════════════════════

ws_bt = wb.create_sheet("Backtest Summary", 1)
ws_bt.sheet_view.showGridLines = False
ws_bt.column_dimensions["A"].width = 28
for col in ["B", "C", "D", "E", "F", "G", "H"]:
    ws_bt.column_dimensions[col].width = 18

section_title(ws_bt, 1, 1, "Backtest Summary Statistics — Walk-Forward OOS (2009 Q4 – 2024 Q4, 20Q burn-in)", 8)
header_row(
    ws_bt, 2,
    ["Portfolio", "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio", "Tracking Error vs 60/40"],
    [28, 16, 18, 16, 16, 16, 16, 26]
)

for i, (name, _) in enumerate(portfolios.items()):
    ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, te, pr = metrics[name]
    bg = bg_alt[i % 2]
    data_row(
        ws_bt, 3+i,
        [name, ann_ret/100, ann_vol/100, sharpe, sortino, max_dd/100, calmar, te/100],
        bg=bg,
        fmt_map={2: "0.0%", 3: "0.0%", 4: "0.00", 5: "0.00", 6: "0.0%", 7: "0.00", 8: "0.0%"}
    )
    ws_bt.cell(3+i, 1).font = font("F0F4F8", 10, True)
    ws_bt.cell(3+i, 1).alignment = aln("left")
    
    # Color-code Sharpe ratio
    sharpe_cell = ws_bt.cell(3+i, 4)
    if sharpe >= 0.7:
        sharpe_cell.font = Font(name="Arial", color=GREEN, size=10, bold=True)
    elif sharpe >= 0.4:
        sharpe_cell.font = Font(name="Arial", color=AMBER, size=10)
    else:
        sharpe_cell.font = Font(name="Arial", color=RED, size=10)
    
    # Color-code drawdown
    dd_cell = ws_bt.cell(3+i, 6)
    if max_dd < -25:
        dd_cell.font = Font(name="Arial", color=RED, size=10, bold=True)
    elif max_dd < -15:
        dd_cell.font = Font(name="Arial", color=AMBER, size=10)
    else:
        dd_cell.font = Font(name="Arial", color=GREEN, size=10)

# Rolling Sharpe Chart
fig, ax = plt.subplots(figsize=(11, 4))
window = 8
rf_ann_pct = RISK_FREE_RATE * 100
for name in portfolios:
    pr = wf_rets[name]   # OOS returns — consistent with stats table
    roll_ret = pr.rolling(window).mean() * 4 * 100
    roll_vol = pr.rolling(window).std() * np.sqrt(4) * 100
    roll_sharpe = (roll_ret - rf_ann_pct) / roll_vol
    ax.plot(roll_sharpe.index, roll_sharpe.values, label=name, color="#"+PORT_HEX[name], linewidth=1.5)

ax.axhline(y=0, color="#7A90A8", linewidth=0.8, linestyle="--")
ax.axhline(y=0.5, color="#1E7B45", linewidth=0.8, linestyle=":", alpha=0.6, label="0.5 reference")
ax.set_title("Rolling 2-Year Sharpe Ratio — All Portfolios", fontsize=11, fontweight="bold", color="#F0F4F8")
ax.set_ylabel("Sharpe Ratio")
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(alpha=0.4)

img = img_to_xl(fig)
img.width = 770
img.height = 300
ws_bt.add_image(img, f"A{4+len(portfolios)+2}")

# Active Share Table
row_as = 4 + len(portfolios) + 20
section_title(ws_bt, row_as, 1, "Active Share vs 60/40 Benchmark", 4)
header_row(ws_bt, row_as+1, ["Portfolio", "Tracking Error (ann.)", "Active Share", "Classification"], [28, 22, 18, 30])

for i, (name, w) in enumerate(portfolios.items()):
    _, _, _, _, _, _, te, _ = metrics[name]
    all_assets = list(set(list(w.index) + list(portfolios["60/40"].index)))
    w1 = w.reindex(all_assets).fillna(0)
    w2 = portfolios["60/40"].reindex(all_assets).fillna(0)
    active_share = float(0.5 * np.sum(np.abs(w1.values - w2.values)))
    
    if active_share < 0.2:
        classification = "Closet Indexer"
    elif active_share < 0.5:
        classification = "Moderate Active"
    elif active_share < 0.8:
        classification = "High Conviction Active"
    else:
        classification = "Very High Active"
    
    bg = bg_alt[i % 2]
    data_row(ws_bt, row_as+2+i, [name, te/100, active_share, classification], bg=bg, fmt_map={2: "0.0%", 3: "0.0%"})
    ws_bt.cell(row_as+2+i, 1).font = font("F0F4F8", 10, True)
    ws_bt.cell(row_as+2+i, 1).alignment = aln("left")
    ws_bt.cell(row_as+2+i, 4).alignment = aln("left")
    
    # Color-code active share
    as_cell = ws_bt.cell(row_as+2+i, 3)
    if active_share > 0.5:
        as_cell.font = Font(name="Arial", color=GREEN, size=10, bold=True)
    elif active_share > 0.2:
        as_cell.font = Font(name="Arial", color=AMBER, size=10)
    else:
        as_cell.font = Font(name="Arial", color=RED, size=10)

# ═══════════════════════════════════════════════════════════════════
# SHEET 3: PORTFOLIO CHARTS
# ═══════════════════════════════════════════════════════════════════

ws2 = wb.create_sheet("Portfolio Analytics")
ws2.sheet_view.showGridLines = False
section_title(ws2, 1, 1, "Portfolio Analytics — Charts", 1)
ws2.column_dimensions["A"].width = 2
ws2.column_dimensions["B"].width = 2

port_names = list(portfolios.keys())
port_colors = ["#"+PORT_HEX[n] for n in port_names]
factor_cols_all = [c for c in ["Equity Premium", "Term Premium", "Credit Spread", "Inflation", "Liquidity", "Idiosyncratic"] if c in decomp.columns]
fcolors = ["#"+FACTOR_HEX.get(c, "566573") for c in factor_cols_all]

# Stacked Factor Risk Chart
fig, ax = plt.subplots(figsize=(10, 4))
bottom = np.zeros(len(port_names))
for col, color in zip(factor_cols_all, fcolors):
    vals = np.array([float(decomp.loc[n, col]) for n in port_names])
    bars = ax.bar(port_names, vals, bottom=bottom, color=color, label=col, width=0.5)
    for bar, b, v in zip(bars, bottom, vals):
        if abs(v) > 4:
            ax.text(bar.get_x()+bar.get_width()/2, b+v/2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    bottom += vals

ax.axhline(y=0, color="#1E2D3D", linewidth=0.8)
ax.set_ylabel("% of Total Portfolio Risk")
ax.set_title("Factor Risk Attribution by Construction Method", fontsize=11, fontweight="bold", color="#F0F4F8")
ax.legend(loc="upper right", fontsize=8, framealpha=0.3)
ax.set_ylim(-10, 115)
ax.grid(axis="y", alpha=0.4)

img = img_to_xl(fig)
img.width = 700
img.height = 300
ws2.add_image(img, "C3")

# Equity Premium Concentration
fig, ax = plt.subplots(figsize=(8, 3))
erp_vals = [float(decomp.loc[n, "Equity Premium"]) for n in port_names]
colors = [("#B03A2E" if v > 70 else "#B7770D" if v > 55 else "#1E7B45") for v in erp_vals]
bars = ax.bar(port_names, erp_vals, color=colors, width=0.5)

for bar, v in zip(bars, erp_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=bar.get_facecolor())

ax.axhline(y=50, color="#7A90A8", linewidth=1, linestyle="--", alpha=0.7, label="50% benchmark")
ax.set_ylabel("Equity Premium Share (%)")
ax.set_title("Equity Premium Concentration", fontsize=11, fontweight="bold", color="#F0F4F8")
ax.set_ylim(0, 105)
ax.legend(fontsize=8, framealpha=0.3)
ax.grid(axis="y", alpha=0.4)

img = img_to_xl(fig)
img.width = 560
img.height = 230
ws2.add_image(img, "C22")

# Max Drawdown & Diversification Ratio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

mdd_vals = [metrics[n][4] for n in port_names]
bars = ax1.bar(port_names, mdd_vals, color=port_colors, width=0.5)
for bar, v in zip(bars, mdd_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, v-0.5, f"{v:.1f}%", ha="center", va="top", fontsize=8, color="white")

ax1.set_title("Maximum Drawdown (%)", fontsize=10, fontweight="bold", color="#F0F4F8")
ax1.set_xticklabels(port_names, rotation=20, ha="right", fontsize=8)
ax1.grid(axis="y", alpha=0.4)

div_vals = []
for n, w in portfolios.items():
    wv = w.reindex(cov.index).fillna(0).values
    if wv.sum() != 0:
        wv = wv / wv.sum()
    wvol = float(np.sum(wv * np.sqrt(np.diag(cov.values))))
    pvol = float(np.sqrt(wv @ cov.values @ wv))
    div_vals.append(wvol/pvol if pvol > 0 else 1.0)

bars = ax2.bar(port_names, div_vals, color=port_colors, width=0.5)
for bar, v in zip(bars, div_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}x", ha="center", va="bottom", fontsize=8, color="white")

ax2.set_title("Diversification Ratio", fontsize=10, fontweight="bold", color="#F0F4F8")
ax2.set_xticklabels(port_names, rotation=20, ha="right", fontsize=8)
ax2.grid(axis="y", alpha=0.4)

plt.tight_layout()
img = img_to_xl(fig)
img.width = 700
img.height = 260
ws2.add_image(img, "C38")

# Cumulative Growth Chart
fig, ax = plt.subplots(figsize=(11, 4))
for n, w in portfolios.items():
    _, _, _, _, _, _, _, pr = metrics[n]
    cum = (1 + pr).cumprod() * 100
    ax.plot(cum.index, cum.values, label=n, color="#"+PORT_HEX[n], linewidth=1.5)

ax.set_title("Cumulative Growth of $100 — All Portfolios", fontsize=11, fontweight="bold", color="#F0F4F8")
ax.set_ylabel("Portfolio Value ($)")
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(alpha=0.4)

img = img_to_xl(fig)
img.width = 770
img.height = 310
ws2.add_image(img, "C58")

# ═══════════════════════════════════════════════════════════════════
# SHEET 4: FACTOR MODEL
# ═══════════════════════════════════════════════════════════════════

ws3 = wb.create_sheet("Factor Model")
ws3.sheet_view.showGridLines = False
ws3.column_dimensions["A"].width = 28
for col in ["B", "C", "D", "E", "F", "G", "H", "I", "J"]:
    ws3.column_dimensions[col].width = 16

section_title(ws3, 1, 1, "Asset-Level Factor Model Results", 9)
header_row(
    ws3, 2,
    ["Asset Class", "Equity Premium", "Term Premium", "Credit Spread", "Inflation", "Liquidity", "Alpha", "R-Squared", "Liq. Proxy"],
    [28, 16, 16, 16, 16, 16, 14, 14, 20]
)

from factors.factor_model import EQUITY_LIQUIDITY_ASSETS

for i, asset in enumerate(result.betas.index):
    r2v = float(result.r_squared.loc[asset]) if asset in result.r_squared.index else 0
    alpha = float(result.alphas.loc[asset]) if asset in result.alphas.index else 0
    beta_vals = [float(result.betas.loc[asset, f]) for f in factor_col_keys]
    liq_proxy = "Pastor-Stambaugh" if asset in EQUITY_LIQUIDITY_ASSETS else "AAA-BAA Spread"
    
    bg = bg_alt[i % 2]
    data_row(
        ws3, 3+i,
        [dn(asset)] + beta_vals + [alpha, r2v, liq_proxy],
        bg=bg,
        fmt_map={2: "0.000", 3: "0.000", 4: "0.000", 5: "0.000", 6: "0.000", 7: "0.0%", 8: "0.0%"}
    )
    ws3.cell(3+i, 1).alignment = aln("left")
    ws3.cell(3+i, 1).font = font("F0F4F8", 10)

# Stress Beta Table
n_assets = len(result.betas.index)
row_start = 4 + n_assets

section_title(ws3, row_start, 1, "Equity Premium Beta: OLS (Mean) vs Q10 Quantile Regression (Left-Tail Conditional)", 5)
header_row(
    ws3, row_start+1,
    ["Asset Class", "OLS Beta", "Q10 Beta", "Uplift", "Left-Tail Interpretation"],
    [28, 16, 20, 16, 36]
)

ols_erp = result.betas["equity_premium"]
q10_result = qm.results.get(0.10, None)
q10_erp = q10_result.betas["equity_premium"].reindex(ols_erp.index) if q10_result is not None else ols_erp.copy()

def _stress_label(uplift: float) -> str:
    # Q10 quantile regression = conditional 10th percentile of asset return given factors.
    # Positive uplift: left tail MORE factor-driven (co-crash with equity premium).
    # Negative uplift: left tail driven by IDIOSYNCRATIC shocks independent of equity premium.
    if uplift > 0.15:
        return "Co-crash: factor-driven downside"
    elif uplift > 0.05:
        return "Moderate co-crash tendency"
    elif uplift >= -0.05:
        return "Symmetric tail sensitivity"
    elif uplift >= -0.15:
        return "Idiosyncratic downside risk"
    else:
        return "Strongly idiosyncratic downside"

stress_rows = []
for asset in ols_erp.index:
    ols_b = float(ols_erp.loc[asset])
    q10_b = float(q10_erp.loc[asset]) if asset in q10_erp.index else ols_b
    uplift = q10_b - ols_b
    stress_rows.append((dn(asset), ols_b, q10_b, uplift, _stress_label(uplift)))

stress_rows.sort(key=lambda x: x[3], reverse=True)

for i, row in enumerate(stress_rows):
    bg = bg_alt[i % 2]
    data_row(ws3, row_start+2+i, list(row), bg=bg, fmt_map={2: "0.000", 3: "0.000", 4: "0.000"})
    ws3.cell(row_start+2+i, 1).alignment = aln("left")
    ws3.cell(row_start+2+i, 1).font = font("F0F4F8", 10)

# Factor Model Charts
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# R-squared chart
r2_sorted = result.r_squared.sort_values(ascending=True)
asset_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(r2_sorted)))
axes[0].barh([dn(a) for a in r2_sorted.index], r2_sorted.values, color=asset_colors)
axes[0].axvline(x=0.7, color="#B7770D", linestyle="--", linewidth=1, label="0.70 reference")

for i, v in enumerate(r2_sorted.values):
    axes[0].text(v+0.01, i, f"{v:.3f}", va="center", fontsize=8, color="#D0D8E4")

axes[0].set_title("Adjusted R-Squared by Asset Class", fontsize=10, fontweight="bold", color="#F0F4F8")
axes[0].set_xlim(0, 1.15)
axes[0].grid(axis="x", alpha=0.4)

# Equity beta comparison
sorted_idx = sorted(ols_erp.index, key=lambda a: float(ols_erp.loc[a]))
x = np.arange(len(sorted_idx))

axes[1].barh(x - 0.2, [float(ols_erp.loc[a]) for a in sorted_idx], height=0.35, label="OLS β (mean)", color="#2980B9")
axes[1].barh(x + 0.2, [float(q10_erp.loc[a]) if a in q10_erp.index else float(ols_erp.loc[a]) for a in sorted_idx], height=0.35, label="Q10 β (left-tail)", color="#B03A2E", alpha=0.85)

axes[1].set_yticks(x)
axes[1].set_yticklabels([dn(a) for a in sorted_idx], fontsize=8)
axes[1].set_title("Equity Beta: OLS Mean vs Q10 Left-Tail Conditional", fontsize=10, fontweight="bold", color="#F0F4F8")
axes[1].legend(fontsize=8, framealpha=0.3)
axes[1].grid(axis="x", alpha=0.4)

plt.tight_layout()
img = img_to_xl(fig)
img.width = 910
img.height = 310
ws3.add_image(img, f"A{row_start+2+len(stress_rows)+2}")

# ═══════════════════════════════════════════════════════════════════
# SHEET 5: STRESS TESTING
# ═══════════════════════════════════════════════════════════════════

ws4 = wb.create_sheet("Stress Testing")
ws4.sheet_view.showGridLines = False
ws4.column_dimensions["A"].width = 34
for col in ["B", "C", "D", "E", "F"]:
    ws4.column_dimensions[col].width = 20

section_title(ws4, 1, 1, "Portfolio Stress Testing — Defined Periods", 6)
header_row(
    ws4, 2,
    ["Portfolio"] + list(stress_periods.keys()) + ["Recovery: 2009 Q2 – 2010 Q4"],
    [34] + [20]*len(stress_periods) + [28]
)

for i, (name, w) in enumerate(portfolios.items()):
    _, _, _, _, _, _, _, pr = metrics[name]
    row_data = [name]
    for label, (start, end) in stress_periods.items():
        row_data.append(period_return(pr, start, end)/100)
    row_data.append(period_return(pr, "2009-04-01", "2010-12-31")/100)
    
    bg = bg_alt[i % 2]
    data_row(ws4, 3+i, row_data, bg=bg, fmt_map={j+2: "0.0%" for j in range(len(stress_periods)+1)})
    ws4.cell(3+i, 1).alignment = aln("left")
    ws4.cell(3+i, 1).font = font("F0F4F8", 10, True)

# ═══════════════════════════════════════════════════════════════════
# SHEET 6: RISK DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════

ws5 = wb.create_sheet("Risk Decomposition")
ws5.sheet_view.showGridLines = False
ws5.column_dimensions["A"].width = 28
for col in ["B", "C", "D", "E", "F", "G"]:
    ws5.column_dimensions[col].width = 18

section_title(ws5, 1, 1, "Asset Marginal Risk Contribution by Portfolio", 6)

for port_idx, (name, w) in enumerate(portfolios.items()):
    row_off = 2 + port_idx * (len(assets) + 3)
    section_title(ws5, row_off, 1, f"{name} — Marginal and Total Risk Contribution", 6)
    header_row(
        ws5, row_off+1,
        ["Asset Class", "Weight", "Marginal Risk Contrib.", "Total Risk Contrib.", "% of Portfolio Risk", "Factor Risk Share"],
        [28, 14, 22, 22, 22, 20]
    )
    
    wv = w.reindex(cov.index).fillna(0).values
    if wv.sum() != 0:
        wv = wv / wv.sum()
    
    cov_v = cov.values
    port_vol = float(np.sqrt(wv @ cov_v @ wv))
    mrc = (cov_v @ wv) / port_vol if port_vol > 0 else wv*0
    trc = wv * mrc
    
    for i, asset in enumerate(cov.index):
        wt = float(w.get(asset, 0))
        # MRC = (Σw)_i / σ_p is non-zero even for zero-weight assets (it is the
        # marginal cost of adding the asset). Zero it out in the display: an asset
        # not held contributes nothing to the current portfolio's risk budget.
        mrc_i = float(mrc[i]) if wt > 0 else 0.0
        trc_i = float(trc[i])
        pct_i = trc_i / port_vol * 100 if port_vol > 0 else 0
        
        # Factor risk share: clamp to [0,1] — POET diagonal and factor model
        # variance can be inconsistent, so syst_i may exceed tot_i.
        if asset in result.betas.index:
            betas_i = result.betas.loc[asset, factor_col_keys].values
            syst_i = float(betas_i @ factor_cov.values @ betas_i.T)
            tot_i = float(cov_v[i, i])
            fshr = min(syst_i / tot_i, 1.0) if tot_i > 0 else 0.0
        else:
            fshr = 0.0
        
        bg = bg_alt[i % 2]
        data_row(
            ws5, row_off+2+i,
            [dn(asset), wt, mrc_i, trc_i, pct_i/100, fshr],
            bg=bg,
            fmt_map={2: "0.0%", 3: "0.000", 4: "0.000", 5: "0.0%", 6: "0.0%"}
        )
        ws5.cell(row_off+2+i, 1).alignment = aln("left")
        ws5.cell(row_off+2+i, 1).font = font("F0F4F8", 10)

# Asset Risk Contribution Charts
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, (name, w) in enumerate(portfolios.items()):
    ax = axes[idx]
    wv = w.reindex(cov.index).fillna(0).values
    if wv.sum() != 0:
        wv = wv / wv.sum()
    
    cov_v = cov.values
    port_vol = float(np.sqrt(wv @ cov_v @ wv))
    trc = wv * ((cov_v @ wv) / port_vol) if port_vol > 0 else wv
    pct = trc / port_vol * 100
    
    sorted_idx = np.argsort(pct)[::-1]
    labels_s = [dn(cov.index[i]) for i in sorted_idx]
    vals_s = [pct[i] for i in sorted_idx]
    c_list = ["#B03A2E" if v > 15 else "#2980B9" if v > 5 else "#1E7B45" for v in vals_s]
    
    ax.barh(labels_s, vals_s, color=c_list)
    ax.set_title(name, fontsize=9, fontweight="bold", color="#F0F4F8")
    ax.set_xlabel("% of Portfolio Risk", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.4)

if len(portfolios) < 6:
    for idx in range(len(portfolios), 6):
        axes[idx].set_visible(False)

plt.suptitle("Asset Marginal Risk Contribution by Portfolio", fontsize=11, fontweight="bold", color="#F0F4F8", y=1.01)
plt.tight_layout()

img = img_to_xl(fig)
img.width = 980
img.height = 600
ws5.add_image(img, f"A{2 + len(portfolios) * (len(assets) + 3) + 1}")

# ═══════════════════════════════════════════════════════════════════
# SHEET 7: COVARIANCE
# ═══════════════════════════════════════════════════════════════════

ws6 = wb.create_sheet("Covariance")
ws6.sheet_view.showGridLines = False
asset_labels = [dn(a) for a in cov.index]
n = len(asset_labels)

ws6.column_dimensions["A"].width = 26
for i in range(2, n+2):
    ws6.column_dimensions[get_column_letter(i)].width = 12

section_title(ws6, 1, 1, "POET Correlation Matrix", n+1)

for j, lab in enumerate(asset_labels, 2):
    cell = ws6.cell(2, j, lab)
    cell.fill = fill(NAVY)
    cell.font = font("F0F4F8", 8, True)
    cell.alignment = aln("center", wrap=True)
    cell.border = border_thin()

ws6.row_dimensions[2].height = 56

std = np.sqrt(np.diag(cov.values))
corr = cov.values / np.outer(std, std)

for i, asset_row in enumerate(cov.index):
    cell = ws6.cell(3+i, 1, dn(asset_row))
    cell.fill = fill(NAVY)
    cell.font = font("F0F4F8", 9, True)
    cell.alignment = aln("left")
    cell.border = border_thin()
    
    for j in range(n):
        val = float(corr[i, j])
        cell = ws6.cell(3+i, 2+j, round(val, 3))
        
        if i == j:
            bg_c = "1E7B45"
        elif val > 0.7:
            bg_c = "6B1A1A"
        elif val > 0.4:
            bg_c = "8B2E1A"
        elif val < -0.2:
            bg_c = "1A3A6B"
        else:
            bg_c = PANEL
        
        cell.fill = fill(bg_c)
        cell.font = font("F0F4F8", 9)
        cell.alignment = aln("center")
        cell.border = border_thin()
        cell.number_format = "0.000"

ws6.row_dimensions[3+n].height = 8
row_pairs = 4 + n

section_title(ws6, row_pairs, 1, "Notable Correlation Pairs", 5)
header_row(ws6, row_pairs+1, ["Asset A", "Asset B", "Correlation", "Type", "Implication"], [26, 26, 14, 20, 40])

pairs = []
for i in range(n):
    for j in range(i+1, n):
        pairs.append((cov.index[i], cov.index[j], float(corr[i, j])))

pairs.sort(key=lambda x: abs(x[2]), reverse=True)

for k, (a, b, c_val) in enumerate(pairs[:15]):
    if c_val > 0.6:
        corr_type = "Strong Positive"
    elif c_val > 0.3:
        corr_type = "Moderate Positive"
    elif c_val < -0.1:
        corr_type = "Negative"
    else:
        corr_type = "Low"
    
    impl = "Low diversification benefit" if c_val > 0.6 else "Hedging relationship" if c_val < -0.1 else "Moderate diversification benefit"
    
    bg = bg_alt[k % 2]
    data_row(ws6, row_pairs+2+k, [dn(a), dn(b), c_val, corr_type, impl], bg=bg, fmt_map={3: "0.000"})
    ws6.cell(row_pairs+2+k, 1).alignment = aln("left")
    ws6.cell(row_pairs+2+k, 2).alignment = aln("left")

# ═══════════════════════════════════════════════════════════════════
# SHEET 8: NET FACTOR EXPOSURE
# ═══════════════════════════════════════════════════════════════════

ws7 = wb.create_sheet("Net Factor Exposure")
ws7.sheet_view.showGridLines = False
ws7.column_dimensions["A"].width = 28
for col in ["B", "C", "D", "E", "F"]:
    ws7.column_dimensions[col].width = 18

section_title(ws7, 1, 1, "Look-Through Portfolio Beta (Net Factor Exposure)", 6)
header_row(ws7, 2, ["Portfolio"] + factor_col_keys, [28] + [18]*len(factor_col_keys))

for i, (name, w) in enumerate(portfolios.items()):
    w_aligned = w.reindex(result.betas.index).fillna(0)
    if w_aligned.sum() > 0:
        w_aligned = w_aligned / w_aligned.sum()
    net_betas = result.betas[factor_col_keys].T @ w_aligned
    row_data = [name] + [float(net_betas.loc[f]) for f in factor_col_keys]
    
    bg = bg_alt[i % 2]
    data_row(ws7, 3+i, row_data, bg=bg, fmt_map={j+2: "0.000" for j in range(len(factor_col_keys))})
    ws7.cell(3+i, 1).font = font("F0F4F8", 10, True)
    ws7.cell(3+i, 1).alignment = aln("left")

# ═══════════════════════════════════════════════════════════════════
# SAVE WORKBOOK
# ═══════════════════════════════════════════════════════════════════

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics_report.xlsx")
wb.save(output_path)
print(f"\n{'='*60}")
print(f"✓ Report saved successfully to {output_path}")
print(f"{'='*60}\n")