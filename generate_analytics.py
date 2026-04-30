"""
Factor-Based SAA Analytics Report
Run from project root: python generate_analytics.py
Outputs: analytics_report.xlsx
"""
from __future__ import annotations
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

# ── Load pipeline ─────────────────────────────────────────────────
print("Loading data and running models...")
from data.data_manager import DataManager
from factors.factor_model import OLSFactorModel, QuantileFactorModel
from models.covariance import POETCovariance
from models.returns import ExpectedReturns
from portfolio.mvo import MVO
from portfolio.risk_parity import RiskParity
from portfolio.hrp import EnhancedHRP
from portfolio.risk_decomp import FactorRiskDecomposition
from config.settings import ASSET_DISPLAY_NAMES

def dn(a): return ASSET_DISPLAY_NAMES.get(a, a.replace("_"," ").title())

dm = DataManager(use_cache=True); dm.load_cached()
ols = OLSFactorModel(dm.factor_returns_t1, dm.asset_returns_t1, credit_liquidity=dm.credit_liquidity)
result = ols.fit()
poet   = POETCovariance(dm.factor_returns_t1, dm.asset_returns_t1_complete, result.betas)
poet.fit()
cov    = poet.as_dataframe()
F      = dm.factor_returns_t1.astype(float)
factor_cov = pd.DataFrame(np.cov(F.values.T), index=F.columns, columns=F.columns)
er     = ExpectedReturns(assets=list(cov.index)); mu = er.quarterly()
mvo_w  = MVO(mu, cov).fit()
rp_w   = RiskParity(cov).fit()
hrp_w  = EnhancedHRP(cov, result.betas).fit()
rd     = FactorRiskDecomposition(result.betas, factor_cov, cov)
assets     = list(cov.index)
eq_assets  = [a for a in assets if a in ["us_large_cap","us_mid_cap","us_small_cap","em_equity","reits"]]
bnd_assets = [a for a in assets if a in ["long_treasury","tips","ig_credit"]]

portfolios = {
    "Equal Weight": rd.equal_weight(assets),
    "60/40":        rd.sixty_forty(eq_assets, bnd_assets),
    "MVO":          mvo_w,
    "Risk Parity":  rp_w,
    "Enhanced HRP": hrp_w,
}
decomp = rd.compare(portfolios)
qm = QuantileFactorModel(dm.factor_returns_t1, dm.asset_returns_t1); qm.fit()
print("Models complete.")

# ── Style helpers ─────────────────────────────────────────────────
NAVY   = "0A1628"; BLUE  = "1A3A5C"; PANEL  = "16202E"
MUTED  = "7A90A8"; RED   = "B03A2E"; AMBER  = "B7770D"; GREEN  = "1E7B45"

PORT_HEX = {
    "Equal Weight": "7F8C8D", "60/40": "B03A2E", "MVO": "2980B9",
    "Risk Parity": "E67E22", "Enhanced HRP": "1E7B45",
}
FACTOR_HEX = {
    "Equity Premium": "C0392B", "Term Premium": "2980B9", "Credit Spread": "E67E22",
    "Inflation": "27AE60", "Liquidity": "8E44AD", "Idiosyncratic": "566573",
}

def fill(hex_): return PatternFill("solid", start_color=hex_, end_color=hex_)
def font(hex_="F0F4F8", sz=10, bold=False): return Font(name="Arial", color=hex_, size=sz, bold=bold)
def aln(h="left", v="center", wrap=False): return Alignment(horizontal=h, vertical=v, wrap_text=wrap)
def border_thin(): s = Side(style="thin", color="1E2D3D"); return Border(left=s, right=s, top=s, bottom=s)
def header_row(ws, row, labels, widths=None, bg=NAVY):
    for c, label in enumerate(labels, 1):
        cell = ws.cell(row=row, column=c, value=label)
        cell.fill = fill(bg); cell.font = font("F0F4F8", 10, True)
        cell.alignment = aln("center"); cell.border = border_thin()
        if widths: ws.column_dimensions[get_column_letter(c)].width = widths[c-1]
def data_row(ws, row, values, bg=PANEL, fmt_map=None):
    for c, val in enumerate(values, 1):
        cell = ws.cell(row=row, column=c, value=val)
        cell.fill = fill(bg); cell.font = font("D0D8E4", 10)
        cell.alignment = aln("center"); cell.border = border_thin()
        if fmt_map and c in fmt_map: cell.number_format = fmt_map[c]
def section_title(ws, row, col, text, span=1):
    cell = ws.cell(row=row, column=col, value=text)
    cell.fill = fill(BLUE); cell.font = font("F0F4F8", 11, True)
    cell.alignment = aln("left"); cell.border = border_thin()
    if span > 1: ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col+span-1)

def img_to_xl(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0e1117", edgecolor="none")
    buf.seek(0); plt.close(fig)
    return XLImage(buf)

def fig_style():
    plt.rcParams.update({
        "figure.facecolor": "#0e1117", "axes.facecolor": "#16202E", "axes.edgecolor": "#1E2D3D",
        "axes.labelcolor": "#D0D8E4", "xtick.color": "#D0D8E4", "ytick.color": "#D0D8E4",
        "text.color": "#D0D8E4", "grid.color": "#1E2D3D", "grid.linewidth": 0.6,
        "font.family": "Arial", "font.size": 9,
    })
fig_style()

# ── Base Setup ──────────────────────────────────────────────────
factor_col_keys = list(factor_cov.index)

# Benchmark definition (60/40)
w_6040 = portfolios["60/40"]
asset_rets = dm.asset_returns_t1_complete.dropna()
w_al_6040 = w_6040.reindex(asset_rets.columns).fillna(0)
if w_al_6040.sum() != 0: w_al_6040 = w_al_6040 / w_al_6040.sum()
benchmark_ret = asset_rets @ w_al_6040

def port_metrics(w):
    wv = w.reindex(cov.index).fillna(0).values
    if wv.sum() != 0: wv = wv / wv.sum()
    mu_v = mu.reindex(cov.index).fillna(0).values

    ann_ret = float(wv @ mu_v) * 4 * 100
    ann_vol = float(np.sqrt(wv @ cov.values @ wv)) * 2 * 100
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    w_al = w.reindex(asset_rets.columns).fillna(0)
    if w_al.sum() != 0: w_al = w_al / w_al.sum()
    pr = asset_rets @ w_al

    cum = (1 + pr).cumprod()
    roll_max = cum.cummax()
    max_dd = float(((cum - roll_max) / roll_max).min()) * 100
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0

    excess_down = np.minimum(pr.values, 0.0)
    downside_vol = float(np.std(excess_down) * np.sqrt(4) * 100)
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    diff = pr - benchmark_ret
    tracking_error = float(np.std(diff) * np.sqrt(4) * 100)

    return ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, tracking_error, pr

metrics = {n: port_metrics(w) for n, w in portfolios.items()}

stress_periods = {
    "GFC (2008 Q3 - 2009 Q1)": ("2008-07-01", "2009-03-31"),
    "COVID Crash (2020 Q1)":   ("2020-01-01", "2020-06-30"),
    "Rate Shock (2022)":       ("2022-01-01", "2022-12-31"),
}

def period_return(port_ret, start, end):
    mask = (port_ret.index >= pd.Timestamp(start)) & (port_ret.index <= pd.Timestamp(end))
    sub  = port_ret[mask]
    if len(sub) == 0: return float("nan")
    return float((1 + sub).prod() - 1) * 100

# ═════════════════════════════════════════════════════════════════
# BUILD WORKBOOK
# ═════════════════════════════════════════════════════════════════
wb = Workbook()
wb.remove(wb.active)

print("Building Excel workbook...")

# ═══ SHEET 1: EXECUTIVE SUMMARY ═══════════════════════════════════
ws1 = wb.create_sheet("Executive Summary")
ws1.sheet_view.showGridLines = False
ws1.column_dimensions["A"].width = 32
for col in ["B","C","D","E","F"]: ws1.column_dimensions[col].width = 18

ws1.merge_cells("A1:F1")
c = ws1["A1"]; c.value = "FACTOR-BASED STRATEGIC ASSET ALLOCATION — ANALYTICS REPORT"
c.fill = fill(NAVY); c.font = font("F0F4F8", 14, True); c.alignment = aln("center")

ws1.merge_cells("A2:F2")
c = ws1["A2"]; c.value = "POET Covariance  |  JPMorgan 2026 LTCMA  |  Enhanced HRP  |  13 Asset Classes  |  2004 Q1 – 2024 Q4"
c.fill = fill(BLUE); c.font = font(MUTED, 10); c.alignment = aln("center")

ws1.row_dimensions[3].height = 8

erp_60 = float(decomp.loc["60/40","Equity Premium"]) if "60/40" in decomp.index else 0
erp_hrp = float(decomp.loc["Enhanced HRP","Equity Premium"]) if "Enhanced HRP" in decomp.index else 0
ws1.merge_cells("A4:F4"); c = ws1["A4"]
c.value = (f"Key Finding: A conventional 60/40 portfolio allocates {erp_60:.1f}% of total risk to Equity Premium. "
           f"Enhanced HRP reduces this to {erp_hrp:.1f}% — a {erp_60-erp_hrp:.1f} pp reduction — "
           f"while maintaining balanced systematic risk exposure across all five factors.")
c.fill = fill(BLUE); c.font = font("F0F4F8", 10); c.alignment = aln("left", wrap=True)
ws1.row_dimensions[4].height = 36
ws1.row_dimensions[5].height = 8

section_title(ws1, 6, 1, "Portfolio Performance Metrics", 8)
header_row(ws1, 7, ["Portfolio","Ann. Return","Ann. Volatility","Sharpe Ratio","Sortino Ratio","Max Drawdown","Calmar Ratio","Tracking Error"], [32, 18, 18, 18, 18, 18, 18, 18])
bg_alt = [PANEL, "111A27"]
for i, (name, w) in enumerate(portfolios.items()):
    ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, tracking_error, _ = metrics[name]
    bg = bg_alt[i % 2]
    data_row(ws1, 8+i, [name, ann_ret/100, ann_vol/100, sharpe, sortino, max_dd/100, calmar, tracking_error/100], bg=bg, fmt_map={2:"0.0%",3:"0.0%",4:"0.00",5:"0.00",6:"0.0%",7:"0.00",8:"0.0%"})
    ws1.cell(8+i, 1).font = font("F0F4F8", 10, True); ws1.cell(8+i, 1).alignment = aln("left")

ws1.row_dimensions[13].height = 8

section_title(ws1, 14, 1, "Factor Risk Attribution (% of Total Portfolio Risk)", 6)
factor_cols = [c for c in ["Equity Premium","Term Premium","Credit Spread","Inflation","Liquidity","Idiosyncratic"] if c in decomp.columns]
header_row(ws1, 15, ["Portfolio"] + factor_cols, [32]+[18]*len(factor_cols))
for i, (name, _) in enumerate(portfolios.items()):
    if name not in decomp.index: continue
    row_data = [name] + [decomp.loc[name, c]/100 for c in factor_cols]
    bg = bg_alt[i % 2]
    data_row(ws1, 16+i, row_data, bg=bg, fmt_map={j+2: "0.0%" for j in range(len(factor_cols))})
    ws1.cell(16+i, 1).font = font("F0F4F8", 10, True); ws1.cell(16+i, 1).alignment = aln("left")
    erp_val = decomp.loc[name,"Equity Premium"] if "Equity Premium" in decomp.columns else 0
    erp_cell = ws1.cell(16+i, 2)
    if erp_val > 70: erp_cell.font = Font(name="Arial", color=RED, size=10, bold=True)
    elif erp_val > 55: erp_cell.font = Font(name="Arial", color=AMBER, size=10, bold=True)
    else: erp_cell.font = Font(name="Arial", color=GREEN, size=10, bold=True)

ws1.row_dimensions[21].height = 8

B_factors = result.betas.reindex(columns=factor_col_keys).values.astype(float)
total_var  = np.trace(cov.values)
factor_var = np.trace(B_factors @ factor_cov.values @ B_factors.T)
sys_share  = factor_var / total_var
idio_share = 1 - sys_share

section_title(ws1, 22, 1, "POET Covariance Diagnostics", 4)
header_row(ws1, 23, ["Parameter","Value","Description"], [32,18,40])
diag_rows = [
    ("Sample Observations (T)", len(F), "Complete-case balanced panel"),
    ("Asset Classes (p)", cov.shape[0], "Number of assets"),
    ("Systematic Factors (k)", F.shape[1], "Factor model dimension"),
    ("Systematic Variance Share", f"{sys_share:.1%}", "Variance explained by factors"),
    ("Idiosyncratic Share", f"{idio_share:.1%}", "Residual variance"),
]
for i, (param, val, desc) in enumerate(diag_rows):
    bg = bg_alt[i % 2]
    ws1.cell(24+i, 1, param).fill = fill(bg); ws1.cell(24+i, 1).font = font("F0F4F8", 10); ws1.cell(24+i, 1).border = border_thin(); ws1.cell(24+i, 1).alignment = aln("left")
    ws1.cell(24+i, 2, val).fill = fill(bg); ws1.cell(24+i, 2).font = font("D0D8E4", 10); ws1.cell(24+i, 2).border = border_thin(); ws1.cell(24+i, 2).alignment = aln("center")
    ws1.cell(24+i, 3, desc).fill = fill(bg); ws1.cell(24+i, 3).font = font(MUTED, 9); ws1.cell(24+i, 3).border = border_thin(); ws1.cell(24+i, 3).alignment = aln("left")

# ═══ SHEET 2: BACKTEST SUMMARY ════════════════════════════════════
ws_bt = wb.create_sheet("Backtest Summary", 1)
ws_bt.sheet_view.showGridLines = False
ws_bt.column_dimensions["A"].width = 28
for col in ["B","C","D","E","F","G","H"]: ws_bt.column_dimensions[col].width = 18

section_title(ws_bt, 1, 1, "Backtest Summary Statistics — All Portfolios (2004 Q4 – 2024 Q4)", 8)
header_row(ws_bt, 2, ["Portfolio","Ann. Return","Ann. Volatility","Sharpe Ratio","Sortino Ratio","Max Drawdown","Calmar Ratio","Tracking Error vs 60/40"], [28,16,18,16,16,16,16,26])

for i, (name, _) in enumerate(portfolios.items()):
    ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, te, pr = metrics[name]
    bg = bg_alt[i % 2]
    data_row(ws_bt, 3+i, [name, ann_ret/100, ann_vol/100, sharpe, sortino, max_dd/100, calmar, te/100], bg=bg, fmt_map={2:"0.0%",3:"0.0%",4:"0.00",5:"0.00",6:"0.0%",7:"0.00",8:"0.0%"})
    ws_bt.cell(3+i, 1).font = font("F0F4F8", 10, True); ws_bt.cell(3+i, 1).alignment = aln("left")
    sharpe_cell = ws_bt.cell(3+i, 4)
    if sharpe >= 0.7: sharpe_cell.font = Font(name="Arial", color=GREEN, size=10, bold=True)
    elif sharpe >= 0.4: sharpe_cell.font = Font(name="Arial", color=AMBER, size=10)
    else: sharpe_cell.font = Font(name="Arial", color=RED, size=10)
    dd_cell = ws_bt.cell(3+i, 6)
    if max_dd < -25: dd_cell.font = Font(name="Arial", color=RED, size=10, bold=True)
    elif max_dd < -15: dd_cell.font = Font(name="Arial", color=AMBER, size=10)
    else: dd_cell.font = Font(name="Arial", color=GREEN, size=10)

fig, ax = plt.subplots(figsize=(11, 4))
window = 8
for name, _ in portfolios.items():
    pr = metrics[name][-1]
    roll_ret = pr.rolling(window).mean() * 4 * 100
    roll_vol = pr.rolling(window).std() * np.sqrt(4) * 100
    roll_sharpe = roll_ret / roll_vol
    ax.plot(roll_sharpe.index, roll_sharpe.values, label=name, color="#"+PORT_HEX[name], linewidth=1.5)
ax.axhline(y=0, color="#7A90A8", linewidth=0.8, linestyle="--")
ax.axhline(y=0.5, color="#1E7B45", linewidth=0.8, linestyle=":", alpha=0.6, label="0.5 reference")
ax.set_title("Rolling 2-Year Sharpe Ratio — All Portfolios", fontsize=11, fontweight="bold", color="#F0F4F8")
ax.set_ylabel("Sharpe Ratio"); ax.legend(fontsize=9, framealpha=0.3); ax.grid(alpha=0.4)
img = img_to_xl(fig); img.width = 770; img.height = 300
ws_bt.add_image(img, f"A{4+len(portfolios)+2}")

row_as = 4 + len(portfolios) + 20
section_title(ws_bt, row_as, 1, "Active Share vs 60/40 Benchmark", 4)
header_row(ws_bt, row_as+1, ["Portfolio","Tracking Error (ann.)","Active Share","Classification"], [28, 22, 18, 30])
for i, (name, w) in enumerate(portfolios.items()):
    _, _, _, _, _, _, te, _ = metrics[name]
    all_assets = list(set(list(w.index) + list(w_6040.index)))
    w1 = w.reindex(all_assets).fillna(0)
    w2 = w_6040.reindex(all_assets).fillna(0)
    active_share = float(0.5 * np.sum(np.abs(w1.values - w2.values)))
    classification = "Closet Indexer" if active_share < 0.2 else "Moderate Active" if active_share < 0.5 else "High Conviction Active" if active_share < 0.8 else "Very High Active"
    bg = bg_alt[i % 2]
    data_row(ws_bt, row_as+2+i, [name, te/100, active_share, classification], bg=bg, fmt_map={2:"0.0%", 3:"0.0%"})
    ws_bt.cell(row_as+2+i, 1).font = font("F0F4F8", 10, True); ws_bt.cell(row_as+2+i, 1).alignment = aln("left"); ws_bt.cell(row_as+2+i, 4).alignment = aln("left")
    as_cell = ws_bt.cell(row_as+2+i, 3)
    if active_share > 0.5: as_cell.font = Font(name="Arial", color=GREEN, size=10, bold=True)
    elif active_share > 0.2: as_cell.font = Font(name="Arial", color=AMBER, size=10)
    else: as_cell.font = Font(name="Arial", color=RED, size=10)

# ═══ SHEET 3: PORTFOLIO CHARTS ════════════════════════════════════
ws2 = wb.create_sheet("Portfolio Analytics")
ws2.sheet_view.showGridLines = False
section_title(ws2, 1, 1, "Portfolio Analytics — Charts", 1)
ws2.column_dimensions["A"].width = 2; ws2.column_dimensions["B"].width = 2

port_names = list(portfolios.keys())
port_colors = ["#"+PORT_HEX[n] for n in port_names]
factor_cols_all = [c for c in ["Equity Premium","Term Premium","Credit Spread","Inflation","Liquidity","Idiosyncratic"] if c in decomp.columns]
fcolors = ["#"+FACTOR_HEX.get(c,"566573") for c in factor_cols_all]

fig, ax = plt.subplots(figsize=(10, 4))
bottom = np.zeros(len(port_names))
for col, color in zip(factor_cols_all, fcolors):
    vals = np.array([float(decomp.loc[n,col]) for n in port_names])
    bars = ax.bar(port_names, vals, bottom=bottom, color=color, label=col, width=0.5)
    for bar, b, v in zip(bars, bottom, vals):
        if abs(v) > 4: ax.text(bar.get_x()+bar.get_width()/2, b+v/2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    bottom += vals
ax.axhline(y=0, color="#1E2D3D", linewidth=0.8); ax.set_ylabel("% of Total Portfolio Risk")
ax.set_title("Factor Risk Attribution by Construction Method", fontsize=11, fontweight="bold", color="#F0F4F8")
ax.legend(loc="upper right", fontsize=8, framealpha=0.3); ax.set_ylim(-10, 115); ax.grid(axis="y", alpha=0.4)
img = img_to_xl(fig); img.width = 700; img.height = 300; ws2.add_image(img, "C3")

fig, ax = plt.subplots(figsize=(8, 3))
erp_vals = [float(decomp.loc[n,"Equity Premium"]) for n in port_names]
colors   = [("#B03A2E" if v>70 else "#B7770D" if v>55 else "#1E7B45") for v in erp_vals]
bars = ax.bar(port_names, erp_vals, color=colors, width=0.5)
for bar, v in zip(bars, erp_vals): ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=bar.get_facecolor())
ax.axhline(y=50, color="#7A90A8", linewidth=1, linestyle="--", alpha=0.7, label="50% benchmark")
ax.set_ylabel("Equity Premium Share (%)"); ax.set_title("Equity Premium Concentration", fontsize=11, fontweight="bold", color="#F0F4F8")
ax.set_ylim(0, 105); ax.legend(fontsize=8, framealpha=0.3); ax.grid(axis="y", alpha=0.4)
img = img_to_xl(fig); img.width = 560; img.height = 230; ws2.add_image(img, "C22")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
mdd_vals = [metrics[n][4] for n in port_names]
bars = ax1.bar(port_names, mdd_vals, color=port_colors, width=0.5)
for bar, v in zip(bars, mdd_vals): ax1.text(bar.get_x()+bar.get_width()/2, v-0.5, f"{v:.1f}%", ha="center", va="top", fontsize=8, color="white")
ax1.set_title("Maximum Drawdown (%)", fontsize=10, fontweight="bold", color="#F0F4F8"); ax1.set_xticklabels(port_names, rotation=20, ha="right", fontsize=8); ax1.grid(axis="y", alpha=0.4)

div_vals = []
for n, w in portfolios.items():
    wv = w.reindex(cov.index).fillna(0).values
    if wv.sum() != 0: wv = wv / wv.sum()
    wvol = float(np.sum(wv * np.sqrt(np.diag(cov.values))))
    pvol = float(np.sqrt(wv @ cov.values @ wv))
    div_vals.append(wvol/pvol if pvol>0 else 1.0)
bars = ax2.bar(port_names, div_vals, color=port_colors, width=0.5)
for bar, v in zip(bars, div_vals): ax2.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}x", ha="center", va="bottom", fontsize=8, color="white")
ax2.set_title("Diversification Ratio", fontsize=10, fontweight="bold", color="#F0F4F8"); ax2.set_xticklabels(port_names, rotation=20, ha="right", fontsize=8); ax2.grid(axis="y", alpha=0.4)
plt.tight_layout(); img = img_to_xl(fig); img.width = 700; img.height = 260; ws2.add_image(img, "C38")

fig, ax = plt.subplots(figsize=(11, 4))
for n, w in portfolios.items():
    _, _, _, _, _, _, _, pr = metrics[n]
    cum = (1 + pr).cumprod() * 100
    ax.plot(cum.index, cum.values, label=n, color="#"+PORT_HEX[n], linewidth=1.5)
ax.set_title("Cumulative Growth of $100 — All Portfolios", fontsize=11, fontweight="bold", color="#F0F4F8"); ax.set_ylabel("Portfolio Value ($)"); ax.legend(fontsize=9, framealpha=0.3); ax.grid(alpha=0.4)
img = img_to_xl(fig); img.width = 770; img.height = 310; ws2.add_image(img, "C58")

# ═══ SHEET 4: FACTOR MODEL ════════════════════════════════════════
ws3 = wb.create_sheet("Factor Model")
ws3.sheet_view.showGridLines = False
ws3.column_dimensions["A"].width = 28
for col in ["B","C","D","E","F","G","H","I","J"]: ws3.column_dimensions[col].width = 16

section_title(ws3, 1, 1, "Asset-Level Factor Model Results", 9)
header_row(ws3, 2, ["Asset Class","Equity Premium","Term Premium","Credit Spread","Inflation","Liquidity","Alpha","R-Squared","Liq. Proxy"], [28,16,16,16,16,16,14,14,20])
for i, asset in enumerate(result.betas.index):
    r2v   = float(result.r_squared.loc[asset]) if asset in result.r_squared.index else 0
    alpha = float(result.betas.loc[asset,"alpha"]) if "alpha" in result.betas.columns else 0
    beta_vals = [float(result.betas.loc[asset, f]) for f in result.betas.columns if f != "alpha"]
    from factors.factor_model import EQUITY_LIQUIDITY_ASSETS
    liq_proxy = "Pastor-Stambaugh" if asset in EQUITY_LIQUIDITY_ASSETS else "BAA-Treasury"
    bg = bg_alt[i % 2]
    data_row(ws3, 3+i, [dn(asset)] + beta_vals + [alpha, r2v, liq_proxy], bg=bg, fmt_map={2:"0.000",3:"0.000",4:"0.000",5:"0.000",6:"0.000",7:"0.0%",8:"0.0%"})
    ws3.cell(3+i, 1).alignment = aln("left"); ws3.cell(3+i, 1).font = font("F0F4F8", 10)

n_assets = len(result.betas.index)
row_start = 4 + n_assets
section_title(ws3, row_start, 1, "Equity Premium Beta: Full-Sample OLS vs Stress (Q10 Quantile Regression)", 5)
header_row(ws3, row_start+1, ["Asset Class","OLS Beta","Stress Beta (Q10)","Uplift","Interpretation"], [28,16,20,16,30])
ols_erp = result.betas["equity_premium"]
q10_result2 = qm.results.get(0.10, None)
q10_erp  = q10_result2.betas["equity_premium"].reindex(ols_erp.index) if q10_result2 is not None else ols_erp.copy()
stress_rows = []
for asset in ols_erp.index:
    ols_b   = float(ols_erp.loc[asset])
    q10_b   = float(q10_erp.loc[asset]) if asset in q10_erp.index else ols_b
    uplift  = q10_b - ols_b
    interp  = "Significant stress uplift" if uplift > 0.15 else "Moderate stress uplift" if uplift > 0.05 else "Stable across regimes"
    stress_rows.append((dn(asset), ols_b, q10_b, uplift, interp))
stress_rows.sort(key=lambda x: x[3], reverse=True)
for i, row in enumerate(stress_rows):
    bg = bg_alt[i % 2]
    data_row(ws3, row_start+2+i, list(row), bg=bg, fmt_map={2:"0.000",3:"0.000",4:"0.000"})
    ws3.cell(row_start+2+i, 1).alignment = aln("left"); ws3.cell(row_start+2+i, 1).font = font("F0F4F8", 10)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
r2_sorted = result.r_squared.sort_values(ascending=True)
asset_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(r2_sorted)))
axes[0].barh([dn(a) for a in r2_sorted.index], r2_sorted.values, color=asset_colors)
axes[0].axvline(x=0.7, color="#B7770D", linestyle="--", linewidth=1, label="0.70 reference")
for i, v in enumerate(r2_sorted.values): axes[0].text(v+0.01, i, f"{v:.3f}", va="center", fontsize=8, color="#D0D8E4")
axes[0].set_title("Adjusted R-Squared by Asset Class", fontsize=10, fontweight="bold", color="#F0F4F8"); axes[0].set_xlim(0, 1.15); axes[0].grid(axis="x", alpha=0.4)

sorted_idx = sorted(ols_erp.index, key=lambda a: float(ols_erp.loc[a]))
x = np.arange(len(sorted_idx))
axes[1].barh(x - 0.2, [float(ols_erp.loc[a]) for a in sorted_idx], height=0.35, label="OLS (Full Sample)", color="#2980B9")
axes[1].barh(x + 0.2, [float(q10_erp.loc[a]) if a in q10_erp.index else float(ols_erp.loc[a]) for a in sorted_idx], height=0.35, label="Stress Q10", color="#B03A2E", alpha=0.85)
axes[1].set_yticks(x); axes[1].set_yticklabels([dn(a) for a in sorted_idx], fontsize=8)
axes[1].set_title("Equity Beta: Normal vs Stress", fontsize=10, fontweight="bold", color="#F0F4F8"); axes[1].legend(fontsize=8, framealpha=0.3); axes[1].grid(axis="x", alpha=0.4)
plt.tight_layout(); img = img_to_xl(fig); img.width = 910; img.height = 310
ws3.add_image(img, f"A{row_start+2+len(stress_rows)+2}")

# ═══ SHEET 5: STRESS TESTING ══════════════════════════════════════
ws4 = wb.create_sheet("Stress Testing")
ws4.sheet_view.showGridLines = False
ws4.column_dimensions["A"].width = 34
for col in ["B","C","D","E","F"]: ws4.column_dimensions[col].width = 20

section_title(ws4, 1, 1, "Portfolio Stress Testing — Defined Periods", 6)
header_row(ws4, 2, ["Portfolio"] + list(stress_periods.keys()) + ["Recovery: 2009 Q2 – 2010 Q4"], [34]+[20]*len(stress_periods)+[28])

for i, (name, w) in enumerate(portfolios.items()):
    _, _, _, _, _, _, _, pr = metrics[name]
    row_data = [name]
    for label, (start, end) in stress_periods.items(): row_data.append(period_return(pr, start, end)/100)
    row_data.append(period_return(pr, "2009-04-01", "2010-12-31")/100)
    bg = bg_alt[i % 2]
    data_row(ws4, 3+i, row_data, bg=bg, fmt_map={j+2:"0.0%" for j in range(len(stress_periods)+1)})
    ws4.cell(3+i, 1).alignment = aln("left"); ws4.cell(3+i, 1).font = font("F0F4F8", 10, True)

# ═══ SHEET 6: RISK DECOMPOSITION ══════════════════════════════════
ws5 = wb.create_sheet("Risk Decomposition")
ws5.sheet_view.showGridLines = False
ws5.column_dimensions["A"].width = 28
for col in ["B","C","D","E","F","G"]: ws5.column_dimensions[col].width = 18

section_title(ws5, 1, 1, "Asset Marginal Risk Contribution by Portfolio", 6)

for port_idx, (name, w) in enumerate(portfolios.items()):
    row_off = 2 + port_idx * (len(assets) + 3)
    section_title(ws5, row_off, 1, f"{name} — Marginal and Total Risk Contribution", 6)
    header_row(ws5, row_off+1, ["Asset Class","Weight","Marginal Risk Contrib.","Total Risk Contrib.","% of Portfolio Risk","Factor Risk Share"], [28,14,22,22,22,20])
    
    wv = w.reindex(cov.index).fillna(0).values
    if wv.sum() != 0: wv = wv / wv.sum()
    cov_v = cov.values
    port_vol = float(np.sqrt(wv @ cov_v @ wv))
    mrc = (cov_v @ wv) / port_vol if port_vol > 0 else wv*0
    trc = wv * mrc
    
    for i, asset in enumerate(cov.index):
        wt    = float(w.get(asset, 0))
        mrc_i = float(mrc[i])
        trc_i = float(trc[i])
        pct_i = trc_i / port_vol * 100 if port_vol > 0 else 0
        
        # FIXED: Slice betas perfectly to factor_col_keys to avoid Alpha dimension mismatch
        betas_i = result.betas.loc[asset, factor_col_keys].values if asset in result.betas.index else np.zeros(len(factor_col_keys))
        syst_i  = float(betas_i @ factor_cov.values @ betas_i.T)
        tot_i   = float(cov_v[i,i])
        fshr    = syst_i/tot_i if tot_i > 0 else 0
        
        bg = bg_alt[i % 2]
        data_row(ws5, row_off+2+i, [dn(asset), wt, mrc_i, trc_i, pct_i/100, fshr], bg=bg, fmt_map={2:"0.0%",3:"0.000",4:"0.000",5:"0.0%",6:"0.0%"})
        ws5.cell(row_off+2+i, 1).alignment = aln("left"); ws5.cell(row_off+2+i, 1).font = font("F0F4F8", 10)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for idx, (name, w) in enumerate(portfolios.items()):
    ax = axes[idx]
    wv = w.reindex(cov.index).fillna(0).values
    if wv.sum() != 0: wv = wv / wv.sum()
    cov_v = cov.values
    port_vol = float(np.sqrt(wv @ cov_v @ wv))
    trc = wv * ((cov_v @ wv) / port_vol) if port_vol > 0 else wv
    pct = trc / port_vol * 100
    sorted_idx = np.argsort(pct)[::-1]
    labels_s   = [dn(cov.index[i]) for i in sorted_idx]
    vals_s     = [pct[i] for i in sorted_idx]
    c_list     = ["#B03A2E" if v > 15 else "#2980B9" if v > 5 else "#1E7B45" for v in vals_s]
    ax.barh(labels_s, vals_s, color=c_list)
    ax.set_title(name, fontsize=9, fontweight="bold", color="#F0F4F8"); ax.set_xlabel("% of Portfolio Risk", fontsize=8)
    ax.tick_params(axis="y", labelsize=7); ax.grid(axis="x", alpha=0.4)
if len(portfolios) < 6:
    for idx in range(len(portfolios), 6): axes[idx].set_visible(False)
plt.suptitle("Asset Marginal Risk Contribution by Portfolio", fontsize=11, fontweight="bold", color="#F0F4F8", y=1.01)
plt.tight_layout()
img = img_to_xl(fig); img.width = 980; img.height = 600
ws5.add_image(img, f"A{2 + len(portfolios) * (len(assets) + 3) + 1}")

# ═══ SHEET 7: COVARIANCE ══════════════════════════════════════════
ws6 = wb.create_sheet("Covariance")
ws6.sheet_view.showGridLines = False
asset_labels = [dn(a) for a in cov.index]
n = len(asset_labels)

ws6.column_dimensions["A"].width = 26
for i in range(2, n+2): ws6.column_dimensions[get_column_letter(i)].width = 12

section_title(ws6, 1, 1, "POET Correlation Matrix", n+1)
for j, lab in enumerate(asset_labels, 2):
    cell = ws6.cell(2, j, lab)
    cell.fill = fill(NAVY); cell.font = font("F0F4F8", 8, True)
    cell.alignment = aln("center", wrap=True); cell.border = border_thin()
ws6.row_dimensions[2].height = 56

std  = np.sqrt(np.diag(cov.values))
corr = cov.values / np.outer(std, std)

for i, asset_row in enumerate(cov.index):
    cell = ws6.cell(3+i, 1, dn(asset_row))
    cell.fill = fill(NAVY); cell.font = font("F0F4F8", 9, True)
    cell.alignment = aln("left"); cell.border = border_thin()
    for j in range(n):
        val  = float(corr[i,j])
        cell = ws6.cell(3+i, 2+j, round(val, 3))
        if i == j: bg_c = "1E7B45"
        elif val > 0.7: bg_c = "6B1A1A"
        elif val > 0.4: bg_c = "8B2E1A"
        elif val < -0.2: bg_c = "1A3A6B"
        else: bg_c = PANEL
        cell.fill = fill(bg_c); cell.font = font("F0F4F8", 9)
        cell.alignment = aln("center"); cell.border = border_thin(); cell.number_format = "0.000"

ws6.row_dimensions[3+n].height = 8
row_pairs = 4 + n
section_title(ws6, row_pairs, 1, "Notable Correlation Pairs", 5)
header_row(ws6, row_pairs+1, ["Asset A","Asset B","Correlation","Type","Implication"], [26,26,14,20,40])
pairs = []
for i in range(n):
    for j in range(i+1, n): pairs.append((cov.index[i], cov.index[j], float(corr[i,j])))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for k, (a, b, c_val) in enumerate(pairs[:15]):
    corr_type = "Strong Positive" if c_val > 0.6 else "Moderate Positive" if c_val > 0.3 else "Negative" if c_val < -0.1 else "Low"
    impl = "Low diversification benefit" if c_val > 0.6 else "Hedging relationship" if c_val < -0.1 else "Moderate diversification benefit"
    bg = bg_alt[k % 2]
    data_row(ws6, row_pairs+2+k, [dn(a), dn(b), c_val, corr_type, impl], bg=bg, fmt_map={3:"0.000"})
    ws6.cell(row_pairs+2+k, 1).alignment = aln("left"); ws6.cell(row_pairs+2+k, 2).alignment = aln("left")

# ═══ SHEET 8: NET FACTOR EXPOSURE (LOOK-THROUGH) ══════════════════
ws7 = wb.create_sheet("Net Factor Exposure")
ws7.sheet_view.showGridLines = False
ws7.column_dimensions["A"].width = 28
for col in ["B","C","D","E","F"]: ws7.column_dimensions[col].width = 18

section_title(ws7, 1, 1, "Look-Through Portfolio Beta (Net Factor Exposure)", 6)
header_row(ws7, 2, ["Portfolio"] + factor_col_keys, [28] + [18]*len(factor_col_keys))

for i, (name, w) in enumerate(portfolios.items()):
    w_aligned = w.reindex(result.betas.index).fillna(0)
    net_betas = result.betas[factor_col_keys].T @ w_aligned
    row_data = [name] + [float(net_betas.loc[f]) for f in factor_col_keys]
    bg = bg_alt[i % 2]
    data_row(ws7, 3+i, row_data, bg=bg, fmt_map={j+2:"0.000" for j in range(len(factor_col_keys))})
    ws7.cell(3+i, 1).font = font("F0F4F8", 10, True); ws7.cell(3+i, 1).alignment = aln("left")

# Save Workbook
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics_report.xlsx")
wb.save(output_path)
print(f"Report saved successfully to {output_path}")