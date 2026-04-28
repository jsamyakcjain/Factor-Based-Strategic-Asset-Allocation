# Hidden Factor Concentration in Multi-Asset Portfolios

## Overview
A factor-based Strategic Asset Allocation framework demonstrating that conventional 60/40 portfolios allocate 80.6% of total risk to equity premium. An Enhanced HRP approach using factor-loading-based clustering reduces this to 42.0% without explicit constraints.

## Key Findings
- 60/40 allocates 80.6% of total risk to a single factor — equity premium
- Equal weighting across 13 diverse assets still concentrates 65.1% in equity premium
- Enhanced HRP (factor-distance clustering) reduces equity concentration to 42.0%
- Private real estate R2 = 0.089 — Geltner unsmoothing amplifies noise, validating use as genuine diversifier

## Asset Universe (13 assets)
Public Markets: US Large/Mid/Small Cap Equity, EM Equity, Long Treasury, TIPS, IG Credit, REITs, Commodities
Alternatives: High Yield Credit, Hedge Funds, Private Equity (unsmoothed), Private Real Estate (unsmoothed)

## Methodology
- Factor Model: 5-factor OLS with Newey-West HAC (Equity Premium, Term Premium, Credit Spread, Inflation, Liquidity)
- Covariance: POET estimator (Fan, Liao and Mincheva, 2013) — 81 complete quarters, 65.8% systematic variance share
- Liquidity Proxies: Pastor-Stambaugh innovation (equity assets) / BAA-Treasury spread change (credit and alternative assets)
- Portfolio Methods: Equal Weight, 60/40, MVO (JPM 2026 LTCMA), Risk Parity, Enhanced HRP
- HRP Innovation: Factor-loading Euclidean distance with z-score standardisation replaces price correlation clustering

## Data Sources
- WRDS: CRSP (market, treasury), Fama-French factors, Pastor-Stambaugh liquidity
- FRED: Moodys BAA yield, GS10, TIPS breakeven, CPI, NBER recession
- yfinance: ETF total returns (IJH, IWM, EEM, TLT, TIP, LQD, VNQ, GSG)
- JPMorgan 2026 LTCMA: Expected return inputs for MVO
- Professor dataset: HY credit, hedge funds, private equity (unsmoothed), private real estate (unsmoothed)

## Project Structure
- config/: Settings, asset universe, factor names, JPM LTCMA
- data/: WRDS, FRED, yfinance, professor data loaders
- factors/: OLS, quantile, and rolling factor models
- models/: POET covariance, expected returns
- portfolio/: MVO, Risk Parity, Enhanced HRP, risk decomposition
- generate_analytics.py: Produces analytics_report.xlsx (8 sheets)

## Requirements
Python 3.11, wrds, pandas, numpy, scipy, cvxpy, matplotlib, openpyxl, yfinance, fredapi, statsmodels

## Period
2004 Q4 to 2024 Q4, 81 complete quarters, all 13 assets present
