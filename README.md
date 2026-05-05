# Factor-Based Strategic Asset Allocation — Project Documentation

**MFE 2026 Spring | Factor-Based Approaches to Portfolio Management**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How to Run](#2-how-to-run)
3. [Data Sources & Pipeline](#3-data-sources--pipeline)
4. [Five-Factor Model Construction](#4-five-factor-model-construction)
5. [Asset Universe](#5-asset-universe)
6. [Factor Model Estimation](#6-factor-model-estimation)
7. [POET Covariance Matrix](#7-poet-covariance-matrix)
8. [Expected Returns](#8-expected-returns)
9. [Portfolio Construction](#9-portfolio-construction)
10. [Factor Risk Decomposition](#10-factor-risk-decomposition)
11. [Walk-Forward Backtest](#11-walk-forward-backtest)
12. [Performance Metrics](#12-performance-metrics)
13. [Analytics Report (Excel)](#13-analytics-report-excel)
14. [Code Architecture](#14-code-architecture)
15. [Key Design Decisions & Rationale](#15-key-design-decisions--rationale)
16. [Known Limitations](#16-known-limitations)

---

## 1. Project Overview

This project implements a **Factor-Based Strategic Asset Allocation (SAA)** framework across 13 asset classes. The core idea: instead of treating assets as black boxes and optimizing purely on return and volatility histories, we decompose each asset's return into five systematic risk premia (factors) plus an idiosyncratic residual. Portfolio construction then explicitly manages *factor risk exposure*, not just asset-level weights.

**What gets produced:**
- An 8-sheet Excel analytics report (`analytics_report.xlsx`)
- Five portfolio strategies compared side-by-side on factor risk, performance, and drawdown
- Honest out-of-sample performance via a walk-forward backtest (no look-ahead bias)

**Pipeline at a glance:**

```
WRDS ──► FF Factors, CRSP market, PS Liquidity, Treasury returns
FRED ──► BAA/AAA yields, CPI, T-bill rate, Recession indicator
yfinance ► ETF returns (mid cap, small cap, EM, bonds, REITs, commodities)
Professor► Private market returns (PE, RE, HF, HY credit)
               │
               ▼
          DataManager ── builds quarterly 81×5 factor panel, 81×13 asset panel
               │
               ▼
    OLSFactorModel ──► β matrix (13×5), α, R², HAC t-stats
    QuantileFactorModel ► Q10/Q50/Q90 conditional betas
               │
               ▼
    POETCovariance ──► 13×13 Σ_POET (EWM-weighted, soft-thresholded)
               │
               ▼
    MVO | RiskParity | EnhancedHRP | EqualWeight | 60/40
               │
               ▼
    FactorRiskDecomposition ──► % risk by factor per portfolio
               │
               ▼
    WalkForwardBacktest ──► 61 OOS quarters, honest Sharpe/Sortino/TE
               │
               ▼
    generate_analytics.py ──► analytics_report.xlsx (8 sheets)
```

---

## 2. How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
FRED_API_KEY=your_fred_api_key_here
WRDS_USERNAME=your_wrds_username_here
```

Also requires `data_cache/professor_dataset.xlsx` — the professor-provided private market dataset (copy `dataset.xlsx` from the project root into `data_cache/` and rename it).

### First run (fetches all data from scratch)

```bash
python generate_analytics.py
```

On first run the script fetches from WRDS (requires institutional VPN access), FRED, and yfinance. This takes approximately 5–10 minutes. All data is cached as `.parquet` files under `data_cache/` so subsequent runs load instantly.

### Subsequent runs

```bash
python generate_analytics.py
```

Loads from cache. Completes in approximately 2–3 minutes (dominated by the 61-step walk-forward backtest).

### Force a cache rebuild

Delete the relevant cache files and re-run. To rebuild everything from raw sources:

```bash
rm data_cache/*.parquet
python generate_analytics.py
```

To rebuild only the data that changed (e.g., after updating the liquidity or commodities proxy):

```bash
rm data_cache/fred_credit_spreads.parquet data_cache/credit_liquidity.parquet
rm data_cache/commodities_spliced.parquet data_cache/market_etf_returns.parquet
python generate_analytics.py
```

The script automatically detects incomplete caches and calls `dm.build()` to rebuild. If all 8 required parquet files exist, it calls `dm.load_cached()` instead.

---

## 3. Data Sources & Pipeline

### 3.1 Data Sources

| Source | Series fetched | Used for |
|--------|---------------|---------|
| **WRDS / CRSP** | `crsp.msi` — value-weighted return (vwretd) | US Large Cap proxy |
| **WRDS / CRSP** | `crsp.cs20yr` — 20Y Treasury total return, `crsp.cs90d` — 90D T-bill | Term premium factor |
| **WRDS / Fama-French** | `ff.factors_monthly` — MktRF, SMB, HML, RF | Equity premium factor |
| **WRDS / Pastor-Stambaugh** | `ff.liq_ps` — PS innovations (`ps_innov`) | Liquidity factor for equity assets |
| **FRED** | BAA (`BAA`) and AAA (`AAA`) Moody's corporate yields | Credit spread + quality spread |
| **FRED** | 10-year Treasury yield (`GS10`) | Credit spread denominator |
| **FRED** | CPI all-items (`CPIAUCSL`) | Inflation factor |
| **FRED** | 3-month T-bill rate (`TB3MS`) | Risk-free rate |
| **FRED** | NBER recession indicator (`USREC`) | Regime analysis |
| **yfinance** | IJH, IWM, EEM, TLT, TIP, LQD, VNQ | ETF returns for 7 public assets |
| **yfinance** | PCRIX (primary) + ^BCOM (backfill) | Commodities total return |
| **Professor dataset** | HY credit, PE, RE, Hedge Funds | Private/alternative assets |

### 3.2 Two-Tier Structure

| Tier | Assets | Date Range | Observations |
|------|--------|-----------|-------------|
| **Tier 1** | 9 public assets only | 2004-Q1 to 2024-Q4 | Up to 84 quarters |
| **Tier 2 (used for everything)** | All 13 assets | 2004-Q4 to 2024-Q4 | **81 quarters** |

The final aligned panel is Tier 2 after `dropna()` on the complete-case intersection. The first three quarters (2004-Q1 through Q3) are dropped because the professor's private market dataset starts in Q4 2004. All 13 assets have exactly **81 quarterly observations** (verified).

### 3.3 Frequency Conversion (Monthly → Quarterly)

The method of aggregation depends on what the series represents:

| Series type | Aggregation | Why |
|-------------|------------|-----|
| Asset returns, equity premium, PS liquidity | Geometric compounding: `(1+r).prod() - 1` | Returns compound multiplicatively |
| Credit spread changes, inflation (ΔCPI) | Arithmetic sum: `Δ.sum()` | Changes in levels add, not compound |
| Liquidity levels (for snapshotting) | Quarter-end: `.resample('QE').last()` | Level variable — take the point-in-time value |

**Common mistake:** Compounding spread differences gives wrong numbers (e.g., three months of +1bp, +1bp, +1bp compounded gives ~3.0003bp instead of 3bp). The code uses separate aggregation paths for each factor type.

---

## 4. Five-Factor Model Construction

Each factor is a stationary quarterly return series representing a specific risk premium. All factors are tested for stationarity (ADF) and multicollinearity (VIF) before use.

### Factor 1 — Equity Premium

```
equity_premium_t = MktRF_t   (Fama-French excess market return, quarterly geometric)
```

The excess return of the CRSP value-weighted market index over the 1-month T-bill rate. Source: WRDS `ff.factors_monthly`. This is the dominant factor — virtually all equity-like assets load heavily and positively on it.

### Factor 2 — Term Premium

```
term_premium_t = r(20Y Treasury Total Return)_t  −  r(90-day T-bill)_t
```

Represents compensation for bearing duration risk. Long-duration bonds load positively (they profit when term premium rises = rates fall); equities typically have near-zero or small negative loadings. Source: CRSP `cs20yr` and `cs90d`.

### Factor 3 — Credit Spread

```
credit_spread_t = −Δ(BAA yield − GS10 yield)_t    [quarterly arithmetic sum]
```

The **negative** change in the BAA-to-10Y spread. Sign convention: when credit spreads *narrow* (improvement in credit conditions), the factor is *positive*, and assets with positive credit betas benefit. HY credit and IG credit load positively; treasuries load negatively.

Source: FRED `BAA` and `GS10`. We use Moody's BAA–GS10 as the HY OAS proxy because the ICE BofA OAS series was restricted on FRED starting 2022.

### Factor 4 — Inflation

```
inflation_t = ΔCPI_t    [quarterly arithmetic sum of monthly CPI percent changes]
```

The rate of change of CPI all-items. TIPS load positively (inflation protection); nominal bonds load negatively. Source: FRED `CPIAUCSL`.

### Factor 5 — Liquidity

Two different proxies depending on asset type:

| Asset group | Proxy | Source | Economic meaning |
|-------------|-------|--------|-----------------|
| US LC, MC, SC, EM equity, REITs, Commodities | Pastor-Stambaugh innovations (`ps_innov`) | WRDS `ff.liq_ps` | Equity market microstructure liquidity: bid-ask tightness, order-flow reversals |
| Long Treasury, TIPS, IG, HY, PE, RE, Hedge Funds | `−Δ(BAA − AAA)` | FRED `BAA` and `AAA` | Credit market flight-to-quality: when quality spread widens, credit liquidity is tightening |

**Why two proxies?** Equity assets respond to *equity market* liquidity (can I trade my S&P 500 position at tight spreads?). Credit and alternative assets respond to *credit market* liquidity (are funding markets open? are investors fleeing to safety within credit?). NFCI was previously used for credit assets but was replaced because it blends equity and credit market signals — BAA-AAA quality spread is purely credit-specific and has a 100-year FRED history.

**Sign convention for quality spread:** `−Δ(BAA − AAA)` means the factor is positive when the quality spread *narrows* (credit conditions ease) and negative when it *widens* (flight to safety). Assets that benefit from easy credit conditions (HY credit, PE, private RE) load positively on this factor.

---

## 5. Asset Universe

All 13 assets have **81 quarterly observations from 2004-Q4 to 2024-Q4** — a fully balanced panel.

| # | Asset | Source/Ticker | Class | Liquidity proxy |
|---|-------|--------------|-------|----------------|
| 1 | US Large Cap Equity | CRSP vwretd | Public equity | Pastor-Stambaugh |
| 2 | US Mid Cap Equity | IJH | Public equity | Pastor-Stambaugh |
| 3 | US Small Cap Equity | IWM | Public equity | Pastor-Stambaugh |
| 4 | Emerging Market Equity | EEM | Public equity | Pastor-Stambaugh |
| 5 | Long Duration Treasury | TLT | Public bond | AAA-BAA spread |
| 6 | TIPS | TIP | Inflation-linked bond | AAA-BAA spread |
| 7 | Investment Grade Credit | LQD | Corporate bond | AAA-BAA spread |
| 8 | REITs | VNQ | Real assets / equity | Pastor-Stambaugh |
| 9 | Commodities | PCRIX + ^BCOM | Real assets | Pastor-Stambaugh |
| 10 | High Yield Credit | Professor dataset | Alternative credit | AAA-BAA spread |
| 11 | Hedge Funds | Professor dataset | Liquid alternatives | AAA-BAA spread |
| 12 | Private Equity (unsmoothed) | Professor dataset | Private equity | AAA-BAA spread |
| 13 | Private Real Estate (unsmoothed) | Professor dataset | Private real estate | AAA-BAA spread |

**Commodities data construction:** PCRIX (PIMCO Bloomberg Commodity Total Return Fund, inception June 2002) is the primary series from 2002-Q4 onwards. The Bloomberg Commodity Index (^BCOM) backfills pre-2002. This replaced the former S&P GSCI (GSG/^SPGSCI) which had approximately 65-70% energy weighting — BCOM caps any single commodity at 33% and any sector at 33%, making it a more representative broad-commodity index.

**Private asset unsmoothing — Geltner (1993):** Private equity and real estate indices are reported with appraisal smoothing. Appraisers use stale or averaged valuations, which artificially reduces measured volatility and correlations with public markets. We recover the "true" economic return series using:

```
r_true_t = (r_observed_t − α × r_observed_{t-1}) / (1 − α)
```

where α is the first-order autocorrelation coefficient of the observed series. Without unsmoothing, MVO would over-allocate to PE/RE (thinking they are low-risk), and factor betas would be biased toward zero (making the factor model invalid).

---

## 6. Factor Model Estimation

### 6.1 OLS with HAC Standard Errors

For each of the 13 assets, we estimate:

```
R_{i,t} = α_i + β_{i,1}·equity_premium_t + β_{i,2}·term_premium_t
                + β_{i,3}·credit_spread_t  + β_{i,4}·inflation_t
                + β_{i,5}·liquidity_t      + ε_{i,t}
```

**Two regressions per asset:**
1. **Raw (unstandardized) factors** → gives true alpha (in return units) and unstandardized betas used for POET covariance, risk decomposition, and portfolio construction
2. **Standardized factors** → gives comparable t-statistics and VIF across factors (standardized using each factor's own mean and std — for credit assets, the quality spread's mean/std is used for the liquidity column, not PS liquidity's parameters)

**HAC standard errors:** `sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})` — Newey-West with 4 lags corrects for serial correlation in quarterly return data. Without HAC, t-statistics would be inflated in the presence of autocorrelated residuals.

**Liquidity proxy swap:** For credit/alternative assets, the `liquidity` column in the factor matrix is replaced with `credit_liquidity` (quality spread changes) before any estimation, ensuring each asset is regressed against the economically appropriate proxy.

### 6.2 Observed OLS Betas (Current Values)

```
Asset                    ERP β   TERM β   CRED β   INFL β   LIQ β    R²
US Large Cap             0.972   small    small    small    small    high
US Mid Cap               1.000   small    small    small    small    high
US Small Cap             1.143   small    small    small    small    high
EM Equity                0.791   small    small    small    small    moderate
Long Treasury            0.009   positive negative small    small    moderate
TIPS                     0.103   positive negative positive small    moderate
IG Credit                0.246   moderate negative small    small    moderate
REITs                    0.957   small    small    small    small    moderate
Commodities              0.445   small    small    positive small    low
HY Credit                0.240   small    positive small    positive moderate
Hedge Funds              0.281   small    small    small    small    low
Private Equity           0.416   small    small    small    small    moderate
Private Real Estate      0.174   small    small    small    small    low
```

Key observations: Large/Mid/Small cap are essentially levered versions of the equity premium. Long Treasury has near-zero equity beta and positive term beta (as expected). TIPS are inflation-protected. Private assets have lower R² because unsmoothing doesn't fully recover the timing of economic shocks.

### 6.3 Quantile Factor Model — Q10/Q50/Q90

We additionally estimate quantile regressions at τ = {0.10, 0.50, 0.90} using `statsmodels.QuantReg`:

```
Q_τ[R_{i,t} | F_t] = α_{i,τ} + β_{i,τ}' F_t
```

The Q10 quantile regression estimates the conditional **10th percentile** of the asset's return distribution given the factor values. This is used to analyze whether an asset's *left tail* is predominantly factor-driven (systematic) or idiosyncratic.

**Correct interpretation (important):** Q10 quantile regression ≠ "beta during market crashes." It estimates the relationship between factors and the 10th percentile of the conditional return distribution across all observations. A positive Q10 uplift (Q10β > OLS β) means the asset's worst returns are *more* factor-sensitive than its average returns — co-crash behavior. A negative uplift means the worst returns are driven by idiosyncratic shocks independent of the factor.

| Q10 uplift = Q10β − OLSβ | Interpretation |
|--------------------------|----------------|
| > +0.15 | Co-crash: factor-driven downside (left tail aligns with equity crashes) |
| +0.05 to +0.15 | Moderate co-crash tendency |
| −0.05 to +0.05 | Symmetric tail sensitivity |
| −0.15 to −0.05 | Idiosyncratic downside risk |
| < −0.15 | Strongly idiosyncratic downside |

**Our results and economic rationale:**

| Asset | OLS β | Q10 β | Uplift | Interpretation | Economic rationale |
|-------|-------|-------|--------|---------------|-------------------|
| EM Equity | 0.791 | 1.001 | +0.210 | Co-crash | EM crashed disproportionately in GFC and COVID — tail IS factor driven |
| Priv. RE (unsmthd) | 0.174 | 0.369 | +0.195 | Co-crash | GFC was a real estate crisis — worst RE periods coincide with equity crashes |
| HY Credit | 0.240 | 0.335 | +0.095 | Moderate co-crash | HY spreads blow out sharply during equity market stress |
| REITs | 0.957 | 0.790 | −0.167 | Strongly idiosyncratic | REITs crash from rising rates (2022: bad REITs, flat equities) — not purely equity-premium driven |
| Commodities | 0.445 | 0.302 | −0.143 | Idiosyncratic | Supply-driven crashes (oil gluts, COVID demand shock) occur independent of equity premium |
| Priv. Equity | 0.416 | 0.303 | −0.113 | Idiosyncratic | PE deal-specific collapses don't always coincide with market-wide equity crashes |
| US Large Cap | 0.972 | 0.969 | −0.003 | Symmetric | US LC essentially IS the market — symmetric by construction |

---

## 7. POET Covariance Matrix

### 7.1 The POET Decomposition

POET (Principal Orthogonal complEment Thresholding) — Fan, Liao & Mincheva (2013) — estimates:

```
Σ_POET = B Σ_f B' + Σ_u
         ─────────   ────
         systematic  sparse idiosyncratic
```

- **B** (13×5): beta matrix from OLS
- **Σ_f** (5×5): factor covariance, exponentially weighted (decay = 0.94)
- **Σ_u** (13×13): residual covariance after soft-thresholding

This structure forces the covariance to respect the factor model. Systematic co-movements (driven by shared factor exposures) are separated from idiosyncratic noise (asset-specific residuals), which is then de-noised by sparsification.

### 7.2 EWM Factor Covariance (Σ_f)

```
Σ_f = exponentially weighted covariance of F with decay λ = 0.94
w_t = λ^(T-t),   normalized so Σw_t = 1
```

Effective sample size ≈ 1/(1−λ²) ≈ 8.5 quarters. This gives more weight to recent factor co-movements — appropriate for a SAA model that should be responsive to current market regimes.

**Critical:** The same `poet.factor_cov` object is used everywhere downstream — portfolio-level risk decomposition, per-asset factor risk shares, net factor exposure. If a different Σ_f were used (e.g., `np.cov()` with equal weights), the systematic variance could exceed the POET diagonal for some assets, producing factor risk shares above 100%.

### 7.3 Adaptive Thresholding (Fan & Liao 2013)

After computing the residual covariance `Σ_u`, convert to correlations, apply soft thresholding, and convert back:

```
τ = C · √(log p / T_eff)

C = clip(2.0 + 0.5·√(max(excess_kurtosis, 0)), 1.5, 4.0)
T_eff = 1 / Σ(w_t²)      [effective sample size under EWM]
```

C ≥ 2.0 is the theoretical minimum (Fan & Liao 2013). Heavier-tailed residuals (financial data) use larger C to threshold more aggressively — the adaptive component. A fixed C was previously used (incorrect); the current implementation computes it from the actual excess kurtosis of the residuals.

Soft thresholding:
```
Corr_u_thresh[i,j] = sign(Corr_u[i,j]) · max(|Corr_u[i,j]| − τ, 0)   for i ≠ j
Corr_u_thresh[i,i] = 1.0   (preserve diagonal)
```

Then convert back to covariance: `Σ_u_thresh = D · Corr_u_thresh · D`.

### 7.4 Positive Definiteness Fix

```python
Σ_POET = Σ_systematic + Σ_u_thresh
if min_eigenvalue(Σ_POET) < 1e-8:
    Σ_POET += (|min_eigenvalue| + 1e-6) · I
```

This ensures the matrix is strictly positive definite for the MVO solver.

---

## 8. Expected Returns

Source: **JPMorgan 2026 Long-Term Capital Market Assumptions (LTCMA)**

These are 10–15 year forward-looking annualized return forecasts calibrated to starting valuations. Used as MVO inputs only — realized backtest performance uses actual historical returns.

Conversion to quarterly excess returns for MVO:

```
μ_quarterly = (1 + μ_annual)^(1/4) − 1
rf_quarterly = (1 + 0.031)^(1/4) − 1     [RISK_FREE_RATE = 3.1% annual]
μ_excess_quarterly = μ_quarterly − rf_quarterly
```

**Selected JPM LTCMA values (annual):**

| Asset | Annual return |
|-------|-------------|
| US Large Cap | 6.7% |
| US Mid Cap | 6.8% |
| US Small Cap | 6.9% |
| EM Equity | 7.8% |
| Long Treasury | 5.2% |
| TIPS | 4.3% |
| IG Credit | 5.2% |
| REITs | 8.8% |
| Commodities | 4.6% |
| HY Credit | 6.1% |
| Hedge Funds | 4.97% |
| Private Equity | 10.2% |
| Private Real Estate | 9.15% |

**Why forward-looking and not historical?** Historical realized returns over 2004–2024 are dominated by the prolonged equity bull market (2009–2021) and the bond bull market (2009–2021). Using them in MVO produces extreme, unstable concentrations in whatever happened to do well. Forward-looking consensus estimates produce more stable, economically-grounded allocations for a 10+ year SAA horizon.

---

## 9. Portfolio Construction

Five strategies are constructed using the same Σ_POET and JPM μ:

### 9.1 Mean-Variance Optimization (MVO)

**Objective:** Maximize quadratic utility:
```
max_w   μ'w − (λ/2) · w'Σw
```

with λ = 3.0 (moderate risk aversion).

**Constraints:**
```
Σ w_i = 1            (fully invested)
w_i ≥ 0              (long only)
w_i ≤ 0.35           (max 35% single asset)
Σ_{i ∈ private} w_i ≤ 0.30   (max 30% private/alternatives)
```

Private assets = hedge_funds, private_equity_unsmthd, real_estate_unsmthd.

**Solver:** CVXPY with CLARABEL. The problem is convex (quadratic objective, linear constraints) and always has a unique global optimum when Σ is PD.

**Result:** MVO tilts toward high-return assets (PE, EM equity, HY credit, private RE) up to their caps. Both the 35% weight cap (HY credit) and the 30% private cap (PE+RE) are binding at the solution — MVO would allocate even more without constraints.

### 9.2 Risk Parity (Equal Risk Contribution)

**Objective:** Each asset contributes equally to total portfolio volatility:
```
TRC_i = w_i · (Σw)_i / σ_p = σ_p / n    for all i
```

Equivalently, minimize `Σᵢ(TRC_i − TRC_mean)²` via SLSQP on the full POET covariance.

```
Constraints: Σw_i = 1,  w_i ≥ 1e-6 (long only, no explicit caps)
Initial guess: inverse-volatility weights (1/σᵢ normalized)
```

**Implementation note — why not factor-level ERC?** An intuitive extension would be to target equal *factor* risk contributions (20% per factor). In practice this degenerates: equity premium return variance is 10–100× larger than inflation or liquidity return variance in `Σ_f`. The 20%-each target is mathematically unreachable for the small-variance factors, so the optimizer collapses to a corner solution with only 2 active assets. Standard asset-level ERC on the POET covariance is the industry-standard implementation of factor-aware risk parity (cf. Bridgewater All Weather) — factor diversification is achieved implicitly because highly correlated equity assets collectively compete for a shared risk budget through the covariance structure.

**Result:** Highly stable, low-volatility assets (TIPS 15.7%, Hedge Funds 13.1%, Private RE 12.6%) receive the largest allocations because equal risk contribution requires more capital in low-vol assets. All 13 assets receive meaningful weight. Expected returns are ignored entirely — this is a pure risk-balancing approach.

### 9.3 Enhanced HRP (Hierarchical Risk Parity)

Full Lopez de Prado (2016) HRP enhanced with a factor-residual distance metric:

**Step 1 — Factor-adjusted distance matrix:**

Remove the factor-driven covariance to isolate idiosyncratic co-movement:
```
Σ_idio = Σ_POET − B Σ_f B'        (strip out systematic co-movement)
ρ_idio = corr(Σ_idio)              (idiosyncratic correlation matrix)
d(i,j) = √(0.5 · (1 − ρ_idio[i,j]))   (Lopez de Prado distance)
```

Clustering on `ρ_idio` rather than raw returns groups assets by their *idiosyncratic* similarity. Without this adjustment, assets with similar high equity betas (US LC, MC, SC, EM, REITs) cluster together for the wrong reason — their shared factor exposure — and equity assets as a block dominate the left half of the dendrogram.

**Step 2 — Hierarchical clustering:** Ward linkage on the condensed distance matrix (`scipy.cluster.hierarchy.linkage`).

**Step 3 — Quasi-diagonalization:** Leaf ordering from the dendrogram (`scipy.cluster.hierarchy.leaves_list`) reorders assets so similar assets are adjacent.

**Step 4 — Recursive bisection:**
```
For each left/right cluster split at each level of the dendrogram:
    Var(cluster) = w_IVP' Σ_POET w_IVP    (IVP weights within the cluster)
    α = Var(right) / (Var(left) + Var(right))
    w_left  *= α                    (lower-variance cluster gets more capital)
    w_right *= 1 − α
```

**Step 5 — Weight cap:** After bisection, weights are clipped to a 20% per-asset maximum via iterative redistribution. This prevents extreme concentration in the lowest-volatility asset (TIPS) that pure IVP bisection produces when asset volatilities are heterogeneous (TIPS quarterly vol ≈ 3%, equities ≈ 8–10%).

**Result (current run):** 13 active assets; TIPS 20%, Hedge Funds 20% (capped); HY Credit 12.3%, Private RE 9.9%, IG Credit 9.4%. The factor-adjusted clustering produces a dendrogram order that groups US equity assets together (they share idiosyncratic structure), giving them a smaller collective budget than an equal-weight assignment would.

### 9.4 Equal Weight

```
w_i = 1/13 ≈ 7.69%   for all 13 assets
```

The "no information" benchmark. All assets equally weighted regardless of expected return, risk, or factor exposure.

### 9.5 60/40 Traditional Benchmark

```
60% in 5 equity assets:  US LC, MC, SC, EM Equity, REITs → 12% each
40% in 3 bond assets:    Long Treasury, TIPS, IG Credit   → 13.33% each
0%  in 5 alternatives:   Commodities, HY, HF, PE, RE      → 0% each
```

A traditional institutional benchmark with no alternatives. Used as the reference for active share and tracking error calculations.

---

## 10. Factor Risk Decomposition

### 10.1 Portfolio-Level Decomposition (Euler)

For portfolio weights w, betas B, factor covariance Σ_f:

```
β_p = B'w                               (5×1 portfolio factor exposure)
σ²_total = w'Σ_POET w                  (total portfolio variance)
σ²_systematic = β_p' Σ_f β_p           (variance explained by factors)
σ²_idiosyncratic = max(σ²_total − σ²_systematic, 0)

Per-factor contribution (Euler decomposition):
FC_k = β_p,k · (Σ_f β_p)_k            (contribution of factor k)

Normalization to 100%:
share_k = FC_k / (Σ_k FC_k + σ²_idiosyncratic)
```

The denominator is the sum of all components (not σ²_total), so that shares always sum to exactly 100% even in the edge case where σ²_systematic slightly exceeds σ²_total due to floating-point or model inconsistency.

### 10.2 Asset-Level Factor Risk Share

For individual asset i:
```
syst_i = β_i' Σ_f β_i          (systematic variance of asset i alone)
tot_i  = Σ_POET[i,i]           (total variance from POET diagonal)
share_i = min(syst_i / tot_i, 1.0)
```

The `min(..., 1.0)` clamp handles the case where the EWM Σ_f (down-weighting old observations) implies slightly higher systematic variance than the full-sample OLS betas predict — the clamp is a belt-and-suspenders safeguard.

### 10.3 Marginal and Total Risk Contribution (MRC/TRC)

```
MRC_i = (Σ_POET · w)_i / σ_p     (sensitivity of portfolio vol to small increase in w_i)
TRC_i = w_i · MRC_i               (total contribution of current position)
%Risk_i = TRC_i / σ_p × 100
```

**Note on zero-weight assets:** MRC is mathematically non-zero for assets with w_i = 0 (it measures the cost of *adding* that asset to the portfolio). In the Excel report, MRC is displayed as 0 for zero-weight assets to avoid confusion — the displayed value represents the current portfolio's risk budget, not a hypothetical addition.

### 10.4 Net Factor Exposure (Look-Through Beta)

```
β_portfolio[k] = Σ_i w_i · β_{i,k}   = (B'w)_k
```

This is the portfolio's effective beta to each factor — the "look-through" exposure that tells you what systematic risks you actually own.

---

## 11. Walk-Forward Backtest

### 11.1 Why Walk-Forward?

If we computed Sharpe, Sortino, and TE on full-sample returns using weights estimated on the full sample, we would have look-ahead bias: the portfolio weights at 2005 would implicitly know 2024's market data. The walk-forward backtest eliminates this by re-estimating the entire model at each step using only historically available data.

### 11.2 Procedure

```
Burn-in:    first 20 quarters (2004-Q4 to 2009-Q3)   — minimum estimation window
OOS period: next 61 quarters (2009-Q4 to 2024-Q4)   — honest performance evaluation

For t = 20, 21, ..., 80  (t indexes the next return to be predicted):
    F_t = factor_data.iloc[:t]          ← all factors up to end of quarter t-1
    R_t = asset_data.iloc[:t]           ← all assets up to end of quarter t-1

    1. Fit OLSFactorModel(F_t, R_t) → β_t, α_t
    2. Fit POETCovariance(F_t, R_t, β_t, α_t) → Σ_t
    3. Load JPM LTCMA μ (fixed — not re-estimated)
    4. Solve MVO, RiskParity, EnhancedHRP using Σ_t and μ → weights w_t
    5. r_{portfolio,t+1} = w_t · R_{t+1}    ← actual next-quarter return
```

Logging is suppressed during the loop (`logging.disable(logging.WARNING)`) to prevent RiskParity convergence messages from cluttering the output.

### 11.3 What Uses OOS vs Full-Sample Returns

| Metric / Output | Return series used | Reason |
|----------------|-------------------|--------|
| Sharpe, Sortino, Volatility | OOS (`wf_rets`) | No look-ahead bias |
| Max Drawdown, Calmar | OOS (`wf_rets`) | Honest drawdown |
| Tracking Error | OOS (`wf_rets`) | OOS TE vs OOS benchmark |
| Rolling Sharpe chart | OOS (`wf_rets`) | Consistent with stats table |
| Cumulative growth chart | Full-sample (`full_returns`) | Must show GFC 2008 (pre-OOS) |
| Stress testing | Full-sample (`full_returns`) | GFC occurs in burn-in period |

---

## 12. Performance Metrics

All statistics computed on **61 OOS quarters (2009-Q4 to 2024-Q4)** unless noted:

```
Annualized Return  = mean(r_OOS) × 4 × 100%
Annualized Vol     = std(r_OOS) × √4 × 100%

Sharpe Ratio       = (Ann Return − RF%) / Ann Vol
  RF = 3.1% annual (RISK_FREE_RATE in config/settings.py)

Sortino Ratio      = (Ann Return − RF%) / Downside Dev
  Downside Dev     = √(mean(min(r_t, 0)²)) × √4 × 100%
  [semi-deviation: only negative quarterly returns enter]

Max Drawdown       = min_t [ (cumulative_t − cumulative_peak_t) / cumulative_peak_t ]
Calmar Ratio       = Ann Return / |Max Drawdown|

Tracking Error     = std(r_portfolio − r_60/40) × √4 × 100%
Active Share       = 0.5 × Σᵢ |w_i^portfolio − w_i^{60/40}|
```

**Annualized volatility formula:** Vol scales as `σ_quarterly × √4` — not `(1+σ)^4 − 1`. The latter formula is for compounding returns, not scaling standard deviations. Using it would overstate annualized volatility for assets with high quarterly vol.

---

## 13. Analytics Report (Excel)

`analytics_report.xlsx` contains 8 sheets with dark-theme formatting:

| Sheet | Contents |
|-------|---------|
| **Executive Summary** | OOS performance table (Sharpe, Sortino, TE, etc.); factor risk attribution table with equity premium concentration color-coding; POET diagnostics |
| **Backtest Summary** | Full OOS metrics table; rolling 2-year Sharpe ratio chart (OOS); active share vs 60/40 table |
| **Portfolio Analytics** | Stacked factor risk bar chart; equity premium concentration bar chart; max drawdown and diversification ratio charts; cumulative growth of $100 (full-sample) |
| **Factor Model** | OLS betas, R², alpha, liquidity proxy per asset; Q10 vs OLS equity beta comparison with left-tail interpretation labels |
| **Stress Testing** | Cumulative returns during: GFC 2008Q3–2009Q1, COVID crash 2020Q1, Rate Shock 2022, GFC recovery 2009Q2–2010Q4 |
| **Risk Decomposition** | Weight, MRC, TRC, % portfolio risk, factor risk share per asset — for all 5 portfolios |
| **Covariance** | POET correlation matrix (heat-mapped); top 15 most extreme correlation pairs with classification |
| **Net Factor Exposure** | Look-through portfolio beta (B'w) per factor for all 5 portfolios |

---

## 14. Code Architecture

```
Factor_SAA/
├── config/
│   └── settings.py              constants: dates, FACTOR_NAMES, JPM_LTCMA, constraints
│
├── data/
│   ├── data_manager.py          orchestrates all loading and factor construction
│   ├── wrds_loader.py           CRSP (vwretd, 20Y treasury, 90D T-bill), FF factors, PS liquidity
│   ├── fred_loader.py           BAA/AAA/GS10, CPI, T-bill rate, NBER recession
│   ├── market_loader.py         7 ETFs via yfinance; PCRIX+^BCOM commodities splice
│   └── professor_loader.py      private market Excel — keyword header search + integer fallback
│
├── factors/
│   ├── factor_model.py          OLSFactorModel, QuantileFactorModel, RollingFactorModel
│   └── factor_proxies.py        ADF stationarity tests, VIF check, summary statistics
│
├── models/
│   ├── covariance.py            POETCovariance (EWM Σ_f, adaptive threshold Σ_u)
│   └── returns.py               ExpectedReturns (JPM LTCMA, quarterly excess conversion)
│
├── portfolio/
│   ├── mvo.py                   CVXPY/CLARABEL mean-variance optimization
│   ├── risk_parity.py           SLSQP asset-level ERC on POET covariance (FactorRiskParity class)
│   ├── hrp.py                   factor-residual distance → Ward linkage → quasi-diag → recursive bisection → 20% cap
│   └── risk_decomp.py           Euler risk decomposition, MRC/TRC, 60/40 and EW helpers
│
├── generate_analytics.py        master script: data → models → portfolios → Excel report
├── data_cache/                  parquet cache (auto-created; delete to force rebuild)
└── analytics_report.xlsx        output
```

### Class relationships

```python
# 1. Load data
dm = DataManager(use_cache=True)
dm.load_cached()               # or dm.build() on first run

# 2. Factor model
ols = OLSFactorModel(F, R, credit_liquidity=dm.credit_liquidity)
result = ols.fit()             # BetaResult: betas, t_stats, r_squared, alphas

qm = QuantileFactorModel(F, R, credit_liquidity=dm.credit_liquidity, quantiles=[0.10, 0.50, 0.90])
qm.fit()                       # qm.results[0.10].betas for Q10 equity betas

# 3. Covariance
poet = POETCovariance(F, R, beta_matrix=result.betas, decay=0.94, alphas=result.alphas)
poet.fit()
cov        = poet.as_dataframe()          # 13×13 Σ_POET
factor_cov = pd.DataFrame(poet.factor_cov, index=FACTOR_NAMES, columns=FACTOR_NAMES)  # 5×5 Σ_f

# 4. Expected returns
mu = ExpectedReturns(assets=list(cov.index)).excess_quarterly()

# 5. Portfolio construction
mvo_w  = MVO(mu, cov, risk_aversion=3.0).fit()
rp_w   = FactorRiskParity(cov, result.betas, factor_cov).fit()   # asset-level ERC on POET cov
hrp_w  = EnhancedHRP(cov, result.betas, factor_cov=factor_cov).fit()  # LdP HRP, 20% cap

# 6. Risk decomposition
rd = FactorRiskDecomposition(beta_matrix=result.betas, factor_cov=factor_cov, asset_cov=cov)
decomp = rd.compare(portfolios)    # DataFrame: portfolios × factors (values in %)
```

---

## 15. Key Design Decisions & Rationale

### Why POET and not Ledoit-Wolf or sample covariance?

| Method | Pros | Cons for this use case |
|--------|------|----------------------|
| Sample covariance | Unbiased (p=13 << T=81) | Ignores factor structure; residual noise enters full matrix |
| Ledoit-Wolf | Well-conditioned, shrinks toward constant-correlation | Shrinkage target (identity/constant correlation) is arbitrary — doesn't respect factor model |
| **POET** | Explicitly separates systematic from idiosyncratic; residual sparsified; theoretically motivated | Requires factor model estimated first |

Given we have a well-specified 5-factor model, POET is the theoretically correct choice. It enforces that factor-driven co-movements go through the factor structure, and residual noise is soft-thresholded.

### Why EWM (decay=0.94) for Σ_f?

Equal-weighted covariance over 81 quarters gives too much weight to 2008–2009 GFC (which dominated volatility in the sample). EWM with λ=0.94 gives roughly 8.5 effective quarters of weight — a covariance that is responsive to the current regime. This matches industry SAA practice (RiskMetrics convention).

### Why unsmoothed PE and RE?

Without unsmoothing:
- Measured volatility of PE/RE is ~5x lower than true economic volatility
- Beta vs equity premium approaches zero (artificially)
- MVO wildly over-allocates (perceived high return, low risk, low correlation)
- Risk decomposition shows <5% equity premium share for PE — economically wrong

With Geltner unsmoothing, PE and RE show more realistic betas (~0.4 and ~0.2 respectively), correlations, and factor exposures. This is academic best practice for private asset factor models.

### Why two liquidity proxies (PS and quality spread)?

PS liquidity measures equity *market microstructure* — primarily relevant for assets that trade on exchanges (equities, REITs, commodities). It captures bid-ask spreads and order flow reversals.

Credit and alternative assets are more sensitive to *credit market funding liquidity*. When the BAA-AAA quality spread widens, institutional investors are demanding more safety premium within credit markets — a direct measure of credit market stress. NFCI was previously used for credit assets but was replaced because NFCI blends equity, credit, and funding conditions and has lower signal specificity for credit-specific liquidity.

### Why JPM LTCMA and not historical returns for MVO?

Historical mean returns over 2004–2024 are upward-biased due to the longest equity bull market in history (2009–2021) and a bond market that benefited from falling rates for 40 years. Using historical means in MVO produces extreme, unstable, and backward-looking allocations. JPM's 10–15 year forward-looking estimates are calibrated to starting valuations (Shiller CAPE, yield levels, credit spreads) and represent equilibrium premia appropriate for SAA.

### Why a 20-quarter burn-in for walk-forward?

Rule of thumb: T ≥ 4k for OLS stability. With k = 5 factors + constant = 6 parameters, we need at least 24 observations, but factor betas are more stable above ~20 observations. Below 20 quarters, the POET covariance matrix becomes poorly conditioned (near-singular), causing MVO and HRP to produce nonsensical weights.

### Why asset-level ERC rather than factor-level ERC for Risk Parity?

Factor-level ERC (targeting 20% of systematic variance per factor) sounds appealing but fails in practice because factor return variances differ by orders of magnitude. Equity premium quarterly variance ≈ 0.0064 (8% vol); inflation quarterly variance ≈ 0.000009 (0.3% vol). To get 20% from inflation, the optimizer would need the portfolio's inflation beta to be ~27× its equity beta — impossible with real assets. The optimization collapses to 2 assets (TIPS + one other). Standard asset-level ERC on the POET covariance achieves factor diversification implicitly: because equity assets are highly correlated through shared factor structure, they collectively receive less budget than their count suggests, which is the economically correct outcome.

### Why a 20% weight cap in Enhanced HRP?

Pure recursive bisection uses IVP weights within each cluster. When asset volatilities are heterogeneous — TIPS quarterly vol ≈ 3%, equities ≈ 8–10% — IVP gives extreme weight to the lowest-vol asset within any cluster it touches. In practice this drives TIPS to 32% and hedge funds to 28% before capping. The 20% cap (with iterative proportional redistribution to uncapped assets) prevents any single asset from dominating while preserving the HRP rank ordering — the highest-ranked asset by HRP still receives the most weight (up to the cap). This is standard practice in production HRP implementations.

### Why forward-looking JPM returns are fixed across all OOS steps?

In a realistic SAA, investors would update capital market assumptions at each rebalancing. However, re-estimating μ at each OOS step would require a proprietary forecasting model (Bayesian updating, factor valuation model, etc.). Using fixed JPM 2026 LTCMA for all steps is a simplifying assumption that focuses the walk-forward on demonstrating the value of the covariance model and portfolio construction methodology, not of return forecasting.

### Why full-sample returns for stress testing and cumulative charts?

GFC (2008-Q3 to 2009-Q1) falls entirely within the 20-quarter burn-in period. There are no OOS returns for this period. If stress testing used OOS returns, GFC would simply not appear — making the stress test useless for the most important stress event in the sample. Full-sample returns (applying final portfolio weights backward) are labeled as "full-sample" in the report and used explicitly for illustrative purposes only.

---

## 16. Known Limitations

| Limitation | Impact level | Note |
|-----------|-------------|------|
| **Single Σ_f for all OOS steps** in walk-forward (each step re-estimates Σ but from growing history) | Low | Correct by design — each step uses all available data |
| **JPM LTCMA fixed across all OOS periods** (2010 weights use 2026 forecasts) | Moderate | Simplifying assumption; acknowledged in report |
| **Professor dataset selection/survivorship bias** (HY, HF returns may be upward-biased) | Moderate | Industry-standard datasets used; disclosed |
| **Unsmoothing removes bias but not all illiquidity premium** (PE/RE R² still understated) | Low | Academic best practice applied |
| **Commodities splice (^BCOM pre-2002, PCRIX post)** creates a structural break point | Low | Both track Bloomberg Commodity Index; break documented |
| **Q10 quantile regression with n=81** has ~8 effective observations at tail | Moderate | Presented as indicative, not precise; economic interpretation robust |
| **Hedge funds in private asset cap** (conservatively counts HF as illiquid) | Low | HF are liquid alternatives; conservative assumption |
| **No leverage or shorting constraints beyond long-only** for RP and HRP | None | By design — SAA is long-only |
| **Stress test periods use full-sample returns** (not OOS) | Disclosed | Required to show GFC which falls in burn-in period |
