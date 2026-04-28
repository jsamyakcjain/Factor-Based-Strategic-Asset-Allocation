from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from config.settings import FACTOR_COLORS, ASSET_COLORS, METHOD_COLORS
from data.data_manager import DataManager
from factors.factor_model import OLSFactorModel
from models.covariance import POETCovariance
from models.returns import ExpectedReturns
from portfolio.mvo import MVO
from portfolio.risk_parity import RiskParity
from portfolio.hrp import EnhancedHRP
from portfolio.risk_decomp import FactorRiskDecomposition

st.set_page_config(
    page_title="Factor-Based SAA",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading data and running models...")
def load_all():
    dm = DataManager(use_cache=True)
    dm.load_cached()
    ols = OLSFactorModel(
        dm.factor_returns_t1,
        dm.asset_returns_t1,
        credit_liquidity=dm.credit_liquidity,
    )
    result = ols.fit()
    poet = POETCovariance(
        dm.factor_returns_t1,
        dm.asset_returns_t1_complete,
        result.betas,
    )
    poet.fit()
    cov = poet.as_dataframe()
    F = dm.factor_returns_t1.astype(float)
    factor_cov = pd.DataFrame(
        np.cov(F.values.T),
        index=F.columns,
        columns=F.columns,
    )
    er  = ExpectedReturns(assets=list(cov.index))
    mu  = er.quarterly()
    mvo_w = MVO(mu, cov).fit()
    rp_w  = RiskParity(cov).fit()
    hrp_w = EnhancedHRP(cov, result.betas).fit()
    rd = FactorRiskDecomposition(result.betas, factor_cov, cov)
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
    return {
        "dm": dm, "result": result, "cov": cov,
        "factor_cov": factor_cov, "mu": mu,
        "portfolios": portfolios, "decomp": decomp, "ols": ols,
    }

data       = load_all()
dm         = data["dm"]
result     = data["result"]
cov        = data["cov"]
portfolios = data["portfolios"]
decomp     = data["decomp"]

st.title("Hidden Factor Concentration in Multi-Asset Portfolios")
st.markdown("**Factor-Based Strategic Asset Allocation**")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Portfolio Overview",
    "Factor Attribution",
    "Asset Deep Dive",
    "Construction Comparison",
    "Factor Model",
    "Cluster Stability Monitor",
])

# ── TAB 1 ──────────────────────────────────────────────────────────
with tab1:
    st.header("Portfolio Allocation")

    def weight_bar(weights, title, color):
        w = weights.dropna().sort_values(ascending=True)
        w = w[w > 0.001]
        fig = go.Figure(go.Bar(
            x=w.values*100, y=w.index, orientation="h",
            marker_color=color,
            text=[f"{v:.1f}%" for v in w.values*100],
            textposition="outside",
        ))
        fig.update_layout(
            title=title, height=350,
            margin=dict(l=130,r=60,t=40,b=20),
            xaxis_title="Weight (%)",
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white",
        )
        return fig

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(weight_bar(portfolios["MVO"],"MVO",METHOD_COLORS["mvo"]), use_container_width=True)
    with c2:
        st.plotly_chart(weight_bar(portfolios["Risk Parity"],"Risk Parity",METHOD_COLORS["risk_parity"]), use_container_width=True)
    with c3:
        st.plotly_chart(weight_bar(portfolios["Enhanced HRP"],"Enhanced HRP",METHOD_COLORS["hrp"]), use_container_width=True)

    st.divider()
    st.subheader("Portfolio Metrics")
    er = ExpectedReturns(assets=list(cov.index))
    mu_q = er.quarterly()

    def port_metrics(weights):
        w = weights.reindex(cov.index).fillna(0).values
        mu_v = mu_q.reindex(cov.index).fillna(0).values
        ann_ret = float(w @ mu_v) * 4 * 100
        ann_vol = float(np.sqrt(w @ cov.values @ w)) * 2 * 100
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
        return {"Ann. Return": f"{ann_ret:.1f}%", "Ann. Vol": f"{ann_vol:.1f}%", "Sharpe": f"{sharpe:.2f}"}

    cols = st.columns(5)
    for i, (name, w) in enumerate(portfolios.items()):
        m = port_metrics(w)
        with cols[i]:
            st.markdown(f"**{name}**")
            for k, v in m.items():
                st.metric(k, v)

# ── TAB 2 ──────────────────────────────────────────────────────────
with tab2:
    st.header("Factor Risk Decomposition")

    factor_cols = ["Equity Premium","Term Premium","Credit Spread","Inflation","Liquidity","Idiosyncratic"]
    colors = [
        FACTOR_COLORS["equity_premium"], FACTOR_COLORS["term_premium"],
        FACTOR_COLORS["credit_spread"],  FACTOR_COLORS["inflation"],
        FACTOR_COLORS["liquidity"],      FACTOR_COLORS["idiosyncratic"],
    ]

    fig = go.Figure()
    for col, color in zip(factor_cols, colors):
        if col in decomp.columns:
            fig.add_trace(go.Bar(
                name=col, x=decomp.index.tolist(),
                y=decomp[col].values, marker_color=color,
                text=[f"{v:.1f}%" for v in decomp[col].values],
                textposition="inside",
            ))
    fig.update_layout(
        barmode="stack", height=450,
        title="Factor Risk Attribution by Portfolio Method",
        yaxis_title="% of Total Portfolio Risk",
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decomposition Table")
    st.dataframe(decomp.style.format("{:.1f}%"), use_container_width=True)

    st.subheader("Equity Premium Concentration")
    erp_vals = decomp["Equity Premium"] if "Equity Premium" in decomp.columns else decomp.iloc[:,0]
    fig2 = go.Figure(go.Bar(
        x=erp_vals.index.tolist(), y=erp_vals.values,
        marker_color=["#C0392B" if v>50 else "#E67E22" if v>40 else "#27AE60" for v in erp_vals.values],
        text=[f"{v:.1f}%" for v in erp_vals.values], textposition="outside",
    ))
    fig2.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% threshold")
    fig2.update_layout(
        height=350, yaxis_title="Equity Premium Risk Share (%)",
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3 ──────────────────────────────────────────────────────────
with tab3:
    st.header("Asset Class Factor Profiles")
    selected = st.selectbox("Select Asset", list(result.betas.index))

    c1, c2 = st.columns(2)
    with c1:
        betas = result.betas.loc[selected]
        cats  = list(betas.index)
        vals  = [abs(v) for v in betas.values]
        fig3 = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself",
            fillcolor=ASSET_COLORS.get(selected,"#4CAF50"),
            opacity=0.3,
            line_color=ASSET_COLORS.get(selected,"#4CAF50"),
        ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True), bgcolor="#1e2130"),
            title=f"{selected} Factor Profile", height=400,
            paper_bgcolor="#1e2130", font_color="white",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.subheader("Factor Betas")
        beta_df = pd.DataFrame({
            "Beta":    result.betas.loc[selected].round(4),
            "T-Stat":  result.t_stats.loc[selected].round(2),
            "P-Value": result.p_values.loc[selected].round(4),
            "Sig":     result.p_values.loc[selected].apply(
                lambda p: "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
            ),
        })
        st.dataframe(beta_df, use_container_width=True)
        st.metric("Adjusted R²", f"{result.r_squared[selected]:.3f}")

    st.subheader("Historical Quarterly Returns")
    ar = dm.asset_returns_t1[selected].dropna()
    fig4 = go.Figure(go.Bar(
        x=ar.index, y=ar.values*100,
        marker_color=[ASSET_COLORS.get(selected,"#4CAF50") if v>=0 else "#C0392B" for v in ar.values],
    ))
    fig4.update_layout(
        height=300, yaxis_title="Quarterly Return (%)",
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── TAB 4 ──────────────────────────────────────────────────────────
with tab4:
    st.header("Portfolio Construction")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Diversification Ratio")
        div_ratios = {}
        for name, w in portfolios.items():
            wv = w.reindex(cov.index).fillna(0).values
            weighted_vol = float(np.sum(wv * np.sqrt(np.diag(cov.values))))
            port_vol     = float(np.sqrt(wv @ cov.values @ wv))
            div_ratios[name] = weighted_vol/port_vol if port_vol>0 else 1
        fig5 = go.Figure(go.Bar(
            x=list(div_ratios.keys()), y=list(div_ratios.values()),
            marker_color=[METHOD_COLORS.get(k.lower().replace(" ","_").replace("/","_"),"#4CAF50") for k in div_ratios],
            text=[f"{v:.2f}" for v in div_ratios.values()], textposition="outside",
        ))
        fig5.update_layout(
            height=350, yaxis_title="Diversification Ratio",
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white",
        )
        st.plotly_chart(fig5, use_container_width=True)

    with c2:
        st.subheader("POET Correlation Matrix")
        std  = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(std, std)
        corr_df = pd.DataFrame(corr, index=cov.index, columns=cov.columns)
        fig6 = px.imshow(corr_df.round(2), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig6.update_layout(height=350, plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white")
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Weight Comparison")
    wdf = pd.DataFrame({
        name: w.reindex(cov.index).fillna(0)*100
        for name, w in portfolios.items()
    })
    fig7 = go.Figure()
    for col in wdf.columns:
        color = METHOD_COLORS.get(col.lower().replace(" ","_").replace("/","_"),"#4CAF50")
        fig7.add_trace(go.Bar(name=col, x=wdf.index.tolist(), y=wdf[col].values, marker_color=color))
    fig7.update_layout(
        barmode="group", height=400, yaxis_title="Weight (%)",
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white",
        legend=dict(orientation="h", y=1.02), xaxis_tickangle=-30,
    )
    st.plotly_chart(fig7, use_container_width=True)

# ── TAB 5 ──────────────────────────────────────────────────────────
with tab5:
    st.header("Factor Model Results")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("OLS Beta Matrix")
        st.dataframe(result.betas.style.format("{:.3f}"), use_container_width=True)

    with c2:
        st.subheader("Adjusted R-Squared")
        fig8 = go.Figure(go.Bar(
            x=result.r_squared.index.tolist(),
            y=result.r_squared.values,
            marker_color=[ASSET_COLORS.get(a,"#4CAF50") for a in result.r_squared.index],
            text=[f"{v:.3f}" for v in result.r_squared.values], textposition="outside",
        ))
        fig8.update_layout(
            height=350, yaxis_title="Adjusted R²", yaxis_range=[0,1.1],
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white", xaxis_tickangle=-30,
        )
        st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Rolling Equity Premium Beta")
    from factors.factor_model import RollingFactorModel

    @st.cache_resource
    def get_rolling():
        rm = RollingFactorModel(dm.factor_returns_t1, dm.asset_returns_t1, window=20)
        rm.fit()
        return rm

    rm = get_rolling()
    asset_roll = st.selectbox("Select asset", list(rm.rolling_betas.keys()), key="roll")
    if asset_roll in rm.rolling_betas:
        rdf = rm.rolling_betas[asset_roll]
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(
            x=rdf.index, y=rdf["equity_premium"],
            mode="lines", name="Rolling ERP Beta",
            line=dict(color=FACTOR_COLORS["equity_premium"], width=2),
        ))
        fig9.add_hline(
            y=result.betas.loc[asset_roll,"equity_premium"],
            line_dash="dash", line_color="white", annotation_text="Full-sample OLS",
        )
        fig9.update_layout(
            height=350, yaxis_title="Equity Premium Beta",
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white",
        )
        st.plotly_chart(fig9, use_container_width=True)

    st.subheader("Factor Summary Statistics")
    stats = (dm.factor_returns_t1*100).describe().loc[["mean","std","min","max"]].T
    stats.columns = ["Mean%","Std%","Min%","Max%"]
    st.dataframe(stats.round(3), use_container_width=True)

# ── TAB 6 ──────────────────────────────────────────────────────────
with tab6:
    st.header("Stochastic Rebalancing Monitor")
    st.markdown(
        "Detects when factor distance between clustered assets has contracted — "
        "indicating the diversification benefit is eroding. "
        "Uses trailing 8-quarter baseline and 2-quarter persistence filter."
    )

    @st.cache_resource
    def get_monitor():
        from portfolio.cluster_monitor import ClusterMonitor
        from factors.factor_model import RollingFactorModel

        rm = RollingFactorModel(
            dm.factor_returns_t1,
            dm.asset_returns_t1,
            window=20,
        )
        rm.fit()

        portfolio_clusters = {}

        hrp_order = list(portfolios["Enhanced HRP"].sort_values(ascending=False).index)
        portfolio_clusters["Enhanced HRP"] = ClusterMonitor.clusters_from_hrp(hrp_order, n_clusters=3)

        mvo_sorted = portfolios["MVO"].sort_values(ascending=False)
        top_mvo  = list(mvo_sorted[mvo_sorted > 0.10].index)
        rest_mvo = list(mvo_sorted[mvo_sorted <= 0.10].index)
        portfolio_clusters["MVO"] = {
            "core":      top_mvo  if top_mvo  else list(mvo_sorted.index[:3]),
            "satellite": rest_mvo if rest_mvo else list(mvo_sorted.index[3:]),
        }

        rp_order = list(portfolios["Risk Parity"].sort_values(ascending=False).index)
        portfolio_clusters["Risk Parity"] = ClusterMonitor.clusters_from_hrp(rp_order, n_clusters=3)

        monitors = {}
        for port_name, clusters in portfolio_clusters.items():
            m = ClusterMonitor(
                rolling_betas=rm.rolling_betas,
                cluster_assignments=clusters,
                baseline_window=8,
                monitor_threshold=0.50,
                rebalance_threshold=0.85,
                persistence=2,
            )
            m.compute()
            monitors[port_name] = (m, clusters)

        return monitors
    with st.spinner("Computing cluster stability..."):
        monitors = get_monitor()
    port_select = st.selectbox(
        "Select Portfolio",
        list(monitors.keys()),
        key="monitor_port"
    )
    monitor, clusters = monitors[port_select]
    
    status = monitor.current_status()
    signal_colors = {"HOLD":"#27AE60","MONITOR":"#E67E22","REBALANCE":"#C0392B"}

    st.subheader("Current Status")
    cols = st.columns(len(status))
    for i, (cluster, info) in enumerate(status.items()):
        sig   = info["signal"]
        prob  = info["probability"]
        color = signal_colors.get(sig,"#27AE60")
        with cols[i]:
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"border-radius:8px;padding:16px;'>"
                f"<p><b>{cluster.upper()}</b></p>"
                f"<p style='font-size:20px'>P = {prob:.1%}</p>"
                f"<p style='font-size:11px;color:#888'>{info['date']}</p>"
                f"<p style='font-size:11px'>{', '.join(clusters[cluster])}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Rebalancing Probability Over Time")
    cluster_colors = {"growth":"#E74C3C","diversifier":"#F39C12","defensive":"#2ECC71"}

    fig10 = go.Figure()
    for cluster in monitor.probabilities.columns:
        ps = monitor.probabilities[cluster].dropna()
        fig10.add_trace(go.Scatter(
            x=ps.index, y=ps.values*100, mode="lines",
            name=cluster.capitalize(),
            line=dict(color=cluster_colors.get(cluster,"#4CAF50"), width=2),
        ))
    fig10.add_hline(y=85, line_dash="dash", line_color="#C0392B", annotation_text="REBALANCE 85%")
    fig10.add_hline(y=50, line_dash="dot",  line_color="#E67E22", annotation_text="MONITOR 50%")

    if dm.recession is not None:
        rec = dm.recession.reindex(monitor.probabilities.index).fillna(0)
        in_rec = False
        rec_start = None
        for date, val in rec.items():
            if val==1 and not in_rec:
                rec_start=date; in_rec=True
            elif val==0 and in_rec:
                fig10.add_vrect(x0=rec_start, x1=date, fillcolor="gray", opacity=0.15, line_width=0)
                in_rec=False

    fig10.update_layout(
        height=400, yaxis_title="Rebalancing Probability (%)", yaxis_range=[0,105],
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white",
        legend=dict(orientation="h", y=1.02), hovermode="x unified",
    )
    st.plotly_chart(fig10, use_container_width=True)

    st.subheader("Signal History")
    signal_map  = {"HOLD":0,"MONITOR":1,"REBALANCE":2}
    signal_nums = monitor.signals.replace(signal_map)
    fig11 = go.Figure(go.Heatmap(
        z=signal_nums.T.values,
        x=signal_nums.index.astype(str),
        y=signal_nums.columns.tolist(),
        colorscale=[[0.0,"#27AE60"],[0.5,"#E67E22"],[1.0,"#C0392B"]],
        showscale=False,
    ))
    fig11.update_layout(
        height=200, plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
        font_color="white", xaxis=dict(tickangle=-45, nticks=20),
    )
    st.plotly_chart(fig11, use_container_width=True)

    st.subheader("Within-Cluster Factor Distance")
    fig12 = go.Figure()
    for cluster in monitor.cluster_stability.columns:
        stab = monitor.cluster_stability[cluster].dropna()
        fig12.add_trace(go.Scatter(
            x=stab.index, y=stab.values, mode="lines",
            name=cluster.capitalize(),
            line=dict(color=cluster_colors.get(cluster,"#4CAF50"), width=2),
        ))
    fig12.update_layout(
        height=350, yaxis_title="Mean Factor Distance",
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white",
        legend=dict(orientation="h", y=1.02), hovermode="x unified",
    )
    st.plotly_chart(fig12, use_container_width=True)

    with st.expander("Methodology"):
        st.markdown("""
        **Signal Formula**

        P(rebalance, t) = Φ((d_baseline − d_current) / σ)

        - d_baseline = trailing 8-quarter mean within-cluster distance
        - d_current  = current within-cluster distance
        - σ = trailing 8-quarter std
        - Φ = standard normal CDF

        **Persistence Filter:** REBALANCE requires P ≥ 85% for 2 consecutive quarters.

        **Trailing Baseline:** Adapts to regime changes — signals identify acute shocks not slow drift.
        """)
