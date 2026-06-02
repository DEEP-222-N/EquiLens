"""
EquiLens — Automated Equity Research & Financial Health Analyzer
Main Streamlit application for NSE-listed Indian companies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

from data_fetcher import get_stock_info, get_historical_prices, get_financials, extract_key_metrics, get_peer_data
from ratio_engine import (
    compute_all_ratios, compute_health_score, get_traffic_light,
    compute_dupont, compute_altman_z_score,
)
from dcf_model import compute_dcf, run_scenarios, sensitivity_matrix
from visualizations import (
    price_chart, ratio_trend_chart, dupont_chart, radar_chart,
    football_field_chart, sensitivity_heatmap, altman_gauge,
    health_score_gauge, peer_bar_chart, COLORS,
)
from narrative_generator import generate_narrative
from pdf_exporter import generate_pdf

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="EquiLens — Equity Research",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for dark theme polish ─────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-card {
        background: #1E2130;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #00D4AA;
    }
    .metric-label { color: #8892B0; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #FAFAFA; font-size: 24px; font-weight: 700; }
    .traffic-green { color: #00D4AA; font-weight: 600; }
    .traffic-amber { color: #FFE66D; font-weight: 600; }
    .traffic-red { color: #FF6B6B; font-weight: 600; }
    .section-header {
        font-size: 20px; font-weight: 700; color: #FAFAFA;
        border-bottom: 2px solid #00D4AA; padding-bottom: 8px; margin: 20px 0 16px;
    }
    div[data-testid="stSidebar"] { background-color: #161B22; }
    .stTabs [data-baseweb="tab"] { color: #8892B0; }
    .stTabs [aria-selected="true"] { color: #00D4AA; }
    .landing-container {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 60vh; text-align: center;
    }
    .landing-title { font-size: 48px; font-weight: 800; color: #00D4AA; margin-bottom: 8px; }
    .landing-sub { font-size: 18px; color: #8892B0; max-width: 500px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔍 EquiLens")
    st.markdown("*Equity Research Analyzer*")
    st.divider()

    ticker = st.text_input("NSE Ticker", value="TCS.NS", help="Enter ticker with .NS suffix (e.g., RELIANCE.NS)")
    if not ticker.endswith((".NS", ".BO")):
        ticker = ticker + ".NS"

    st.divider()
    st.markdown("### Peer Comparison")
    peers_input = st.text_input("Peer Tickers (comma-separated)", value="INFY.NS, WIPRO.NS, HCLTECH.NS")
    peer_tickers = [p.strip() for p in peers_input.split(",") if p.strip()]

    st.divider()
    st.markdown("### DCF Assumptions")
    revenue_growth = st.slider("Revenue Growth Rate (%)", 0.0, 30.0, 12.0, 0.5) / 100
    ebitda_margin = st.slider("EBITDA Margin (%)", 5.0, 50.0, 25.0, 0.5) / 100
    wacc = st.slider("WACC (%)", 5.0, 20.0, 11.0, 0.5) / 100
    terminal_growth = st.slider("Terminal Growth Rate (%)", 1.0, 6.0, 3.5, 0.25) / 100
    tax_rate = st.slider("Tax Rate (%)", 10.0, 40.0, 25.0, 1.0) / 100

    st.divider()
    groq_key = os.getenv("GROQ_API_KEY", "")

    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")


# ── Helpers ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_company_data(ticker: str):
    info = get_stock_info(ticker)
    hist = get_historical_prices(ticker)
    financials = get_financials(ticker)
    metrics = extract_key_metrics(financials)
    return info, hist, financials, metrics


@st.cache_data(ttl=3600, show_spinner=False)
def load_peer_metrics(peer_tickers_tuple):
    return get_peer_data(list(peer_tickers_tuple))


def render_metric_card(label: str, value: str, delta: str = None, color: str = None):
    delta_html = ""
    if delta:
        delta_color = COLORS["green"] if not delta.startswith("-") else COLORS["red"]
        delta_html = f'<div style="color:{delta_color}; font-size:13px; margin-top:2px;">{delta}</div>'
    border_color = color or COLORS["primary"]
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {border_color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ── Analyze button triggers data load ────────────────────────────
if analyze_btn:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            info, hist, financials, metrics = load_company_data(ticker)
            st.session_state["data_loaded"] = True
            st.session_state["info"] = info
            st.session_state["hist"] = hist
            st.session_state["financials"] = financials
            st.session_state["metrics"] = metrics
            st.session_state["ticker"] = ticker
        except Exception as e:
            st.error(f"Failed to fetch data for {ticker}: {e}")
            st.stop()

# ── Landing page when no analysis done yet ───────────────────────
if not st.session_state.get("data_loaded", False):
    st.markdown("""
    <div class="landing-container">
        <div class="landing-title">EquiLens</div>
        <div class="landing-sub">
            Enter an NSE ticker in the sidebar and click <b>Analyze</b> to generate
            a full equity research report with ratio analysis, DCF valuation,
            peer benchmarking, and AI-powered insights.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── From here on, data is loaded ─────────────────────────────────
info = st.session_state["info"]
hist = st.session_state["hist"]
metrics = st.session_state["metrics"]
current_ticker = st.session_state["ticker"]

if metrics.empty:
    st.warning("No financial data available for this ticker. Try a different company.")
    st.stop()

# Compute ratios
market_cap = info.get("market_cap", 0)
all_ratios = compute_all_ratios(metrics, market_cap)
health_score = compute_health_score(all_ratios)

# Latest year metrics for DCF
latest_year = metrics.columns[-1]
last_revenue = metrics.loc["Revenue", latest_year]
shares = info.get("shares_outstanding", 1)
net_debt = metrics.loc["Total Debt", latest_year] - metrics.loc["Cash", latest_year]

# DCF
dcf_result = compute_dcf(
    last_revenue=last_revenue,
    revenue_growth=revenue_growth,
    ebitda_margin=ebitda_margin,
    wacc=wacc,
    terminal_growth=terminal_growth,
    tax_rate=tax_rate,
    shares_outstanding=shares,
    net_debt=net_debt,
)
scenarios = run_scenarios(
    last_revenue=last_revenue,
    base_growth=revenue_growth,
    base_margin=ebitda_margin,
    wacc=wacc,
    terminal_growth=terminal_growth,
    shares_outstanding=shares,
    net_debt=net_debt,
    tax_rate=tax_rate,
)

# Altman Z-Score (latest year)
altman_df = all_ratios.get("altman", pd.DataFrame())
z_score = 0.0
z_zone = "N/A"
if not altman_df.empty and "Z-Score" in altman_df.index:
    z_score = altman_df.loc["Z-Score", altman_df.columns[-1]]
    z_zone = altman_df.loc["Zone", altman_df.columns[-1]]

# ── Header ───────────────────────────────────────────────────────
cmp = info.get("cmp", 0)
st.markdown(f"## {info.get('name', current_ticker)}")
st.markdown(f"**{info.get('sector', '')}** · **{info.get('industry', '')}** · {current_ticker}")

cols = st.columns(6)
with cols[0]:
    render_metric_card("CMP", f"₹{cmp:,.2f}")
with cols[1]:
    render_metric_card("Market Cap", f"₹{market_cap/1e7:,.0f} Cr" if market_cap > 0 else "N/A")
with cols[2]:
    render_metric_card("P/E Ratio", f"{info.get('pe_ratio', 0):.1f}")
with cols[3]:
    render_metric_card("P/B Ratio", f"{info.get('pb_ratio', 0):.1f}")
with cols[4]:
    render_metric_card("52W Range", f"₹{info.get('52w_low', 0):,.0f} – ₹{info.get('52w_high', 0):,.0f}")
with cols[5]:
    intrinsic = dcf_result["intrinsic_per_share"]
    upside = ((intrinsic - cmp) / cmp * 100) if cmp > 0 else 0
    color = COLORS["green"] if upside > 0 else COLORS["red"]
    render_metric_card("DCF Value", f"₹{intrinsic:,.0f}", f"{upside:+.1f}% to CMP", color)

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────
tab_overview, tab_ratios, tab_valuation, tab_peers, tab_risk = st.tabs(
    ["📊 Overview", "📈 Ratios", "💰 Valuation", "🏢 Peer Comp", "⚠️ Risk"]
)

# ── TAB: Overview ────────────────────────────────────────────────
with tab_overview:
    col1, col2 = st.columns([2, 1])

    with col1:
        if not hist.empty:
            st.plotly_chart(price_chart(hist, current_ticker), use_container_width=True, key="overview_price")

    with col2:
        st.plotly_chart(health_score_gauge(health_score), use_container_width=True, key="overview_health")
        st.plotly_chart(altman_gauge(float(z_score), str(z_zone)), use_container_width=True, key="overview_altman")

    # AI Narrative
    st.markdown('<div class="section-header">AI Investment Summary</div>', unsafe_allow_html=True)
    narrative = generate_narrative(
        company_name=info.get("name", current_ticker),
        ticker=current_ticker,
        ratios=all_ratios,
        health_score=health_score,
        cmp=cmp,
        intrinsic_value=dcf_result["intrinsic_per_share"],
        z_score=float(z_score),
        z_zone=str(z_zone),
        api_key=groq_key if groq_key else None,
    )
    st.markdown(narrative)

    # Football Field Valuation
    st.markdown('<div class="section-header">Football Field Valuation</div>', unsafe_allow_html=True)
    st.plotly_chart(
        football_field_chart(
            cmp=cmp, scenarios=scenarios,
            high_52w=info.get("52w_high", 0), low_52w=info.get("52w_low", 0),
        ),
        use_container_width=True, key="overview_football",
    )

# ── TAB: Ratios ──────────────────────────────────────────────────
with tab_ratios:
    st.markdown('<div class="section-header">Profitability Ratios (5Y Trend)</div>', unsafe_allow_html=True)
    prof = all_ratios.get("profitability", pd.DataFrame())
    if not prof.empty:
        st.plotly_chart(ratio_trend_chart(prof, "Profitability Ratios"), use_container_width=True, key="ratio_prof")

        st.markdown("**Latest Values:**")
        tl_cols = st.columns(len(prof.index))
        for i, ratio_name in enumerate(prof.index):
            val = prof.loc[ratio_name, prof.columns[-1]]
            light = get_traffic_light(ratio_name, val)
            emoji = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(light, "⚪")
            tl_cols[i].metric(f"{emoji} {ratio_name}", f"{val:.1f}")

    st.divider()

    st.markdown('<div class="section-header">Liquidity Ratios</div>', unsafe_allow_html=True)
    liq = all_ratios.get("liquidity", pd.DataFrame())
    if not liq.empty:
        st.plotly_chart(ratio_trend_chart(liq, "Liquidity Ratios"), use_container_width=True, key="ratio_liq")

    st.divider()

    st.markdown('<div class="section-header">Efficiency Ratios</div>', unsafe_allow_html=True)
    eff = all_ratios.get("efficiency", pd.DataFrame())
    if not eff.empty:
        st.plotly_chart(ratio_trend_chart(eff, "Efficiency Ratios"), use_container_width=True, key="ratio_eff")

    st.divider()

    st.markdown('<div class="section-header">DuPont Decomposition</div>', unsafe_allow_html=True)
    dupont = all_ratios.get("dupont", pd.DataFrame())
    if not dupont.empty:
        st.plotly_chart(dupont_chart(dupont), use_container_width=True, key="ratio_dupont")
        st.dataframe(dupont.T.style.format("{:.4f}"), use_container_width=True)

# ── TAB: Valuation ───────────────────────────────────────────────
with tab_valuation:
    st.markdown('<div class="section-header">DCF Valuation</div>', unsafe_allow_html=True)

    scen_cols = st.columns(3)
    for i, (name, val) in enumerate(scenarios.items()):
        with scen_cols[i]:
            upside = ((val - cmp) / cmp * 100) if cmp > 0 else 0
            color = COLORS["green"] if upside > 0 else COLORS["red"]
            render_metric_card(f"{name} Case", f"₹{val:,.0f}", f"{upside:+.1f}%", color)

    st.markdown("**Projected Free Cash Flows:**")
    if dcf_result.get("projections") is not None:
        proj_display = dcf_result["projections"].copy()
        for col in proj_display.columns:
            if col != "Year":
                proj_display[col] = proj_display[col].apply(lambda x: f"₹{x/1e7:,.0f} Cr")
        st.dataframe(proj_display, use_container_width=True, hide_index=True)

    st.markdown("**Valuation Summary:**")
    val_data = {
        "Component": ["Sum of PV(FCF)", "PV(Terminal Value)", "Enterprise Value", "Less: Net Debt", "Equity Value", "Shares Outstanding", "Intrinsic Value/Share"],
        "Value": [
            f"₹{dcf_result['sum_pv_fcf']/1e7:,.0f} Cr",
            f"₹{dcf_result['pv_terminal']/1e7:,.0f} Cr",
            f"₹{dcf_result['enterprise_value']/1e7:,.0f} Cr",
            f"₹{net_debt/1e7:,.0f} Cr",
            f"₹{dcf_result['equity_value']/1e7:,.0f} Cr",
            f"{shares/1e7:,.2f} Cr",
            f"₹{dcf_result['intrinsic_per_share']:,.2f}",
        ],
    }
    st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)

    st.divider()

    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown("*Adjust WACC and Terminal Growth in the sidebar to see real-time changes.*")

    sens_matrix = sensitivity_matrix(
        last_revenue=last_revenue,
        ebitda_margin=ebitda_margin,
        revenue_growth=revenue_growth,
        shares_outstanding=shares,
        net_debt=net_debt,
        tax_rate=tax_rate,
    )
    st.plotly_chart(sensitivity_heatmap(sens_matrix, cmp), use_container_width=True, key="val_heatmap")

    st.markdown("**Quick Sensitivity:**")
    sens_col1, sens_col2, sens_col3 = st.columns(3)
    with sens_col1:
        quick_wacc = st.slider("Quick WACC (%)", 6.0, 18.0, wacc * 100, 0.5, key="quick_wacc") / 100
    with sens_col2:
        quick_tgr = st.slider("Quick TGR (%)", 1.0, 6.0, terminal_growth * 100, 0.25, key="quick_tgr") / 100
    with sens_col3:
        if quick_wacc > quick_tgr:
            quick_dcf = compute_dcf(
                last_revenue=last_revenue, revenue_growth=revenue_growth,
                ebitda_margin=ebitda_margin, wacc=quick_wacc,
                terminal_growth=quick_tgr, tax_rate=tax_rate,
                shares_outstanding=shares, net_debt=net_debt,
            )
            quick_val = quick_dcf["intrinsic_per_share"]
            quick_upside = ((quick_val - cmp) / cmp * 100) if cmp > 0 else 0
            color = COLORS["green"] if quick_upside > 0 else COLORS["red"]
            render_metric_card("Intrinsic Value", f"₹{quick_val:,.0f}", f"{quick_upside:+.1f}% vs CMP", color)
        else:
            st.warning("WACC must exceed Terminal Growth Rate")

# ── TAB: Peer Comparison ─────────────────────────────────────────
with tab_peers:
    st.markdown('<div class="section-header">Peer Benchmarking</div>', unsafe_allow_html=True)

    if peer_tickers:
        with st.spinner("Loading peer data..."):
            all_peer_tickers = [current_ticker] + [t for t in peer_tickers if t != current_ticker]
            peer_data = load_peer_metrics(tuple(all_peer_tickers))

            peer_ratios = {}
            for t, data in peer_data.items():
                t_metrics = data.get("metrics", pd.DataFrame())
                if t_metrics.empty:
                    continue
                t_all = compute_all_ratios(t_metrics, data.get("info", {}).get("market_cap", 0))
                latest = {}
                for cat_df in t_all.values():
                    if isinstance(cat_df, pd.DataFrame) and not cat_df.empty:
                        yr = cat_df.columns[-1]
                        for idx in cat_df.index:
                            v = cat_df.loc[idx, yr]
                            if isinstance(v, (int, float)):
                                latest[idx] = round(v, 2)
                peer_ratios[t] = latest

            if peer_ratios:
                radar_metrics = ["ROE %", "Net Margin %", "Current Ratio", "Asset Turnover", "ROCE %"]
                available_radar = [r for r in radar_metrics if any(r in v for v in peer_ratios.values())]

                if available_radar:
                    st.plotly_chart(radar_chart(peer_ratios, available_radar), use_container_width=True, key="peer_radar")

                compare_ratios = ["ROE %", "Net Margin %", "Debt/Equity", "Current Ratio", "ROCE %", "Asset Turnover"]
                bar_cols = st.columns(2)
                for i, ratio_name in enumerate(compare_ratios):
                    if any(ratio_name in v for v in peer_ratios.values()):
                        with bar_cols[i % 2]:
                            st.plotly_chart(peer_bar_chart(peer_ratios, ratio_name), use_container_width=True, key=f"peer_bar_{i}")

                st.markdown("**Detailed Comparison:**")
                comp_df = pd.DataFrame(peer_ratios).T
                comp_df.index = [t.replace(".NS", "") for t in comp_df.index]
                st.dataframe(comp_df.style.format("{:.2f}", na_rep="—"), use_container_width=True)
            else:
                st.info("Could not load peer data. Check ticker symbols.")
    else:
        st.info("Enter peer tickers in the sidebar to compare.")

# ── TAB: Risk ────────────────────────────────────────────────────
with tab_risk:
    st.markdown('<div class="section-header">Altman Z-Score Analysis</div>', unsafe_allow_html=True)

    col_z1, col_z2 = st.columns([1, 1])
    with col_z1:
        st.plotly_chart(altman_gauge(float(z_score), str(z_zone)), use_container_width=True, key="risk_altman")
    with col_z2:
        st.markdown(f"""
        **Score: {z_score}** — **{z_zone} Zone**

        | Zone | Z-Score Range | Interpretation |
        |------|--------------|----------------|
        | 🟢 Safe | > 2.99 | Low bankruptcy risk |
        | 🟡 Grey | 1.81 – 2.99 | Moderate risk, needs monitoring |
        | 🔴 Distress | < 1.81 | High risk of financial distress |
        """)

    if not altman_df.empty:
        st.markdown("**Z-Score Components Over Time:**")
        st.plotly_chart(ratio_trend_chart(altman_df.drop(["Zone"], errors="ignore"), "Altman Z-Score Components"), use_container_width=True, key="risk_altman_trend")

    st.divider()

    st.markdown('<div class="section-header">Solvency & Leverage</div>', unsafe_allow_html=True)
    solv = all_ratios.get("solvency", pd.DataFrame())
    if not solv.empty:
        st.plotly_chart(ratio_trend_chart(solv, "Solvency Ratios"), use_container_width=True, key="risk_solv")

    st.divider()
    st.markdown('<div class="section-header">Traffic Light Dashboard</div>', unsafe_allow_html=True)

    tl_data = []
    for category, df in all_ratios.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        latest_yr = df.columns[-1]
        for ratio_name in df.index:
            val = df.loc[ratio_name, latest_yr]
            if not isinstance(val, (int, float)):
                continue
            light = get_traffic_light(ratio_name, val)
            emoji = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(light, "⚪")
            tl_data.append({
                "Category": category.title(),
                "Ratio": ratio_name,
                "Value": f"{val:.2f}",
                "Signal": emoji,
            })

    if tl_data:
        st.dataframe(pd.DataFrame(tl_data), use_container_width=True, hide_index=True)

# ── PDF Export ───────────────────────────────────────────────────
st.divider()
col_pdf, _ = st.columns([1, 3])
with col_pdf:
    if st.button("📄 Export PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            pdf_path = generate_pdf(
                company_name=info.get("name", current_ticker),
                ticker=current_ticker,
                stock_info=info,
                all_ratios=all_ratios,
                health_score=health_score,
                dcf_result=dcf_result,
                scenarios=scenarios,
                narrative=narrative if "narrative" in dir() else "Analysis not generated.",
                z_score=float(z_score),
                z_zone=str(z_zone),
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Report",
                    data=f.read(),
                    file_name=f"EquiLens_{current_ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
