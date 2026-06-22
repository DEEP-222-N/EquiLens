"""
EquiLens — Automated Equity Research & Financial Health Analyzer
Main Streamlit application for NSE-listed Indian companies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from data_fetcher import get_stock_info, get_historical_prices, get_financials, extract_key_metrics, get_peer_data, get_shareholding_pattern, get_historical_returns
from ratio_engine import (
    compute_all_ratios, compute_health_score, get_traffic_light,
    compute_dupont, compute_altman_z_score,
)
from dcf_model import compute_dcf, run_scenarios, sensitivity_matrix, compute_comparable_valuation, compute_target_price_and_rating
from visualizations import (
    price_chart, ratio_trend_chart, dupont_chart, radar_chart,
    football_field_chart, sensitivity_heatmap, altman_gauge,
    health_score_gauge, peer_bar_chart, comps_waterfall_chart,
    shareholding_donut, returns_comparison_chart, COLORS,
)
from narrative_generator import generate_narrative
from pdf_exporter import generate_pdf
from assumption_engine import compute_all_assumptions

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

    ticker = st.text_input("Stock Name", value="", placeholder="e.g., TCS.NS or RELIANCE.NS", help="Enter ticker with .NS suffix (e.g., RELIANCE.NS)")
    if not ticker.endswith((".NS", ".BO")):
        ticker = ticker + ".NS"

    st.divider()
    st.markdown("### Peer Comparison")
    peers_input = st.text_input("Peer Tickers (comma-separated)", value="", placeholder="e.g., INFY.NS, WIPRO.NS")
    peer_tickers = []
    for p in peers_input.split(","):
        p = p.strip()
        if p:
            if not p.endswith((".NS", ".BO")):
                p = p + ".NS"
            peer_tickers.append(p)

    st.divider()
    groq_key = os.getenv("GROQ_API_KEY", "") or st.secrets.get("GROQ_API_KEY", "")

    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    export_btn = st.button("📄 Export PDF Report", use_container_width=True, disabled=not st.session_state.get("data_loaded", False))
    # Show download button if PDF was already generated
    if st.session_state.get("pdf_data"):
        st.download_button(
            label="⬇️ Download Report",
            data=st.session_state["pdf_data"],
            file_name=st.session_state.get("pdf_filename", "EquiLens_Report.pdf"),
            mime="application/pdf",
            use_container_width=True,
        )


# ── Helpers ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_company_data(ticker: str):
    info = get_stock_info(ticker)
    hist = get_historical_prices(ticker)
    financials = get_financials(ticker)
    metrics = extract_key_metrics(financials)
    assumptions = compute_all_assumptions(info, metrics) if not metrics.empty else None
    shareholding = get_shareholding_pattern(ticker)
    hist_returns = get_historical_returns(ticker)
    return info, hist, financials, metrics, assumptions, shareholding, hist_returns


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
            st.session_state.pop("pdf_data", None)
            st.session_state.pop("pdf_filename", None)
            info, hist, financials, metrics, assumptions, shareholding, hist_returns = load_company_data(ticker)
            st.session_state["data_loaded"] = True
            st.session_state["info"] = info
            st.session_state["hist"] = hist
            st.session_state["financials"] = financials
            st.session_state["metrics"] = metrics
            st.session_state["assumptions"] = assumptions
            st.session_state["shareholding"] = shareholding
            st.session_state["hist_returns"] = hist_returns
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
assumptions = st.session_state["assumptions"]
current_ticker = st.session_state["ticker"]

if metrics.empty:
    st.warning("No financial data available for this ticker. Try a different company.")
    st.stop()

# Use auto-computed assumptions
revenue_growth = assumptions["revenue_growth"]
ebitda_margin = assumptions["ebitda_margin"]
wacc = assumptions["wacc"]
terminal_growth = assumptions["terminal_growth"]
tax_rate = assumptions["tax_rate"]
capex_pct = assumptions["capex_pct"]
nwc_pct = assumptions["nwc_pct"]
da_pct = assumptions["da_pct"]

# Compute ratios
market_cap = info.get("market_cap", 0)
all_ratios = compute_all_ratios(metrics, market_cap)
stock_sector = info.get("sector", "")
health_result = compute_health_score(all_ratios, sector=stock_sector)
health_score = health_result["score"]

# Latest year metrics for DCF
latest_year = metrics.columns[-1]
last_revenue = metrics.loc["Revenue", latest_year]
shares = info.get("shares_outstanding", 1)
net_debt = metrics.loc["Total Debt", latest_year] - metrics.loc["Cash", latest_year]

# DCF — skip for financial services (banks' revenue/debt/EBITDA break DCF assumptions)
is_financial = stock_sector.lower() in ("financial services",)
if is_financial:
    dcf_result = {"intrinsic_per_share": 0, "enterprise_value": 0, "equity_value": 0,
                  "projections": pd.DataFrame(), "terminal_value": 0, "sum_pv_fcf": 0}
else:
    dcf_result = compute_dcf(
        last_revenue=last_revenue,
        revenue_growth=revenue_growth,
        ebitda_margin=ebitda_margin,
        wacc=wacc,
        terminal_growth=terminal_growth,
        tax_rate=tax_rate,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        da_pct=da_pct,
        shares_outstanding=shares,
        net_debt=net_debt,
    )
if is_financial:
    scenarios = {}
else:
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


# Altman Z-Score (latest year) — only meaningful for manufacturing/industrial sectors
ALTMAN_APPLICABLE_SECTORS = {"industrials", "basic materials", "energy", "consumer cyclical",
                             "consumer defensive", "utilities", "real estate"}
altman_applicable = stock_sector.lower() in ALTMAN_APPLICABLE_SECTORS
altman_df = all_ratios.get("altman", pd.DataFrame())
z_score = 0.0
z_zone = "N/A"
if not altman_df.empty and "Z-Score" in altman_df.index:
    z_score = altman_df.loc["Z-Score", altman_df.columns[-1]]
    z_zone = altman_df.loc["Zone", altman_df.columns[-1]]

# ── Pre-compute comps for rating ────────────────────────────────
cmp = info.get("cmp", 0)
comps_result = {"pe_comps": {"applicable": False}, "ev_ebitda_comps": {"applicable": False}}
if peer_tickers:
    all_peer_tickers_ff = [current_ticker] + [t for t in peer_tickers if t != current_ticker]
    peer_data_ff = load_peer_metrics(tuple(all_peer_tickers_ff))
    peer_data_for_comps = {t: d for t, d in peer_data_ff.items() if t != current_ticker}
    comps_result = compute_comparable_valuation(info, metrics, peer_data_for_comps)
st.session_state["comps_result"] = comps_result

# ── Blended Target Price & Rating ───────────────────────────────
rating_result = compute_target_price_and_rating(
    cmp=cmp,
    dcf_result=dcf_result,
    scenarios=scenarios,
    comps_result=comps_result,
    health_score=health_score,
)

# ── Pre-compute narrative (needed by both Overview tab and PDF export) ──
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
    scenarios=scenarios,
    comps_result=comps_result,
    rating_result=rating_result,
    shareholding=st.session_state.get("shareholding", {}),
    hist_returns=st.session_state.get("hist_returns", {}),
    sector=info.get("sector", ""),
)
st.session_state["narrative"] = narrative

# ── Pre-compute sensitivity matrix ──
sens_matrix = sensitivity_matrix(
    last_revenue=last_revenue,
    ebitda_margin=ebitda_margin,
    revenue_growth=revenue_growth,
    shares_outstanding=shares,
    net_debt=net_debt,
    tax_rate=tax_rate,
)

# ── Header ───────────────────────────────────────────────────────
st.markdown(f"## {info.get('name', current_ticker)}")
st.markdown(f"**{info.get('sector', '')}** · **{info.get('industry', '')}** · {current_ticker}")

# Rating badge
rating = rating_result["rating"]
target_price = rating_result["target_price"]
target_upside = rating_result["upside"]
confidence = rating_result["confidence"]
rating_colors = {"BUY": "#00D4AA", "HOLD": "#FFE66D", "SELL": "#FF6B6B", "N/A": "#8892B0"}
rating_color = rating_colors.get(rating, "#8892B0")

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 24px; margin: 8px 0 16px;">
    <div style="background: {rating_color}; color: #0E1117; font-size: 28px; font-weight: 900;
                padding: 8px 28px; border-radius: 8px; letter-spacing: 2px;">{rating}</div>
    <div>
        <div style="color: #8892B0; font-size: 13px;">Target Price</div>
        <div style="color: #FAFAFA; font-size: 24px; font-weight: 700;">₹{target_price:,.0f}
            <span style="color: {rating_color}; font-size: 16px; margin-left: 8px;">{target_upside:+.1f}%</span>
        </div>
        <div style="color: #8892B0; font-size: 12px;">Confidence: {confidence} · {rating_result['methods_used']} methods</div>
    </div>
</div>
""", unsafe_allow_html=True)

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
    if is_financial:
        render_metric_card("Valuation", "Via Comps", "DCF N/A for banks")
    else:
        intrinsic = dcf_result["intrinsic_per_share"]
        upside = ((intrinsic - cmp) / cmp * 100) if cmp > 0 else 0
        color = COLORS["green"] if upside > 0 else COLORS["red"]
        render_metric_card("DCF Value", f"₹{intrinsic:,.0f}", f"{upside:+.1f}% to CMP", color)

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
        st.plotly_chart(health_score_gauge(health_score, health_result.get("grade", "")), use_container_width=True, key="overview_health")
        if altman_applicable:
            st.plotly_chart(altman_gauge(float(z_score), str(z_zone)), use_container_width=True, key="overview_altman")

        # Category sub-scores
        cat_scores = health_result.get("category_scores", {})
        if cat_scores:
            cat_cols = st.columns(len(cat_scores))
            for i, (cat, data) in enumerate(cat_scores.items()):
                with cat_cols[i]:
                    st.metric(cat, f"{data['score']}", delta=data["grade"])

        # Strengths / Weaknesses
        strengths = health_result.get("strengths", [])
        weaknesses = health_result.get("weaknesses", [])
        if strengths:
            st.markdown(f"**Strengths:** {', '.join(strengths)}")
        if weaknesses:
            st.markdown(f"**Weaknesses:** {', '.join(weaknesses)}")

    # Auto-computed DCF Assumptions
    st.markdown('<div class="section-header">Auto-Computed DCF Assumptions</div>', unsafe_allow_html=True)
    a_row1 = st.columns(4)
    rationale = assumptions["rationale"]
    with a_row1[0]:
        render_metric_card("Revenue Growth", f"{revenue_growth*100:.1f}%")
        st.caption(rationale["revenue_growth"])
    with a_row1[1]:
        render_metric_card("EBITDA Margin", f"{ebitda_margin*100:.1f}%")
        st.caption(rationale["ebitda_margin"])
    with a_row1[2]:
        render_metric_card("WACC", f"{wacc*100:.1f}%")
        st.caption(rationale["wacc"])
    with a_row1[3]:
        render_metric_card("Terminal Growth", f"{terminal_growth*100:.1f}%")
        st.caption(rationale["terminal_growth"])
    a_row2 = st.columns(4)
    with a_row2[0]:
        render_metric_card("Tax Rate", f"{tax_rate*100:.1f}%")
        st.caption(rationale["tax_rate"])
    with a_row2[1]:
        render_metric_card("Capex/Revenue", f"{capex_pct*100:.1f}%")
        st.caption(rationale["capex_pct"])
    with a_row2[2]:
        render_metric_card("NWC/Revenue", f"{nwc_pct*100:.1f}%")
        st.caption(rationale["nwc_pct"])
    with a_row2[3]:
        render_metric_card("D&A/Revenue", f"{da_pct*100:.1f}%")
        st.caption(rationale["da_pct"])

    # AI Narrative
    st.markdown('<div class="section-header">AI Investment Summary</div>', unsafe_allow_html=True)
    st.markdown(st.session_state.get("narrative", ""))

    # Football Field Valuation
    st.markdown('<div class="section-header">Football Field Valuation</div>', unsafe_allow_html=True)
    st.plotly_chart(
        football_field_chart(
            cmp=cmp, scenarios=scenarios,
            high_52w=info.get("52w_high", 0), low_52w=info.get("52w_low", 0),
            pe_comps=comps_result.get("pe_comps"),
            ev_ebitda_comps=comps_result.get("ev_ebitda_comps"),
        ),
        use_container_width=True, key="overview_football",
    )

    # Shareholding Pattern & Historical Returns
    sh_col, ret_col = st.columns(2)

    shareholding = st.session_state.get("shareholding", {})
    with sh_col:
        if any(v > 0 for v in shareholding.values()):
            st.plotly_chart(shareholding_donut(shareholding), use_container_width=True, key="overview_shareholding")
        else:
            st.info("Shareholding data not available for this ticker.")

    hist_returns = st.session_state.get("hist_returns", {})
    with ret_col:
        if hist_returns and any(d.get("stock") is not None for d in hist_returns.values()):
            st.plotly_chart(returns_comparison_chart(hist_returns, current_ticker), use_container_width=True, key="overview_returns")
        else:
            st.info("Historical returns data not available.")

# ── TAB: Ratios ──────────────────────────────────────────────────
with tab_ratios:
    sub_liq, sub_prof, sub_eff, sub_solv = st.tabs(
        ["💧 Liquidity", "💰 Profitability", "⚙️ Efficiency", "🏦 Solvency"]
    )

    # ── Liquidity ──
    with sub_liq:
        st.markdown('<div class="section-header">Liquidity Ratios (5Y Trend)</div>', unsafe_allow_html=True)
        liq = all_ratios.get("liquidity", pd.DataFrame())
        if not liq.empty:
            st.plotly_chart(ratio_trend_chart(liq, "Liquidity Ratios"), use_container_width=True, key="ratio_liq")

            st.markdown("**Latest Values:**")
            tl_cols = st.columns(len(liq.index))
            for i, ratio_name in enumerate(liq.index):
                val = liq.loc[ratio_name, liq.columns[-1]]
                light = get_traffic_light(ratio_name, val, stock_sector)
                emoji = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(light, "⚪")
                tl_cols[i].metric(f"{emoji} {ratio_name}", f"{val:.2f}")
        else:
            st.info("Liquidity ratio data not available.")

    # ── Profitability ──
    with sub_prof:
        st.markdown('<div class="section-header">Profitability Ratios (5Y Trend)</div>', unsafe_allow_html=True)
        prof = all_ratios.get("profitability", pd.DataFrame())
        if not prof.empty:
            st.plotly_chart(ratio_trend_chart(prof, "Profitability Ratios"), use_container_width=True, key="ratio_prof")

            st.markdown("**Latest Values:**")
            tl_cols = st.columns(len(prof.index))
            for i, ratio_name in enumerate(prof.index):
                val = prof.loc[ratio_name, prof.columns[-1]]
                light = get_traffic_light(ratio_name, val, stock_sector)
                emoji = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(light, "⚪")
                tl_cols[i].metric(f"{emoji} {ratio_name}", f"{val:.1f}")
        else:
            st.info("Profitability ratio data not available.")

        st.divider()

        st.markdown('<div class="section-header">DuPont Decomposition</div>', unsafe_allow_html=True)
        dupont = all_ratios.get("dupont", pd.DataFrame())
        if not dupont.empty:
            st.plotly_chart(dupont_chart(dupont), use_container_width=True, key="ratio_dupont")
            st.dataframe(dupont.T.style.format("{:.4f}"), use_container_width=True)

    # ── Efficiency ──
    with sub_eff:
        st.markdown('<div class="section-header">Efficiency Ratios (5Y Trend)</div>', unsafe_allow_html=True)
        eff = all_ratios.get("efficiency", pd.DataFrame())
        if not eff.empty:
            st.plotly_chart(ratio_trend_chart(eff, "Efficiency Ratios"), use_container_width=True, key="ratio_eff")

            st.markdown("**Latest Values:**")
            tl_cols = st.columns(min(len(eff.index), 5))
            for i, ratio_name in enumerate(eff.index):
                val = eff.loc[ratio_name, eff.columns[-1]]
                if not isinstance(val, (int, float)):
                    continue
                light = get_traffic_light(ratio_name, val, stock_sector)
                emoji = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(light, "⚪")
                tl_cols[i % len(tl_cols)].metric(f"{emoji} {ratio_name}", f"{val:.1f}")
        else:
            st.info("Efficiency ratio data not available.")

    # ── Solvency ──
    with sub_solv:
        st.markdown('<div class="section-header">Solvency & Leverage Ratios (5Y Trend)</div>', unsafe_allow_html=True)
        solv = all_ratios.get("solvency", pd.DataFrame())
        if not solv.empty:
            st.plotly_chart(ratio_trend_chart(solv, "Solvency Ratios"), use_container_width=True, key="ratio_solv")

            st.markdown("**Latest Values:**")
            tl_cols = st.columns(len(solv.index))
            for i, ratio_name in enumerate(solv.index):
                val = solv.loc[ratio_name, solv.columns[-1]]
                light = get_traffic_light(ratio_name, val, stock_sector)
                emoji = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(light, "⚪")
                tl_cols[i].metric(f"{emoji} {ratio_name}", f"{val:.2f}")
        else:
            st.info("Solvency ratio data not available.")

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
        "Component": [
            "Sum of PV(FCF)", "PV(Terminal Value)", "Enterprise Value",
            "Less: Net Debt", "Equity Value", "Shares Outstanding",
            "Intrinsic Value/Share",
        ],
        "Value": [
            f"₹{dcf_result['sum_pv_fcf']/1e7:,.0f} Cr",
            f"₹{dcf_result['pv_terminal']/1e7:,.0f} Cr",
            f"₹{dcf_result['enterprise_value']/1e7:,.0f} Cr",
            f"₹{net_debt/1e7:,.0f} Cr",
            f"₹{dcf_result['equity_value']/1e7:,.0f} Cr",
            f"{shares/1e7:,.2f} Cr",
            f"₹{dcf_result['intrinsic_per_share']:,.2f}",
        ],
        "Note": [
            f"Capex: {capex_pct*100:.1f}% of Rev (historical)",
            f"NWC: {nwc_pct*100:.1f}% of Rev (historical)",
            "", "", "", "", "",
        ],
    }
    st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)

    # ── Comparable Valuation Section ──
    st.divider()
    st.markdown('<div class="section-header">Comparable Valuation</div>', unsafe_allow_html=True)

    comps_result = st.session_state.get("comps_result", {})
    pe_comps = comps_result.get("pe_comps", {})
    ev_ebitda_comps = comps_result.get("ev_ebitda_comps", {})

    if pe_comps.get("applicable") or ev_ebitda_comps.get("applicable"):
        comps_cols = st.columns(2)

        if pe_comps.get("applicable"):
            with comps_cols[0]:
                pe_val = pe_comps["fair_value_median"]
                pe_upside = ((pe_val - cmp) / cmp * 100) if cmp > 0 else 0
                color = COLORS["green"] if pe_upside > 0 else COLORS["red"]
                render_metric_card(
                    "PE Comps Fair Value",
                    f"₹{pe_val:,.0f}",
                    f"{pe_upside:+.1f}% vs CMP (Median PE: {pe_comps['peer_median_pe']:.1f}x)",
                    color,
                )
                st.plotly_chart(
                    comps_waterfall_chart(pe_comps, "P/E Comps", cmp),
                    use_container_width=True, key="comps_pe_chart",
                )

        if ev_ebitda_comps.get("applicable"):
            with comps_cols[1 if pe_comps.get("applicable") else 0]:
                ev_val = ev_ebitda_comps["fair_value_median"]
                ev_upside = ((ev_val - cmp) / cmp * 100) if cmp > 0 else 0
                color = COLORS["green"] if ev_upside > 0 else COLORS["red"]
                render_metric_card(
                    "EV/EBITDA Fair Value",
                    f"₹{ev_val:,.0f}",
                    f"{ev_upside:+.1f}% vs CMP (Median: {ev_ebitda_comps['peer_median_multiple']:.1f}x)",
                    color,
                )
                st.plotly_chart(
                    comps_waterfall_chart(ev_ebitda_comps, "EV/EBITDA Comps", cmp),
                    use_container_width=True, key="comps_ev_chart",
                )

        # Summary table
        st.markdown("**Valuation Summary Across Methods:**")
        summary_rows = []
        summary_rows.append({"Method": "DCF (Base)", "Fair Value": f"₹{dcf_result['intrinsic_per_share']:,.0f}",
                             "vs CMP": f"{((dcf_result['intrinsic_per_share'] - cmp) / cmp * 100):+.1f}%" if cmp > 0 else "N/A"})
        if pe_comps.get("applicable"):
            summary_rows.append({"Method": "PE Comps (Median)", "Fair Value": f"₹{pe_comps['fair_value_median']:,.0f}",
                                 "vs CMP": f"{((pe_comps['fair_value_median'] - cmp) / cmp * 100):+.1f}%" if cmp > 0 else "N/A"})
        if ev_ebitda_comps.get("applicable"):
            summary_rows.append({"Method": "EV/EBITDA Comps (Median)", "Fair Value": f"₹{ev_ebitda_comps['fair_value_median']:,.0f}",
                                 "vs CMP": f"{((ev_ebitda_comps['fair_value_median'] - cmp) / cmp * 100):+.1f}%" if cmp > 0 else "N/A"})
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Add peer tickers in the sidebar to enable PE and EV/EBITDA comparable valuation.")

    st.divider()

    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)

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
                capex_pct=capex_pct, nwc_pct=nwc_pct, da_pct=da_pct,
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

            failed_peers = [t.replace(".NS", "") for t in all_peer_tickers if t not in peer_ratios]
            if failed_peers:
                st.warning(f"Could not load data for: {', '.join(failed_peers)}")

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
    if altman_applicable:
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
    else:
        st.info(f"Altman Z-Score is not shown for **{stock_sector}** companies — it was designed for manufacturing firms and gives misleading results for this sector.")
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
            light = get_traffic_light(ratio_name, val, stock_sector)
            emoji = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(light, "⚪")
            tl_data.append({
                "Category": category.title(),
                "Ratio": ratio_name,
                "Value": f"{val:.2f}",
                "Signal": emoji,
            })

    if tl_data:
        st.dataframe(pd.DataFrame(tl_data), use_container_width=True, hide_index=True)

# ── PDF Export (triggered from sidebar) ─────────────────────────
if export_btn:
    with st.spinner("Generating PDF report..."):
        # Build peer_ratios for PDF
        pdf_peer_ratios = {}
        if peer_tickers:
            all_peer_tickers_pdf = [current_ticker] + [t for t in peer_tickers if t != current_ticker]
            peer_data_pdf = load_peer_metrics(tuple(all_peer_tickers_pdf))
            for t, data in peer_data_pdf.items():
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
                pdf_peer_ratios[t] = latest

        pdf_path = generate_pdf(
            company_name=info.get("name", current_ticker),
            ticker=current_ticker,
            stock_info=info,
            all_ratios=all_ratios,
            health_score=health_result,
            dcf_result=dcf_result,
            scenarios=scenarios,
            narrative=st.session_state.get("narrative", "Analysis not generated."),
            z_score=float(z_score),
            z_zone=str(z_zone),
            assumptions=assumptions,
            comps_result=st.session_state.get("comps_result"),
            rating_result=rating_result,
            shareholding=st.session_state.get("shareholding", {}),
            hist_returns=st.session_state.get("hist_returns", {}),
            peer_ratios=pdf_peer_ratios if pdf_peer_ratios else None,
            sensitivity_df=sens_matrix,
        )
        with open(pdf_path, "rb") as f:
            st.session_state["pdf_data"] = f.read()
            st.session_state["pdf_filename"] = f"EquiLens_{current_ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf"
        st.rerun()
