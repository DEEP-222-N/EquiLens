"""
Visualization module for EquiLens.
All charts use Plotly with a consistent dark theme.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Consistent dark color palette
COLORS = {
    "primary": "#00D4AA",
    "secondary": "#FF6B6B",
    "accent": "#4ECDC4",
    "warning": "#FFE66D",
    "bg": "#0E1117",
    "card_bg": "#1E2130",
    "text": "#FAFAFA",
    "grid": "#2D3250",
    "green": "#00D4AA",
    "amber": "#FFE66D",
    "red": "#FF6B6B",
}

CHART_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)


def price_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick chart with volume bars for historical prices."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        name="Price",
    ))

    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"],
        marker_color="rgba(0, 212, 170, 0.2)",
        name="Volume", yaxis="y2",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{ticker} — 5Y Price History",
        yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        height=400,
    )
    return fig


def ratio_trend_chart(ratio_df: pd.DataFrame, title: str) -> go.Figure:
    """Multi-line chart showing ratio trends across years."""
    fig = go.Figure()
    colors_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                    COLORS["warning"], "#A78BFA", "#F472B6"]

    for i, ratio_name in enumerate(ratio_df.index):
        values = ratio_df.loc[ratio_name]
        # Skip non-numeric rows (like the Zone row in Altman)
        if not all(isinstance(v, (int, float, np.floating)) for v in values):
            continue
        fig.add_trace(go.Scatter(
            x=list(ratio_df.columns), y=list(values),
            mode="lines+markers", name=ratio_name,
            line=dict(color=colors_cycle[i % len(colors_cycle)], width=2),
            marker=dict(size=8),
        ))

    fig.update_layout(**CHART_LAYOUT, title=title, height=380, legend=dict(
        orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
    ))
    return fig


def dupont_chart(dupont_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing DuPont decomposition components over time."""
    fig = go.Figure()
    components = ["Net Margin", "Asset Turnover", "Equity Multiplier"]
    colors = [COLORS["primary"], COLORS["accent"], COLORS["warning"]]

    for comp, color in zip(components, colors):
        if comp in dupont_df.index:
            fig.add_trace(go.Bar(
                x=list(dupont_df.columns),
                y=list(dupont_df.loc[comp]),
                name=comp, marker_color=color,
            ))

    fig.update_layout(
        **CHART_LAYOUT, title="DuPont Decomposition",
        barmode="group", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


def radar_chart(peer_ratios: dict, ratio_names: list) -> go.Figure:
    """Radar chart comparing key ratios across peers."""
    fig = go.Figure()
    colors_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                    COLORS["warning"], "#A78BFA"]

    for i, (ticker, ratios) in enumerate(peer_ratios.items()):
        values = [ratios.get(r, 0) for r in ratio_names]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=ratio_names + [ratio_names[0]],
            fill="toself",
            name=ticker.replace(".NS", ""),
            line=dict(color=colors_cycle[i % len(colors_cycle)]),
            opacity=0.7,
        ))

    fig.update_layout(
        **CHART_LAYOUT, title="Peer Comparison Radar",
        polar=dict(
            bgcolor=COLORS["bg"],
            angularaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
            radialaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
        ),
        height=450,
    )
    return fig


def football_field_chart(
    cmp: float, scenarios: dict, high_52w: float, low_52w: float,
    peer_avg_pe: float = 0, eps: float = 0
) -> go.Figure:
    """
    Football field valuation chart showing ranges across valuation methods.
    Horizontal bars from low to high, with CMP as a vertical reference line.
    """
    methods = []
    lows = []
    highs = []
    mids = []

    # 52-Week Range
    methods.append("52-Week Range")
    lows.append(low_52w)
    highs.append(high_52w)
    mids.append((low_52w + high_52w) / 2)

    # DCF scenarios
    if "Bear" in scenarios and "Bull" in scenarios:
        methods.append("DCF Valuation")
        lows.append(scenarios["Bear"])
        highs.append(scenarios["Bull"])
        mids.append(scenarios.get("Base", (scenarios["Bear"] + scenarios["Bull"]) / 2))

    # PE Comps (if peer data available)
    if peer_avg_pe > 0 and eps > 0:
        low_pe_val = eps * peer_avg_pe * 0.8
        high_pe_val = eps * peer_avg_pe * 1.2
        methods.append("PE Comps")
        lows.append(low_pe_val)
        highs.append(high_pe_val)
        mids.append(eps * peer_avg_pe)

    fig = go.Figure()

    for i, method in enumerate(methods):
        fig.add_trace(go.Bar(
            y=[method], x=[highs[i] - lows[i]],
            base=lows[i], orientation="h",
            marker=dict(
                color=f"rgba(0, 212, 170, 0.3)",
                line=dict(color=COLORS["primary"], width=1),
            ),
            name=method, showlegend=False,
            hovertemplate=f"{method}<br>Low: ₹{lows[i]:,.0f}<br>High: ₹{highs[i]:,.0f}<extra></extra>",
        ))
        # Midpoint marker
        fig.add_trace(go.Scatter(
            x=[mids[i]], y=[method],
            mode="markers", marker=dict(size=12, color=COLORS["accent"], symbol="diamond"),
            showlegend=False,
        ))

    # CMP vertical line
    fig.add_vline(x=cmp, line=dict(color=COLORS["secondary"], width=2, dash="dash"))
    fig.add_annotation(
        x=cmp, y=len(methods) - 0.5, text=f"CMP: ₹{cmp:,.0f}",
        showarrow=False, font=dict(color=COLORS["secondary"], size=12),
    )

    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(
        **layout, title="Football Field Valuation",
        xaxis_title="Price (₹)", height=300,
        yaxis=dict(gridcolor=COLORS["grid"]),
    )
    return fig


def sensitivity_heatmap(matrix: pd.DataFrame, cmp: float) -> go.Figure:
    """
    Heatmap for WACC × Terminal Growth sensitivity analysis.
    Cells colored green when intrinsic > CMP (undervalued), red otherwise.
    """
    values = matrix.values.astype(float)
    # Cap infinities for display
    values = np.where(np.isinf(values), 0, values)

    # Color: green if above CMP, red if below
    colors = np.where(values >= cmp, 1, 0).astype(float)

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=list(matrix.columns),
        y=list(matrix.index),
        text=[[f"₹{v:,.0f}" if v > 0 else "N/A" for v in row] for row in values],
        texttemplate="%{text}",
        colorscale=[[0, COLORS["red"]], [0.5, COLORS["warning"]], [1, COLORS["green"]]],
        zmin=cmp * 0.5,
        zmax=cmp * 2.0,
        showscale=False,
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title="Sensitivity Analysis: Intrinsic Value (₹)",
        xaxis_title="Terminal Growth Rate",
        yaxis_title="WACC",
        height=400,
    )
    return fig


def altman_gauge(z_score: float, zone: str) -> go.Figure:
    """Gauge chart showing Altman Z-Score with zone coloring."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=z_score,
        title={"text": f"Z-Score — {zone}", "font": {"color": COLORS["text"], "size": 14}},
        number={"font": {"color": COLORS["text"], "size": 28}},
        gauge=dict(
            axis=dict(range=[0, 6], tickcolor=COLORS["text"], tickfont=dict(size=10)),
            bar=dict(color=COLORS["primary"]),
            bgcolor=COLORS["card_bg"],
            steps=[
                dict(range=[0, 1.81], color="rgba(255, 107, 107, 0.3)"),
                dict(range=[1.81, 2.99], color="rgba(255, 230, 109, 0.3)"),
                dict(range=[2.99, 6], color="rgba(0, 212, 170, 0.3)"),
            ],
            threshold=dict(line=dict(color=COLORS["secondary"], width=3), thickness=0.8, value=z_score),
        ),
    ))

    fig.update_layout(paper_bgcolor=COLORS["bg"], font=dict(color=COLORS["text"]), height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def health_score_gauge(score: int) -> go.Figure:
    """Overall financial health gauge (0-100)."""
    if score >= 70:
        bar_color = COLORS["green"]
    elif score >= 40:
        bar_color = COLORS["amber"]
    else:
        bar_color = COLORS["red"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Health Score", "font": {"color": COLORS["text"], "size": 14}},
        number={"font": {"color": COLORS["text"], "size": 28}, "suffix": "/100"},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=COLORS["text"], tickfont=dict(size=10)),
            bar=dict(color=bar_color),
            bgcolor=COLORS["card_bg"],
            steps=[
                dict(range=[0, 40], color="rgba(255, 107, 107, 0.15)"),
                dict(range=[40, 70], color="rgba(255, 230, 109, 0.15)"),
                dict(range=[70, 100], color="rgba(0, 212, 170, 0.15)"),
            ],
        ),
    ))
    fig.update_layout(paper_bgcolor=COLORS["bg"], font=dict(color=COLORS["text"]), height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def peer_bar_chart(peer_data: dict, ratio_name: str) -> go.Figure:
    """Horizontal bar chart comparing a single ratio across peers."""
    tickers = []
    values = []
    for ticker, ratios in peer_data.items():
        tickers.append(ticker.replace(".NS", ""))
        values.append(ratios.get(ratio_name, 0))

    fig = go.Figure(go.Bar(
        y=tickers, x=values, orientation="h",
        marker=dict(color=COLORS["primary"]),
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        **CHART_LAYOUT, title=f"{ratio_name} — Peer Comparison",
        height=300, xaxis_title=ratio_name,
    )
    return fig
