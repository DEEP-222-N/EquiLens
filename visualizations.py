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
    pe_comps: dict = None, ev_ebitda_comps: dict = None,
) -> go.Figure:
    """
    Football field valuation chart showing ranges across all valuation methods.
    Horizontal bars from low to high, with CMP as a vertical reference line.
    """
    methods = []
    lows = []
    highs = []
    mids = []
    bar_colors = []

    color_palette = [
        ("rgba(0, 212, 170, 0.3)", COLORS["primary"]),
        ("rgba(78, 205, 196, 0.3)", COLORS["accent"]),
        ("rgba(255, 230, 109, 0.3)", COLORS["warning"]),
        ("rgba(167, 139, 250, 0.3)", "#A78BFA"),
        ("rgba(244, 114, 182, 0.3)", "#F472B6"),
    ]

    # 52-Week Range
    if high_52w > 0 and low_52w > 0:
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

    # PE Comps
    if pe_comps and pe_comps.get("applicable"):
        methods.append("PE Comps")
        lows.append(pe_comps["fair_value_low"])
        highs.append(pe_comps["fair_value_high"])
        mids.append(pe_comps["fair_value_median"])

    # EV/EBITDA Comps
    if ev_ebitda_comps and ev_ebitda_comps.get("applicable"):
        methods.append("EV/EBITDA Comps")
        lows.append(ev_ebitda_comps["fair_value_low"])
        highs.append(ev_ebitda_comps["fair_value_high"])
        mids.append(ev_ebitda_comps["fair_value_median"])

    fig = go.Figure()

    for i, method in enumerate(methods):
        fill_color, border_color = color_palette[i % len(color_palette)]
        fig.add_trace(go.Bar(
            y=[method], x=[highs[i] - lows[i]],
            base=lows[i], orientation="h",
            marker=dict(color=fill_color, line=dict(color=border_color, width=1)),
            name=method, showlegend=False,
            hovertemplate=f"{method}<br>Low: ₹{lows[i]:,.0f}<br>Mid: ₹{mids[i]:,.0f}<br>High: ₹{highs[i]:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[mids[i]], y=[method],
            mode="markers", marker=dict(size=12, color=border_color, symbol="diamond"),
            showlegend=False,
        ))

    # CMP vertical line
    fig.add_vline(x=cmp, line=dict(color=COLORS["secondary"], width=2, dash="dash"))
    fig.add_annotation(
        x=cmp, y=len(methods) - 0.5, text=f"CMP: ₹{cmp:,.0f}",
        showarrow=False, font=dict(color=COLORS["secondary"], size=12),
    )

    chart_height = max(280, 80 + len(methods) * 50)
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(
        **layout, title="Football Field Valuation",
        xaxis_title="Price (₹)", height=chart_height,
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


def health_score_gauge(score: int, grade: str = "") -> go.Figure:
    """Overall financial health gauge (0-100) with letter grade."""
    if score >= 70:
        bar_color = COLORS["green"]
    elif score >= 40:
        bar_color = COLORS["amber"]
    else:
        bar_color = COLORS["red"]

    title_text = f"Health Score — {grade}" if grade else "Health Score"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title_text, "font": {"color": COLORS["text"], "size": 14}},
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


def comps_waterfall_chart(comps: dict, method_name: str, cmp: float) -> go.Figure:
    """Waterfall-style chart showing peer multiples vs implied fair value."""
    peers = comps.get("peers", [])
    if not peers:
        return go.Figure()

    if "pe" in peers[0]:
        key = "pe"
        mult_label = "P/E"
    else:
        key = "ev_ebitda"
        mult_label = "EV/EBITDA"

    tickers = [p["ticker"].replace(".NS", "") for p in peers]
    values = [p[key] for p in peers]

    median_val = comps.get("peer_median_pe", comps.get("peer_median_multiple", 0))
    fair_value = comps.get("fair_value_median", 0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tickers, y=values,
        marker_color=[COLORS["primary"] if v <= median_val else COLORS["accent"] for v in values],
        text=[f"{v:.1f}x" for v in values],
        textposition="outside",
        name="Peer Multiple",
    ))

    fig.add_hline(
        y=median_val,
        line=dict(color=COLORS["warning"], width=2, dash="dash"),
        annotation_text=f"Median: {median_val:.1f}x",
        annotation_font=dict(color=COLORS["warning"]),
    )

    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{method_name} — Peer Multiples (Fair Value: ₹{fair_value:,.0f})",
        yaxis_title=f"{mult_label} Multiple",
        height=350,
        showlegend=False,
    )
    return fig


def peer_bar_chart(peer_data: dict, ratio_name: str) -> go.Figure:
    """Horizontal bar chart comparing a single ratio across peers, each in a distinct color."""
    tickers = []
    values = []
    for ticker, ratios in peer_data.items():
        tickers.append(ticker.replace(".NS", ""))
        values.append(ratios.get(ratio_name, 0))

    bar_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                  COLORS["warning"], "#A78BFA", "#F472B6", "#38BDF8", "#FB923C"]
    colors = [bar_colors[i % len(bar_colors)] for i in range(len(tickers))]

    fig = go.Figure(go.Bar(
        y=tickers, x=values, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        **CHART_LAYOUT, title=f"{ratio_name} — Peer Comparison",
        height=max(250, 60 + len(tickers) * 40), xaxis_title=ratio_name,
    )
    return fig


def shareholding_donut(shareholding: dict) -> go.Figure:
    """Donut chart for shareholding pattern (Promoters/FII/DII/Retail)."""
    labels = []
    values = []
    for k, v in shareholding.items():
        if v > 0:
            labels.append(k)
            values.append(v)

    if not values:
        return go.Figure()

    donut_colors = [COLORS["primary"], "#A78BFA", COLORS["accent"], COLORS["warning"]]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=donut_colors[:len(labels)]),
        textinfo="label+percent",
        textfont=dict(color=COLORS["text"], size=12),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        title=dict(text="Shareholding Pattern", font=dict(color=COLORS["text"], size=16)),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(font=dict(color=COLORS["text"], size=11)),
    )
    return fig


def returns_comparison_chart(returns: dict, ticker: str) -> go.Figure:
    """Grouped bar chart comparing stock vs Nifty returns over 1Y/3Y/5Y."""
    periods = []
    stock_returns = []
    bench_returns = []

    for period in ["1Y", "3Y", "5Y"]:
        data = returns.get(period, {})
        s = data.get("stock")
        b = data.get("benchmark")
        if s is not None:
            periods.append(period)
            stock_returns.append(s)
            bench_returns.append(b if b is not None else 0)

    if not periods:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=periods, y=stock_returns,
        name=ticker.replace(".NS", ""),
        marker_color=COLORS["primary"],
        text=[f"{v:+.1f}%" for v in stock_returns],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        x=periods, y=bench_returns,
        name="Nifty 50",
        marker_color=COLORS["accent"],
        text=[f"{v:+.1f}%" for v in bench_returns],
        textposition="outside",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title="Historical Returns vs Nifty 50",
        yaxis_title="Return (%)",
        barmode="group",
        height=350,
        legend=dict(font=dict(color=COLORS["text"])),
    )
    return fig
