# EquiLens

Automated equity research tool for NSE-listed Indian companies. Enter a ticker, get a full research report — ratio analysis, multi-method valuation, peer benchmarking, AI narrative, and a Buy/Hold/Sell rating with target price.

Built with Streamlit, yfinance, and Plotly. Deployed on Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://equilens.streamlit.app)

---

## Features

### Valuation
- **DCF Model** — 5-year free cash flow projections with auto-computed assumptions (WACC, growth, margins) derived from historical financials
- **Dividend Discount Model (DDM)** — Gordon Growth Model for dividend-paying stocks
- **Comparable Valuation** — PE and EV/EBITDA multiples from user-defined peers
- **Blended Target Price** — Weighted average across DCF (40%), Comps (35%), DDM (15%), and Scenarios (10%)
- **Buy/Hold/Sell Rating** — Threshold-adjusted by financial health score, with confidence level based on method agreement
- **Sensitivity Analysis** — WACC vs Terminal Growth heatmap + interactive sliders
- **Football Field Chart** — All valuation ranges on one visual with CMP reference line

### Financial Analysis
- **13 Financial Ratios** — Profitability, liquidity, efficiency, and solvency with 5-year trend charts
- **Sector-Adjusted Scoring** — Traffic-light thresholds vary by sector (Tech, Financials, Energy, etc.)
- **Weighted Health Score** — Continuous 0-100 scoring with category sub-scores, letter grades (A+ to F), trend detection, and auto-identified strengths/weaknesses
- **DuPont Decomposition** — 3-factor ROE breakdown (margin x turnover x leverage)
- **Altman Z-Score** — Bankruptcy risk assessment with Safe/Grey/Distress zones

### Market Context
- **Shareholding Pattern** — Promoter/FII/DII/Retail breakdown as a donut chart
- **Historical Returns vs Nifty 50** — 1Y/3Y/5Y stock performance compared to the benchmark index
- **Peer Benchmarking** — Radar charts, bar comparisons, and detailed ratio tables across competitors

### AI & Export
- **AI Investment Summary** — LLM-generated narrative via Groq API (Qwen 3.6 27B) with strengths and red flags; rule-based fallback when no API key is set
- **PDF Report Export** — Branded equity research PDF with rating, health score, narrative, and ratio tables

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/DEEP-222-N/EquiLens.git
cd EquiLens
pip install -r requirements.txt
```

### 2. (Optional) Set up Groq API key for AI narratives

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com). The app works without it — you just get a rule-based summary instead of AI-generated text.

### 3. Run

```bash
streamlit run app.py
```

### 4. Use

1. Enter an NSE ticker in the sidebar (e.g., `TCS.NS`, `RELIANCE.NS`)
2. Add peer tickers for comparison (e.g., `INFY.NS, WIPRO.NS`)
3. Click **Analyze**

---

## Project Structure

```
EquiLens/
├── app.py                  # Streamlit UI — 5-tab dashboard
├── data_fetcher.py         # yfinance data retrieval, shareholding, returns
├── ratio_engine.py         # Ratio computation, health scoring, sector thresholds
├── dcf_model.py            # DCF, DDM, comps, target price & rating
├── assumption_engine.py    # Auto-derives DCF inputs from historical data
├── narrative_generator.py  # AI narrative (Groq) + rule-based fallback
├── visualizations.py       # Plotly charts with dark theme
├── pdf_exporter.py         # PDF report generation (fpdf2)
├── .devcontainer/          # Dev Container configuration
├── requirements.txt
└── .env                    # API keys (not committed)
```

---

## How the Rating Works

The target price blends multiple valuation methods:

| Method | Weight | Source |
|--------|--------|--------|
| DCF | 40% | 5Y projected FCF discounted at WACC |
| Comps | 35% | Average of PE-implied and EV/EBITDA-implied fair values from peers |
| DDM | 15% | Gordon Growth Model (dividend-paying stocks only) |
| Scenario | 10% | Midpoint of bear/bull DCF scenarios |

Rating thresholds are adjusted by the health score — companies with stronger fundamentals get a slightly lower bar for a BUY rating. Confidence is based on how many methods are available and how tightly they agree.

---

## Health Score

Each of the 13 ratios is scored on a continuous 0-10 scale (not hard green/amber/red buckets), weighted by importance, and adjusted for sector:

| Category | Ratios | Top Weight |
|----------|--------|------------|
| Profitability | Gross Margin, EBITDA Margin, Net Margin, ROE, ROA, ROCE | ROE, ROCE (3.0x) |
| Liquidity | Current Ratio, Quick Ratio | Current Ratio (1.5x) |
| Efficiency | Asset Turnover, Debtor Days, CCC | Asset Turnover (1.0x) |
| Solvency | Debt/Equity, Interest Coverage | D/E (2.5x) |

Trend detection gives improving ratios a +0.5 bonus and declining ones a -0.5 penalty.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Data | yfinance, BeautifulSoup (Screener.in scraping) |
| Analysis | pandas, NumPy |
| Visualization | Plotly |
| AI | Groq API (Qwen 3.6 27B) |
| PDF | fpdf2 |

---

## Requirements

- Python 3.10+
- Internet connection (fetches live market data)
- Groq API key (optional, for AI narratives)

---

## Limitations

- **NSE/BSE only** — Designed for Indian markets (`.NS`/`.BO` tickers)
- **yfinance dependency** — Data quality depends on Yahoo Finance availability; some fields may be missing for smaller companies
- **Shareholding data** — Approximated from yfinance major holders; not as granular as BSE/NSE filings
- **No real-time data** — Prices refresh on each analysis run, not live streaming
- **Sector thresholds** — Cover 10 sectors; unlisted sectors fall back to defaults

---

## License

MIT
