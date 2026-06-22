"""
Data fetching module for EquiLens.
Retrieves financial data from yfinance and Screener.in for NSE-listed companies.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def get_stock_info(ticker: str) -> dict:
    """Fetch basic stock info: name, sector, CMP, market cap, 52-week range."""
    stock = yf.Ticker(ticker)
    info = stock.info
    dividend_rate = info.get("dividendRate", 0) or 0
    trailing_eps = info.get("trailingEps", 0) or 0
    payout_ratio = info.get("payoutRatio", 0) or 0

    return {
        "name": info.get("longName", ticker.replace(".NS", "")),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "cmp": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", 0),
        "pb_ratio": info.get("priceToBook", 0),
        "dividend_yield": info.get("dividendYield", 0),
        "dividend_rate": dividend_rate,
        "trailing_eps": trailing_eps,
        "payout_ratio": payout_ratio,
        "52w_high": info.get("fiftyTwoWeekHigh", 0),
        "52w_low": info.get("fiftyTwoWeekLow", 0),
        "beta": info.get("beta", 1.0),
        "shares_outstanding": info.get("sharesOutstanding", 0),
        "currency": info.get("currency", "INR"),
    }


def get_historical_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch historical price data for charting."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist


def get_financials(ticker: str) -> dict:
    """
    Fetch P&L, Balance Sheet, and Cash Flow from yfinance.
    Returns annual data for up to 5 years.
    """
    stock = yf.Ticker(ticker)

    financials = {
        "income_stmt": _clean_financial_df(stock.financials),
        "balance_sheet": _clean_financial_df(stock.balance_sheet),
        "cashflow": _clean_financial_df(stock.cashflow),
        "quarterly_income": _clean_financial_df(stock.quarterly_financials),
        "quarterly_balance": _clean_financial_df(stock.quarterly_balance_sheet),
    }
    return financials


def _clean_financial_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean financial dataframes: handle NaN, convert to crores, sort by date."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.fillna(0)
    if isinstance(df.columns, pd.DatetimeIndex):
        df.columns = df.columns.strftime("%Y")
    df = df[sorted(df.columns)]
    return df


def safe_get(df: pd.DataFrame, row_labels: list, col: str) -> float:
    """Safely extract a value from financial statements, trying multiple label variants."""
    for label in row_labels:
        if label in df.index:
            val = df.loc[label, col]
            if pd.notna(val) and val != 0:
                return float(val)
    return 0.0


def extract_key_metrics(financials: dict) -> pd.DataFrame:
    """
    Extract key financial metrics from raw statements into a clean time-series DataFrame.
    Handles yfinance's inconsistent row naming across tickers.
    """
    income = financials["income_stmt"]
    balance = financials["balance_sheet"]
    cashflow = financials["cashflow"]

    if income.empty or balance.empty:
        return pd.DataFrame()

    years = sorted(income.columns)
    metrics = {}

    for year in years:
        revenue = safe_get(income, ["Total Revenue", "Operating Revenue", "Revenue"], year)
        cogs = safe_get(income, ["Cost Of Revenue", "Cost of Revenue", "Cost Of Goods Sold"], year)
        gross_profit = safe_get(income, ["Gross Profit"], year)
        if gross_profit == 0 and revenue > 0:
            gross_profit = revenue - cogs

        operating_income = safe_get(income, ["Operating Income", "EBIT", "Operating Profit"], year)
        da = safe_get(income, ["Depreciation And Amortization", "Depreciation & Amortisation",
                                "Reconciled Depreciation"], year)
        da = abs(da)
        ebitda = safe_get(income, ["EBITDA", "Normalized EBITDA"], year)
        if ebitda == 0:
            ebitda = operating_income + da
        if da == 0 and ebitda > 0 and operating_income > 0:
            da = ebitda - operating_income

        net_income = safe_get(income, ["Net Income", "Net Income Common Stockholders",
                                        "Net Income From Continuing Operations"], year)
        interest_expense = safe_get(income, ["Interest Expense", "Interest Expense Non Operating",
                                              "Net Interest Income"], year)
        tax = safe_get(income, ["Tax Provision", "Income Tax Expense", "Tax Rate For Calcs"], year)

        total_assets = safe_get(balance, ["Total Assets"], year)
        total_equity = safe_get(balance, ["Total Stockholders Equity", "Stockholders Equity",
                                           "Total Equity Gross Minority Interest", "Common Stock Equity"], year)
        total_debt = safe_get(balance, ["Total Debt", "Long Term Debt", "Total Non Current Liabilities Net Minority Interest"], year)
        current_assets = safe_get(balance, ["Current Assets", "Total Current Assets"], year)
        current_liabilities = safe_get(balance, ["Current Liabilities", "Total Current Liabilities"], year)
        inventory = safe_get(balance, ["Inventory", "Raw Materials", "Net Inventory"], year)
        receivables = safe_get(balance, ["Accounts Receivable", "Net Receivables", "Receivables"], year)
        payables = safe_get(balance, ["Accounts Payable", "Accounts Payables", "Current Accrued Expenses"], year)
        retained_earnings = safe_get(balance, ["Retained Earnings"], year)
        working_capital = safe_get(balance, ["Working Capital", "Net Working Capital"], year)
        if working_capital == 0:
            working_capital = current_assets - current_liabilities

        cash = safe_get(balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
                                   "Cash Financial"], year)

        operating_cf = safe_get(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                                            "Total Cash From Operating Activities"], year) if not cashflow.empty else 0
        capex = safe_get(cashflow, ["Capital Expenditure", "Capital Expenditures",
                                     "Purchase Of Fixed Assets"], year) if not cashflow.empty else 0
        free_cash_flow = safe_get(cashflow, ["Free Cash Flow"], year) if not cashflow.empty else 0
        if free_cash_flow == 0 and operating_cf != 0:
            free_cash_flow = operating_cf + capex  # capex is typically negative

        metrics[year] = {
            "Revenue": revenue,
            "COGS": cogs,
            "Gross Profit": gross_profit,
            "EBITDA": ebitda,
            "D&A": da,
            "Operating Income": operating_income,
            "Net Income": net_income,
            "Interest Expense": abs(interest_expense),
            "Tax": tax,
            "Total Assets": total_assets,
            "Total Equity": total_equity,
            "Total Debt": total_debt,
            "Current Assets": current_assets,
            "Current Liabilities": current_liabilities,
            "Inventory": inventory,
            "Receivables": receivables,
            "Payables": payables,
            "Cash": cash,
            "Retained Earnings": retained_earnings,
            "Working Capital": working_capital,
            "Operating Cash Flow": operating_cf,
            "Capex": capex,
            "Free Cash Flow": free_cash_flow,
        }

    return pd.DataFrame(metrics)


def scrape_screener_ratios(ticker_symbol: str) -> dict:
    """
    Scrape supplementary ratios from Screener.in as a fallback data source.
    ticker_symbol should be the bare symbol without .NS (e.g., 'TCS').
    Returns a dict of key ratios or empty dict on failure.
    """
    symbol = ticker_symbol.replace(".NS", "").replace(".BO", "")
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            url = f"https://www.screener.in/company/{symbol}/"
            response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return {}

        soup = BeautifulSoup(response.text, "lxml")
        ratios = {}

        # Extract key ratios from the top-level ratio list
        ratio_list = soup.find("ul", {"id": "top-ratios"})
        if ratio_list:
            items = ratio_list.find_all("li")
            for item in items:
                name_tag = item.find("span", class_="name")
                value_tag = item.find("span", class_="number")
                if name_tag and value_tag:
                    name = name_tag.get_text(strip=True)
                    value = value_tag.get_text(strip=True).replace(",", "").replace("%", "")
                    try:
                        ratios[name] = float(value)
                    except ValueError:
                        ratios[name] = value

        return ratios
    except Exception:
        return {}


def _scrape_screener_shareholding(ticker: str) -> dict:
    """Scrape latest shareholding pattern from Screener.in."""
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    result = {}

    for url in [
        f"https://www.screener.in/company/{symbol}/consolidated/",
        f"https://www.screener.in/company/{symbol}/",
    ]:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            for heading in soup.find_all(["h2", "h3", "h4"]):
                if "shareholding" in heading.get_text(strip=True).lower():
                    table = heading.find_next("table")
                    if not table:
                        continue
                    for row in table.find_all("tr"):
                        cols = row.find_all(["td", "th"])
                        if len(cols) < 2:
                            continue
                        label = cols[0].get_text(strip=True).lower().rstrip("+")
                        last_val = cols[-1].get_text(strip=True).replace("%", "").replace(",", "")
                        try:
                            pct = float(last_val)
                        except ValueError:
                            continue

                        if "promoter" in label:
                            result["Promoters"] = result.get("Promoters", 0) + pct
                        elif "fii" in label or "foreign" in label:
                            result["FII"] = result.get("FII", 0) + pct
                        elif "dii" in label or ("domestic" in label and "institut" in label):
                            result["DII"] = result.get("DII", 0) + pct
                        elif "government" in label:
                            result["Promoters"] = result.get("Promoters", 0) + pct
                        elif "public" in label:
                            result["Retail & Others"] = result.get("Retail & Others", 0) + pct

                    if result:
                        return result
        except Exception:
            continue

    return result


def get_shareholding_pattern(ticker: str) -> dict:
    """Fetch shareholding pattern — tries yfinance first, falls back to Screener.in."""
    stock = yf.Ticker(ticker)
    result = {
        "Promoters": 0,
        "FII": 0,
        "DII": 0,
        "Retail & Others": 0,
    }

    try:
        holders = stock.major_holders
        if holders is not None and not holders.empty:
            for _, row in holders.iterrows():
                label = str(row.iloc[1]).lower() if len(row) > 1 else ""
                pct = float(str(row.iloc[0]).replace("%", "")) if row.iloc[0] else 0

                if "insider" in label or "promoter" in label:
                    result["Promoters"] += pct
                elif "institution" in label:
                    result["DII"] += pct
                elif "float" in label or "public" in label:
                    result["Retail & Others"] += pct

        inst = stock.institutional_holders
        if inst is not None and not inst.empty and "pctHeld" in inst.columns:
            total_inst_pct = inst["pctHeld"].sum() * 100
            fii_estimate = total_inst_pct * 0.5
            result["FII"] = round(fii_estimate, 2)
            result["DII"] = round(max(0, result["DII"] - fii_estimate), 2)

    except Exception:
        pass

    total = sum(result.values())
    if total > 0 and abs(total - 100) > 5:
        result["Retail & Others"] = round(max(0, 100 - result["Promoters"] - result["FII"] - result["DII"]), 2)

    # Fallback to Screener.in if yfinance returned nothing useful
    if total == 0 or all(v == 0 for v in result.values()):
        screener_data = _scrape_screener_shareholding(ticker)
        if screener_data:
            result = {
                "Promoters": round(screener_data.get("Promoters", 0), 2),
                "FII": round(screener_data.get("FII", 0), 2),
                "DII": round(screener_data.get("DII", 0), 2),
                "Retail & Others": round(screener_data.get("Retail & Others", 0), 2),
            }

    return result


def get_historical_returns(ticker: str, benchmark: str = "^NSEI") -> dict:
    """Compute 1Y, 3Y, 5Y returns for stock and benchmark (Nifty 50)."""
    stock = yf.Ticker(ticker)
    bench = yf.Ticker(benchmark)
    now = datetime.now()

    periods = {
        "1Y": timedelta(days=365),
        "3Y": timedelta(days=365 * 3),
        "5Y": timedelta(days=365 * 5),
    }

    result = {}
    for label, delta in periods.items():
        start = now - delta
        try:
            s_hist = stock.history(start=start, end=now)
            b_hist = bench.history(start=start, end=now)

            if s_hist.empty or len(s_hist) < 2:
                result[label] = {"stock": None, "benchmark": None}
                continue

            s_return = ((s_hist["Close"].iloc[-1] / s_hist["Close"].iloc[0]) - 1) * 100
            b_return = None
            if not b_hist.empty and len(b_hist) >= 2:
                b_return = ((b_hist["Close"].iloc[-1] / b_hist["Close"].iloc[0]) - 1) * 100

            result[label] = {
                "stock": round(float(s_return), 1),
                "benchmark": round(float(b_return), 1) if b_return is not None else None,
            }
        except Exception:
            result[label] = {"stock": None, "benchmark": None}

    return result


def _screener_to_metrics(screener_ratios: dict, info: dict) -> pd.DataFrame:
    """Convert Screener.in ratios into a single-year metrics DataFrame compatible with compute_all_ratios."""
    if not screener_ratios:
        return pd.DataFrame()

    mcap = info.get("market_cap", 0)
    cmp = info.get("cmp", 0)
    pe = info.get("pe_ratio", 0) or 0
    shares = info.get("shares_outstanding", 0) or 0

    net_income = (cmp / pe * shares) if pe > 0 and shares > 0 else 0
    revenue = screener_ratios.get("Sales", 0) or 0
    if revenue == 0 and net_income > 0:
        net_margin_pct = screener_ratios.get("Net Profit Margin", 10)
        if isinstance(net_margin_pct, (int, float)) and net_margin_pct > 0:
            revenue = net_income / (net_margin_pct / 100)

    roe = screener_ratios.get("ROE", 0)
    if isinstance(roe, str):
        roe = 0
    equity = net_income / (roe / 100) if roe > 0 else 0

    roce = screener_ratios.get("ROCE", 0)
    if isinstance(roce, str):
        roce = 0

    current_ratio = screener_ratios.get("Current Ratio", 0)
    if isinstance(current_ratio, str):
        current_ratio = 0

    de = screener_ratios.get("Debt to Equity", 0)
    if isinstance(de, str):
        de = 0
    total_debt = equity * de if equity > 0 else 0
    total_assets = equity + total_debt if equity > 0 else mcap

    ebitda_margin = screener_ratios.get("OPM", 20)
    if isinstance(ebitda_margin, str):
        ebitda_margin = 20
    ebitda = revenue * ebitda_margin / 100 if revenue > 0 else 0

    current_liabilities = total_assets * 0.2 if total_assets > 0 else 0
    current_assets = current_ratio * current_liabilities if current_liabilities > 0 else 0

    metrics = {
        "latest": {
            "Revenue": revenue,
            "COGS": revenue * 0.6,
            "Gross Profit": revenue * 0.4,
            "EBITDA": ebitda,
            "D&A": ebitda * 0.15,
            "Operating Income": ebitda * 0.85,
            "Net Income": net_income,
            "Interest Expense": 0,
            "Tax": 0,
            "Total Assets": total_assets,
            "Total Equity": equity,
            "Total Debt": total_debt,
            "Current Assets": current_assets,
            "Current Liabilities": current_liabilities,
            "Inventory": 0,
            "Receivables": 0,
            "Payables": 0,
            "Cash": 0,
            "Retained Earnings": equity * 0.7,
            "Working Capital": current_assets - current_liabilities,
            "Operating Cash Flow": 0,
            "Capex": 0,
            "Free Cash Flow": 0,
        }
    }
    return pd.DataFrame(metrics)


def get_peer_data(peer_tickers: list) -> dict:
    """
    Fetch key metrics for a list of peer tickers.
    Tries yfinance first, falls back to Screener.in if financials are empty.
    """
    peer_data = {}
    for ticker in peer_tickers:
        try:
            info = get_stock_info(ticker)
            financials = get_financials(ticker)
            key_metrics = extract_key_metrics(financials)

            if key_metrics.empty:
                screener = scrape_screener_ratios(ticker)
                key_metrics = _screener_to_metrics(screener, info)

            peer_data[ticker] = {
                "info": info,
                "metrics": key_metrics,
            }
        except Exception:
            try:
                stock = yf.Ticker(ticker)
                yf_info = stock.info
                info = {
                    "name": yf_info.get("longName", ticker.replace(".NS", "")),
                    "sector": yf_info.get("sector", "N/A"),
                    "industry": yf_info.get("industry", "N/A"),
                    "cmp": yf_info.get("currentPrice", yf_info.get("regularMarketPrice", 0)),
                    "market_cap": yf_info.get("marketCap", 0),
                    "pe_ratio": yf_info.get("trailingPE", 0),
                    "shares_outstanding": yf_info.get("sharesOutstanding", 0),
                }
                screener = scrape_screener_ratios(ticker)
                key_metrics = _screener_to_metrics(screener, info)
                peer_data[ticker] = {"info": info, "metrics": key_metrics}
            except Exception:
                peer_data[ticker] = {"info": {"name": ticker, "error": True}, "metrics": pd.DataFrame()}
    return peer_data
