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
    return {
        "name": info.get("longName", ticker.replace(".NS", "")),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "cmp": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", 0),
        "pb_ratio": info.get("priceToBook", 0),
        "dividend_yield": info.get("dividendYield", 0),
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
        ebitda = safe_get(income, ["EBITDA", "Normalized EBITDA"], year)
        if ebitda == 0:
            da = safe_get(income, ["Depreciation And Amortization", "Depreciation & Amortisation",
                                    "Reconciled Depreciation"], year)
            ebitda = operating_income + abs(da)

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


def get_peer_data(peer_tickers: list) -> dict:
    """Fetch key metrics for a list of peer tickers for benchmarking."""
    peer_data = {}
    for ticker in peer_tickers:
        try:
            info = get_stock_info(ticker)
            financials = get_financials(ticker)
            key_metrics = extract_key_metrics(financials)
            peer_data[ticker] = {
                "info": info,
                "metrics": key_metrics,
            }
        except Exception:
            peer_data[ticker] = {"info": {"name": ticker, "error": True}, "metrics": pd.DataFrame()}
    return peer_data
