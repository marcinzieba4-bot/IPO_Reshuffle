"""Data access: Nasdaq IPO calendar + stock screener, Yahoo daily prices.

Everything is cached on disk under data/cache so re-runs are offline and fast.
"""
from __future__ import annotations

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(os.environ.get("IPO_CACHE", Path(__file__).resolve().parent.parent / "data" / "cache"))
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# Nasdaq screener sector -> liquid sector ETF with a long history.
SECTOR_ETF = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Finance": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "VNQ",
    "Telecommunications": "VOX",
    "Basic Materials": "XLB",
}
HEDGE = "SPY"


def _get(url: str, tries: int = 5) -> requests.Response | None:
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def _money(s) -> float:
    try:
        return float(str(s).replace("$", "").replace(",", ""))
    except ValueError:
        return float("nan")


def ipo_calendar(start: str = "2010-01", end: str | None = None) -> pd.DataFrame:
    """All priced US IPOs between two YYYY-MM months (inclusive)."""
    end = end or pd.Timestamp.today().strftime("%Y-%m")
    this_month = pd.Timestamp.today().strftime("%Y-%m")
    d = CACHE / "ipo"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in pd.period_range(start, end, freq="M").strftime("%Y-%m"):
        f = d / f"{m}.json"
        if f.exists():
            priced = json.loads(f.read_text())
        else:
            r = _get(f"https://api.nasdaq.com/api/ipo/calendar?date={m}")
            if r is None:
                raise RuntimeError(f"IPO calendar download failed for {m}")
            priced = ((r.json().get("data") or {}).get("priced") or {}).get("rows") or []
            if m < this_month:  # only freeze completed months
                f.write_text(json.dumps(priced))
            time.sleep(0.15)
        rows += priced
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "ticker": df["proposedTickerSymbol"].str.strip().str.upper(),
            "company": df["companyName"].str.strip(),
            "exchange": df["proposedExchange"],
            "ipo_date": pd.to_datetime(df["pricedDate"], format="%m/%d/%Y"),
            "ipo_price": df["proposedSharePrice"].map(_money),
            "deal_usd": df["dollarValueOfSharesOffered"].map(_money),
        }
    )
    return out.dropna(subset=["ticker", "deal_usd"]).sort_values("ipo_date").reset_index(drop=True)


def screener() -> pd.DataFrame:
    """Currently listed US stocks with sector / industry (Nasdaq classification)."""
    f = CACHE / "screener.json"
    if f.exists():
        rows = json.loads(f.read_text())
    else:
        r = _get("https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&download=true")
        if r is None:
            raise RuntimeError("screener download failed")
        rows = r.json()["data"]["rows"]
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rows))
    df = pd.DataFrame(rows)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["market_cap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    df["ipoyear"] = pd.to_numeric(df["ipoyear"], errors="coerce")
    junk = df["name"].str.contains(r"Warrant|Unit|Preferred|Notes|Right|Debenture|Acquisition", case=False, na=False)
    df = df[~junk & ~df["symbol"].str.contains(r"\^") & (df["sector"].fillna("") != "")]
    return df[["symbol", "name", "sector", "industry", "market_cap", "ipoyear"]].reset_index(drop=True)


def prices(ticker: str, start: str = "2008-06-01") -> pd.DataFrame | None:
    """Daily bars [open, close, adj, volume]; `adj` is the total-return close."""
    d = CACHE / "px"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{ticker.replace('/', '-')}.csv"
    if f.exists():
        txt = f.read_text()
        if not txt.strip():
            return None
        return pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True)
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp.today().normalize().timestamp())
    sym = ticker.replace("/", "-").replace(".", "-")
    r = _get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div,splits"
    )
    df = None
    try:
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame(
            {
                "open": q["open"],
                "close": q["close"],
                "adj": res["indicators"]["adjclose"][0]["adjclose"],
                "volume": q["volume"],
            },
            index=pd.to_datetime(res["timestamp"], unit="s").normalize(),
        ).dropna(subset=["close", "adj"])
        df = df[~df.index.duplicated()]
    except Exception:
        df = None
    f.write_text("" if df is None or df.empty else df.to_csv())
    return None if df is None or df.empty else df


def price_panel(tickers, workers: int = 6) -> dict[str, pd.DataFrame]:
    tickers = sorted(set(tickers))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(prices, tickers))
    return {t: p for t, p in zip(tickers, res) if p is not None}
