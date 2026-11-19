"""Variants of the IPO trade on mega-deals only (>= $2bn by default).

    python analyze_mega.py            # stats table + P&L chart into results/

Legs are sized per $1 of the "main" leg:
  pre-IPO variants : short $1 sector, long beta x SPY
  IPO-long variants: long $1 of the IPO stock, short $1 sector (dollar-neutral)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ipo_reshuffle import data
from ipo_reshuffle.backtest import Config, build_panel, estimate_beta, pick_peers, portfolio, portfolio_stats
from run_backtest import candidates, load_events

OUT = Path(__file__).parent / "results"
MIN_DEAL = float(sys.argv[1]) if len(sys.argv) > 1 else 2e9
CFG = Config(min_deal_usd=MIN_DEAL)
COST = 5e-4          # per side, per leg
BORROW = 0.01        # per year on the short
# No offer price / no allocation exists for these: excluded from "buy at IPO price" variants.
DIRECT_LISTINGS = {"SPOT", "WORK", "COIN", "RBLX", "PLTR", "ASAN", "SQSP", "AMPL", "WRBY"}


def splits_after(ticker: str, t0) -> float:
    """Cumulative split factor after t0 (to convert the raw offer price to Yahoo's split-adjusted scale)."""
    r = data._get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1=0&period2=9999999999"
                  f"&interval=3mo&events=splits")
    try:
        ev = r.json()["chart"]["result"][0].get("events", {}).get("splits", {})
    except Exception:
        return 1.0
    f = 1.0
    for s in ev.values():
        if pd.Timestamp(s["date"], unit="s") > t0:
            f *= s["numerator"] / s["denominator"]
    return f


def leg(panel, cols, a, b):
    """Equal-weight buy-and-hold value path from close[a] to close[b], 1.0 at a."""
    lv = panel.adj[cols].iloc[a: b + 1].ffill()
    return (lv / lv.iloc[0]).mean(axis=1)


def trade(long_path, short_path, h_long, h_short, cal):
    """Daily net P&L. Paths share an index; the first row is the entry."""
    d = (h_long * long_path.diff() - h_short * short_path.diff()).iloc[1:]
    d.iloc[0] -= COST * (h_long + h_short)
    d.iloc[-1] -= COST * (h_long + h_short) + BORROW * h_short * (d.index[-1] - long_path.index[0]).days / 365
    return d


def build(ev, cands, panel, px):
    """Returns {variant: [daily pnl Series per event]}."""
    cal, H = panel.adj.index, panel.hedge
    out: dict[str, list] = {}
    add = lambda k, d: out.setdefault(k, []).append(d)
    for e in ev.itertuples():
        i0 = int(cal.searchsorted(e.t0))
        if i0 + 61 >= len(cal) or e.sector not in data.SECTOR_ETF:
            continue
        etf = [data.SECTOR_ETF[e.sector]]
        peers = pick_peers(panel, cands[e.ticker], i0 - 20, CFG)
        shorts = {"ETF": etf, "peers": peers if len(peers) >= CFG.min_peers else None}

        for nm, s in shorts.items():
            if s is None:
                continue
            # --- A. original idea: short sector, long SPY, before the IPO -----------------
            for lab, a, b in [("T-10 > IPO close", i0 - 10, i0), ("T-20 > T-1", i0 - 20, i0 - 1)]:
                beta = estimate_beta(panel, s, a, CFG)
                add(f"A. Short {nm} / long SPX, {lab}", trade(leg(panel, [H], a, b), leg(panel, s, a, b), beta, 1, cal))
            # --- D. rebound: long sector, short SPY after the IPO ------------------------
            if nm == "ETF":
                a, b = i0 + 3, i0 + 20
                beta = estimate_beta(panel, s, a, CFG)
                add("D. Rebound: long ETF / short SPX, T+3 > T+20",
                    trade(leg(panel, s, a, b), leg(panel, [H], a, b), 1, beta, cal))

            # --- B/C. long the IPO stock, short the sector --------------------------------
            p = px.get(e.ticker)
            if p is None or p.index[0] != cal[i0]:
                continue
            ipo = panel.adj[e.ticker]
            ratio = p["adj"].iloc[0] / p["close"].iloc[0]
            entries = {"first-day open": (p["open"].iloc[0] * ratio, panel.adj_open[s].iloc[i0])}
            if e.ticker not in DIRECT_LISTINGS:
                entries["IPO price"] = (e.ipo_price / e.split * ratio, panel.adj[s].iloc[i0 - 1])
            for ename, (lp0, sp0) in entries.items():
                for hold, b in [("first-day close", i0), ("T+5", i0 + 5), ("T+20", i0 + 20)]:
                    idx = cal[i0 - 1: b + 1]   # row 0 = entry (overwritten with entry prices)
                    lp = ipo.reindex(idx).ffill(); lp.iloc[0] = lp0
                    sp = panel.adj[s].reindex(idx).ffill(); sp.iloc[0] = sp0
                    lp, sp = lp / lp.iloc[0], (sp / sp.iloc[0]).mean(axis=1)
                    add(f"B. Buy IPO at {ename} / short {nm}, exit {hold}", trade(lp, sp, 1, 1, cal))
                    if nm == "ETF":
                        add(f"B0. Buy IPO at {ename}, unhedged, exit {hold}", trade(lp, sp, 1, 0, cal))
            # after the first day (no allocation needed)
            for hold, b in [("T+20", i0 + 20), ("T+60", i0 + 60)]:
                add(f"C. Buy IPO at first-day close / short {nm}, exit {hold}",
                    trade(leg(panel, [e.ticker], i0, b), leg(panel, s, i0, b), 1, 1, cal))
    return out


def stats(dailies, cal):
    x = pd.Series([d.sum() for d in dailies])
    nav = portfolio(dailies, cal, CFG)
    ps = portfolio_stats(nav)
    return {"n": len(x), "mean_%": x.mean() * 100, "median_%": x.median() * 100,
            "t": x.mean() / x.std(ddof=1) * np.sqrt(len(x)), "hit_%": (x > 0).mean() * 100,
            "worst_%": x.min() * 100, "best_%": x.max() * 100,
            "ann_ret_%": ps["ann_ret"] * 100, "sharpe": ps["sharpe"], "max_dd_%": ps["max_dd"] * 100}, nav


def main():
    ev = load_events("2010-01", MIN_DEAL)
    cands = candidates(ev)
    need = {data.HEDGE, *data.SECTOR_ETF.values(), *ev["ticker"], *itertools.chain(*cands.values())}
    px = data.price_panel(need)
    panel = build_panel(px, data.HEDGE)
    ev["t0"] = [px[t].index[0] if t in px and abs((px[t].index[0] - d).days) <= 7 else d
                for t, d in zip(ev["ticker"], ev["ipo_date"])]
    ev["split"] = [splits_after(t, d) for t, d in zip(ev["ticker"], ev["t0"])]

    res = build(ev, cands, panel, px)
    rows, navs = [], {}
    for k in sorted(res):
        s, navs[k] = stats(res[k], panel.adj.index)
        rows.append({"variant": k, **s})
    tab = pd.DataFrame(rows).set_index("variant")
    tag = f"mega_{MIN_DEAL/1e9:g}bn"
    OUT.mkdir(exist_ok=True)
    tab.to_csv(OUT / f"{tag}_variants.csv", float_format="%.3f")
    pd.DataFrame(navs).fillna(0).to_csv(OUT / f"{tag}_nav.csv", float_format="%.6f")
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(tab.round(2).to_string())
    chart(navs, tab, tag)


def chart(navs, tab, tag):
    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    picks = [("A. Short ETF / long SPX, T-10 > IPO close", "Pre-IPO: short sector ETF, long SPX", "#2a78d6"),
             ("B. Buy IPO at IPO price / short ETF, exit first-day close", "Buy at IPO price, short ETF, exit day 1", "#eb6834"),
             ("B. Buy IPO at first-day open / short ETF, exit T+20", "Buy at first-day open, short ETF, 20 days", "#1baf7a"),
             ("C. Buy IPO at first-day close / short ETF, exit T+60", "Buy at first-day close, short ETF, 60 days", "#6250d6")]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for key, label, col in picks:
        eq = navs[key].cumsum() * 100
        ax.plot(eq.index, eq, color=col, lw=2, label=f"{label}  (n={int(tab.loc[key, 'n'])}):  {eq.iloc[-1]:+.1f}%")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(length=0)
    ax.set_title(f"Cumulative P&L, IPOs ≥ ${MIN_DEAL/1e9:g}bn, net of costs (% of NAV, 10% of NAV per deal)",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_pnl.png", dpi=200)


if __name__ == "__main__":
    main()
