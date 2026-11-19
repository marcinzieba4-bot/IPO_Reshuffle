"""Run the pre-IPO peer-short / SPX-long backtest.

    python run_backtest.py                 # base case, writes results/
    python run_backtest.py --sweep         # robustness grid
    python run_backtest.py --entry 15 --min-deal 2e9 --mode etf --exit t0_open
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ipo_reshuffle import data
from ipo_reshuffle.backtest import (Config, build_panel, event_path, pick_peers, portfolio,
                                    portfolio_stats, run_event, trade_stats)

OUT = Path(__file__).parent / "results"
PRESELECT = 30  # peer candidates downloaded per industry


def load_events(start: str, min_deal: float) -> pd.DataFrame:
    """Large, non-SPAC IPOs that can be mapped to a sector."""
    ipos = data.ipo_calendar(start)
    spac = ipos["company"].str.contains(r"Acquisition|Blank Check|SPAC|Capital Corp", case=False) | (
        (ipos["ipo_price"] == 10.0) & ipos["ticker"].str.endswith("U"))
    ev = ipos[~spac & (ipos["deal_usd"] >= min_deal)].drop_duplicates("ticker")
    scr = data.screener().set_index("symbol")
    ev = ev.join(scr[["sector", "industry"]], on="ticker")
    print(f"IPOs >= ${min_deal/1e9:.1f}bn (ex-SPAC): {len(ev)}; with sector mapping: {ev['sector'].notna().sum()}")
    return ev.dropna(subset=["sector"]).reset_index(drop=True)


def candidates(ev: pd.DataFrame) -> dict[str, list[str]]:
    """Per event: same-industry listed stocks (fallback: same sector), biggest first."""
    scr = data.screener()
    scr = scr[scr["market_cap"] > 0].sort_values("market_cap", ascending=False)
    out = {}
    for e in ev.itertuples():
        older = scr[(scr["symbol"] != e.ticker) & ~(scr["ipoyear"] >= e.ipo_date.year)]
        ind = older[older["industry"] == e.industry]
        if len(ind) < 8:
            ind = older[older["sector"] == e.sector]
        out[e.ticker] = list(ind["symbol"].head(PRESELECT))
    return out


def backtest(ev, cands, panel, cfg: Config, paths: bool = False):
    trades, dailies, curves = [], [], []
    for e in ev[ev["deal_usd"] >= cfg.min_deal_usd].itertuples():
        if cfg.peer_mode == "etf" and e.sector not in data.SECTOR_ETF:
            continue
        pool = [data.SECTOR_ETF[e.sector]] if cfg.peer_mode == "etf" else cands[e.ticker]
        c = replace(cfg, min_dollar_vol=0.0) if cfg.peer_mode == "etf" else cfg
        res = run_event(panel, e.t0, pool, c)
        if res is None:
            continue
        tr, d = res
        trades.append({"ticker": e.ticker, "company": e.company, "sector": e.sector,
                       "industry": e.industry, "deal_usd_bn": e.deal_usd / 1e9, **tr})
        dailies.append(d)
        if paths:
            p = event_path(panel, e.t0, tr["peers"].split(), c)
            if p is not None:
                curves.append(p)
    tr = pd.DataFrame(trades)
    nav = portfolio(dailies, panel.adj.index, cfg)
    return tr, nav, (pd.concat(curves, axis=1) if curves else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--min-deal", type=float, default=1e9)
    ap.add_argument("--entry", type=int, default=10)
    ap.add_argument("--exit", default="t0_close", choices=["t0_close", "t0_open", "tm1_close"])
    ap.add_argument("--mode", default="stocks", choices=["stocks", "etf"])
    ap.add_argument("--hedge", default="beta", choices=["beta", "dollar"])
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    cfg = Config(min_deal_usd=a.min_deal, entry_offset=a.entry, exit=a.exit, peer_mode=a.mode, hedge=a.hedge)

    ev = load_events(a.start, min(a.min_deal, 5e8) if a.sweep else a.min_deal)
    cands = candidates(ev)
    need = {data.HEDGE, *data.SECTOR_ETF.values(), *ev["ticker"], *itertools.chain(*cands.values())}
    print(f"loading prices for {len(need)} tickers ...")
    px = data.price_panel(need)
    panel = build_panel(px, data.HEDGE)

    # T0 = first bar of the IPO ticker when it sits within a week of the calendar date.
    def t0(e):
        p = px.get(e.ticker)
        if p is not None and abs((p.index[0] - e.ipo_date).days) <= 7:
            return p.index[0]
        return e.ipo_date
    ev["t0"] = [t0(e) for e in ev.itertuples()]
    OUT.mkdir(exist_ok=True)

    if a.sweep:
        rows = []
        for mode, n, dmin, ex in itertools.product(["stocks", "etf"], [5, 10, 15, 20], [5e8, 1e9, 2e9],
                                                   ["tm1_close", "t0_open", "t0_close"]):
            c = replace(cfg, peer_mode=mode, entry_offset=n, min_deal_usd=dmin, exit=ex)
            tr, nav, _ = backtest(ev, cands, panel, c)
            rows.append({"mode": mode, "entry_T-": n, "min_deal_bn": dmin / 1e9, "exit": ex,
                         **trade_stats(tr), **portfolio_stats(nav)})
        sw = pd.DataFrame(rows)
        sw.to_csv(OUT / "sweep.csv", index=False, float_format="%.4f")
        pd.set_option("display.width", 250)
        print(sw.round(2).to_string(index=False))
        return

    tr, nav, curves = backtest(ev, cands, panel, cfg, paths=True)
    tag = f"{cfg.peer_mode}_T-{cfg.entry_offset}_{cfg.exit}"
    tr.to_csv(OUT / f"trades_{tag}.csv", index=False, float_format="%.5f")
    nav.rename("nav_ret").to_csv(OUT / f"nav_{tag}.csv", float_format="%.6f")
    if curves is not None:
        pd.DataFrame({"mean": curves.mean(axis=1), "median": curves.median(axis=1),
                      "se": curves.std(axis=1) / np.sqrt(curves.shape[1]),
                      "n": curves.shape[1]}).to_csv(OUT / f"event_path_{tag}.csv", float_format="%.5f")
    print(f"\n== {tag}, deals >= ${cfg.min_deal_usd/1e9:.1f}bn ==")
    for k, v in {**trade_stats(tr), **portfolio_stats(nav)}.items():
        print(f"{k:>18}: {v:,.3f}")
    tr["year"] = tr["t0"].dt.year
    print("\nby year (net bps / n):")
    print(tr.groupby("year")["net"].agg(lambda x: x.mean() * 1e4).round(0).astype(int).to_frame("net_bps")
          .join(tr.groupby("year").size().rename("n")).T.to_string())
    print("\nby sector:")
    print(tr.groupby("sector")["net"].agg(n="size", net_bps=lambda x: x.mean() * 1e4).round(0).to_string())


if __name__ == "__main__":
    main()
