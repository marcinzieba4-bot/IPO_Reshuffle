"""Pre-IPO sector-pressure trade: short listed peers, long SPX, cover at the IPO.

Per event (T0 = first trading day of the IPO):
    entry : close of T0 - entry_offset trading days
    short : equal-weight basket of same-sector peers (or the sector ETF)
    long  : beta x SPY, beta from a pre-entry OLS window
    exit  : at the IPO (T0 close / T0 open / T-1 close)

All functions here are pure (matrices in, numbers out) so they can be unit tested
without network access.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Config:
    min_deal_usd: float = 1e9      # "large IPO"
    entry_offset: int = 10         # trading days before T0
    exit: str = "t0_close"         # t0_close | t0_open | tm1_close
    peer_mode: str = "stocks"      # stocks | etf
    n_peers: int = 10
    min_peers: int = 4
    min_history: int = 250         # trading days of history a peer needs
    min_dollar_vol: float = 5e6    # median daily $ volume (60d) a peer needs
    beta_window: int = 120
    hedge: str = "beta"            # beta | dollar
    cost_bps_peer: float = 5.0     # per side, on short notional
    cost_bps_hedge: float = 1.0    # per side, on hedge notional
    borrow_bps_annual: float = 100.0
    event_weight: float = 0.10     # short notional per event, fraction of NAV
    max_gross_short: float = 0.50  # cap on total short notional
    placebo_shifts: tuple = (-200, -160, -120, -80, -40, 40, 80, 120, 160, 200)


@dataclass
class Panel:
    """Aligned price matrices on the hedge instrument's trading calendar."""
    adj: pd.DataFrame        # total-return close
    adj_open: pd.DataFrame   # open, rescaled to the total-return series
    dvol: pd.DataFrame       # close * volume (unadjusted)
    hedge: str = "SPY"


def build_panel(px: dict[str, pd.DataFrame], hedge: str = "SPY") -> Panel:
    cal = px[hedge].index
    adj = pd.DataFrame({t: p["adj"] for t, p in px.items()}).reindex(cal)
    op = pd.DataFrame({t: p["open"] * p["adj"] / p["close"] for t, p in px.items()}).reindex(cal)
    dv = pd.DataFrame({t: p["close"] * p["volume"] for t, p in px.items()}).reindex(cal)
    return Panel(adj, op, dv, hedge)


def pick_peers(panel: Panel, candidates: list[str], ie: int, cfg: Config) -> list[str]:
    """Liquid peers with enough history, chosen using data up to the entry close only."""
    cands = [c for c in dict.fromkeys(candidates) if c in panel.adj.columns and c != panel.hedge]
    if not cands or ie < cfg.min_history:
        return []
    hist = panel.adj[cands].iloc[ie - cfg.min_history: ie + 1]
    ok = hist.columns[hist.notna().mean() > 0.98]
    liq = panel.dvol[ok].iloc[ie - 59: ie + 1].median()
    liq = liq[liq >= cfg.min_dollar_vol].sort_values(ascending=False)
    return list(liq.index[: cfg.n_peers])


def estimate_beta(panel: Panel, peers: list[str], end: int, cfg: Config) -> float:
    if cfg.hedge == "dollar":
        return 1.0
    r = panel.adj[peers + [panel.hedge]].iloc[end - cfg.beta_window: end + 1].ffill().pct_change().iloc[1:]
    b, m = r[peers].mean(axis=1), r[panel.hedge]
    beta = np.cov(b, m)[0, 1] / np.var(m, ddof=1)
    return float(np.clip(beta, 0.3, 2.5))


def _window_levels(panel: Panel, peers, ie, ix, use_open=False):
    """Basket and hedge value paths, both 1.0 at the entry close[ie]; last point is the exit."""
    cols = peers + [panel.hedge]
    lv = panel.adj[cols].iloc[ie: ix + 1].ffill().copy()
    if use_open:
        o = panel.adj_open[cols].iloc[ix]
        lv.iloc[-1] = o.where(o.notna(), lv.iloc[-2])
    lv = lv / lv.iloc[0]
    return lv[peers].mean(axis=1), lv[panel.hedge]   # buy-and-hold equal-weight basket


def _window_pnl(panel: Panel, peers, beta, ie, ix, use_open=False) -> pd.Series:
    """Daily gross P&L per $1 short: short basket, long beta x hedge."""
    basket, hedge = _window_levels(panel, peers, ie, ix, use_open)
    return (-basket.diff() + beta * hedge.diff()).iloc[1:]


def run_event(panel: Panel, ipo_date, candidates: list[str], cfg: Config):
    """Returns (trade dict, daily net pnl Series per $1 short) or None if untradeable."""
    cal = panel.adj.index
    i0 = int(cal.searchsorted(pd.Timestamp(ipo_date)))
    if i0 >= len(cal):
        return None
    ie = i0 - cfg.entry_offset
    ix = i0 - 1 if cfg.exit == "tm1_close" else i0
    if ix <= ie:
        return None
    peers = pick_peers(panel, candidates, ie, cfg)
    if len(peers) < (1 if cfg.peer_mode == "etf" else cfg.min_peers):
        return None
    beta = estimate_beta(panel, peers, ie, cfg)
    use_open = cfg.exit == "t0_open"
    basket, hedge = _window_levels(panel, peers, ie, ix, use_open)
    daily = (-basket.diff() + beta * hedge.diff()).iloc[1:]
    gross = float(daily.sum())

    side_cost = (cfg.cost_bps_peer + beta * cfg.cost_bps_hedge) / 1e4
    borrow = cfg.borrow_bps_annual / 1e4 * (cal[ix] - cal[ie]).days / 365
    net_daily = daily.copy()
    net_daily.iloc[0] -= side_cost
    net_daily.iloc[-1] -= side_cost + borrow

    # Placebo: identical trade on the same basket at shifted (non-event) dates.
    plc = []
    for s in cfg.placebo_shifts:
        a, b = ie + s, ix + s
        if a - cfg.beta_window > 0 and b < len(cal) and panel.adj[peers].iloc[a].notna().all():
            pb = estimate_beta(panel, peers, a, cfg)
            plc.append(float(_window_pnl(panel, peers, pb, a, b, use_open).sum()))
    placebo = float(np.mean(plc)) if plc else np.nan

    trade = {
        "t0": cal[i0], "entry": cal[ie], "exit": cal[ix], "n_peers": len(peers),
        "peers": " ".join(peers), "beta": round(beta, 3),
        "peer_ret": float(basket.iloc[-1] - 1), "spy_ret": float(hedge.iloc[-1] - 1),
        "gross": gross, "net": float(net_daily.sum()), "placebo": placebo, "abnormal": gross - placebo,
    }
    return trade, net_daily


def event_path(panel: Panel, ipo_date, peers: list[str], cfg: Config, pre: int = 30, post: int = 20):
    """Cumulative beta-hedged return of the (long) peer basket, T-pre .. T+post, 0 at T-pre."""
    cal = panel.adj.index
    i0 = int(cal.searchsorted(pd.Timestamp(ipo_date)))
    a, b = i0 - pre, i0 + post
    if a - cfg.beta_window <= 0 or b >= len(cal) or not peers:
        return None
    beta = estimate_beta(panel, peers, a, cfg)
    r = panel.adj[peers + [panel.hedge]].iloc[a: b + 1].ffill().pct_change().iloc[1:]
    ar = (r[peers].mean(axis=1) - beta * r[panel.hedge]).cumsum().to_numpy()
    return pd.Series(np.r_[0.0, ar], index=range(-pre, post + 1))


def portfolio(dailies: list[pd.Series], cal: pd.DatetimeIndex, cfg: Config) -> pd.Series:
    """Daily NAV return: each event gets event_weight short notional, scaled down under the gross cap."""
    if not dailies:
        return pd.Series(dtype=float)
    m = pd.concat(dailies, axis=1).reindex(cal[(cal >= min(d.index[0] for d in dailies)) &
                                              (cal <= max(d.index[-1] for d in dailies))])
    active = m.notna().sum(axis=1)
    w = np.minimum(cfg.event_weight, cfg.max_gross_short / active.clip(lower=1))
    return m.fillna(0).sum(axis=1) * w


def trade_stats(tr: pd.DataFrame) -> dict:
    def t(x):
        x = x.dropna()
        return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan
    return {
        "n": len(tr),
        "gross_bps": tr["gross"].mean() * 1e4, "t_gross": t(tr["gross"]),
        "net_bps": tr["net"].mean() * 1e4, "t_net": t(tr["net"]),
        "median_net_bps": tr["net"].median() * 1e4,
        "hit": (tr["net"] > 0).mean(),
        "placebo_bps": tr["placebo"].mean() * 1e4,
        "abn_bps": tr["abnormal"].mean() * 1e4, "t_abn": t(tr["abnormal"]),
        "worst_bps": tr["net"].min() * 1e4, "best_bps": tr["net"].max() * 1e4,
    }


def portfolio_stats(nav_ret: pd.Series) -> dict:
    if nav_ret.empty:
        return {}
    eq = nav_ret.cumsum()
    yrs = len(nav_ret) / 252
    vol = nav_ret.std() * np.sqrt(252)
    return {
        "ann_ret": nav_ret.sum() / yrs, "ann_vol": vol,
        "sharpe": nav_ret.mean() / nav_ret.std() * np.sqrt(252) if nav_ret.std() > 0 else np.nan,
        "max_dd": float((eq - eq.cummax()).min()),
        "pct_days_invested": float((nav_ret != 0).mean()),
    }


def cfg_dict(cfg: Config) -> dict:
    return asdict(cfg)
