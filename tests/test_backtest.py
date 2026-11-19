import numpy as np
import pandas as pd
import pytest

from ipo_reshuffle.backtest import Config, Panel, portfolio, run_event, pick_peers

N, T0 = 600, 500


def make_panel(peer_drift=0.0, seed=0):
    """SPY random walk; 6 peers with beta 1.5 plus an extra drift in the 10 days before T0."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2015-01-01", periods=N)
    m = rng.normal(0.0003, 0.01, N)
    rets = {"SPY": m}
    for i in range(6):
        r = 1.5 * m + rng.normal(0, 0.002, N)
        r[T0 - 9: T0 + 1] += peer_drift
        rets[f"P{i}"] = r
    adj = pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in rets.items()}, index=cal)
    dvol = pd.DataFrame(1e8, index=cal, columns=adj.columns)
    return Panel(adj, adj.shift(1).bfill(), dvol, "SPY"), cal


CFG = Config(cost_bps_peer=0, cost_bps_hedge=0, borrow_bps_annual=0)
PEERS = [f"P{i}" for i in range(6)]


def test_beta_hedge_removes_market_and_captures_peer_drop():
    panel, cal = make_panel(peer_drift=-0.002)
    tr, daily = run_event(panel, cal[T0], PEERS, CFG)
    assert tr["entry"] == cal[T0 - 10] and tr["exit"] == cal[T0]
    assert tr["beta"] == pytest.approx(1.5, abs=0.1)
    assert tr["gross"] == pytest.approx(0.02, abs=0.012)   # 10 days x 20bps of peer weakness
    assert tr["abnormal"] > 0.01                            # placebo windows have no drift
    assert daily.sum() == pytest.approx(tr["net"])


def test_no_edge_without_peer_drift():
    panel, cal = make_panel(peer_drift=0.0)
    tr, _ = run_event(panel, cal[T0], PEERS, CFG)
    assert abs(tr["gross"]) < 0.012


def test_costs_reduce_net():
    panel, cal = make_panel()
    cfg = Config(cost_bps_peer=5, cost_bps_hedge=1, borrow_bps_annual=100)
    tr, _ = run_event(panel, cal[T0], PEERS, cfg)
    expected = 2 * (5 + tr["beta"] * 1) / 1e4 + 0.01 * (tr["exit"] - tr["entry"]).days / 365
    assert tr["gross"] - tr["net"] == pytest.approx(expected, abs=1e-6)  # beta is stored rounded


def test_exit_variants_and_no_lookahead():
    panel, cal = make_panel()
    tr, _ = run_event(panel, cal[T0], PEERS, Config(exit="tm1_close"))
    assert tr["exit"] == cal[T0 - 1]
    # Changing prices after the exit must not change the trade.
    base, _ = run_event(panel, cal[T0], PEERS, CFG)
    panel.adj.iloc[T0 + 1:, 1:] *= 3
    again, _ = run_event(panel, cal[T0], PEERS, CFG)
    for k in ("peers", "beta", "gross", "net"):
        assert base[k] == again[k]


def test_peer_filters():
    panel, cal = make_panel()
    panel.dvol["P0"] = 1e5                 # illiquid
    panel.adj.loc[: cal[T0 - 100], "P1"] = np.nan   # recently listed
    assert set(pick_peers(panel, PEERS + ["SPY", "NOPE"], T0 - 10, CFG)) == {"P2", "P3", "P4", "P5"}


def test_portfolio_gross_cap():
    cal = pd.bdate_range("2020-01-01", periods=5)
    d = [pd.Series(0.01, index=cal[1:4]) for _ in range(10)]
    nav = portfolio(d, cal, Config(event_weight=0.10, max_gross_short=0.50))
    assert nav.iloc[0] == pytest.approx(10 * 0.01 * 0.05)   # 10 events share the 50% cap
