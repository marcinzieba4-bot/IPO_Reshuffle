"""Charts for a finished run:  python plot_results.py stocks_T-10_t0_close"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).parent / "results"
tag = sys.argv[1] if len(sys.argv) > 1 else "stocks_T-10_t0_close"
SURFACE, INK, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.tick_params(length=0)


ep = pd.read_csv(OUT / f"event_path_{tag}.csv", index_col=0)
nav = pd.read_csv(OUT / f"nav_{tag}.csv", index_col=0, parse_dates=True)["nav_ret"]

fig, ax = plt.subplots(figsize=(8, 4.2))
m, se = ep["mean"] * 1e4, ep["se"] * 1e4
ax.fill_between(ep.index, m - 2 * se, m + 2 * se, color=BLUE, alpha=0.15, lw=0)
ax.plot(ep.index, m, color=BLUE, lw=2)
ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.annotate("IPO day", (0, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points", color=MUTED, fontsize=9)
style(ax, f"Peer basket vs SPX around large IPOs (beta-hedged, bps, n={int(ep['n'].iloc[0])})")
ax.set_xlabel("trading days relative to IPO  ·  band = ±2 standard errors")
fig.tight_layout(); fig.savefig(OUT / f"event_path_{tag}.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(nav.index, nav.cumsum() * 100, color=BLUE, lw=2)
style(ax, "Strategy cumulative P&L, net of costs (% of NAV)")
fig.tight_layout(); fig.savefig(OUT / f"equity_{tag}.png", dpi=200); plt.close(fig)
print("wrote charts for", tag)
