# IPO Reshuffle

**Idea.** A large IPO forces investors to make room for it. Sector-dedicated money sells
existing holdings in the same sector to fund the new allocation, so listed peers should
underperform the market in the days before the deal. The trade:

| | |
|---|---|
| **Short** | basket of listed peers from the IPO's sector (or the sector ETF) |
| **Long** | S&P 500 (SPY), sized to make the pair beta-neutral |
| **Entry** | close of T-10 trading days (T0 = IPO's first trading day; roadshow is public by then) |
| **Exit** | close everything at the IPO (T0 close; T0 open and T-1 close also tested) |

## Verdict from the backtest

**As specified, the trade has no reliable edge.** 2010 – Nov 2026, US IPOs, net of costs:

| Variant | Events | Gross / trade | Net / trade | t-stat (net) | Hit rate | vs. placebo |
|---|---|---|---|---|---|---|
| Stock peers, deals ≥ $1bn, T-10 → T0 close (base case) | 111 | −17 bps | −33 bps | −0.8 | 47% | +11 bps (t 0.3) |
| Sector ETF, deals ≥ $1bn, T-10 → T0 close | 112 | +13 bps | −3 bps | −0.1 | 52% | +17 bps (t 0.7) |
| Sector ETF, deals ≥ $2bn, T-10 → T0 close | 41 | +46 bps | +29 bps | 0.9 | 59% | +53 bps (t 1.5) |

- Across all 72 tested combinations (peer type × entry day × deal size × exit), **none reaches
  statistical significance** (best net t-stat is 1.2). Full grid: `results/sweep.csv`.
- The only consistent pocket is **mega-IPOs (≥ $2bn) shorted via the sector ETF**: positive net
  in 10 of its 12 variants, +22 to +37 bps per trade for entries at T-10 or earlier (about zero
  when entering as late as T-5). With 41
  events and 72 configurations tried, that is a lead to follow, not a result to trade.
- Around those mega-IPOs the sector ETF drifts about −100 bps versus SPX from T-30 to T-1
  (roughly 2 standard errors) and recovers most of it by T+20
  (`results/event_path_etf_T-10_t0_close.png`). The pressure, if real, starts earlier than T-10
  and reverses after listing.
- Costs matter: round trip is about 16 bps (commissions/slippage plus borrow), which is the same
  size as the gross edge in the broad sample.
- Portfolio level, even the best variant earns ~0.1% a year on NAV at 10% per event, because
  it is invested only ~9% of days.

## Mega-IPOs only (≥ $2bn): variants

`python analyze_mega.py` compares 27 variants on the 41 deals of $2bn or more
(`results/mega_2bn_variants.csv`, chart `results/mega_2bn_pnl.png`). Net of costs, per trade:

| Variant | n | Mean | Median | t-stat | Hit rate | Worst |
|---|---|---|---|---|---|---|
| Pre-IPO: short sector ETF / long SPX, T-10 → IPO close | 41 | +0.2% | +0.5% | 0.7 | 59% | −6% |
| Buy at IPO price / short sector ETF, exit first-day close | 36 | +23.7% | +13.6% | 3.9 | 81% | −52% |
| Buy at IPO price / short sector ETF, exit T+5 | 36 | +27.7% | +24.5% | 4.8 | 86% | −34% |
| Buy at IPO price, no hedge, exit T+5 | 36 | +28.0% | +24.5% | 4.9 | 86% | −34% |
| Buy at first-day open / short sector ETF, exit T+5 | 40 | +3.1% | 0.0% | 1.2 | 50% | −31% |
| Buy at first-day close / short sector ETF, exit T+20 | 40 | +1.8% | −0.3% | 0.6 | 48% | −34% |
| Rebound: long sector ETF / short SPX, T+3 → T+20 | 41 | +0.5% | +0.7% | 1.1 | 66% | −5% |

- The whole edge is the **IPO discount**: it exists only if you receive shares at the offer
  price. Buying in the market on day one (open or close) has no significant edge.
- The **sector short adds nothing** to the IPO-price trade (hedged and unhedged are the same).
- Allocation is the catch. The average is driven by hot deals (Snowflake, Airbnb, DoorDash),
  where allocations are smallest; the backtest assumes an equal fill in every deal.
  2020-21 alone produced about two thirds of the cumulative P&L.
- Direct listings are excluded from the IPO-price variants (no offer price). The −52% worst case
  is Pershing Square Inc. (2026), whose offer terms were not verified and may be a data artefact.

## How to use this

1. **Do not trade the base case.** Shorting single-stock peers before every $1bn+ IPO lost money.
2. **If you pursue it, narrow it:** deals ≥ $2bn only, short the sector ETF (cheaper, no
   single-name squeeze risk), beta-hedged with SPY.
3. **Next tests worth running** before any capital: enter at roadshow launch rather than a fixed
   T-10; scale by deal size relative to sector market cap; test the post-IPO rebound (long
   peers from T+3 to T+20), which looks at least as strong as the pre-IPO leg.

## Known limits of this backtest

- **Survivorship.** Sectors and peers come from today's listed stocks. 46 of 158 large IPOs
  (delisted or acquired since) are dropped, and stock peers are today's survivors, which biases
  the stock-peer short downwards. The "vs. placebo" column (same basket at non-event dates)
  corrects for this; the ETF variant is not affected on the peer side.
- **Deals that never priced** (postponed or withdrawn) are not in the sample; live, you would
  already be short when a deal is pulled.
- **Direct listings** (Coinbase, Spotify, Roblox, Slack) are in the calendar although they raise
  no new money.
- Entry at a fixed T-10 assumes the listing date was known then. That is usually true for large
  deals but was not verified per event.
- Daily closes; no intraday execution model. Costs assumed: 5 bps per side on the short, 1 bp on
  SPY, 1% a year borrow.

## Run it

```bash
pip install -r requirements.txt
python run_backtest.py                              # base case
python run_backtest.py --mode etf --min-deal 2e9    # mega-IPO / sector ETF variant
python run_backtest.py --sweep                      # full robustness grid (~7 min)
python plot_results.py etf_T-10_t0_close            # charts
python analyze_mega.py                              # mega-IPO variants + P&L chart
pytest
```

First run downloads the IPO calendar (Nasdaq) and ~1,800 price histories (Yahoo) into
`data/cache/`, which takes about 20 minutes; later runs are offline.

| File | Contents |
|---|---|
| `ipo_reshuffle/backtest.py` | Peer selection, beta hedge, per-event P&L, placebo, portfolio |
| `ipo_reshuffle/data.py` | IPO calendar, sector classification, prices |
| `results/trades_*.csv` | Every trade: peers, beta, dates, gross/net return |
| `results/sweep.csv` | All 72 parameter combinations |
