# Walkthrough

This follows one run of `scripts/run_backtest.py` on the default synthetic prices (seed 7, 2,000 days, no planted
signal), function by function. Every number here comes from running the code in this repo. To follow along in a
Python shell started at the repo root, first run `import sys; sys.path.insert(0, "src")`.

## 1. Prices

```python
from volhedge.data import make_synthetic_prices
prices = make_synthetic_prices(2000, seed=7)
```

2,000 business days from January 2015, with columns `date, spy, vixy, vix`. The equity has slow-moving
volatility, the VIX follows that volatility, and the ETP moves about -2.5x the equity and loses a little every day.
With `signal_strength=0` tomorrow's return cannot be predicted from the past. `scripts/make_synthetic_data.py`
writes the same data to `data/prices.csv`.

## 2. Features and target

```python
from volhedge.features import build_features, book_return
table, feats = build_features(prices)
```

21 features, all shifted so row t only uses closes up to t-1: lagged returns of both legs (1, 2, 5 days),
realised volatility and momentum over 5, 21 and 63 days, the VIX level, its 5-day change, a 21-day z-score,
implied vs realised volatility, and an expanding z-score of the VIX. `table["ret_t1"]` is the next-day return of
the 80/20 book. That is always what the overlay earns. The model's target is `book_return(prices, h)`, the
h-day forward return.

## 3. Purged walk-forward

```python
from volhedge.walkforward import walk_forward, purge_rows
oos = walk_forward(prices, table, feats, horizon=20)
```

Scoring starts at row 750. The model is refitted every 21 rows on all earlier rows, except the last
`purge_rows(h)` rows. For h = 20 that gap is 22 rows, so the newest training label ends before the day being
predicted. With `leaky=True` the gap is the old 2 rows. Then a T+20 model has training labels that contain the
predicted return on 17 of every 21 days in the refit cycle (`diagnostics.label_overlap`).

## 4. Score to weight

```python
from volhedge.strategy import positions, pnl, stats
w = positions(oos["raw"].values)
```

The score is smoothed with an EMA (span 7). Each day's smoothed score is ranked against all earlier ones, never
later ones, and mapped to `w = 2 * pct - 1`, between -1 and +1. The first 250 days are a warm-up with `w = 0`, so
the evaluation covers the last 999 days. `pnl` charges 2 bp per unit of turnover.

## 5. Compare and test

`run_backtest.py` prints one line per book, all on the same 999 days:

| Book | Sharpe | Perm. p | Mean weight |
|---|---|---|---|
| base (static 80/20) | -0.94 | - | +1 |
| const_short | 0.94 | - | -1 |
| h1 | -0.34 | 0.59 | 0.02 |
| h20 | -0.31 | 0.48 | 0.13 |
| h20_leaky | 0.07 | 0.23 | 0.09 |

`block_permutation_p` shuffles the weights in 21-day blocks 200 times and counts how often the shuffled Sharpe is
at least as good. None of the models is significant, which is right for data with no signal. The constant short
beats them all, because the ETP decays. With `--signal 0.3` the purged T+1 model finds the planted effect.

## 6. Leak diagnostics

`scripts/leakage_diagnostics.py` runs the T+20 model leaky and purged and prints, for each, the IC by lead and the
refit-cycle fingerprint. The fingerprint compares IC on cycle days that have label overlap against days that do
not. The purged model has no overlap days, so there is nothing to compare.

## Things to try

- Run with `--signal 0.3` and compare h1 against h20. Why does the T+20 model miss a one-day effect?
- Change `--cost-bp` to 0 and to 10, and watch which books change most.
- Pass a leaky score file to `leakage_diagnostics.py --scores results/oos_h20_leaky.csv --horizon 20 --gap 2`.
