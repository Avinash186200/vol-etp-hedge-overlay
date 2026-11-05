# Methodology and bias audit

## Problem

The base book is 80% long an S&P 500 ETF and 20% long a short-term VIX-futures ETP. The ETP is a crash hedge, but it loses value most days because of futures roll. The research question: can a daily model decide when to hold the book, hold less of it, or reverse it, and do better than holding it statically?

## Data contract

| Column | Meaning |
|---|---|
| `date` | Trading day |
| `spy` | Equity ETF close |
| `vixy` | VIX-futures ETP close |
| `vix` | VIX index close |

The research version also used intraday 3pm snapshots, cross-asset ETFs, sector dispersion and a set of macro features. I dropped those here: some are not public, and the core finding does not depend on them.

## Timing

- Features in row t use closes up to t-1. A test bumps every price from day t onward and checks that no feature at or before t changes.
- The position for day t is set before the close of t and earns close[t] -> close[t+1].
- The percentile used to map score to weight compares with strictly earlier scores only.

## Validation design

- Expanding walk-forward, retrained every 21 rows, scored row by row.
- **Purge**: for a label of horizon h, the last training row is `h + embargo` rows before the prediction row (embargo = 2). The original code used the embargo alone for every horizon.
- Evaluation starts after a 250-day percentile warm-up, on the same days for every book.
- Significance: block permutation of positions (21-day blocks, 200 draws) against the realised book returns.

## Bias found -> fix -> corrected result

**Found.** Horizon models (T+3, T+5, T+10, T+20) were trained with a 2-day embargo. With h = 20, a training row 3 days before the prediction row carries a label covering the next 20 days, which overlaps the predicted day by about 17 days. Those models then looked like the strongest ensemble members.

**Fix.** `purge_rows(h) = h + embargo`. Tested for h in {1, 5, 20}. The leaky gap stays available (`leaky=True`) so the comparison can be reproduced.

**Corrected result.** On real data, after the purge and the fixes below, the overlay showed no reliable next-day edge. Information coefficients were close to zero and balanced accuracy was close to 0.5. The best-looking variants were net-short the decaying ETP, which a constant short earns without a model, and they did not clear a significance test against that baseline. Earlier headline figures from the leaky pipeline are withdrawn and not published. On synthetic noise, the leaky T+20 model still ranks above the purged models; with a planted signal, the purged T+1 model recovers it.

## Other issues fixed

| Issue | Effect | Fix |
|---|---|---|
| Ensemble built by adding/swapping members on out-of-sample Sharpe | Best-of-N selection bias | No member search; one model per run |
| Position scale (3.5) chosen after results | Inflated returns | Fixed scale 1.0 |
| Target winsorised with full-sample std | Mild look-ahead | Removed |
| NaN fill with medians spanning evaluation years | Mild look-ahead | Native NaN handling in the model |
| Day-by-day score permutation | Weak null for autocorrelated signals | Block permutation |
| Sub-1 bp blended cost | Optimistic | 2 bp per unit turnover default |
| No constant-short baseline | Decay harvest mistaken for skill | Mean weight reported; constant short is the baseline to beat |

## What I dropped

- An 11-model ensemble across horizons, algorithms (gradient boosting, LightGBM, calibrated classifiers) and feature sets of 300 to 700 columns. It was reached by sequential selection on the evaluation window.
- Presentation decks, explainer workbooks, score histories and cached data. These were either proprietary or built on the biased results.
- Many exploratory studies: selective/abstaining rules, regime routing, encoders, threshold objectives. All were negative or did not clear significance and deflated-Sharpe gates.

## Known limitations

- The synthetic generator is a stylised world. It shows the method behaves correctly; it says nothing about real-market profitability.
- One retrain cadence and one model family are used for speed in CI.
- No final untouched hold-out exists for the real data, which is one more reason no real-data figures are published.
