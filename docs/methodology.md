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

## Later independent review

A later review covered the full research pipeline: the original models, a richer feature set and a futures implementation of the same book. I read it against this repo. What follows is what it confirmed, what it added, and what I did and did not port.

### Label overlap, confirmed by score-only tests (ported)

A feature-timestamp audit passes a label-overlap leak, because no feature looks ahead: the fitted model does. The review used two tests that need only the out-of-sample scores. I've implemented both in `volhedge/diagnostics.py`.

- **Overlap count.** On a refit at row i, with purge gap g, the last training row is `i - g - 1`. The return predicted on day `i + c` of the refit cycle is `close[i+c] -> close[i+c+1]`. The number of training labels of horizon h that contain it is `max(h - g - 1 - c, 0)`. With the original 2-day embargo and a 5-day refit, that is 2, 1, 0, 0, 0 for a T+5 model and 7 down to 3 for a T+10 model. With `g = h + embargo` it is 0 on every day. A test checks the formula against a brute-force count.
- **IC by lead.** Correlate the score at t with the return on day t+k, for k = 1, 2, .... In the review, the leaky members' IC fell linearly and reached zero exactly at the lead where no training label still contained that day's return. The honest members showed no such break.
- **Refit-cycle fingerprint.** Compare IC on cycle days with overlap against days without (Fisher z). In the review, the T+5 member forecast well on the two days after each refit and not at all on the other three, a difference far beyond chance. No economic mechanism follows a refit calendar.
- **Causal replication.** The same recipe with only the embargo changed (2 days vs 10 days on a T+10 target) lost most of its apparent skill.

On this repo's synthetic data (seed 7, T+20, refit every 21 days), the leaky model has overlap on 17 of 21 cycle days. Its fingerprint points the right way, IC +0.04 on overlap days against -0.06 on clean days, but z = 1.3 and p = 0.18. I also tried a deeper, research-sized model, with the same result. The synthetic world has little persistent structure for a model to memorise, so the leak transfers weakly. The tests therefore check detection on planted cases (a leaky score is found with p < 1e-6, an honest one is not flagged), and the script runs leaky vs purged so the overlap count is visible. The review also noted that a live system cannot benefit from this leak, because a live label cannot contain a return that has not happened yet. Live results should be expected to track the non-leaking members.

### Findings (documented, no code change)

- **Residual signal.** The non-leaking members kept a small positive IC on the book, measured on a richer feature set than I could rebuild. It is preliminary, declined over the years, and was not validated on a final untouched hold-out. This refines my earlier "close to zero" description. It does not change the verdict.
- **Where the honest improvements came from.** Score smoothing (lower turnover), a cheaper instrument (ETFs instead of futures; gross results were similar across instruments, while net results tracked cost), and reducing a position as its sign ages beyond the signal's short information horizon. None of these is new predictive information. The P&L was concentrated in a small share of post-shock days. The programme evaluated hundreds of configurations and reused its hold-out year for promotions, so its performance figures are not published here.
- **What the book is.** The futures leg's beta to equities is strongly negative, so 80/20 is close to a beta-neutral position in the futures leg. The signals predicted that residual, not either leg.
- **Roll accounting (futures version only).** Futures returns were computed across a contract switch on roll days, because the code used the wrong expiry rule. That booked the calendar spread as P&L, and it also made long-only rules look better than long/short. The ETF data here is a continuous product, so this repo is not affected. The lesson is general: compute returns on the same contract.

### Tested in the review, not ported

- **Position-age decay** (exposure falls as the ensemble's sign persists) and a **longer smoothing span**. Both passed plateau, CPCV and deflated-Sharpe checks in the review, but their parameters were chosen on the real data, and I have no fresh hold-out to confirm them. Adding them here would add parameters I cannot validate on synthetic data, which goes against the fixed-in-advance rule above.
- Many other rules were rejected in the review (regime gates, sizing grids, member weighting, curve features on top of the signal, holding longer). They are consistent with the "not significant" exploratory work listed below.

## What I dropped

- An 11-model ensemble across horizons, algorithms (gradient boosting, LightGBM, calibrated classifiers) and feature sets of 300 to 700 columns. It was reached by sequential selection on the evaluation window.
- Presentation decks, explainer workbooks, score histories and cached data. These were either proprietary or built on the biased results.
- Many exploratory studies: selective/abstaining rules, regime routing, encoders, threshold objectives. All were negative or did not clear significance and deflated-Sharpe gates.

## Known limitations

- The synthetic generator is a stylised world. It shows the method behaves correctly; it says nothing about real-market profitability.
- One retrain cadence and one model family are used for speed in CI.
- No final untouched hold-out exists for the real data, which is one more reason no real-data figures are published.
