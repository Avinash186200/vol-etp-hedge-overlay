# Volatility-ETP Hedge Overlay

An ML timing overlay on a simple hedged book: 80% long an S&P 500 ETF (SPY) and 20% long a VIX-futures ETP (VIXY). Each day a model scores the book's next-day return before the close, and the score is turned into a position weight between -1 and +1.

I built this as a research project. After I found and removed a look-ahead bias in the original walk-forward, **the overlay showed no reliable edge on real data**. This repo is the corrected method, run on synthetic prices so anyone can reproduce it, plus an honest account of what went wrong.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

## At a glance

| | |
|---|---|
| **Input** | Daily closes: `date, spy, vixy, vix` (public market data; a synthetic generator with the same columns is included) |
| **Method** | Backward-looking volatility, momentum and implied-vs-realised features; gradient-boosted regression on forward book returns; expanding walk-forward; past-only percentile mapping to a weight |
| **Output** | Out-of-sample score and weight per day, overlay PnL after costs, Sharpe / drawdown / block-permutation p-value against the static 80/20 book |
| **Leakage controls** | Features use closes up to t-1 only; purge gap = label horizon + embargo; past-only percentile; fixed scale; noise and planted-signal tests in CI |
| **Status** | Research complete. No reliable edge after the bias fix. Results below are synthetic |

## How it works

```
closes ──► features (<= t-1) ──► purged walk-forward ──► score ──► EMA + past-only pct ──► weight w ──► w x book return - costs
 data.py    features.py          walkforward.py                    strategy.py
```

1. **Features** in row t use closes up to t-1: lagged returns, realised vol (5/21/63d), momentum vs moving averages, VIX level / change / z-score, VIX minus realised vol, and an expanding (never full-sample) VIX z-score.
2. **Target**: forward h-day return of the 80/20 book from close[t]. PnL always uses the T+1 return.
3. **Walk-forward**: retrain on an expanding window. The last training row is at least `horizon + embargo` rows before the prediction row, so no training label overlaps the return being predicted.
4. **Position**: EMA(7) of the score, ranked against strictly earlier smoothed scores, then `w = clip(2*pct - 1, -1, 1)`. Costs are 2 bp per unit of turnover.

## Bias found -> fix -> corrected result

This is the most important part of the project.

**Bias found.** The original pipeline ensembled models trained on several horizons, from T+1 up to T+20. Every model used the same 2-day embargo between the last training row and the prediction row. For a T+1 label that is fine. For a T+20 label it is not: the label of a row 3 days back is the return over the next 20 days, so it contains the next ~17 days of returns, including the one being predicted. The long-horizon models were partly trained on the answer. They were also the members that "added the most" to the ensemble, which should have been a warning sign.

**Fix.** The purge gap is now `horizon + embargo` rows (`walkforward.purge_rows`). A unit test checks that the last training label ends before the prediction row for every horizon, and another keeps the old leaky gap so the overlap stays visible.

**Corrected result.** With the purge in place and the other issues below fixed, the real-data overlay did not beat simple baselines in a way that survived significance testing. Most of what looked like skill was the overlay being net-short a product that decays every day, which a constant short position earns without any model. The earlier headline Sharpe figures came from the biased pipeline and I have withdrawn them. I don't publish any real-data performance numbers here.

The same effect shows on synthetic data (`python scripts/run_backtest.py`, seed 7, 2,000 days, 999 evaluated days, 2 bp costs):

| Book | Sharpe | Total | Max DD | Perm. p | Mean weight |
|---|---|---|---|---|---|
| Static 80/20 | -0.94 | -24.9% | -30.9% | - | +1 |
| Constant short book (no model) | 0.94 | 30.3% | -9.2% | - | -1 |
| T+1 model, purged | -0.34 | -5.7% | -11.6% | 0.59 | 0.02 |
| T+20 model, purged | -0.31 | -5.2% | -12.6% | 0.48 | 0.13 |
| T+20 model, **leaky** (old 2-day embargo) | 0.07 | 0.8% | -9.6% | 0.23 | 0.09 |

There is no signal in this data, so every model should be at noise level. Among the models, the leaky T+20 one ranks first; with no signal in the data, the only thing separating it from the purged T+20 model is the label overlap. This is one seed, and none of the p-values is significant, so I read it as the direction of the bias, not its size. None of them beats the constant short, which earns the ETP decay with no model at all. That is the same baseline the real-data overlay failed to beat. With a planted signal (`--signal 0.3`), the purged T+1 model recovers it (Sharpe 2.17, p = 0.005), while the purged T+20 model, which targets the wrong horizon, does not (Sharpe -0.28, p = 0.55). So the pipeline finds real structure when it is there and stays at noise level when it isn't.

## Leakage and bias audit

All of these were in the original research code. The details are in [docs/methodology.md](docs/methodology.md).

| Issue | Effect | Fix |
|---|---|---|
| **2-day embargo on T+3..T+20 labels** (look-ahead) | Training labels overlapped the predicted return; long-horizon members looked strong | Purge = horizon + embargo; tested in CI |
| Ensemble members added or swapped one at a time by out-of-sample Sharpe | Selection bias: the reported figure was best-of-many on the evaluation window | One pre-declared model per horizon; no member search |
| Position scale chosen after looking at results (3.5 in the ensemble, 1.0 elsewhere) | Inflated returns and drawdown control | Scale fixed at 1.0 in advance |
| Target winsorised with a full-sample standard deviation | Mild look-ahead in the label | No winsorising; a robust loss can be added per fold if needed |
| Missing values filled with medians that included out-of-sample years | Mild look-ahead | Model handles NaN natively; no global fill |
| Permutation test shuffled scores day by day | Null too easy to beat, since smoothed scores are autocorrelated | Block permutation (21-day blocks) of positions |
| Blended cost under 1 bp | Optimistic for a vol ETP | 2 bp per unit turnover by default (`--cost-bp`) |
| Comparison only against the static book | Net-short decay harvest looked like skill | Constant-short book reported alongside every run, plus mean weight |

## Repo layout

```
src/volhedge/   data.py · features.py · walkforward.py · strategy.py
scripts/        run_backtest.py
tests/          feature timing, purge, leaky-gap overlap, past-only positions, noise, planted signal
docs/           methodology.md
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/run_backtest.py                     # synthetic, no signal
python scripts/run_backtest.py --signal 0.3        # synthetic, planted signal
python scripts/run_backtest.py --prices my.csv     # your own date,spy,vixy,vix closes
```

## License

MIT
