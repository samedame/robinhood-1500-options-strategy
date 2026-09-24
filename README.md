# $1,500 Robinhood options strategy research package

Companion code for the Claude Doc "Robinhood $1,500 Options Strategy Book" (2026-09-23). Everything here is
research code plus one live helper script. Nothing in it places orders.

## Result in one paragraph

Champion: **IBS Dip Calls**. When SPY, QQQ or IWM trades in the bottom 10% of its day's range at 3:45-3:55 pm ET,
buy one call at least 10 days out, taking the highest delta (0.50 / 0.40 / 0.30) whose single contract costs
<= 12% of equity (target 8%); sell on the first 3:45 pm reading above the prior day's high or on the session before
expiry. Holdout 2022-2025 on real SPY chains: +19% per trade (80 trades). At $1,500 the simulated 12-month median is
$1,611 (p10 $1,174, p90 $3,128) with a 35% p95 drawdown. Evidence tier B: the deflated Sharpe gate fails and the
affordable IWM leg is proxy-priced. The same signal through 3x ETF shares beats the options version at this size.

## Files

| Path | What it does |
|---|---|
| `deliver/ibs_dip_calls_signal.py` | Live helper: run at 3:45 pm ET; prints exit/entry tickets per the rulebook |
| `deliver/journal_schema.sql` | Supabase/Postgres trade journal, kill switch, proposals, `live_monitor` view |
| `deliver/agentic/` | Next.js/Vercel sketch: cron -> proposals -> human approval -> agent places via Robinhood MCP; hard limits in `lib/limits.ts` |
| `preregistration.md` | Frozen pre-registration + Addendum A (SHA-256 recorded in the book) |
| `src/build_chains.py` | Converts the raw vendor files into a compact chain store (`data/chains/*.parquet`) |
| `src/fetch_dolthub.py` | Pulls DoltHub SPY chains one day at a time (narrow queries only) |
| `src/core.py` | Data access, signals (IBS, RSI2, streaks, VIX expiries), option selection, trade engine with Robinhood fees |
| `src/stats.py` | Kelly growth score, stationary block bootstrap, Newey-West t, PSR/DSR |
| `src/run_is.py` | In-sample grids for families F1-F8 (loads only 2010-2021 chains) |
| `src/selection.py`, `src/walkforward.py` | Neighbourhood-averaged selection and the 2014-2021 walk-forward |
| `src/random_timing.py`, `src/g8_champion.py`, `src/g8b_same_exit.py` | Random-timing gate (and the same-exit variant) |
| `src/proxy.py`, `src/calib_proxy_bias.py`, `src/validate_proxy.py`, `src/proxy_bias_*.py` | Proxy pricer for QQQ/IWM/DIA, its calibration and its bias checks |
| `src/run_breadth_is.py`, `src/champion_is.py`, `src/wf_f9.py` | Basket (F9) in-sample, affordability and walk-forward |
| `src/run_holdout.py`, `src/run_holdout_d30.py` | The one-time holdout run for H1-H9, stress variants and vendor checks |
| `src/champion_data.py` | Assembles the champion's trade history with proxy haircuts |
| `src/cash_engine.py` | Cash-account simulator (settled/unsettled ledgers, T+1, whole contracts, K slots) |
| `src/mc.py`, `src/mc_alt.py`, `src/mc_shares.py`, `src/mc_hybrid.py`, `src/mc_thresholds.py`, `src/mc_checks.py` | Monte Carlo frontiers, 3x-ETF benchmark, hybrid, capital thresholds, robustness |
| `results/research_log_full.csv` | Every variant run (133 rows) with n, mean, CI, t, Kelly |
| `results/mc_*.csv` | Frontier tables used in section 6 of the book |

## Reproduce

```bash
pip install pandas numpy scipy pyarrow yfinance requests statsmodels
# 1. data (anonymous public downloads)
mkdir -p ~/data/kaggle && cd ~/data/kaggle
curl -L -o spy_optionsdx_2010_2023.zip https://www.kaggle.com/api/v1/datasets/download/dudesurfin/spy-options-eod-volatility-surface-2010-2023
curl -L -o qqq_optionsdx_2020_2022.zip https://www.kaggle.com/api/v1/datasets/download/kylegraupe/qqq-daily-option-chains-q1-2020-to-q4-2022
curl -L -o spy_av_2014_2025.zip       https://www.kaggle.com/api/v1/datasets/download/shankerabhigyan/s-and-p500-options-spy-implied-volatility-2019-24
unzip -q spy_optionsdx_2010_2023.zip -d spy_odx
# Cboe vol indices: https://cdn.cboe.com/api/global/us_indices/daily_prices/<VIX|VIX9D|VIX3M|VXN|RVX>_History.csv
# Yahoo daily OHLC for SPY QQQ IWM DIA UPRO TQQQ TNA ^IRX via yfinance (see the book's audit section)
# 2. build the chain store, then run the phases in order
python src/build_chains.py spy_odx && python src/build_chains.py qqq_odx && python src/build_chains.py spy_av
python src/run_is.py F1,F4,F5,F6,F7,F8 && python src/run_is.py F2,F3
python src/walkforward.py && python src/run_breadth_is.py && python src/run_breadth_is.py extra
python src/run_holdout.py && python src/run_holdout_d30.py && python src/champion_data.py
python src/mc_alt.py basket 4000 && python src/mc_shares.py && python src/mc_thresholds.py
```

Paths in the scripts assume `/home/claude/research` and `/home/claude/data`; edit `ROOT`/`MKT` in `src/core.py`.

## Caveats

* Dataset licences on the Kaggle mirrors are listed as unknown; use for personal research.
* QQQ before 2021-03 and all IWM/DIA option prices are proxy (Black-Scholes on Cboe vol indices with SPY-calibrated
  smile, spread and level corrections), haircut 6/11/15 points per trade at 0.50/0.40/0.30 delta.
* Not financial or tax advice.
