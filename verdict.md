# $1,500 Robinhood cash-account options strategy: verdict

**Date:** 2026-09-23/24 · **Verdict: Tier B options champion ("IBS Dip Calls"), deployable only in stages; at $1,500 the
same signal through 3x ETF shares beats every options version at the same drawdown.**

Strategy book (Claude Doc): https://claude.ai/code/artifact/a15fe087-cb2c-401a-afa7-07b741b403ff
Pre-registration: `claude/robinhood-1500-options-preregistration.md` — original frozen 2026-09-24T02:29:54Z
(SHA-256 `74b897da...fbe`), Addendum A frozen 2026-09-24T03:06:46Z (`022133238e...186`) before any holdout data was loaded.
Code + research log (133 variants): `robinhood_1500_options_research.zip` delivered in chat.

## Question
Best options strategy a $1,500 Robinhood **cash** account (Level 2: long calls/puts, covered calls, cash-secured puts)
can run, maximizing log growth subject to P(equity <= $750 in 12 months) <= 5% and p95 max drawdown <= 35%.

## Data (new to this project)
Real EOD SPY option chains **2010-2025 from two vendors** (optionsdx 2010-2023; Alpha Vantage-format 2014-2025) via
anonymous public Kaggle mirrors — fixes the project's long-standing DoltHub limitation (3-4 expiries, 11-70 DTE, contracts
rarely re-observable). QQQ real chains 2021-03 to 2022-12. **No free IWM history found**; IWM (and QQQ before 2021-03)
priced by an SPY-calibrated proxy, then haircut 6/11/15 points per trade at 0.50/0.40/0.30 delta (worst bias measured on
real overlaps). Vendors agree: 2022-23 champion trades +7.9% vs +7.3% per trade, corr 0.994; AV 2025 vs DoltHub median
mid difference 0.6%.

## Champion rule
IBS = (last - low)/(high - low) at 3:45-3:55 pm ET on SPY, QQQ, IWM. If IBS < 0.10 and a slot is free (K = 1 below
$3,000), buy the first listed expiry >= 10 calendar days out, highest delta rung (0.50 -> 0.40 -> 0.30) whose single
contract costs <= 12% of equity (target 8%). Exit at 3:45 pm on the first day price > prior session's high, or on the
session before expiry. No stops, no rolls, settled cash only. At $1,500 this is mostly one IWM 0.30-delta call (~$136).

## Results (holdout 2022-2025, run once)
| # | Hypothesis | Holdout mean/trade | Holm (m=9) |
|---|---|---|---|
| H9 | Basket calls 10-DTE 0.50-delta (champion) | +21.8% (242, proxy haircut); SPY real +19.0% (80), CI +2.5% to +34.7% | Pass (p<0.001) |
| H1 | SPY 20-DTE 0.70-delta calls | +7.6% | Fail (p=0.041) |
| H2 | 3-down-days trigger | +7.6% | Pass, but ~$1,600/contract: unplaceable |
| H4 | Puts into VIX expiry | +23.8% | Pass, rejected: IS -5.6% (sign flip) |
| H6/H7 | IBS put credit / call debit spreads (lever B) | +4.9% / +15.5% | Fail (p=0.042 / 0.016) |
| H3/H5/H8 | weekly calls / cheap-vol straddles / weekly put spreads | +3.1% / -2.4% / +1.0% | Fail |

Robustness (SPY real): 2x costs +18.0%, 3x +17.0%, one-day-late +7.9% (CI crosses 0), DTE 8/12 and delta 0.40/0.60
all positive. Random timing: holdout 96.8th pct (same-exit variant 99.5th); in-sample 85.8th (92.5th). DSR 0.48 (N=112)
/ 0.70-0.75 (long-call family) -> **fails 0.95**; PSR 0.99. ~570 effectively independent trades. Edge grows with VIX
(SPY: +2% per trade at VIX<15, +27% at VIX>30); loses in grinding declines (Jan-Oct 2022 -12%/trade, Feb 2018 -31%).

## Sizing (MC, monthly block bootstrap, cash simulator, whole contracts at 2026-09-23 prices)
Recommended **8% of equity per trade, K=1** (largest size inside both limits): 12-month p10/p50/p90 $1,174 / $1,611 /
$3,128; p95 drawdown 35%; P(touch $750) ~0; P(2x in 12 mo) 13.7%; ~9 trades/yr. Era sensitivity: median $1,552 if the
year looks like 2010-2021, $2,125 if like 2022-2025. 10% sizing breaks the drawdown limit; 15% breaks ruin.
**3x ETF shares on the same signals** (UPRO/TQQQ/TNA, fractional): median $1,969-1,976 at p95 DD 37-38%, P(2x) 5-7%.
Options become competitive near $5,000-10,000 (f=4%: median +17% to +27%, DD 29-31%).
Lever B (margin, no borrowing): unsettled funds change nothing (identical MC); spreads failed Holm -> not recommended yet.

## Recommendation
Stage 0 paper with live Robinhood quotes (>= 4 weeks, 6 signals) -> Stage 1 one contract <= $180 until 20 live trades
-> Stage 2 scale by rule. Kill switch: equity < $1,050 or DD > 35%; 20-trade mean < -12%; 30-signal ETF move < +0.20%.
Highest-value next step: forward-log real IWM quotes for every signal (free) to replace the proxy — the only path to Tier A.
Current reading: last 30 signals (2026) average +0.41% ETF move per trade (IWM/QQQ +0.78%, SPY -0.33%) vs +0.69% 2010-2025.

## Lessons worth keeping
- Free dense SPY option history exists (Kaggle mirrors of optionsdx / Alpha Vantage); use it instead of DoltHub for
  anything needing daily re-marks or short DTE.
- At $1,500 the binding constraint is contract lumpiness, not the edge: most signals are unaffordable at risk-limited size.
- The IBS edge survives an option wrapper (unlike RSI timing on spreads, which was pure VRP), but the wrapper pays
  roughly half the per-trade Sharpe in theta/vega; 3x ETFs keep more of it when leverage is capped by a cash account.
