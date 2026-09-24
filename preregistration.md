# $1,500 Robinhood cash-account options strategy: pre-registration

**Written:** 2026-09-23 (evening, Mountain time), before any strategy P&L, trade return, or signal-conditional
option price was computed. Work done before freezing: tool/data audit, downloads, a vendor-vs-vendor quote
comparison (mid-price agreement only), coverage counts (expiries per day, spreads by moneyness/DTE), and a live
snapshot of current option prices. None of these condition on a strategy signal.

**Owner:** Sam. **Analyst:** Claude. **Book:** "Robinhood $1,500 Options Strategy Book" (Claude Doc).

**Known contamination, disclosed up front.** (1) This project already validated the IBS index-ETF rule on the
*underlying* (callout-bot-verdict, OOS 2019-2026, Sharpe 0.97-1.26). I know that result; I have NOT seen any
option-wrapped version of it (the indicator-suite's "2E-V" option wrapper was NOT TESTABLE on DoltHub data).
(2) Prior project results on 0DTE buying, event straddles, gamma levels, calendar effects and intraday
technicals are known and are used only as [PRIOR] reasons to discard candidates.

## 1. Account and objective (from the brief)

$1,500 Robinhood cash account, Level 2 (long calls/puts, covered calls, cash-secured puts), no deposits.
Objective: maximize expected log growth per calendar time subject to P(equity <= $750 within 12 months) <= 5%
and 95th-percentile max drawdown <= 35%. Level 3 structures are evaluated only under lever B (margin account,
no borrowing).

## 2. Data (fixed now)

| Block | Source | Role |
|---|---|---|
| SPY chains 2010-01 to 2023-12 | optionsdx EOD 4pm (Kaggle mirror) | primary vendor through 2023 |
| SPY chains 2014-01 to 2025-12 | Alpha Vantage-format EOD (Kaggle mirror) | primary vendor 2024-2025; independent re-run 2014-2023 |
| QQQ chains 2021-03 to 2022-12 | optionsdx EOD | cross-underlying check (2021 = IS, 2022 = holdout) |
| SPY chains 2025-06 to 2026-09 | DoltHub (11-70 DTE only) | descriptive post-holdout check only if the structure fits |
| Underlying OHLC | Yahoo daily, unadjusted OHLC for signals | signals |
| Vol indices | Cboe VIX/VIX9D/VIX3M/VXN/RVX | filters, proxy pricer |
| Rates | ^IRX | Kelly/cash (idle cash earns 0% in the base case) |

Contract selection filters: bid > 0, ask > bid, (ask - bid)/mid <= 0.30, |vendor delta - target| <= 0.10.

## 3. Sample splits (fixed now)

- **In-sample (selection):** 2010-01-01 to 2021-12-31.
- **Walk-forward inside IS:** expanding window, re-select each year: train 2010..Y-1, trade Y, for Y = 2014..2021.
- **Sealed holdout:** 2022-01-01 to 2025-12-31 (last 25% of the SPY record; contains the 2022 bear market,
  the Aug-2024 vol spike and the Apr-2025 tariff crash). Touched once, in Phase 5, for the pre-registered
  hypotheses only.
- **Post-holdout descriptive:** 2026-01 to 2026-09 (DoltHub) if the selected structure's DTE fits its slots.

## 4. Execution and cost model (fixed now)

- Signals use the official daily close/high/low. Entry and exit at the 4:00 pm ET EOD quote of the signal/exit day.
- Base fill: buy at mid + 25% of spread, sell at mid - 25% of spread. **2x-cost stress = natural fills**
  (buy at ask, sell at bid). 3x stress = natural fills plus 25% of spread extra each side.
- Robinhood fees per contract (live 2026-09-23): $0.04 per side (ORF/OCC), $0.00329 FINRA TAF on sells, $0.0003 CAT
  per side, SEC $20.60 per $1M of sell principal, plus $0.01 per order rounding. Index (XSP) variant adds $0.50/side.
- Exit quote missing: use the next trading day's quote (logged); if expired, intrinsic value less 1%.
- One position per underlying at a time; signals while in a position are ignored.
- Options are always exited no later than the close of the last trading day before expiration (no exercise,
  no assignment, no pin risk in a cash account).
- Returns are on premium paid (long options), on max loss (spreads), on collateral (cash-secured puts).

## 5. Candidate families and grids (confirmatory set)

"IBS" = (close - low)/(high - low). The IBS rule is the project's existing rule, not re-tuned.

| Fam | Candidate | Grid | Variants | Placeable |
|---|---|---|---|---|
| F1 | IBS < 0.10 at close -> buy SPY call | DTE target {2,5,10,20,35} (first expiry with calendar DTE >= target) x call delta {0.30,0.50,0.70} x exit {IBS10: first close > prior high else 10 td; IBS5: same, max 5 td; H1: next close; H3: 3rd close} | 60 | base |
| F2 | Other reversal triggers with F1's selected structure | trigger {RSI(2) < 10; 3 consecutive down closes; close = 5-day low close} | 3 | base |
| F3 | Equity-premium call carry (no timing skill) | buy F1-selected DTE, 0.50-delta call every Monday close, hold 5 td; {unconditional; only if SPY > SMA200} | 2 | base |
| F4 | VIX-expiration-day SPY puts | buy put at close of the trading day before VIX final settlement, sell at close of settlement day; DTE target {2,7} x put delta {0.50,0.30} | 4 | base |
| F5 | Cheap-vol long straddle (legged as 2 buys) | enter ATM straddle, DTE target 7, when VIX9D / RV10 < {0.9, 1.0}; hold {1, 3} td | 4 | base |
| F6 | IBS < 0.10 -> bull put credit spread | short put delta {0.30,0.50}, long put width {1%,2%} of spot below, DTE target {5,10}, exit IBS10 | 8 | lever B |
| F7 | IBS < 0.10 -> bull call debit spread | long call delta 0.50, short call +{1%,2%} of spot, DTE {5,10}, exit IBS10 | 4 | lever B |
| F8 | Weekly VRP put credit spread | Monday close, short 0.30-delta put, long put 2% lower, DTE target 7, hold to last day before expiry or 50% of credit; {ungated; VIX3M/VIX > 1 gate} | 2 | lever B |

Total confirmatory-grid variants: **87** (+1 pre-registered filter test below = **88**). Every run variant is
logged in `logs/research_log.csv`; the DSR uses N = 88 plus any logged extras.

**Pre-registered filter test (one, not a grid):** F1 selected variant with an SMA200 trend filter (enter only if
SPY close > 200-day SMA). Adopted only if it raises walk-forward growth AND holdout growth; otherwise dropped.

**Discarded before compute (no coherent counterparty or already settled):** event straddles (FOMC/CPI: we would
pay the event premium; project B01 lost 70/70), 0DTE unconditional buying (project F01), dealer-gamma/OPEX pin
timing and calendar effects (project indicator suite, callout sweep), intraday technicals (project family A),
option-implied cross-sectional signals and pre-earnings straddles (no free point-in-time multi-name chain
history reachable; logged as "not yet"), cash-secured puts / covered calls on <= $15 names (whole account in one
single-name position; evaluated with project wheel evidence and break-even math, not re-backtested),
dispersion/correlation (infeasible at size).

**Benchmarks (reported, not in the Holm family):** IBS on SPY shares (1x) and on UPRO (3x) with identical
entry/exit dates; random-timing call buying (G8).

**Proxy extension (descriptive, not in the Holm family):** QQQ and IWM with the F1-selected structure priced by
a Black-Scholes proxy whose IV = vol index (VXN / RVX) x term/skew ratios calibrated on SPY IS data only, with
a spread model calibrated on SPY IS data; validated on real QQQ 2021-2022 before use.

## 6. Selection rule (IS only)

For each variant: net per-trade returns r_i (base costs). Kelly fraction f* = argmax_f mean(log(1 + f r_i)),
f in [0, 0.99]. Score = trades/year x mean(log(1 + f_h r_i)) with f_h = min(0.5 f*, 0.5) (capped half-Kelly
log growth per year). Neighborhood-average the score over adjacent grid cells (F1: DTE +/-1 step and delta +/-1
step, same exit). Pick the max; ties -> fewer parameters/longer DTE. Minimum 30 IS trades. Each family
selects one variant -> one confirmatory hypothesis H1..H8.

**Champion rule:** among base-case families (F1-F5) whose selected variant has positive walk-forward
(2014-2021) score, the champion is the highest walk-forward scorer; if two or more base-case families qualify
and their same-day P&L correlation < 0.5, the equal-risk ensemble is also tested and wins only if its
walk-forward score is higher. The holdout is then evaluated once. Lever B: the champion is re-run with
unsettled funds allowed, and the best lever-B family is compared on identical gates.

## 7. Gates (reported pass/fail for the champion)

1. Net expectancy per trade > 0; 95% stationary-block-bootstrap CI lower bound > 0 (i.e. gross edge above the
   one-contract break-even), holdout.
2. Probabilistic/deflated Sharpe (per-trade Sharpe, N = 88 variants) >= 0.95; effective independent trades
   reported (target ~300 across IS walk-forward + holdout).
3. Holdout consistent with IS (same sign, holdout mean within the IS 90% bootstrap band or better); walk-forward
   positive in >= 60% of years and through a volatility shock (2015, 2018 or 2020).
4. Positive net expectancy at 2x costs (natural fills), with one-day-late entry and exit, and at +/-25% parameter
   perturbations (DTE and delta neighbours, max-hold 8 and 12 days).
5. Top trade < 15% of total net P&L; top 5 trades < 50%; worst day, worst week, CVaR95/99 at the recommended size
   (whole contracts) inside the 35% drawdown budget.
6. Named counterparty and persistence argument; any filter pre-registered and improving OOS.
7. Cash-cycle feasibility: the account simulation never spends unsettled cash and the assumed trade count is
   achievable with K tranches.
8. (Project standard) Beats >= 95% of 2,000 random-timing simulations with the same structure, trade count and
   holding-period distribution.

Holm-Bonferroni across H1..H8 (one-sided holdout test of mean net return > 0, block-bootstrap p-value), alpha 0.05.
Tiers: A = all gates on real data; B = limited or proxy data; C = design supported by [PRIOR] only.

## 8. Sizing and growth-math protocol (Phase 6)

Monthly-block bootstrap of the champion's trade list (IS walk-forward + holdout), 10,000 paths x 12 months,
starting equity $1,500, whole contracts at *today's* contract prices (premium as % of spot x current spot),
fraction-of-equity sizing f in {5%..60%}, K = 1..5 tranches, T+1 settlement on the NYSE calendar. Report
P(ruin <= $750), P(equity < $150), p95 max drawdown, p10/p50/p90 at 1/3/6/12 months, median/p90 time to $3,000,
P(2x) at 3/6/12 months, pre-tax and after-tax (12%, 24%; XSP 60/40 variant). Recommended f = the largest f that
satisfies both risk constraints after a 50% Kelly shrink, then rounded down to the nearest whole-contract rule.

## 9. Deviations log

Appended as encountered, before any affected result is reported.

---

## Addendum A — frozen before the holdout is opened (2026-09-24, same session)

**What had been seen when this was written (IS 2010-2021 and WF 2014-2021 only; no 2022+ data loaded):**
F1-F8 IS grids and walk-forward; F1 selected = d20/0.70-delta/IBS10 (IS mean +4.7%/trade, random-timing 84th
percentile); F3 (weekly unconditional calls) and F1 tie on WF growth (0.063 vs 0.061); F4/F5 negative; F6 (lever B)
strongest (WF mean +5.9% on max risk, NW t 3.6); IBS on UPRO shares beats IBS calls on growth (IS 0.23 vs 0.18/yr at
Kelly). Exploratory (logged, counted in N): IBS calls on QQQ/IWM/DIA priced by the SPY-calibrated proxy, 4 structures
each (IS means: QQQ +10% to +21%, IWM +7% to +17%, DIA +2% to +5% per trade).

**Proxy pricer status:** calibrated on SPY IS only; validated on real QQQ 2021 (16 trades: real +7.3% vs proxy +7.5%
per trade, corr 0.996) and SPY 2019-2021 (58 trades: real +14.4% vs proxy +16.2%, corr 0.99). Proxy results carry a
PROXY tag and a -2 percentage-point per-trade haircut when used for sizing.

**A1. New base-case family F9 — IBS basket calls.** Universe SPY, QQQ, IWM (DIA excluded by an objective liquidity rule
fixed now: median relative spread of 0.3-0.8-delta, 5-25-DTE calls must be < 4%; DIA = 5.6% vs 1.5-2.9% for the others,
Cboe snapshot 2026-09-22; DIA option volume is 0.6% of SPY's). Signal IBS < 0.10 at the close of that ETF; buy that
ETF's call; one position per ETF; exit IBS10. Structure selected from {d10/0.50, d10/0.70, d20/0.50, d20/0.70} by the
same capped-half-Kelly log-growth score on the pooled basket (IS), neighbourhood = adjacent DTE or delta. Pricing:
SPY real chains; QQQ and IWM proxy (QQQ real chains replace the proxy where they exist, 2021-03 onward). WF as before.
Portfolio growth is computed with the cash-account simulator (Section A4).

**A2. Champion rule (clarified, applies unchanged to F1-F5 and F9):** the champion is the base-case (Level 2) option
family with the highest WF growth from the cash-account simulator at the IS-estimated half-Kelly fraction. If the top
two are within 0.01/yr, both go to the holdout as co-candidates and the one with the higher holdout growth *that also
passes more gates* is named, with the other reported. The pre-registered F1+F3 ensemble test is run and reported.

**A3. Non-option implementations (reported with the same gates, not eligible as the options champion):** IBS on the same
ETFs executed with 3x ETF shares (UPRO, TQQQ, TNA), and a core-satellite hybrid (3x ETF core + calls satellite) with
weights fixed on IS data under the risk constraints. If a non-option implementation dominates the options champion on
growth under the same constraints, the Verdict says so plainly.

**A4. Cash-account simulator (now fixed):** daily loop on the NYSE session calendar; purchases only from settled cash;
sale proceeds settle next session; whole contracts at the contract cost implied by each trade's entry premium as a
percentage of spot times the *current* (2026-09-23) spot price; per-trade budget = f x equity (marked to market);
at most K concurrent positions (K = tranches); a trade whose single contract exceeds the budget is skipped unless
the "one-contract allowance" is on (then 1 contract if it costs <= 1.5 x budget and <= settled cash).

**Multiple-testing count update:** N = 88 (original) + 16 exploratory breadth cells + 4 F9 structures + 1 hybrid
weight search + 3 benchmark sets = **112** variants for the deflated Sharpe.

### Addendum A result log (written before the holdout is loaded)
- F9 IS structure selection: d10/0.50-delta (pooled capped half-Kelly scores d10_D50 0.94, d20_D50 0.86, d10_D70 0.66,
  d20_D70 0.48; neighbourhood-averaged winner d10_D50). WF 2014-2021 re-selects d10_D50 every year.
- Affordability at 2026-09-23 prices: F1's SPY 20-DTE 0.70-delta call (~$1,600/contract) and F3's SPY 20-DTE ATM call
  (~$880) cannot be bought at f <= 0.3 on $1,500 (0 and ~0 trades in the simulator). Their simulator WF growth is 0.
- **Champion (by rule A2): F9, IBS basket calls, 10-DTE ~0.50-delta, SPY/QQQ/IWM, IBS10 exit.** WF 2014-2021 (cash
  simulator, K=3, f=0.10): log growth 0.237/yr, max drawdown -36%; per-trade mean +19% (SPY real +13%).
- Holm family for the holdout: H1-H8 as pre-registered plus H9 = F9 champion (m = 9).
