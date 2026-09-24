"""
Cash-account simulator (Addendum A4).

* Daily loop over NYSE sessions (the SPY trading-day index).
* Purchases only from SETTLED cash. Sale proceeds settle on the next session (T+1). settle_lag=0 models lever B /
  the Agentic limited-margin account (unsettled funds usable immediately).
* Position size: budget = f x equity (marked to market at the prior close); whole contracts at today's contract cost
  (entry premium as % of spot at entry x current spot x 100); fractional units allowed for ETF shares.
* At most K concurrent positions (tranches).
* Optional one-contract allowance: if 0 contracts fit the budget, buy 1 if it costs <= allow_mult x budget.
* Returns the daily equity curve, the executed trade log and skip counts.
"""
import json
import numpy as np, pandas as pd

CURRENT_SPOT = {"SPY": 767.81, "QQQ": 741.21, "IWM": 281.92, "DIA": 514.30, "UPRO": 150.30, "TQQQ": 78.66, "TNA": 60.43,
                "SPXL": 287.55, "UDOW": 65.27}

def unit_cost_today(row, scale_to_today=True):
    """Capital at risk for one unit (contract x100, or one share) at today's price level."""
    if row.get("kind", "option") == "share":
        return 1.0                                        # fractional shares: $1 of exposure per unit
    frac = row["risk"] / 100.0 / row["und_entry"]         # risk per share as % of spot
    if not scale_to_today:
        return row["risk"]
    return frac * CURRENT_SPOT.get(row["sym"], row["und_entry"]) * 100.0

def simulate_account(trades, sessions, f=0.2, K=3, start_equity=1500.0, settle_lag=1, allow_one=True, allow_mult=1.5,
                     max_total_risk=1.0, scale_to_today=True, ret_col="ret", fractional_shares=True, start=None, end=None):
    tr = trades.copy()
    tr = tr.sort_values(["entry", "sym"]).reset_index(drop=True)
    if start is not None:
        tr = tr[tr.entry >= pd.Timestamp(start)]
    if end is not None:
        tr = tr[tr.entry <= pd.Timestamp(end)]
    sessions = pd.DatetimeIndex([d for d in sessions if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end) + pd.Timedelta(days=30))])
    spos = {d: i for i, d in enumerate(sessions)}
    by_entry = {}
    for i, r in tr.iterrows():
        by_entry.setdefault(r.entry, []).append(r)
    settled = start_equity
    pending = []                  # (settle_session_index, amount)
    open_pos = []                 # dicts
    eq = []; log = []; skips = {"no_budget": 0, "no_cash": 0, "max_K": 0, "risk_cap": 0}
    prev_equity = start_equity
    for i, d in enumerate(sessions):
        # 1. settle pending cash
        still = []
        for (si, amt) in pending:
            if si <= i:
                settled += amt
            else:
                still.append((si, amt))
        pending = still
        # 2. exits at today's close
        keep = []
        for p in open_pos:
            if p["exit"] == d:
                proceeds = p["capital"] * (1.0 + p["ret"])
                proceeds = max(proceeds, 0.0) if p["kind"] != "credit" else proceeds
                if settle_lag == 0:
                    settled += proceeds
                else:
                    pending.append((i + settle_lag, proceeds))
                log.append(dict(sym=p["sym"], entry=p["entry"], exit=d, units=p["units"], capital=p["capital"],
                                pnl=p["capital"] * p["ret"], ret=p["ret"]))
            else:
                keep.append(p)
        open_pos = keep
        # 3. mark to market (for sizing and drawdown)
        mtm = 0.0
        for p in open_pos:
            m = p["marks"].get(d)
            if m is not None:
                p["last_mark"] = m
            mtm += p["capital"] * p["last_mark"]
        equity = settled + sum(a for _, a in pending) + mtm
        # 4. entries at today's close (sized on today's equity)
        for r in by_entry.get(d, []):
            if len(open_pos) >= K:
                skips["max_K"] += 1; continue
            risk_open = sum(p["capital"] for p in open_pos)
            if risk_open >= max_total_risk * equity:
                skips["risk_cap"] += 1; continue
            budget = min(f * equity, max_total_risk * equity - risk_open)
            kind = r.get("kind", "option")
            uc = unit_cost_today(r, scale_to_today)
            if kind == "share" and fractional_shares:
                units = min(budget, settled) / uc
            else:
                units = np.floor(budget / uc)
                if units < 1 and allow_one and uc <= allow_mult * budget:
                    units = 1
            cap = units * uc
            if units <= 0:
                skips["no_budget"] += 1; continue
            if cap > settled + 1e-9:
                units2 = np.floor(settled / uc) if not (kind == "share" and fractional_shares) else settled / uc
                if units2 <= 0:
                    skips["no_cash"] += 1; continue
                units = units2; cap = units * uc
            settled -= cap
            # marks: value ratio relative to entry cost (debit structures): mid_t / entry_px
            marks = {}
            if isinstance(r.get("marks"), str) and r.get("entry_px", 0) and r["entry_px"] > 0:
                for ds, mv in json.loads(r["marks"]):
                    marks[pd.Timestamp(ds)] = mv / r["entry_px"]
            open_pos.append(dict(sym=r["sym"], entry=d, exit=r["exit"], units=units, capital=cap, ret=r[ret_col],
                                 kind=kind, marks=marks, last_mark=1.0))
            equity = settled + sum(a for _, a in pending) + sum(p["capital"] * p["last_mark"] for p in open_pos)
        eq.append((d, equity))
    curve = pd.Series(dict(eq))
    return curve, pd.DataFrame(log), skips

def curve_stats(curve):
    if len(curve) < 2:
        return {}
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    lg = np.log(curve.iloc[-1] / curve.iloc[0]) / yrs
    dd = (curve / curve.cummax() - 1).min()
    daily = curve.pct_change().dropna()
    return dict(log_growth=lg, cagr=np.exp(lg) - 1, max_dd=dd, final=curve.iloc[-1], vol=daily.std() * np.sqrt(252),
                worst_day=daily.min(), worst_week=(curve.resample("W").last().pct_change().min()))
