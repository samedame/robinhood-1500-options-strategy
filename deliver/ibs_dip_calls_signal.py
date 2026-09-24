#!/usr/bin/env python3
"""
IBS Dip Calls -- daily signal helper for a $1,500 Robinhood cash account (Level 2).

Run it at 3:45 pm US/Eastern (1:45 pm Mountain) on trading days. It prints what the rulebook says to do today:
  * EXIT tickets for open positions (bounce exit or time exit), then
  * ENTRY tickets for new IBS < 0.10 signals on SPY / QQQ / IWM, choosing the highest affordable delta rung.

It never places orders. Confirm every price in the Robinhood app (Cboe delayed quotes lag ~15 minutes; Yahoo
intraday bars can lag). All thresholds below are the frozen rulebook values from the strategy book, section 4.

Files it uses (created next to the script if missing):
  positions.csv   open positions you entered: etf,expiry,strike,contracts,entry_date,entry_price
  signals_log.csv every run's readings (append-only), useful for the paper stage and decay monitoring

Usage:
  python ibs_dip_calls_signal.py --equity 1500 --settled 1500
  python ibs_dip_calls_signal.py --equity 1500 --settled 1500 --discord $DISCORD_WEBHOOK_URL
Requires: pip install yfinance pandas numpy requests
"""
import argparse, csv, datetime as dt, json, math, os, sys
import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance")

# ------------------------------- frozen rulebook parameters --------------------------------------------------
ETFS = ["SPY", "QQQ", "IWM"]            # DIA excluded (spreads ~5.6%, volume 0.6% of SPY)
IBS_ENTRY = 0.10                        # signal if IBS < 0.10
DTE_MIN = 10                            # first listed expiry >= 10 calendar days away
RUNGS = [0.50, 0.40, 0.30]              # delta ladder, highest affordable first
DELTA_TOL = 0.06                        # accept |delta - rung| <= 0.06
TARGET_F = 0.08                         # target premium at risk per trade (fraction of equity)
ALLOW_MULT = 1.5                        # one contract allowed if it costs <= 1.5 x target (i.e. <= 12% of equity)
MAX_REL_SPREAD = 0.04                   # (ask - bid) / mid <= 4%
MIN_OI = 500
EVIDENCE_ORDER = {"SPY": 0, "QQQ": 1, "IWM": 2}
HERE = os.path.dirname(os.path.abspath(__file__))
POS_FILE = os.path.join(HERE, "positions.csv")
LOG_FILE = os.path.join(HERE, "signals_log.csv")

# NYSE full-day holidays 2026-2027 (for the time exit); verify annually at nyse.com
HOLIDAYS = {"2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
            "2026-09-07", "2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
            "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}

def k_slots(equity):
    return 1 if equity < 3000 else (2 if equity < 5000 else 3)

def is_session(d):
    return d.weekday() < 5 and d.isoformat() not in HOLIDAYS

def last_session_before(expiry):
    d = expiry - dt.timedelta(days=1)
    while not is_session(d):
        d -= dt.timedelta(days=1)
    return d

def ny_now():
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return dt.datetime.utcnow() - dt.timedelta(hours=4)

# ------------------------------- market data ------------------------------------------------------------------
def intraday_state(sym):
    """Today's regular-session high, low, last; prior session's high (Yahoo)."""
    m = yf.download(sym, period="1d", interval="1m", progress=False, auto_adjust=False, prepost=False)
    d = yf.download(sym, period="10d", interval="1d", progress=False, auto_adjust=False)
    for x in (m, d):
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
    if m.empty or len(d) < 2:
        return None
    hi, lo, last = float(m["High"].max()), float(m["Low"].min()), float(m["Close"].iloc[-1])
    today = m.index[-1].date()
    prior = d[d.index.date < today]
    prior_high = float(prior["High"].iloc[-1])
    ibs = (last - lo) / (hi - lo) if hi > lo else 0.5
    return dict(sym=sym, last=last, high=hi, low=lo, ibs=ibs, prior_high=prior_high, bar_time=str(m.index[-1]))

def cboe_chain(sym):
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
    j = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).json()
    o = pd.DataFrame(j["data"]["options"])
    p = o.option.str.extract(r"^(?P<root>[A-Z]+)(?P<exp>\d{6})(?P<cp>[CP])(?P<k>\d{8})$")
    o = pd.concat([o, p], axis=1)
    o["strike"] = o.k.astype(float) / 1000
    o["expiry"] = pd.to_datetime(o.exp, format="%y%m%d").dt.date
    return o[o.cp == "C"].copy(), (j.get("timestamp"), float(j["data"].get("current_price") or np.nan))

def pick_contract(sym, equity, settled, today, live_last=None):
    calls, (stamp, chain_spot) = cboe_chain(sym)
    stale = live_last is not None and np.isfinite(chain_spot) and abs(chain_spot / live_last - 1) > 0.005
    exps = sorted(e for e in calls.expiry.unique() if (e - today).days >= DTE_MIN and last_session_before(e) > today)
    if not exps:
        return None, "no expiry >= 10 days"
    e = exps[0]
    g = calls[(calls.expiry == e) & (calls.bid > 0) & (calls.ask > calls.bid)].copy()
    g["mid"] = (g.bid + g.ask) / 2
    g["rel_spread"] = (g.ask - g.bid) / g.mid
    budget = TARGET_F * equity
    for rung in RUNGS:
        c = g.iloc[(g.delta - rung).abs().argsort()[:1]]
        if c.empty:
            continue
        c = c.iloc[0]
        if abs(c.delta - rung) > DELTA_TOL or c.rel_spread > MAX_REL_SPREAD or c.open_interest < MIN_OI:
            continue
        cost = float(c.ask) * 100
        n = math.floor(budget / cost)
        if n < 1 and cost <= ALLOW_MULT * budget:
            n = 1
        if n >= 1 and n * cost <= settled:
            return dict(sym=sym, expiry=str(e), strike=float(c.strike), rung=rung, delta=round(float(c.delta), 3),
                        bid=float(c.bid), ask=float(c.ask), mid=round(float(c.mid), 3),
                        max_limit=round(float(c.mid + 0.25 * (c.ask - c.bid)), 2), contracts=int(n),
                        cost_at_ask=round(n * cost, 2), pct_equity=round(n * cost / equity * 100, 1),
                        quote_time=stamp, stale=stale, chain_spot=chain_spot), None
    return None, "no rung affordable (<= 12% of equity) with spread <= 4% and OI >= 500"

# ------------------------------- positions & logging ----------------------------------------------------------
def read_positions():
    if not os.path.exists(POS_FILE):
        with open(POS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["etf", "expiry", "strike", "contracts", "entry_date", "entry_price"])
        return []
    return list(csv.DictReader(open(POS_FILE)))

def log_row(row):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, required=True, help="total account value now")
    ap.add_argument("--settled", type=float, required=True, help="settled cash available to buy")
    ap.add_argument("--discord", default=os.environ.get("DISCORD_WEBHOOK_URL", ""))
    a = ap.parse_args()
    now = ny_now(); today = now.date()
    lines = [f"IBS Dip Calls | {now:%Y-%m-%d %H:%M} ET | equity ${a.equity:,.0f} | settled ${a.settled:,.0f}"]
    if not (dt.time(15, 40) <= now.time() <= dt.time(15, 58)):
        lines.append("WARNING: outside the 3:40-3:58 pm ET window; readings are for reference only.")
    if not is_session(today):
        lines.append("Market holiday or weekend: nothing to do."); print("\n".join(lines)); return
    states = {s: intraday_state(s) for s in ETFS}
    for s, st in states.items():
        if st:
            lines.append(f"  {s}: last {st['last']:.2f} | high {st['high']:.2f} low {st['low']:.2f} | IBS {st['ibs']:.2f}"
                         f" | prior high {st['prior_high']:.2f}")
            log_row(dict(run_time=now.isoformat(), etf=s, **{k: st[k] for k in ['last', 'high', 'low', 'ibs', 'prior_high', 'bar_time']}))
    # exits first
    positions = read_positions(); held = set()
    for p in positions:
        s = p["etf"]; held.add(s); st = states.get(s)
        exp = dt.date.fromisoformat(p["expiry"])
        if st and st["last"] > st["prior_high"]:
            lines.append(f"EXIT (bounce): SELL TO CLOSE {p['contracts']} {s} {p['expiry']} {p['strike']}C, limit at mid, "
                         f"walk toward bid; at 3:58 pm sell at the bid.")
        elif today >= last_session_before(exp):
            lines.append(f"EXIT (time): SELL TO CLOSE {p['contracts']} {s} {p['expiry']} {p['strike']}C today; last session before expiry.")
        else:
            lines.append(f"HOLD {s} {p['expiry']} {p['strike']}C (price {st['last'] if st else '?'} not above prior high "
                         f"{st['prior_high'] if st else '?'}).")
    # entries
    slots = k_slots(a.equity) - len(positions)
    cands = []
    for s, st in states.items():
        if st and st["ibs"] < IBS_ENTRY and s not in held:
            c, why = pick_contract(s, a.equity, a.settled, today, st["last"])
            if c:
                cands.append(c)
            else:
                lines.append(f"  {s} signalled (IBS {st['ibs']:.2f}) but skipped: {why}")
    cands.sort(key=lambda c: (-c["rung"], EVIDENCE_ORDER[c["sym"]]))
    if slots <= 0 and cands:
        lines.append(f"Signals present but no free slot (K = {k_slots(a.equity)}).")
    for c in cands[:max(slots, 0)]:
        lines.append(f"ENTRY: BUY TO OPEN {c['contracts']} {c['sym']} {c['expiry']} {c['strike']:.1f}C (~{c['rung']:.2f} delta, "
                     f"Cboe {c['bid']}/{c['ask']} at {c['quote_time']}). Limit at the LIVE mid; walk $0.01 every 30-60 s; "
                     f"never above mid + 25% of the live spread (now ~{c['max_limit']}); cancel at 3:58 pm. "
                     f"Cost at ask ${c['cost_at_ask']} = {c['pct_equity']}% of equity."
                     + (f" CAUTION: Cboe chain is stale (its spot {c['chain_spot']:.2f} vs live {states[c['sym']]['last']:.2f});"
                        f" pick the strike by delta in the app using the same rungs." if c['stale'] else ""))
    if len(lines) == 1 + len(states):
        lines.append("No action today.")
    msg = "\n".join(lines)
    print(msg)
    if a.discord:
        try:
            requests.post(a.discord, json={"content": msg[:1900]}, timeout=15)
        except Exception as e:
            print("discord post failed:", e)

if __name__ == "__main__":
    main()
