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
                  YOU maintain it: add a line when an ENTRY fills, delete the line after you sell. The ENTRY and EXIT
                  messages print the exact line / instruction. No line = no EXIT messages for that trade; a line left
                  behind after you sold = repeated EXIT/HOLD messages and a slot that never frees up.
  signals_log.csv every run's readings (append-only), useful for the paper stage and decay monitoring

Usage:
  python ibs_dip_calls_signal.py --equity 1500 --settled 1500
  python ibs_dip_calls_signal.py --equity 1500 --settled 1500 --discord $DISCORD_WEBHOOK_URL
Requires: pip install yfinance pandas numpy requests

Optional real-time prices: set the secret TRADIER_API_KEY (and TRADIER_BASE_URL=https://sandbox.tradier.com/v1 only for a
sandbox key). A production key makes Tradier the first source for the 1-minute bars and the option chain (real bid/ask and
greeks), with Yahoo / Cboe as the fallback. A sandbox key is ~15 minutes delayed, so Yahoo / Cboe stay first and the sandbox
is only a last resort. Without a key nothing changes. deliver/tradier.py is read-only (market data only, never orders).

Exit code: 0 = the full check ran and the message was delivered; 1 = something could not be checked (Yahoo / Cboe
down) or Discord refused the message. In GitHub Actions a missing webhook counts as a failure, not a quiet success.

Audit 2026-10-14 (the frozen rulebook values below are untouched): the Discord post is now checked and retried; a Yahoo or
Cboe outage on a signal day is reported in the message instead of crashing before anything is sent; stale 1-minute bars
are flagged; a skipped signal says WHICH rule blocked each rung; the ENTRY/EXIT messages say how to maintain
positions.csv; expired / malformed / same-day position lines are handled.
"""
import argparse, csv, datetime as dt, json, math, os, sys, time
import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance")

import tradier      # read-only market-data client (deliver/tradier.py); used only when TRADIER_API_KEY is set

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
NOTIFIED_FILE = os.path.join(os.path.dirname(HERE), ".notified")   # tells notify_failure.py the message already explained it
STALE_BAR_MIN = 6                       # a 1-minute bar older than this at 3:45 pm is Yahoo lag, not "the latest reading"

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
class DataProblem(Exception):
    """Something today's decision needs could not be read; the message says what to check by hand."""


def _to_et(ts):
    ts = pd.Timestamp(ts)
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        return ts.tz_localize(et) if ts.tzinfo is None else ts.tz_convert(et)
    except Exception:
        return ts


def _sources():
    """Order in which to try the price feeds. A production Tradier key is real-time, so it goes first; a sandbox key is
    ~15 minutes delayed, so Yahoo stays first and the sandbox is only a last resort (its age warning then fires)."""
    if not tradier.configured():
        return ["yahoo"]
    try:
        return ["yahoo", "tradier"] if tradier.is_delayed() else ["tradier", "yahoo"]
    except tradier.TradierError:           # a bad TRADIER_BASE_URL must not silence the bot
        return ["yahoo"]


def _tradier_label():
    try:
        return "Tradier sandbox (delayed)" if tradier.is_delayed() else "Tradier"
    except tradier.TradierError:
        return "Tradier"


def _tradier_intraday(sym, now):
    today = now.date()
    rows = tradier.timesales(sym, f"{today:%Y-%m-%d} 09:30", f"{today:%Y-%m-%d} {now:%H:%M}", interval="1min")
    bars = []
    for r in rows:
        try:
            bars.append((_to_et(r["time"]), float(r["high"]), float(r["low"]), float(r["close"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not bars:
        raise DataProblem("Tradier returned no 1-minute bars")
    bars.sort(key=lambda b: b[0])
    last_ts = bars[-1][0]
    if last_ts.date() != today:
        raise DataProblem(f"Tradier's newest 1-minute bar is from {last_ts.date():%a %m-%d}, not today")
    bars = [b for b in bars if b[0].date() == today]
    hi, lo, last = max(b[1] for b in bars), min(b[2] for b in bars), bars[-1][3]
    d = tradier.daily_bars(sym, 10, today=today)
    prior = d[d.index.date < today]
    if prior.empty:
        raise DataProblem("Tradier returned no prior-session bar")
    prior_high = float(prior["high"].iloc[-1])
    ibs = (last - lo) / (hi - lo) if hi > lo else 0.5
    bar_dt = last_ts.to_pydatetime()
    if now.tzinfo is None:
        bar_dt = bar_dt.replace(tzinfo=None)
    age = max((now - bar_dt).total_seconds() / 60.0, 0.0)
    return dict(sym=sym, last=last, high=hi, low=lo, ibs=ibs, prior_high=prior_high, bar_time=str(last_ts),
                age_min=round(age, 1), source=_tradier_label())


def _yahoo_intraday(sym, now):
    m = yf.download(sym, period="1d", interval="1m", progress=False, auto_adjust=False, prepost=False)
    d = yf.download(sym, period="10d", interval="1d", progress=False, auto_adjust=False)
    for x in (m, d):
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
    if m.empty:
        raise DataProblem("Yahoo returned no 1-minute bars")
    if len(d) < 2:
        raise DataProblem("Yahoo returned no daily history")
    last_ts = _to_et(m.index[-1])
    today = last_ts.date()
    if today != now.date():
        raise DataProblem(f"Yahoo's newest 1-minute bar is from {today:%a %m-%d}, not today")
    hi, lo, last = float(m["High"].max()), float(m["Low"].min()), float(m["Close"].iloc[-1])
    prior = d[d.index.date < today]
    if prior.empty:
        raise DataProblem("Yahoo returned no prior-session bar")
    prior_high = float(prior["High"].iloc[-1])
    ibs = (last - lo) / (hi - lo) if hi > lo else 0.5
    bar_dt = last_ts.to_pydatetime()
    if now.tzinfo is None:                  # ny_now() fallback returns a naive Eastern wall-clock time
        bar_dt = bar_dt.replace(tzinfo=None)
    age = max((now - bar_dt).total_seconds() / 60.0, 0.0)
    return dict(sym=sym, last=last, high=hi, low=lo, ibs=ibs, prior_high=prior_high, bar_time=str(m.index[-1]),
                age_min=round(age, 1))


def intraday_state(sym, now=None):
    """Today's regular-session high, low, last; prior session's high. Tradier first when a production key is set,
    else Yahoo (and each is the fallback for the other). Raises DataProblem when no feed has anything for TODAY (an
    outage, or a holiday missing from HOLIDAYS) -- it used to silently use the last bar it found. A Tradier failure
    never silences the bot: it falls through to Yahoo."""
    now = now or ny_now()
    errors = []
    for src in _sources():
        try:
            return _yahoo_intraday(sym, now) if src == "yahoo" else _tradier_intraday(sym, now)
        except Exception as e:
            errors.append((src, e))
    if len(errors) == 1:
        raise errors[0][1]
    raise DataProblem("; ".join(f"{'Yahoo' if s == 'yahoo' else 'Tradier'}: "
                                f"{e if isinstance(e, DataProblem) else type(e).__name__ + ': ' + str(e)}"
                                for s, e in errors))


def cboe_chain(sym):
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
    j = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).json()
    o = pd.DataFrame(j["data"]["options"])
    p = o.option.str.extract(r"^(?P<root>[A-Z]+)(?P<exp>\d{6})(?P<cp>[CP])(?P<k>\d{8})$")
    o = pd.concat([o, p], axis=1)
    o["strike"] = o.k.astype(float) / 1000
    o["expiry"] = pd.to_datetime(o.exp, format="%y%m%d").dt.date
    return o[o.cp == "C"].copy(), (j.get("timestamp"), float(j["data"].get("current_price") or np.nan))


def tradier_chain(sym, today):
    """The nearest eligible expiry's calls from Tradier, shaped like cboe_chain's output:
    (calls_df, (quote_time, spot)). Raises when Tradier has nothing usable so the caller falls back to Cboe."""
    exps = []
    for x in tradier.expirations(sym):
        try:
            e = dt.date.fromisoformat(str(x)[:10])
        except ValueError:
            continue
        if (e - today).days >= DTE_MIN and last_session_before(e) > today:
            exps.append(e)
    if not exps:
        raise LookupError("Tradier lists no expiry at least 10 days out")
    e = min(exps)
    rows = []
    stamps = []
    for o in tradier.chain(sym, e.isoformat(), greeks=True):
        if str(o.get("option_type", "")).lower() != "call":
            continue
        g = o.get("greeks") or {}
        try:
            row = dict(option=o.get("symbol"), strike=float(o["strike"]), bid=float(o.get("bid") or 0.0),
                       ask=float(o.get("ask") or 0.0), delta=float(g["delta"]),
                       open_interest=float(o.get("open_interest") or 0.0), expiry=e, cp="C")
        except (KeyError, TypeError, ValueError):
            continue            # no greeks / unparsable strike: the contract cannot be ranked by delta
        rows.append(row)
        for k in ("bid_date", "ask_date"):
            if o.get(k):
                stamps.append(o[k])
    if not rows:
        raise LookupError(f"Tradier returned no usable calls for {e}")
    stamp = None
    if stamps:
        stamp = _to_et(pd.Timestamp(max(stamps), unit="ms", tz="UTC")).strftime("%Y-%m-%d %H:%M:%S")
    try:
        q = tradier.quotes([sym]).get(sym.upper()) or {}
        spot = float(q["last"]) if q.get("last") is not None else float("nan")
    except Exception:
        spot = float("nan")
    return pd.DataFrame(rows), (stamp, spot)


def load_chain(sym, today):
    """(calls, (quote_time, spot), source). Production Tradier first, then Cboe; a sandbox key only if Cboe fails."""
    order = ["cboe"]
    if tradier.configured():
        try:
            order = ["cboe", "tradier"] if tradier.is_delayed() else ["tradier", "cboe"]
        except tradier.TradierError:
            pass
    errors = []
    for src in order:
        try:
            if src == "cboe":
                calls, meta = cboe_chain(sym)
                return calls, meta, "Cboe"
            calls, meta = tradier_chain(sym, today)
            return calls, meta, _tradier_label()
        except Exception as e:
            errors.append(e)
    if len(errors) == 1:
        raise errors[0]
    raise RuntimeError("; ".join(f"{type(e).__name__}: {e}" for e in errors))


def _rung_check(c, rung, budget, settled):
    """(contracts, cost per contract at the ask, [reasons it failed]) for the contract nearest `rung` in delta -- the
    frozen rule, kept in one place so the skip message can name exactly which part of it blocked the rung."""
    problems = []
    if abs(c.delta - rung) > DELTA_TOL:
        problems.append(f"nearest delta is {c.delta:.2f}")
    if c.rel_spread > MAX_REL_SPREAD:
        problems.append(f"bid-ask spread {c.rel_spread * 100:.1f}% is over {MAX_REL_SPREAD * 100:.0f}%")
    if c.open_interest < MIN_OI:
        problems.append(f"open interest {int(c.open_interest)} is under {MIN_OI}")
    if problems:
        return 0, None, problems
    cost = float(c.ask) * 100
    n = math.floor(budget / cost)
    if n < 1 and cost <= ALLOW_MULT * budget:
        n = 1
    if n < 1:
        return 0, cost, [f"costs ${cost:.0f} at the ask"]
    if n * cost > settled:
        return 0, cost, [f"needs ${n * cost:.0f} of settled cash, you have ${settled:.0f}"]
    return int(n), cost, []


def pick_contract(sym, equity, settled, today, live_last=None):
    calls, (stamp, chain_spot), source = load_chain(sym, today)
    stale = live_last is not None and np.isfinite(chain_spot) and abs(chain_spot / live_last - 1) > 0.005
    exps = sorted(e for e in calls.expiry.unique() if (e - today).days >= DTE_MIN and last_session_before(e) > today)
    if not exps:
        return None, f"the {source} chain lists no expiry at least 10 days out"
    e = exps[0]
    g = calls[(calls.expiry == e) & (calls.bid > 0) & (calls.ask > calls.bid)].copy()
    g["mid"] = (g.bid + g.ask) / 2
    g["rel_spread"] = (g.ask - g.bid) / g.mid
    budget = TARGET_F * equity
    why = []
    for rung in RUNGS:
        c = g.iloc[(g.delta - rung).abs().argsort()[:1]]
        if c.empty:
            why.append(f"{rung:.2f}-delta: no quoted contract")
            continue
        c = c.iloc[0]
        n, cost, problems = _rung_check(c, rung, budget, settled)
        if n >= 1:
            out = dict(sym=sym, expiry=str(e), strike=float(c.strike), rung=rung, delta=round(float(c.delta), 3),
                       bid=float(c.bid), ask=float(c.ask), mid=round(float(c.mid), 3),
                       max_limit=round(float(c.mid + 0.25 * (c.ask - c.bid)), 2), contracts=int(n),
                       cost_at_ask=round(n * cost, 2), pct_equity=round(n * cost / equity * 100, 1),
                       quote_time=stamp, stale=stale, chain_spot=chain_spot)
            if source != "Cboe":
                out["source"] = source
            return out, None
        why.append(f"{rung:.2f}-delta {c.strike:g}C {e:%a %m-%d}: " + "; ".join(problems))
    return None, (f"no call passed the rules (cost at most ${ALLOW_MULT * budget:,.0f} per contract = "
                  f"{ALLOW_MULT * TARGET_F * 100:.0f}% of equity, bid-ask spread <= {MAX_REL_SPREAD * 100:.0f}%, "
                  f"open interest >= {MIN_OI}). Checked:\n      " + "\n      ".join(why))


# ------------------------------- positions & logging ----------------------------------------------------------
def read_positions():
    if not os.path.exists(POS_FILE):
        with open(POS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["etf", "expiry", "strike", "contracts", "entry_date", "entry_price"])
        return []
    with open(POS_FILE, newline="") as f:
        return list(csv.DictReader(f))


def log_row(row):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ------------------------------- Discord (checked, retried, never silent) -------------------------------------
class PostFailed(Exception):
    """Discord did not accept the message."""


SHORTENED_NOTE = "\n...(shortened to fit Discord's 2000-character limit)"


def fit_discord(text, limit=1980):
    """Discord rejects (does not truncate) anything over 2000 characters, so an overlong message would vanish whole.
    The result, shortening note included, is never longer than `limit`."""
    if len(text) <= limit:
        return text
    cut = text[:limit - len(SHORTENED_NOTE)].rsplit("\n", 1)[0]
    return cut + SHORTENED_NOTE


def post_discord(webhook, content, tries=4, sleep=None):
    """POST one message; returns when Discord accepts it (204/200), raises PostFailed otherwise. 429 / 5xx / network
    errors are retried with back-off (429 honours retry_after); any other 4xx (deleted webhook, rejected payload)
    fails at once because retrying cannot fix it."""
    sleep = sleep or time.sleep
    payload = {"content": fit_discord(content), "allowed_mentions": {"parse": []}}
    last = "no attempt made"
    for i in range(tries):
        wait = 2 ** i
        try:
            r = requests.post(webhook, json=payload, timeout=15)
        except requests.RequestException as e:
            last = f"network error: {e}"
        else:
            if r.status_code in (200, 204):
                return
            if r.status_code == 429:
                last = "HTTP 429 (rate limited)"
                try:
                    wait = max(wait, float(r.json().get("retry_after", 0)))
                except Exception:
                    pass
            elif r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            else:
                raise PostFailed(f"HTTP {r.status_code}: {str(r.text)[:200]}")
        if i < tries - 1:
            sleep(min(wait, 30))
    raise PostFailed(f"gave up after {tries} tries ({last})")


# ------------------------------- the report -------------------------------------------------------------------
POSITIONS_HOWTO = ("Edit deliver/positions.csv on GitHub (open the file, pencil icon, add or delete a line, "
                   "Commit changes).")


def run(equity, settled, now=None):
    """Build today's message. Returns (lines, send, degraded): `send` False on a market holiday/weekend (nothing to
    post); `degraded` True when something could not be checked, so the run should end red."""
    now = now or ny_now()
    today = now.date()
    lines = [f"IBS Dip Calls | {now:%Y-%m-%d %H:%M} ET | equity ${equity:,.0f} | settled ${settled:,.0f}"]
    if not (dt.time(15, 40) <= now.time() <= dt.time(15, 58)):
        lines.append("WARNING: outside the 3:40-3:58 pm ET window; readings are for reference only.")
    if not is_session(today):
        lines.append("Market holiday or weekend: nothing to do.")
        return lines, False, False
    degraded = False
    states, problems = {}, {}
    for s in ETFS:
        try:
            states[s] = intraday_state(s, now)
        except Exception as e:       # Yahoo down / throttled / empty: say so instead of crashing before any message
            states[s] = None
            problems[s] = str(e) if isinstance(e, DataProblem) else f"{type(e).__name__}: {e}"
    for s, st in states.items():
        if st:
            lines.append(f"  {s}: last {st['last']:.2f} | high {st['high']:.2f} low {st['low']:.2f} | IBS {st['ibs']:.2f}"
                         f" | prior high {st['prior_high']:.2f}")
            log_row(dict(run_time=now.isoformat(), etf=s, **{k: st[k] for k in ['last', 'high', 'low', 'ibs', 'prior_high', 'bar_time']}))
            if st["age_min"] > STALE_BAR_MIN:
                lines.append(f"  WARNING: {s}'s newest {st.get('source', 'Yahoo')} bar is {st['age_min']:.0f} min old ({st['bar_time'][11:16]}); "
                             f"the reading above may be stale -- check {s} in the Robinhood app before acting on it.")
        else:
            degraded = True
            lines.append(f"  {s}: NO READING ({problems[s]}) -- {s} was NOT checked today; look at it in the app: "
                         f"IBS = (last - low) / (high - low) of today's range.")
    acts = []          # the lines that say what to do (or why not)
    # exits first
    positions = read_positions(); held = set(); sold_lines = False
    for p in positions:
        try:
            s = p["etf"].strip().upper(); exp = dt.date.fromisoformat(p["expiry"].strip())
            float(p["strike"])
            if s not in ETFS or int(float(p["contracts"])) < 1:
                raise ValueError("unknown ETF or contracts < 1")
            label = f"{p['contracts']} {s} {p['expiry']} {p['strike']}C"
        except (KeyError, TypeError, ValueError, AttributeError):
            acts.append(f"BAD LINE in positions.csv: {dict(p)} -- fix or delete it (columns: "
                        f"etf,expiry,strike,contracts,entry_date,entry_price; expiry as YYYY-MM-DD). {POSITIONS_HOWTO}")
            continue
        held.add(s); st = states.get(s)
        if exp < today:
            acts.append(f"STALE LINE in positions.csv: {label} expired on {p['expiry']}. If it is no longer in your account, "
                        f"delete the line -- it is using up your free slot. {POSITIONS_HOWTO}")
        elif (p.get("entry_date") or "").strip() == today.isoformat():
            acts.append(f"HOLD {s} {p['expiry']} {p['strike']}C (entered today; the first exit check is the next session).")
        elif st and st["last"] > st["prior_high"]:
            sold_lines = True
            acts.append(f"EXIT (bounce): SELL TO CLOSE {p['contracts']} {s} {p['expiry']} {p['strike']}C, limit at mid, "
                        f"walk toward bid; at 3:58 pm sell at the bid.")
        elif today >= last_session_before(exp):
            sold_lines = True
            acts.append(f"EXIT (time): SELL TO CLOSE {p['contracts']} {s} {p['expiry']} {p['strike']}C today; last session before expiry.")
        elif st is None:
            acts.append(f"CHECK BY HAND: {s} {p['expiry']} {p['strike']}C -- no {s} prices today, so the exit test could not run. "
                        f"Sell if {s} is trading above yesterday's high (or on {last_session_before(exp):%a %m-%d}, the last session before expiry).")
        else:
            acts.append(f"HOLD {s} {p['expiry']} {p['strike']}C (price {st['last']} not above prior high "
                        f"{st['prior_high']}).")
    if sold_lines:
        acts.append("After you sell, delete that position's line from deliver/positions.csv -- otherwise tomorrow's run keeps "
                    f"telling you to exit and no new entry can use the slot. {POSITIONS_HOWTO}")
    # entries
    slots = k_slots(equity) - len(positions)
    cands = []
    for s, st in states.items():
        if not st or st["ibs"] >= IBS_ENTRY:
            continue
        if s in held:
            acts.append(f"  {s} signalled (IBS {st['ibs']:.2f}) but you already hold {s}: ignored (one position per ETF).")
            continue
        try:
            c, why = pick_contract(s, equity, settled, today, st["last"])
        except Exception as e:
            degraded = True
            acts.append(f"  {s} signalled (IBS {st['ibs']:.2f}) but the {'Tradier/Cboe' if tradier.configured() else 'Cboe'} option chain could not be loaded "
                        f"({type(e).__name__}) -- no ticket for {s}. Look in the Robinhood app: nearest expiry >= 10 days out, "
                        f"highest of the 0.50 / 0.40 / 0.30-delta calls that costs <= ${ALLOW_MULT * TARGET_F * equity:,.0f} per contract "
                        f"with a spread <= 4% and open interest >= 500, and place it by the usual rules before 3:58 pm.")
            continue
        if c:
            c["bar_age"] = st["age_min"]
            cands.append(c)
        else:
            acts.append(f"  {s} signalled (IBS {st['ibs']:.2f}) but skipped: {why}")
    cands.sort(key=lambda c: (-c["rung"], EVIDENCE_ORDER[c["sym"]]))
    if slots <= 0 and cands:
        acts.append(f"Signals present ({', '.join(c['sym'] for c in cands)}) but no free slot: K = {k_slots(equity)} and "
                    f"positions.csv has {len(positions)} line(s) (a call you hold, or a line you forgot to delete).")
    for c in cands[:max(slots, 0)]:
        acts.append(f"ENTRY: BUY TO OPEN {c['contracts']} {c['sym']} {c['expiry']} {c['strike']:.1f}C (~{c['rung']:.2f} delta, "
                    f"{c.get('source', 'Cboe')} {c['bid']}/{c['ask']} at {c['quote_time']}). Limit at the LIVE mid; walk $0.01 every 30-60 s; "
                    f"never above mid + 25% of the live spread (now ~{c['max_limit']}); cancel at 3:58 pm. "
                    f"Cost at ask ${c['cost_at_ask']} = {c['pct_equity']}% of equity."
                    + (f" CAUTION: {c.get('source', 'Cboe')} chain is stale (its spot {c['chain_spot']:.2f} vs live {states[c['sym']]['last']:.2f});"
                       f" pick the strike by delta in the app using the same rungs." if c['stale'] else "")
                    + (f" CAUTION: the IBS reading came from a {c['bar_age']:.0f}-minute-old {states[c['sym']].get('source', 'Yahoo')} bar; confirm IBS < 0.10 in the app first."
                       if c['bar_age'] > STALE_BAR_MIN else ""))
        acts.append(f"  If it FILLS, add this line to deliver/positions.csv (replace FILL_PRICE with your fill price per share, "
                    f"e.g. 1.45): {c['sym']},{c['expiry']},{c['strike']:.1f},{c['contracts']},{today.isoformat()},FILL_PRICE  "
                    f"-- no line means no EXIT messages for this trade. {POSITIONS_HOWTO} "
                    f"Not filled by 3:58 pm? Cancel the order; there is nothing to record.")
    if not acts:
        missing = [x for x, st in states.items() if st is None]
        if not missing:
            acts.append("No action today (no ETF has IBS below 0.10 and you hold nothing).")
        elif len(missing) == len(states):
            acts.append("Nothing could be checked today, so this is NOT a 'no signal' day. Look at SPY / QQQ / IWM in the "
                        "app: IBS below 0.10 is a signal.")
        else:
            acts.append(f"No action for the ETFs that were checked, but {', '.join(missing)} could NOT be checked, so this is "
                        f"not a 'no signal' day for them.")
    return lines + acts, True, degraded


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, required=True, help="total account value now")
    ap.add_argument("--settled", type=float, required=True, help="settled cash available to buy")
    ap.add_argument("--discord", default=os.environ.get("DISCORD_WEBHOOK_URL", ""))
    a = ap.parse_args(argv)
    lines, send, degraded = run(a.equity, a.settled)
    msg = "\n".join(lines)
    print(msg)
    code = 1 if degraded else 0
    if send:
        if a.discord:
            try:
                post_discord(a.discord, msg)
                if degraded:      # the message itself says what could not be checked: no generic "run FAILED" notice on top
                    try:
                        with open(NOTIFIED_FILE, "w") as f:
                            f.write("degraded run reported in Discord")
                    except OSError:
                        pass
            except PostFailed as e:
                print(f"ERROR: Discord refused the message ({e}); today's instructions above did NOT reach you.", file=sys.stderr)
                code = 1
        elif os.environ.get("GITHUB_ACTIONS") == "true":
            print("ERROR: no Discord webhook configured (secret DISCORD_WEBHOOK_URL) -- the message above went nowhere.",
                  file=sys.stderr)
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
