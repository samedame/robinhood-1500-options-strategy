"""
Core research harness: data access, signals, option selection, trade simulation with Robinhood costs.

Conventions
-----------
* Dates are pandas Timestamps (normalized). Trading calendar = days present in the underlying's Yahoo history.
* A 'structure' is a list of legs; each leg = dict(cp='C'|'P', side=+1 (long)|-1 (short), sel=('delta', 0.5) |
  ('atm',) | ('offset_pct', 0.02, ref_leg_index)).
* Fill model: long leg pays mid + c*spread, short leg receives mid - c*spread (c=0.25 base, 0.50 natural = 2x cost,
  0.75 = 3x cost). Exits mirror this.
* Fees (Robinhood, verified 2026-09-23): per contract per side $0.04 ORF/OCC + $0.0003 CAT; sells add FINRA TAF
  $0.00329 and SEC $20.60 per $1M principal; +$0.01 per order rounding. Index options (XSP) add $0.50 per side.
"""
from __future__ import annotations
import os, glob, math, json
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

ROOT = "/home/claude/research"
CHAINS = f"{ROOT}/data/chains"
MKT = "/home/claude/data/mkt"

# ----------------------------------------------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------------------------------------------
def load_ohlc(sym: str) -> pd.DataFrame:
    df = pd.read_csv(f"{MKT}/yf_{sym}.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna(subset=["Close"])
    return df

def load_cboe(name: str) -> pd.Series:
    d = pd.read_csv(f"{MKT}/{name}_History.csv")
    d.columns = [c.upper() for c in d.columns]
    d["DATE"] = pd.to_datetime(d["DATE"])
    col = "CLOSE" if "CLOSE" in d.columns else [c for c in d.columns if c != "DATE"][0]
    return d.set_index("DATE")[col].astype(float).sort_index()

# ----------------------------------------------------------------------------------------------------------
# Option chains
# ----------------------------------------------------------------------------------------------------------
class Chains:
    """Holds EOD chains for one symbol as {date: DataFrame indexed by (expiry, cp, strike)}."""
    def __init__(self, symbol: str, vendor_by_year: dict[int, str]):
        self.symbol = symbol
        self.by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        for y, v in sorted(vendor_by_year.items()):
            fn = f"{CHAINS}/{v}_{symbol}_{y}.parquet"
            if not os.path.exists(fn):
                continue
            df = pd.read_parquet(fn)
            df["mid"] = (df.bid + df.ask) / 2
            for d, g in df.groupby("date", sort=True):
                g = g.set_index(["expiry", "cp", "strike"]).sort_index()
                self.by_date[pd.Timestamp(d).normalize()] = g
        self.dates = sorted(self.by_date)

    def has(self, d) -> bool:
        return d in self.by_date

    def day(self, d):
        return self.by_date.get(d)

    def quote(self, d, expiry, cp, strike):
        g = self.by_date.get(d)
        if g is None:
            return None
        try:
            r = g.loc[(expiry, cp, strike)]
        except KeyError:
            return None
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return r

# ----------------------------------------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------------------------------------
def ibs(df):
    rng = (df.High - df.Low).replace(0, np.nan)
    return ((df.Close - df.Low) / rng).fillna(0.5)

def rsi2(close, n=2):
    d = close.diff()
    up = d.clip(lower=0); dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean(); ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def down_streak(close, k=3):
    dn = (close.diff() < 0).astype(int)
    return dn.rolling(k).sum() == k

def at_n_day_low(close, n=5):
    return close <= close.rolling(n).min()

def vix_expirations(start="2009-01-01", end="2027-12-31", trading_days=None):
    """VIX final settlement: the Wednesday 30 days before the third Friday of the following month.
    If that Friday is a holiday -> the Wednesday 30 days before the business day preceding it; if the
    Wednesday itself is a holiday -> the preceding business day. trading_days: set of valid sessions."""
    out = []
    months = pd.date_range(start, end, freq="MS")
    for m in months:
        nm = m + pd.offsets.MonthBegin(1)
        fridays = pd.date_range(nm, nm + pd.offsets.MonthEnd(0), freq="W-FRI")
        tf = fridays[2]
        if trading_days is not None and tf <= max(trading_days) and tf not in trading_days:
            tf = tf - pd.Timedelta(days=1)
        wed = tf - pd.Timedelta(days=30)
        if trading_days is not None and wed <= max(trading_days):
            while wed not in trading_days:
                wed -= pd.Timedelta(days=1)
        out.append(wed.normalize())
    return pd.DatetimeIndex(sorted(set(out)))

# ----------------------------------------------------------------------------------------------------------
# Fees and fills
# ----------------------------------------------------------------------------------------------------------
def fee_per_contract(side_is_sell: bool, price: float, index_option: bool = False) -> float:
    f = 0.04 + 0.0003
    if side_is_sell:
        f += 0.00329 + 20.60e-6 * price * 100
    if index_option:
        f += 0.50
    return f

def fill(bid, ask, buying: bool, c: float):
    mid = (bid + ask) / 2
    spr = max(ask - bid, 0.0)
    return mid + c * spr if buying else max(mid - c * spr, 0.0)

# ----------------------------------------------------------------------------------------------------------
# Trade simulation
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Trade:
    entry: pd.Timestamp
    exit: pd.Timestamp
    expiry: pd.Timestamp
    legs: list            # list of (cp, side, strike)
    entry_px: dict        # c -> net entry cash per share (positive = paid)
    exit_px: dict         # c -> net exit cash per share (positive = received)
    risk: dict            # c -> capital at risk per share
    pnl: dict             # c -> $ P&L per 1 structure (100 multiplier) net of fees
    ret: dict             # c -> return on capital at risk
    marks: list = field(default_factory=list)   # (date, mid value of structure per share)
    und_entry: float = np.nan
    und_exit: float = np.nan
    spread_rel: float = np.nan
    notes: str = ""

class Engine:
    def __init__(self, chains: Chains, ohlc: pd.DataFrame, index_option=False, max_rel_spread=0.30):
        self.ch = chains
        self.px = ohlc
        self.tdays = pd.DatetimeIndex([d for d in ohlc.index])
        self.tpos = {d: i for i, d in enumerate(self.tdays)}
        self.index_option = index_option
        self.max_rel_spread = max_rel_spread

    # --- calendar helpers
    def next_td(self, d, k=1):
        i = self.tpos.get(d)
        if i is None or i + k >= len(self.tdays):
            return None
        return self.tdays[i + k]

    def last_td_before(self, expiry):
        i = self.tdays.searchsorted(expiry) - 1
        return self.tdays[i] if i >= 0 else None

    # --- selection
    def pick_expiry(self, d, dte_target):
        g = self.ch.day(d)
        if g is None:
            return None
        exps = g.index.get_level_values(0).unique()
        cands = sorted([e for e in exps if (e - d).days >= dte_target])
        for e in cands:
            cap = self.last_td_before(e)
            if cap is not None and cap > d:
                return e
        return None

    def pick_strike(self, d, expiry, cp, sel, S, ref_strike=None):
        g = self.ch.day(d)
        try:
            sub = g.loc[(expiry, cp)]
        except KeyError:
            return None
        sub = sub[(sub.bid > 0) & (sub.ask > sub.bid)]
        sub = sub[(sub.ask - sub.bid) / sub.mid <= self.max_rel_spread]
        if sub.empty:
            return None
        kind = sel[0]
        if kind == "delta":
            tgt = sel[1] if cp == "C" else -sel[1]
            dd = (sub.delta - tgt).abs()
            if dd.isna().all():
                return None
            k = dd.idxmin()
            if dd.loc[k] > 0.10:
                return None
            return float(k)
        if kind == "atm":
            return float(sub.index[np.argmin(np.abs(sub.index.values - S))])
        if kind == "offset_pct":        # strike relative to a reference strike: + for calls above, - for puts below
            tgt = ref_strike + sel[1] * S
            ks = sub.index.values
            if sel[1] >= 0:
                ok = ks[ks >= tgt - 1e-9]
                return float(ok.min()) if len(ok) else None
            ok = ks[ks <= tgt + 1e-9]
            return float(ok.max()) if len(ok) else None
        raise ValueError(kind)

    def build(self, d, legs_spec, dte_target):
        """Return (expiry, legs[(cp, side, strike)]) or None."""
        S = float(self.px.Close.loc[d])
        e = self.pick_expiry(d, dte_target)
        if e is None:
            return None
        legs = []
        for spec in legs_spec:
            ref = legs[spec["ref"]][2] if "ref" in spec else None
            k = self.pick_strike(d, e, spec["cp"], spec["sel"], S, ref)
            if k is None:
                return None
            legs.append((spec["cp"], spec["side"], k))
        return e, legs

    def structure_value(self, d, e, legs, c, closing: bool):
        """Cash per share to open (closing=False; positive=paid) or to close (closing=True; positive=received)."""
        tot = 0.0; fees = 0.0; mids = 0.0; rel = []
        for cp, side, k in legs:
            q = self.ch.quote(d, e, cp, k)
            if q is None or not np.isfinite(q.bid) or not np.isfinite(q.ask) or q.ask <= 0:
                return None
            bid, ask = max(float(q.bid), 0.0), float(q.ask)
            if ask < bid:
                return None
            buying = (side > 0) != closing          # opening a long / closing a short = buy
            p = fill(bid, ask, buying, c)
            tot += (p if buying else -p)
            fees += fee_per_contract(not buying, p, self.index_option) / 100 + 0.01 / 100
            mids += side * (bid + ask) / 2
            if (bid + ask) > 0:
                rel.append((ask - bid) / ((bid + ask) / 2))
        # opening: tot = net paid per share; closing: -tot = net received per share
        val = tot if not closing else -tot
        return val, fees, mids, (np.mean(rel) if rel else np.nan)

    def exit_date(self, d0, rule, cap, event_date=None, max_hold=None):
        px = self.px
        i0 = self.tpos[d0]
        if rule == "EVT":
            ex = event_date
        elif rule.startswith("H"):
            ex = self.tdays[min(i0 + int(rule[1:]), len(self.tdays) - 1)]
        elif rule.startswith("IBS"):
            mh = max_hold or int(rule[3:])
            ex = None
            for j in range(1, mh + 1):
                if i0 + j >= len(self.tdays):
                    break
                t = self.tdays[i0 + j]
                if px.Close.iloc[i0 + j] > px.High.iloc[i0 + j - 1]:
                    ex = t; break
            if ex is None:
                ex = self.tdays[min(i0 + mh, len(self.tdays) - 1)]
        else:
            raise ValueError(rule)
        return min(ex, cap)

    def simulate(self, d0, legs_spec, dte_target, rule, cs=(0.25, 0.5, 0.75), delay=0, event_date=None,
                 max_hold=None, pt_credit=None):
        """Simulate one trade signalled at close of d0. delay=k: entry k sessions late, and the exit signal
        (evaluated from the original signal date) is also acted on k sessions late."""
        d_sig = d0
        if d_sig not in self.tpos:
            return None
        if delay:
            d0 = self.next_td(d_sig, delay)
            if d0 is None:
                return None
        if not self.ch.has(d0):
            return None
        b = self.build(d0, legs_spec, dte_target)
        if b is None:
            return None
        e, legs = b
        cap = self.last_td_before(e)
        far = self.tdays[-1]
        ex = self.exit_date(d_sig, rule, far, event_date, max_hold)
        if delay and rule != "EVT":
            nx = self.next_td(ex, delay)
            ex = nx if nx is not None else ex
        ex = min(ex, cap)
        if ex <= d0:
            nx = self.next_td(d0)
            if nx is None or nx > cap:
                return None
            ex = nx
        # entry
        ent = {}
        for c in cs:
            v = self.structure_value(d0, e, legs, c, closing=False)
            if v is None:
                return None
            ent[c] = v
        # daily marks and optional profit-target exit (credit spreads)
        marks = []
        i0, i1 = self.tpos[d0], self.tpos[ex]
        credit = -ent[0.25][2] if pt_credit else None
        for j in range(i0 + 1, i1 + 1):
            t = self.tdays[j]
            mv = self.structure_value(t, e, legs, 0.25, closing=True)
            if mv is None:
                continue
            marks.append((t, mv[2]))
            if pt_credit and credit and credit > 0 and (-mv[2]) <= (1 - pt_credit) * credit and t < ex:
                ex = t; break
        # exit (fall back to the next session's quote, then intrinsic)
        xv = {}; note = ""
        for c in cs:
            v = self.structure_value(ex, e, legs, c, closing=True)
            if v is None:
                alt = self.next_td(ex)
                if alt is not None and alt < e:
                    v = self.structure_value(alt, e, legs, c, closing=True)
                    note = "exit_next_day"
                if v is None:
                    S = float(self.px.Close.loc[ex])
                    intr = sum(side * (max(S - k, 0) if cp == "C" else max(k - S, 0)) for cp, side, k in legs)
                    v = (intr * 0.99 if intr > 0 else intr * 1.01, 0.0, intr, np.nan)
                    note = "exit_intrinsic"
            xv[c] = v
        pnl, ret, risk, entry_px, exit_px = {}, {}, {}, {}, {}
        widths = [k for _, _, k in legs]
        for c in cs:
            paid, fin, _, rel_in = ent[c]
            recv, fout, _, _ = xv[c]
            p = (recv - paid - fin - fout) * 100
            if paid > 0:                 # debit structure: risk = debit + fees
                rk = paid + fin + fout
            else:                        # credit structure: risk = width - credit
                width = max(widths) - min(widths)
                rk = max(width + paid, 0.01) + fin + fout
            pnl[c] = p; risk[c] = rk; ret[c] = p / (rk * 100)
            entry_px[c] = paid; exit_px[c] = recv
        return Trade(entry=d0, exit=ex, expiry=e, legs=legs, entry_px=entry_px, exit_px=exit_px, risk=risk, pnl=pnl,
                     ret=ret, marks=marks, und_entry=float(self.px.Close.loc[d0]), und_exit=float(self.px.Close.loc[ex]),
                     spread_rel=ent[0.25][3], notes=note)

# ----------------------------------------------------------------------------------------------------------
# Running a signal set
# ----------------------------------------------------------------------------------------------------------
def run_signals(engine: Engine, sig_dates, legs_spec, dte_target, rule, start, end, delay=0, event_map=None,
                max_hold=None, pt_credit=None, one_at_a_time=True):
    trades = []; busy_until = pd.Timestamp("1900-01-01"); skipped = 0
    for d in sig_dates:
        if d < pd.Timestamp(start) or d > pd.Timestamp(end):
            continue
        if one_at_a_time and d <= busy_until:
            continue
        ev = event_map.get(d) if event_map else None
        t = engine.simulate(d, legs_spec, dte_target, rule, delay=delay, event_date=ev, max_hold=max_hold,
                            pt_credit=pt_credit)
        if t is None:
            skipped += 1
            continue
        trades.append(t)
        busy_until = t.exit
    return trades, skipped

def trades_df(trades, c=0.25):
    rows = []
    for t in trades:
        rows.append(dict(entry=t.entry, exit=t.exit, expiry=t.expiry, dte=(t.expiry - t.entry).days,
                         hold=len(pd.bdate_range(t.entry, t.exit)) - 1, legs=str(t.legs), ret=t.ret[c], pnl=t.pnl[c],
                         risk=t.risk[c] * 100, ret_2x=t.ret.get(0.5, np.nan), ret_3x=t.ret.get(0.75, np.nan),
                         und_entry=t.und_entry, und_exit=t.und_exit, und_ret=t.und_exit / t.und_entry - 1,
                         spread_rel=t.spread_rel, notes=t.notes,
                         min_mark_ret=(min([m for _, m in t.marks]) / t.entry_px[0.25] - 1) if (t.marks and t.entry_px[0.25] > 0) else np.nan,
                         entry_px=t.entry_px[c],
                         marks=json.dumps([(str(dd.date()), float(m)) for dd, m in t.marks])))
    return pd.DataFrame(rows)
