"""
Monte Carlo growth math (Phase 6). Monthly stationary block bootstrap of the champion's trade history, replayed through
a cash-account simulator on a synthetic business-day calendar (T+1 settlement, whole contracts at today's contract
costs, fraction-of-equity sizing, K concurrent slots, portfolio heat cap, one-contract allowance).
"""
import sys, json, time
sys.path.insert(0, "/home/claude/research/src")
import numpy as np, pandas as pd
from cash_engine import CURRENT_SPOT

def prep_trades(df, ret_col="ret"):
    """Return list of months, each a list of trade tuples (entry_off, exit_off, ret, unit_cost, marks{off:ratio}, sym)."""
    df = df.copy()
    df["month"] = df.entry.dt.to_period("M")
    months = pd.period_range(df.month.min(), df.month.max(), freq="M")
    out = []
    for m in months:
        g = df[df.month == m]
        m0 = m.start_time
        bdays = pd.bdate_range(m0, m0 + pd.Timedelta(days=62))
        pos = {d: i for i, d in enumerate(bdays)}
        items = []
        for _, r in g.iterrows():
            e0 = pos.get(r.entry); e1 = pos.get(r.exit)
            if e0 is None or e1 is None:
                continue
            uc = r["risk"] / 100.0 / r["und_entry"] * CURRENT_SPOT[r["sym"]] * 100.0
            mk = {}
            if isinstance(r.get("marks"), str) and r["entry_px"] > 0:
                for ds, mv in json.loads(r["marks"]):
                    o = pos.get(pd.Timestamp(ds))
                    if o is not None:
                        mk[o] = mv / r["entry_px"]
            items.append((e0, e1, float(r[ret_col]), float(uc), mk, r["sym"]))
        out.append(items)
    return out

def sample_months(n_months, horizon, rng, mean_block=3):
    idx = []
    i = rng.integers(n_months)
    while len(idx) < horizon:
        idx.append(i)
        if rng.random() < 1.0 / mean_block:
            i = rng.integers(n_months)
        else:
            i = (i + 1) % n_months
    return idx

def run_path(months, idx, f, K, heat, eq0=1500.0, allow_mult=1.5, settle_lag=1, days_per_month=21):
    # build the path's trades on a synthetic calendar
    trades = []
    for k, mi in enumerate(idx):
        base = k * days_per_month
        for (e0, e1, r, uc, mk, sym) in months[mi]:
            if e0 >= days_per_month:        # entries beyond the 21st business day are dropped (rare)
                continue
            trades.append((base + e0, base + e1, r, uc, {base + o: v for o, v in mk.items()}, sym))
    trades.sort(key=lambda t: t[0])
    T = len(idx) * days_per_month + 15
    by_day = {}
    for t in trades:
        by_day.setdefault(t[0], []).append(t)
    settled = eq0; pending = []; openp = []
    curve = np.empty(T); n_taken = 0; n_skip = 0
    for d in range(T):
        if pending:
            keep = []
            for (sd, a) in pending:
                if sd <= d:
                    settled += a
                else:
                    keep.append((sd, a))
            pending = keep
        if openp:
            keep = []
            for p in openp:
                if p[1] == d:
                    proceeds = max(p[3] * (1 + p[2]), 0.0)
                    if settle_lag == 0:
                        settled += proceeds
                    else:
                        pending.append((d + settle_lag, proceeds))
                else:
                    keep.append(p)
            openp = keep
        mtm = 0.0
        for p in openp:
            v = p[4].get(d)
            if v is not None:
                p[5] = v
            mtm += p[3] * p[5]
        equity = settled + sum(a for _, a in pending) + mtm
        for t in by_day.get(d, []):
            if len(openp) >= K:
                n_skip += 1; continue
            risk_open = sum(p[3] for p in openp)
            budget = min(f * equity, heat * equity - risk_open)
            if budget <= 0:
                n_skip += 1; continue
            uc = t[3]
            n = np.floor(budget / uc)
            if n < 1 and uc <= allow_mult * budget:
                n = 1
            if n < 1:
                n_skip += 1; continue
            cap = n * uc
            if cap > settled:
                n = np.floor(settled / uc)
                if n < 1:
                    n_skip += 1; continue
                cap = n * uc
            settled -= cap
            openp.append([t[0], t[1], t[2], cap, t[4], 1.0])
            n_taken += 1
            equity = settled + sum(a for _, a in pending) + sum(p[3] * p[5] for p in openp)
        curve[d] = equity
    return curve, n_taken, n_skip

def frontier(months, fs, Ks, heats, n_paths=5000, horizon=12, seed=1, eq0=1500.0, settle_lag=1, allow_mult=1.5):
    rows = []
    for K in Ks:
        for heat in heats:
            for f in fs:
                rng = np.random.default_rng(seed)
                finals = np.empty(n_paths); mdd = np.empty(n_paths); ruin = np.zeros(n_paths, bool); wipe = np.zeros(n_paths, bool)
                t2x = np.full(n_paths, np.nan); m1 = np.empty(n_paths); m3 = np.empty(n_paths); m6 = np.empty(n_paths)
                taken = 0; skipped = 0
                for i in range(n_paths):
                    idx = sample_months(len(months), horizon, rng)
                    c, nt, ns = run_path(months, idx, f, K, heat, eq0=eq0, allow_mult=allow_mult, settle_lag=settle_lag)
                    c = c[: horizon * 21]
                    finals[i] = c[-1]
                    mdd[i] = (c / np.maximum.accumulate(c) - 1).min()
                    ruin[i] = (c <= eq0 * 0.5).any(); wipe[i] = (c <= eq0 * 0.1).any()
                    hit = np.where(c >= 2 * eq0)[0]
                    if len(hit):
                        t2x[i] = hit[0] / 21.0
                    m1[i] = c[20]; m3[i] = c[62]; m6[i] = c[125]
                    taken += nt; skipped += ns
                g = np.log(finals / eq0)
                rows.append(dict(K=K, heat=heat, f=f, median_log_growth=np.median(g), mean_log_growth=g.mean(),
                                 p_ruin=ruin.mean(), p_wipe90=wipe.mean(), mdd_p50=np.median(mdd), mdd_p95=np.quantile(mdd, 0.05),
                                 m1_p10=np.quantile(m1, .1), m1_p50=np.median(m1), m1_p90=np.quantile(m1, .9),
                                 m3_p10=np.quantile(m3, .1), m3_p50=np.median(m3), m3_p90=np.quantile(m3, .9),
                                 m6_p10=np.quantile(m6, .1), m6_p50=np.median(m6), m6_p90=np.quantile(m6, .9),
                                 m12_p10=np.quantile(finals, .1), m12_p50=np.median(finals), m12_p90=np.quantile(finals, .9),
                                 p2x_3m=np.mean(t2x <= 3), p2x_6m=np.mean(t2x <= 6), p2x_12m=np.mean(t2x <= 12),
                                 t2x_median=np.nanmedian(t2x) if np.isfinite(t2x).any() else np.nan,
                                 trades_per_yr=taken / n_paths * 12 / horizon, skip_per_yr=skipped / n_paths * 12 / horizon))
                print(rows[-1]["K"], rows[-1]["heat"], rows[-1]["f"], round(rows[-1]["median_log_growth"], 3),
                      round(rows[-1]["p_ruin"], 3), round(rows[-1]["mdd_p95"], 3), flush=True)
    return pd.DataFrame(rows)
