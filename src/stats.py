"""Statistics used for selection, gates and reporting."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats as st
from scipy.optimize import minimize_scalar

def kelly_fraction(r, fmax=0.99):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) == 0 or r.mean() <= 0:
        return 0.0
    lo = -1.0 / r.max() if r.max() > 0 else fmax
    hi = min(fmax, (-1.0 / r.min()) * 0.999) if r.min() < 0 else fmax
    f = minimize_scalar(lambda f: -np.mean(np.log1p(f * r)), bounds=(0.0, hi), method="bounded").x
    return float(max(f, 0.0))

def growth_score(r, years, cap=0.5, mult=0.5):
    """Trades/yr x mean log(1 + f r) at capped fractional Kelly. Returns (score, f_used, f_kelly)."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) == 0 or years <= 0:
        return np.nan, 0.0, 0.0
    fk = kelly_fraction(r)
    f = min(mult * fk, cap)
    g = np.mean(np.log1p(f * r)) if f > 0 else 0.0
    return float(len(r) / years * g), f, fk

def stationary_bootstrap_idx(n, mean_block, rng):
    idx = np.empty(n, dtype=int)
    i = rng.integers(n)
    p = 1.0 / mean_block
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = rng.integers(n)
        else:
            i = (i + 1) % n
    return idx

def boot_mean_ci(x, B=5000, mean_block=5, alpha=0.05, seed=7):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan
    ms = np.empty(B)
    for b in range(B):
        ms[b] = x[stationary_bootstrap_idx(n, mean_block, rng)].mean()
    lo, hi = np.quantile(ms, [alpha / 2, 1 - alpha / 2])
    p_le0 = (ms <= 0).mean()
    return float(lo), float(hi), float(p_le0)

def nw_tstat(x, lags=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    m = x.mean(); e = x - m
    s = np.dot(e, e) / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * np.dot(e[L:], e[:-L]) / n
    return float(m / np.sqrt(s / n))

def psr(sr, n, skew, kurt, sr0=0.0):
    """Probabilistic Sharpe ratio (per-observation SR). kurt = raw kurtosis (normal=3)."""
    den = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(st.norm.cdf((sr - sr0) * np.sqrt(n - 1) / den))

def dsr(returns, n_trials, sr_var_trials):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado). sr_var_trials: variance of per-obs SR across trials."""
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g = 0.5772156649
    emax = np.sqrt(sr_var_trials) * ((1 - g) * st.norm.ppf(1 - 1 / n_trials) + g * st.norm.ppf(1 - 1 / (n_trials * np.e)))
    return psr(sr, n, st.skew(r), st.kurtosis(r, fisher=False), sr0=emax), float(sr), float(emax)

def summarize(df, years, col="ret"):
    r = df[col].values
    sc, f, fk = growth_score(r, years)
    lo, hi, p = boot_mean_ci(r)
    wins = (r > 0).mean() if len(r) else np.nan
    return dict(n=len(r), per_yr=len(r) / years if years else np.nan, mean=float(np.mean(r)) if len(r) else np.nan,
                median=float(np.median(r)) if len(r) else np.nan, sd=float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
                win=wins, ci_lo=lo, ci_hi=hi, p_boot=p, nw_t=nw_tstat(r), kelly=fk, f_half=f, score=sc,
                worst=float(np.min(r)) if len(r) else np.nan, best=float(np.max(r)) if len(r) else np.nan)
