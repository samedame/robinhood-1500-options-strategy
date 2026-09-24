"""
Proxy option pricer for underlyings without chain history (QQQ before 2021-03, IWM, DIA).
Calibrated ONLY on SPY in-sample chains (2010-2021):
  ATM IV(T) = beta(T) x volindex-term(T); smile IV/ATM by standardized moneyness z and DTE; relative spread by
  |delta| and DTE, times an underlying multiplier from the 2026-09-22 live snapshot.
Expiry calendar: the expiries listed for SPY on the same date (QQQ/IWM listings closely track SPY's).
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from core import load_cboe, fee_per_contract, fill

R = "/home/claude/research"
SMILE_Z = np.array([-3.5, -2.75, -2.25, -1.75, -1.25, -0.75, -0.375, 0.0, 0.375, 0.75, 1.25, 1.75, 2.25, 2.75, 3.5])
SMILE_T = [(3, None), (7, None), (14, None), (21, None), (35, None), (60, None), (80, None)]
SMILE = np.array([
    [1.700, 1.831, 1.950, 2.034, 2.075, 2.111, 2.125],
    [1.516, 1.614, 1.720, 1.795, 1.836, 1.861, 1.867],
    [1.395, 1.477, 1.574, 1.638, 1.674, 1.696, 1.697],
    [1.294, 1.357, 1.433, 1.488, 1.519, 1.539, 1.541],
    [1.201, 1.247, 1.304, 1.345, 1.370, 1.384, 1.391],
    [1.114, 1.144, 1.181, 1.205, 1.224, 1.233, 1.239],
    [1.054, 1.070, 1.089, 1.103, 1.113, 1.118, 1.120],
    [0.999, 0.999, 1.000, 1.000, 0.999, 0.999, 0.999],
    [0.948, 0.929, 0.915, 0.901, 0.891, 0.887, 0.884],
    [0.907, 0.877, 0.850, 0.827, 0.808, 0.805, 0.803],
    [0.890, 0.855, 0.825, 0.805, 0.789, 0.785, 0.783],
    [0.928, 0.909, 0.899, 0.886, 0.878, 0.872, 0.854],
    [1.021, 1.028, 1.034, 1.022, 1.010, 1.032, 1.024],
    [1.193, 1.219, 1.233, 1.216, 1.194, 1.288, 1.270],
    [1.641, 1.711, 1.706, 1.663, 1.726, 1.732, 1.609]])
T_EDGES = [3, 7, 14, 21, 35, 60, 80]
BETA_T = [(3, 0.72), (7, 0.76), (80, 0.80)]
SPREAD_D = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
SPREAD = np.array([   # rows |delta| buckets, cols DTE buckets as T_EDGES (SPY calls 2016-2021 medians)
    [0.400, 0.400, 0.286, 0.222, 0.182, 0.143, 0.105],
    [0.067, 0.047, 0.039, 0.036, 0.038, 0.028, 0.020],
    [0.038, 0.029, 0.025, 0.023, 0.024, 0.018, 0.013],
    [0.026, 0.021, 0.018, 0.017, 0.017, 0.013, 0.010],
    [0.021, 0.016, 0.013, 0.012, 0.012, 0.009, 0.006],
    [0.024, 0.016, 0.014, 0.012, 0.012, 0.010, 0.007],
    [0.025, 0.018, 0.015, 0.013, 0.013, 0.010, 0.008],
    [0.030, 0.021, 0.016, 0.014, 0.013, 0.010, 0.009],
    [0.014, 0.010, 0.009, 0.009, 0.008, 0.007, 0.007]])
CORR = np.array([   # real/proxy median price ratio on SPY IS calls, rows |delta| buckets (SPREAD_D), cols DTE buckets
    [2.687, 1.313, 1.127, 1.088, 1.092, 1.002, 1.002],
    [2.687, 1.313, 1.127, 1.088, 1.092, 1.002, 1.002],
    [2.063, 1.272, 1.126, 1.085, 1.084, 1.020, 1.020],
    [1.695, 1.215, 1.095, 1.062, 1.068, 1.027, 1.027],
    [1.322, 1.114, 1.064, 1.043, 1.048, 1.024, 1.024],
    [1.163, 1.068, 1.041, 1.032, 1.034, 1.019, 1.019],
    [1.085, 1.041, 1.027, 1.022, 1.025, 1.017, 1.017],
    [1.035, 1.020, 1.015, 1.013, 1.015, 1.012, 1.012],
    [1.005, 1.005, 1.006, 1.006, 1.008, 1.008, 1.008]])
USE_CORR = True

def price_corr(absdelta, T):
    if not USE_CORR:
        return 1.0
    i = int(np.searchsorted(SPREAD_D, min(max(absdelta, 1e-6), 1.0)))
    i = min(i, len(SPREAD_D) - 1)
    return float(CORR[i, _tcol(T)])

UNDERLYING = {  # vol index, dividend yield, spread multiplier vs SPY (live snapshot 2026-09-22), strike step
    "SPY": ("VIX", 0.015, 1.0, 1.0),
    "QQQ": ("VXN", 0.007, 1.0, 1.0),
    "IWM": ("RVX", 0.013, 1.6, 1.0),
    "DIA": ("VXD", 0.018, 2.5, 1.0),
}

def _tcol(T):
    for i, e in enumerate(T_EDGES):
        if T <= e:
            return i
    return len(T_EDGES) - 1

def smile(z, T):
    col = SMILE[:, _tcol(T)]
    return float(np.interp(np.clip(z, SMILE_Z[0], SMILE_Z[-1]), SMILE_Z, col))

def beta(T):
    for e, b in BETA_T:
        if T <= e:
            return b
    return BETA_T[-1][1]

def rel_spread(absdelta, T, mult):
    i = int(np.searchsorted(SPREAD_D, min(max(absdelta, 1e-6), 1.0)))
    i = min(i, len(SPREAD_D) - 1)
    return SPREAD[i, _tcol(T)] * mult

def bs(S, K, T, sig, r, q, cp):
    T = max(T, 0.5) / 365.0
    sig = max(sig, 0.01)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if cp == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), np.exp(-q * T) * norm.cdf(d1)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1), -np.exp(-q * T) * norm.cdf(-d1)

class VolTerm:
    def __init__(self, sym):
        idx = UNDERLYING[sym][0]
        self.v30 = load_cboe(idx)
        self.vix = load_cboe("VIX"); self.v9 = load_cboe("VIX9D"); self.v93 = load_cboe("VIX3M")
        irx = pd.read_csv("/home/claude/data/mkt/yf_IRX.csv", index_col=0, parse_dates=True)["Close"] / 100
        irx.index = pd.to_datetime(irx.index).normalize()
        self.r = irx

    def iv_term(self, d, T):
        b = self.v30.get(d, np.nan)
        if not np.isfinite(b):
            prev = self.v30[:d]
            b = prev.iloc[-1] if len(prev) else np.nan
        vx = self.vix.get(d, np.nan); a = self.v9.get(d, np.nan); c = self.v93.get(d, np.nan)
        ra = a / vx if np.isfinite(a) and np.isfinite(vx) else 0.95
        rc = c / vx if np.isfinite(c) and np.isfinite(vx) else 1.08
        pts = np.array([9, 30, 93.0]); vols = np.array([b * ra, b, b * rc]) / 100
        w = vols ** 2 * pts
        Tc = float(np.clip(T, 1, 93))
        wi = np.interp(Tc, pts, w) if Tc >= 9 else w[0] * Tc / 9
        return float(np.sqrt(wi / Tc))

    def rate(self, d):
        v = self.r.get(d, np.nan)
        if not np.isfinite(v):
            prev = self.r[:d]
            v = prev.iloc[-1] if len(prev) else 0.02
        return float(v)

class ProxyQuote:
    __slots__ = ("bid", "ask", "delta", "iv", "und")
    def __init__(self, bid, ask, delta, iv, und):
        self.bid, self.ask, self.delta, self.iv, self.und = bid, ask, delta, iv, und

class ProxyChains:
    """Duck-types core.Chains for the Engine: day(d) returns a DataFrame indexed by (expiry, cp, strike)."""
    def __init__(self, sym, ohlc, expiry_source):
        self.sym = sym
        self.px = ohlc
        self.vt = VolTerm(sym)
        self.q = UNDERLYING[sym][1]; self.mult = UNDERLYING[sym][2]; self.step = UNDERLYING[sym][3]
        self.exp_src = expiry_source      # dict date -> list of expiries (e.g. SPY's listed expiries)
        self._cache = {}

    def has(self, d):
        return d in self.exp_src and d in self.px.index

    def _price(self, d, e, cp, K):
        S = float(self.px.Close.loc[d]); T = max((e - d).days, 0.5)
        atm = self.vt.iv_term(d, T) * beta(T)
        r = self.vt.rate(d)
        F = S * np.exp((r - self.q) * T / 365)
        z = np.log(K / F) / (atm * np.sqrt(T / 365))
        iv = atm * smile(z, T)
        c_px, c_de = bs(S, K, T, iv, r, self.q, "C")
        # SPY-calibrated level correction: real/proxy call price ratio by call delta and DTE. By put-call parity the
        # same additive correction applies to the put at this strike.
        add = c_px * (price_corr(abs(c_de), T) - 1.0)
        if cp == "C":
            p, de = c_px + add, c_de
        else:
            p_px, de = bs(S, K, T, iv, r, self.q, "P")
            p = p_px + add
        rs = rel_spread(abs(de), T, self.mult)
        half = max(p * rs / 2, 0.005)
        bid = max(round(p - half, 2), 0.0); ask = round(p + half, 2)
        return bid, ask, de, iv, S

    def day(self, d):
        if d in self._cache:
            return self._cache[d]
        if not self.has(d):
            return None
        S = float(self.px.Close.loc[d]); r = self.vt.rate(d); frames = []
        for e in self.exp_src[d]:
            T = (e - d).days
            if T < 1 or T > 75:
                continue
            ks = np.arange(np.floor(S * 0.88), np.ceil(S * 1.12) + 1, self.step).astype(float)
            atm = self.vt.iv_term(d, T) * beta(T)
            F = S * np.exp((r - self.q) * T / 365)
            z = np.log(ks / F) / (atm * np.sqrt(T / 365))
            col = SMILE[:, _tcol(T)]
            iv = atm * np.interp(np.clip(z, SMILE_Z[0], SMILE_Z[-1]), SMILE_Z, col)
            Ty = T / 365.0
            d1 = (np.log(S / ks) + (r - self.q + 0.5 * iv * iv) * Ty) / (iv * np.sqrt(Ty)); d2 = d1 - iv * np.sqrt(Ty)
            cpx = S * np.exp(-self.q * Ty) * norm.cdf(d1) - ks * np.exp(-r * Ty) * norm.cdf(d2)
            cde = np.exp(-self.q * Ty) * norm.cdf(d1)
            ppx = ks * np.exp(-r * Ty) * norm.cdf(-d2) - S * np.exp(-self.q * Ty) * norm.cdf(-d1)
            pde = -np.exp(-self.q * Ty) * norm.cdf(-d1)
            corr = np.array([price_corr(a, T) for a in cde])
            add = cpx * (corr - 1.0)
            for cp, px_, de_ in (("C", cpx + add, cde), ("P", ppx + add, pde)):
                rs = np.array([rel_spread(abs(a), T, self.mult) for a in de_])
                half = np.maximum(px_ * rs / 2, 0.005)
                bid = np.maximum(np.round(px_ - half, 2), 0.0); ask = np.round(px_ + half, 2)
                frames.append(pd.DataFrame({"expiry": e, "cp": cp, "strike": ks, "bid": bid, "ask": ask,
                                            "mid": (bid + ask) / 2, "delta": de_, "iv": iv, "und": S}))
        g = pd.concat(frames, ignore_index=True).set_index(["expiry", "cp", "strike"]).sort_index()
        if len(self._cache) > 400:
            self._cache.clear()
        self._cache[d] = g
        return g

    def quote(self, d, expiry, cp, strike):
        if not self.has(d):
            return None
        bid, ask, de, iv, S = self._price(d, expiry, cp, strike)
        return ProxyQuote(bid, ask, de, iv, S)
