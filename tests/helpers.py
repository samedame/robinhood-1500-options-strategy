"""Fakes for the signal script's three outside dependencies (Yahoo, Cboe, Discord) and its Eastern clock.

Nothing here touches the network. `SignalTestCase` points the script's data files at a temp directory, swaps in the
fakes, and gives every test a 3:46 pm ET Wednesday 2026-10-14 clock (a real trading day; the next listed weekly expiry
that is at least 10 days out is Friday 2026-10-30).
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deliver"))
import ibs_dip_calls_signal as sig            # noqa: E402

ET = ZoneInfo("America/New_York")
TODAY = dt.date(2026, 10, 14)
EXPIRY = dt.date(2026, 10, 30)


def et(y, m, d, hh, mm, ss=0):
    return dt.datetime(y, m, d, hh, mm, ss, tzinfo=ET)


NOW = et(2026, 10, 14, 15, 46)


def minute_bars(day=TODAY, high=100.0, low=90.0, last=95.0, last_minute="15:45", multiindex=True):
    """A regular-session 1-minute frame from 09:30 to `last_minute` whose max High, min Low and final Close are given."""
    idx = pd.date_range(f"{day:%Y-%m-%d} 09:30", f"{day:%Y-%m-%d} {last_minute}", freq="1min", tz=ET)
    n = len(idx)
    mid = (high + low) / 2
    df = pd.DataFrame({"Open": mid, "High": mid + 0.1, "Low": mid - 0.1, "Close": mid, "Volume": 1000}, index=idx)
    df.iloc[5, df.columns.get_loc("High")] = high
    df.iloc[10, df.columns.get_loc("Low")] = low
    df.iloc[n - 1, df.columns.get_loc("Close")] = last
    df.iloc[n - 1, df.columns.get_loc("High")] = max(last, df.iloc[n - 1]["High"])
    df.iloc[n - 1, df.columns.get_loc("Low")] = min(last, df.iloc[n - 1]["Low"])
    if multiindex:                            # recent yfinance returns (Price, Ticker) columns even for one ticker
        df.columns = pd.MultiIndex.from_product([df.columns, ["X"]])
    return df


def daily_bars(prior_high, today_high=None, day=TODAY, multiindex=True):
    """Daily frame: a few earlier sessions, yesterday (`prior_high`), and today's still-forming bar."""
    days = [dt.date(2026, 10, 9), dt.date(2026, 10, 12), dt.date(2026, 10, 13), day]
    highs = [prior_high - 3, prior_high - 1, prior_high, today_high if today_high is not None else prior_high + 50]
    df = pd.DataFrame({"High": highs, "Low": [h - 5 for h in highs], "Close": [h - 2 for h in highs]},
                      index=pd.to_datetime(days))
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["X"]])
    return df


class FakeYF:
    """Stands in for the yfinance module: per-symbol frames, or an Exception to raise."""

    def __init__(self):
        self.minute, self.daily, self.requests = {}, {}, []

    def set(self, sym, minute=None, daily=None):
        if minute is not None:
            self.minute[sym] = minute
        if daily is not None:
            self.daily[sym] = daily

    def download(self, sym, period=None, interval=None, **kw):
        self.requests.append((sym, period, interval))
        src = self.minute if interval == "1m" else self.daily
        v = src[sym]
        if isinstance(v, Exception):
            raise v
        return v.copy()


def chain(rows, expiry=EXPIRY, spot=270.0, stamp="2026-10-14 15:40:00"):
    """Fake Cboe call chain. rows = (strike, delta, bid, ask, open_interest)."""
    df = pd.DataFrame(rows, columns=["strike", "delta", "bid", "ask", "open_interest"])
    df["expiry"] = expiry
    df["cp"] = "C"
    return df, (stamp, spot)


# A tidy IWM-like chain: 0.50 delta costs $2.60 (too dear for $1,500), 0.40 costs $2.05, 0.30 costs $1.60 (affordable).
AFFORDABLE_030 = [
    (270.0, 0.51, 2.50, 2.60, 3000),
    (273.0, 0.41, 1.98, 2.05, 2500),
    (276.0, 0.31, 1.55, 1.60, 2000),
    (279.0, 0.21, 0.90, 0.94, 1500),
]


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.text = status, body or {}, str(body or "")

    def json(self):
        return self._body


class SignalTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.pos = os.path.join(self.tmp, "positions.csv")
        self.log = os.path.join(self.tmp, "signals_log.csv")
        self.notified = os.path.join(self.tmp, ".notified")
        for name, value in (("POS_FILE", self.pos), ("LOG_FILE", self.log), ("NOTIFIED_FILE", self.notified)):
            p = mock.patch.object(sig, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.yf = FakeYF()
        p = mock.patch.object(sig, "yf", self.yf)
        p.start()
        self.addCleanup(p.stop)
        self.chains = {}                      # symbol -> (df, meta) or Exception
        p = mock.patch.object(sig, "cboe_chain", side_effect=self._cboe)
        p.start()
        self.addCleanup(p.stop)
        self.now = NOW
        for s in sig.ETFS:                    # every ETF quiet (IBS 0.5) unless a test says otherwise
            self.etf(s, high=100.0, low=90.0, last=95.0, prior_high=101.0)

    def _cboe(self, sym):
        v = self.chains.get(sym)
        if v is None:
            raise AssertionError(f"test did not set a chain for {sym}")
        if isinstance(v, Exception):
            raise v
        return v

    def etf(self, sym, high, low, last, prior_high, last_minute="15:45", day=TODAY):
        self.yf.set(sym, minute=minute_bars(day, high, low, last, last_minute), daily=daily_bars(prior_high))

    def positions(self, *lines):
        with open(self.pos, "w", newline="") as f:
            f.write("etf,expiry,strike,contracts,entry_date,entry_price\n")
            for ln in lines:
                f.write(ln + "\n")

    def run_report(self, equity=1500, settled=1500):
        lines, send, degraded = sig.run(equity, settled, now=self.now)
        return "\n".join(lines), send, degraded
