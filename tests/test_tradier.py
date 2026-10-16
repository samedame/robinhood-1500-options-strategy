"""Tradier as the first data source for the IBS Dip Calls signal. Offline: tradier's network functions are patched."""
import datetime as dt
import os
import unittest
from unittest import mock

import pandas as pd

from tests.helpers import (AFFORDABLE_030, EXPIRY, NOW, ROOT, TODAY, SignalTestCase, chain, et, sig)

import tradier

KEY = {"TRADIER_API_KEY": "SECRET-KEY-123", "TRADIER_BASE_URL": ""}
SANDBOX = {"TRADIER_API_KEY": "SECRET-KEY-123", "TRADIER_BASE_URL": "https://sandbox.tradier.com/v1"}


def bars(high=100.0, low=90.0, last=95.0, last_minute="15:45", day=TODAY):
    """Tradier timesales rows: 09:30..last_minute, with the given session high / low and final close."""
    idx = pd.date_range(f"{day:%Y-%m-%d} 09:30", f"{day:%Y-%m-%d} {last_minute}", freq="1min")
    mid = (high + low) / 2
    rows = [{"time": t.strftime("%Y-%m-%dT%H:%M:%S"), "open": mid, "high": mid + 0.1, "low": mid - 0.1, "close": mid}
            for t in idx]
    rows[5]["high"], rows[10]["low"], rows[-1]["close"] = high, low, last
    return rows


def daily(prior_high=101.0):
    idx = pd.to_datetime([dt.date(2026, 10, 12), dt.date(2026, 10, 13), TODAY])
    return pd.DataFrame({"open": 1.0, "high": [prior_high - 1, prior_high, prior_high + 50], "low": 1.0, "close": 1.0},
                        index=idx)


def ms(y, m, d, hh, mm):
    return int(et(y, m, d, hh, mm).timestamp() * 1000)


def opt(strike, delta, bid, ask, oi, kind="call", stamp=None):
    return {"symbol": f"IWM261030{kind[0].upper()}{int(strike * 1000):08d}", "strike": strike, "option_type": kind,
            "bid": bid, "ask": ask, "open_interest": oi, "bid_date": stamp, "ask_date": stamp,
            "greeks": {"delta": delta} if delta is not None else None}


def tchain(stamp=None):
    stamp = stamp or ms(2026, 10, 14, 15, 44)
    rows = [opt(k, d, b, a, oi, stamp=stamp) for k, d, b, a, oi in AFFORDABLE_030]
    rows.append(opt(276.0, -0.69, 3.0, 3.1, 900, kind="put", stamp=stamp))
    return rows


def tradier_data(**kw):
    ts = mock.Mock(return_value=kw.get("bars", bars()))
    return mock.patch.multiple(tradier, timesales=ts, daily_bars=mock.Mock(return_value=kw.get("daily", daily())),
                               expirations=mock.Mock(return_value=["2026-10-16", "2026-10-23", "2026-10-30"]),
                               chain=mock.Mock(return_value=kw.get("chain", tchain())),
                               quotes=mock.Mock(return_value={"IWM": {"last": 268.2}}))


def env(d):
    return mock.patch.dict(os.environ, d)


class Intraday(SignalTestCase):
    def test_production_key_uses_tradier_bars_and_never_calls_yahoo(self):
        self.yf.download = mock.Mock(side_effect=AssertionError("Yahoo must not be called"))
        with env(KEY), tradier_data(bars=bars(100, 90, 91)):
            st = sig.intraday_state("SPY", now=NOW)
        self.assertEqual((st["last"], st["high"], st["low"], st["prior_high"]), (91.0, 100.0, 90.0, 101.0))
        self.assertAlmostEqual(st["ibs"], 0.1)
        self.assertEqual(st["source"], "Tradier")
        self.assertEqual(st["age_min"], 1.0)
        self.assertEqual(st["bar_time"][11:16], "15:45")

    def test_ignores_yesterdays_daily_bar_for_prior_high(self):
        with env(KEY), tradier_data(daily=daily(prior_high=123.0)):
            self.assertEqual(sig.intraday_state("SPY", now=NOW)["prior_high"], 123.0)

    def test_tradier_failure_falls_back_to_yahoo(self):
        with env(KEY), mock.patch.object(tradier, "timesales", side_effect=tradier.TradierError("HTTP 401")):
            st = sig.intraday_state("SPY", now=NOW)
        self.assertNotIn("source", st)                       # the Yahoo path is untouched
        self.assertEqual(st["last"], 95.0)

    def test_bars_from_a_previous_day_are_refused_and_yahoo_is_used(self):
        old = bars(day=dt.date(2026, 10, 13))
        with env(KEY), tradier_data(bars=old):
            st = sig.intraday_state("SPY", now=NOW)
        self.assertNotIn("source", st)

    def test_both_feeds_down_reports_both(self):
        self.yf.set("SPY", minute=ConnectionError("yahoo down"))
        with env(KEY), mock.patch.object(tradier, "timesales", side_effect=tradier.TradierError("HTTP 500")):
            with self.assertRaises(sig.DataProblem) as cm:
                sig.intraday_state("SPY", now=NOW)
        msg = str(cm.exception)
        self.assertIn("Tradier:", msg)
        self.assertIn("Yahoo:", msg)
        self.assertNotIn("SECRET-KEY-123", msg)

    def test_sandbox_key_keeps_yahoo_first(self):
        with env(SANDBOX), mock.patch.object(tradier, "timesales", side_effect=AssertionError("not first")):
            st = sig.intraday_state("SPY", now=NOW)
        self.assertEqual(st["last"], 95.0)

    def test_sandbox_is_the_last_resort_and_a_stale_reading_is_flagged(self):
        self.yf.set("SPY", minute=ConnectionError("yahoo down"))
        with env(SANDBOX), tradier_data(bars=bars(last_minute="15:30")):
            st = sig.intraday_state("SPY", now=NOW)
        self.assertEqual(st["source"], "Tradier sandbox (delayed)")
        self.assertGreater(st["age_min"], sig.STALE_BAR_MIN)

    def test_a_bad_base_url_does_not_silence_the_bot(self):
        with env({"TRADIER_API_KEY": "k", "TRADIER_BASE_URL": "https://evil.example/v1"}):
            self.assertEqual(sig.intraday_state("SPY", now=NOW)["last"], 95.0)

    def test_no_key_never_touches_tradier(self):
        with mock.patch.object(tradier, "timesales", side_effect=AssertionError("no key")):
            self.assertEqual(sig.intraday_state("SPY", now=NOW)["last"], 95.0)


class Chain(SignalTestCase):
    def test_tradier_chain_is_shaped_like_cboes(self):
        with tradier_data():
            calls, (stamp, spot) = sig.tradier_chain("IWM", TODAY)
        self.assertEqual(sorted(calls.strike), [270.0, 273.0, 276.0, 279.0])      # the put is dropped
        self.assertEqual(set(calls.expiry), {EXPIRY})
        self.assertEqual((stamp, spot), ("2026-10-14 15:44:00", 268.2))
        self.assertEqual(float(calls[calls.strike == 276.0].delta.iloc[0]), 0.31)

    def test_pick_contract_uses_tradier_first_and_names_the_source(self):
        self.chains["IWM"] = AssertionError("Cboe must not be called")
        with env(KEY), tradier_data():
            got, why = sig.pick_contract("IWM", 1500, 1500, TODAY, 268.2)
        self.assertIsNone(why)
        self.assertEqual((got["expiry"], got["strike"], got["rung"], got["source"]), ("2026-10-30", 276.0, 0.30, "Tradier"))

    def test_the_same_contract_as_cboe_for_the_same_data(self):
        self.chains["IWM"] = chain(AFFORDABLE_030)
        want, _ = sig.pick_contract("IWM", 1500, 1500, TODAY, 268.2)
        with env(KEY), tradier_data():
            got, _ = sig.pick_contract("IWM", 1500, 1500, TODAY, 268.2)
        for k in ("expiry", "strike", "rung", "delta", "bid", "ask", "contracts", "cost_at_ask"):
            self.assertEqual(got[k], want[k], k)

    def test_contracts_without_greeks_are_skipped_not_fatal(self):
        rows = tchain() + [opt(281.0, None, 0.5, 0.6, 900)]
        with tradier_data(chain=rows):
            calls, _ = sig.tradier_chain("IWM", TODAY)
        self.assertNotIn(281.0, list(calls.strike))

    def test_tradier_failure_falls_back_to_cboe(self):
        self.chains["IWM"] = chain(AFFORDABLE_030)
        with env(KEY), mock.patch.object(tradier, "expirations", side_effect=tradier.TradierError("HTTP 429")):
            got, why = sig.pick_contract("IWM", 1500, 1500, TODAY, 268.2)
        self.assertNotIn("source", got)
        self.assertEqual(got["strike"], 276.0)

    def test_sandbox_key_uses_cboe_first(self):
        self.chains["IWM"] = chain(AFFORDABLE_030)
        with env(SANDBOX), mock.patch.object(tradier, "expirations", side_effect=AssertionError("not first")):
            got, _ = sig.pick_contract("IWM", 1500, 1500, TODAY, 268.2)
        self.assertEqual(got["strike"], 276.0)

    def test_both_chains_down_raises_and_the_report_says_so(self):
        self.etf("IWM", high=100.0, low=90.0, last=90.5, prior_high=101.0)
        self.chains["IWM"] = ValueError("cboe down")
        with env(KEY), mock.patch.object(tradier, "expirations", side_effect=tradier.TradierError("HTTP 500")):
            text, send, degraded = self.run_report()
        self.assertTrue(degraded)
        self.assertIn("Tradier/Cboe option chain could not be loaded", text)

    def test_the_entry_line_names_tradier(self):
        def by_symbol(sym, *a, **k):
            return bars(100, 90, 90.5) if sym == "IWM" else bars()
        with env(KEY), tradier_data(), mock.patch.object(tradier, "timesales", side_effect=by_symbol):
            text, send, degraded = self.run_report()
        self.assertIn("ENTRY: BUY TO OPEN 1 IWM 2026-10-30 276.0C (~0.30 delta, Tradier 1.55/1.6 at 2026-10-14 15:44:00).", text)


class Wiring(unittest.TestCase):
    def test_workflow_passes_the_key(self):
        with open(os.path.join(ROOT, ".github", "workflows", "daily-signal.yml")) as f:
            text = f.read()
        self.assertIn("TRADIER_API_KEY: ${{ secrets.TRADIER_API_KEY }}", text)
        self.assertIn("TRADIER_BASE_URL: ${{ secrets.TRADIER_BASE_URL }}", text)


if __name__ == "__main__":
    unittest.main()
