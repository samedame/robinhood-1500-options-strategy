"""Offline tests for deliver/ibs_dip_calls_signal.py: what the 3:45 pm Discord message says in every situation,
and that the frozen contract-selection rule still picks exactly what the original code picked."""
import contextlib
import csv
import io
import math
import os
import random
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import requests

from tests.helpers import (AFFORDABLE_030, EXPIRY, NOW, TODAY, FakeResponse, SignalTestCase, chain, daily_bars, et,
                           minute_bars, sig)

SIGNAL_IWM = dict(high=272.0, low=268.0, last=268.2, prior_high=274.0)      # IBS = 0.05
DEAR_CHAIN = [(270.0, 0.51, 3.90, 4.00, 3000), (273.0, 0.41, 3.10, 3.20, 2500),
              (276.0, 0.31, 2.45, 2.50, 2000), (279.0, 0.21, 1.50, 1.56, 1500)]


# --------------------------------------------------------------------------------------------------------------
class Readings(SignalTestCase):
    def test_quiet_day_lists_every_etf_and_says_no_action(self):
        text, send, degraded = self.run_report()
        self.assertTrue(send)
        self.assertFalse(degraded)
        for s in ("SPY", "QQQ", "IWM"):
            self.assertIn(f"  {s}: last 95.00 | high 100.00 low 90.00 | IBS 0.50 | prior high 101.00", text)
        self.assertIn("No action today (no ETF has IBS below 0.10 and you hold nothing).", text)
        self.assertTrue(text.startswith("IBS Dip Calls | 2026-10-14 15:46 ET | equity $1,500 | settled $1,500"))

    def test_signals_log_keeps_its_columns(self):
        self.run_report()
        with open(self.log, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), ["run_time", "etf", "last", "high", "low", "ibs", "prior_high", "bar_time"])
        self.assertEqual([r["etf"] for r in rows], ["SPY", "QQQ", "IWM"])

    def test_holiday_or_weekend_posts_nothing(self):
        self.now = et(2026, 11, 26, 15, 46)           # Thanksgiving
        text, send, degraded = self.run_report()
        self.assertFalse(send)
        self.assertIn("Market holiday or weekend", text)

    def test_outside_the_window_is_flagged(self):
        self.now = et(2026, 10, 14, 15, 59)
        text, _, _ = self.run_report()
        self.assertIn("WARNING: outside the 3:40-3:58 pm ET window", text)


class DataProblems(SignalTestCase):
    def test_one_etf_down_is_named_and_the_others_still_report(self):
        self.yf.set("QQQ", minute=ConnectionError("yahoo down"))
        text, send, degraded = self.run_report()
        self.assertTrue(degraded)
        self.assertIn("QQQ: NO READING", text)
        self.assertIn("QQQ was NOT checked today", text)
        self.assertIn("SPY: last 95.00", text)
        self.assertIn("IWM: last 95.00", text)
        self.assertIn("No action for the ETFs that were checked, but QQQ could NOT be checked", text)
        self.assertNotIn("No action today (", text)

    def test_everything_down_is_never_called_a_quiet_day(self):
        for s in sig.ETFS:
            self.yf.set(s, minute=pd.DataFrame())
        text, send, degraded = self.run_report()
        self.assertTrue(degraded)
        self.assertEqual(text.count("NO READING"), 3)
        self.assertIn("Nothing could be checked today, so this is NOT a 'no signal' day", text)

    def test_yesterdays_bars_are_refused_not_used(self):
        self.yf.set("SPY", minute=minute_bars(day=et(2026, 10, 13, 12, 0).date(), high=100, low=90, last=90.5))
        text, _, degraded = self.run_report()
        self.assertTrue(degraded)
        self.assertIn("SPY: NO READING (Yahoo's newest 1-minute bar is from Tue 10-13, not today)", text)
        self.assertNotIn("SPY signalled", text)

    def test_stale_bar_is_warned_about(self):
        self.etf("SPY", high=100, low=90, last=95, prior_high=101, last_minute="15:36")      # 10 min old at 15:46
        text, _, degraded = self.run_report()
        self.assertFalse(degraded)
        self.assertIn("WARNING: SPY's newest Yahoo bar is 10 min old (15:36)", text)
        self.assertNotIn("QQQ's newest Yahoo bar", text)

    def test_stale_bar_on_a_signal_adds_a_caution_to_the_ticket(self):
        self.etf("IWM", **SIGNAL_IWM, last_minute="15:36")
        self.chains["IWM"] = chain(AFFORDABLE_030, spot=268.2)
        text, _, _ = self.run_report()
        self.assertIn("ENTRY: BUY TO OPEN", text)
        self.assertIn("CAUTION: the IBS reading came from a 10-minute-old Yahoo bar; confirm IBS < 0.10 in the app first.", text)

    def test_no_daily_history_is_a_data_problem(self):
        self.yf.set("IWM", daily=daily_bars(101.0).iloc[:1])
        text, _, degraded = self.run_report()
        self.assertTrue(degraded)
        self.assertIn("IWM: NO READING (Yahoo returned no daily history)", text)

    def test_naive_clock_does_not_crash_the_age_calculation(self):
        st = sig.intraday_state("SPY", now=NOW.replace(tzinfo=None))
        self.assertEqual(st["age_min"], 1.0)


class IntradayState(SignalTestCase):
    def test_numbers_match_the_bars(self):
        self.etf("SPY", high=100, low=90, last=91, prior_high=101)
        st = sig.intraday_state("SPY", now=NOW)
        self.assertAlmostEqual(st["ibs"], 0.1)
        self.assertEqual((st["last"], st["high"], st["low"], st["prior_high"]), (91.0, 100.0, 90.0, 101.0))
        self.assertEqual(st["age_min"], 1.0)           # bar 15:45, clock 15:46

    def test_todays_forming_daily_bar_is_not_the_prior_high(self):
        self.yf.set("SPY", daily=daily_bars(101.0, today_high=999.0))
        self.assertEqual(sig.intraday_state("SPY", now=NOW)["prior_high"], 101.0)

    def test_flat_columns_work_too(self):
        self.yf.set("SPY", minute=minute_bars(high=100, low=90, last=95, multiindex=False),
                    daily=daily_bars(101.0, multiindex=False))
        self.assertAlmostEqual(sig.intraday_state("SPY", now=NOW)["ibs"], 0.5)

    def test_flat_day_range_gives_a_neutral_ibs(self):
        flat = minute_bars(high=100, low=100, last=100)
        flat[("High", "X")] = 100.0
        flat[("Low", "X")] = 100.0
        self.yf.set("SPY", minute=flat)
        self.assertEqual(sig.intraday_state("SPY", now=NOW)["ibs"], 0.5)

    def test_empty_minute_frame(self):
        self.yf.set("SPY", minute=pd.DataFrame())
        with self.assertRaises(sig.DataProblem):
            sig.intraday_state("SPY", now=NOW)


# --------------------------------------------------------------------------------------------------------------
class Entries(SignalTestCase):
    def setUp(self):
        super().setUp()
        self.etf("IWM", **SIGNAL_IWM)
        self.chains["IWM"] = chain(AFFORDABLE_030, spot=268.2)

    def test_affordable_signal_gives_a_ticket_and_the_exact_positions_line(self):
        text, send, degraded = self.run_report()
        self.assertFalse(degraded)
        self.assertIn("ENTRY: BUY TO OPEN 1 IWM 2026-10-30 276.0C (~0.30 delta, Cboe 1.55/1.6 at 2026-10-14 15:40:00).", text)
        self.assertIn("Limit at the LIVE mid; walk $0.01 every 30-60 s; never above mid + 25% of the live spread", text)
        self.assertIn("Cost at ask $160.0 = 10.7% of equity.", text)
        self.assertIn("add this line to deliver/positions.csv", text)
        self.assertIn("IWM,2026-10-30,276.0,1,2026-10-14,FILL_PRICE", text)
        self.assertIn("Not filled by 3:58 pm? Cancel the order; there is nothing to record.", text)
        self.assertIn("no line means no EXIT messages for this trade", text)
        self.assertNotIn("No action today", text)

    def test_the_line_in_the_message_is_a_valid_positions_row(self):
        text, _, _ = self.run_report()
        line = [t for t in text.split() if t.startswith("IWM,2026-10-30")][0]
        self.positions(line.replace("FILL_PRICE", "1.62"))
        row = sig.read_positions()[0]
        self.assertEqual((row["etf"], row["expiry"], float(row["strike"]), int(row["contracts"]), row["entry_date"],
                          float(row["entry_price"])), ("IWM", "2026-10-30", 276.0, 1, "2026-10-14", 1.62))

    def test_stale_cboe_chain_is_called_out(self):
        self.chains["IWM"] = chain(AFFORDABLE_030, spot=275.0)          # chain's spot is 2.5% away from the live price
        text, _, _ = self.run_report()
        self.assertIn("CAUTION: Cboe chain is stale (its spot 275.00 vs live 268.20)", text)

    def test_skip_names_every_rung_and_what_blocked_it(self):
        self.chains["IWM"] = chain(DEAR_CHAIN, spot=268.2)
        text, _, degraded = self.run_report()
        self.assertFalse(degraded)
        self.assertIn("IWM signalled (IBS 0.05) but skipped: no call passed the rules (cost at most $180 per contract = "
                      "12% of equity, bid-ask spread <= 4%, open interest >= 500). Checked:", text)
        self.assertIn("0.50-delta 270C Fri 10-30: costs $400 at the ask", text)
        self.assertIn("0.40-delta 273C Fri 10-30: costs $320 at the ask", text)
        self.assertIn("0.30-delta 276C Fri 10-30: costs $250 at the ask", text)
        self.assertNotIn("ENTRY", text)

    def test_skip_reasons_cover_delta_spread_open_interest_and_cash(self):
        rows = [(270.0, 0.80, 1.50, 1.55, 3000),        # nothing near 0.50 -> nearest delta
                (273.0, 0.41, 1.00, 1.20, 3000),        # 18% spread
                (276.0, 0.31, 1.55, 1.60, 120)]         # thin open interest
        self.chains["IWM"] = chain(rows, spot=268.2)
        text, _, _ = self.run_report()
        self.assertIn("0.50-delta 273C Fri 10-30: nearest delta is 0.41; bid-ask spread 18.2% is over 4%", text)
        self.assertIn("0.40-delta 273C Fri 10-30: bid-ask spread 18.2% is over 4%", text)
        self.assertIn("0.30-delta 276C Fri 10-30: open interest 120 is under 500", text)
        self.chains["IWM"] = chain(AFFORDABLE_030, spot=268.2)     # affordable by the rule, but not with only $100 settled
        text, _, _ = self.run_report(settled=100)
        self.assertIn("needs $160 of settled cash, you have $100", text)

    def test_cboe_outage_on_a_signal_tells_you_what_to_do_by_hand(self):
        self.chains["IWM"] = ValueError("Expecting value: line 1 column 1 (char 0)")
        text, _, degraded = self.run_report()
        self.assertTrue(degraded)
        self.assertIn("IWM signalled (IBS 0.05) but the Cboe option chain could not be loaded (ValueError)", text)
        self.assertIn("costs <= $180 per contract with a spread <= 4% and open interest >= 500", text)
        self.assertIn("before 3:58 pm", text)

    def test_signal_on_an_etf_you_already_hold_is_explained(self):
        self.positions("IWM,2026-10-30,276.0,1,2026-10-12,1.50")
        text, _, _ = self.run_report()
        self.assertIn("IWM signalled (IBS 0.05) but you already hold IWM: ignored (one position per ETF).", text)

    def test_no_free_slot_says_what_is_using_it(self):
        self.etf("QQQ", high=100, low=90, last=95, prior_high=101)
        self.positions("QQQ,2026-10-30,500.0,1,2026-10-12,1.50")
        text, _, _ = self.run_report()
        self.assertIn("Signals present (IWM) but no free slot: K = 1 and positions.csv has 1 line(s)", text)
        self.assertNotIn("BUY TO OPEN", text)

    def test_two_signals_one_slot_takes_the_higher_rung_then_evidence_order(self):
        self.etf("SPY", high=700.0, low=690.0, last=690.4, prior_high=705.0)
        spy_rows = [(690.0, 0.50, 4.00, 4.08, 5000), (700.0, 0.40, 2.60, 2.66, 5000), (710.0, 0.30, 1.50, 1.55, 5000)]
        self.chains["SPY"] = chain(spy_rows, spot=690.4)        # only the 0.30 rung is affordable ($155)
        text, _, _ = self.run_report()                           # IWM also qualifies at 0.30 -> SPY wins on evidence order
        self.assertEqual(text.count("BUY TO OPEN"), 1)
        self.assertIn("BUY TO OPEN 1 SPY", text)

    def test_bigger_account_gets_two_slots_and_more_contracts(self):
        self.etf("SPY", high=700.0, low=690.0, last=690.4, prior_high=705.0)
        self.chains["SPY"] = chain([(690.0, 0.50, 4.00, 4.08, 5000)], spot=690.4)
        text, _, _ = self.run_report(equity=4000, settled=4000)  # K = 2; budget $320 -> 0.50-delta at $408 -> 1 contract (<= $480)
        self.assertEqual(text.count("BUY TO OPEN"), 2)
        self.assertIn("BUY TO OPEN 1 SPY 2026-10-30 690.0C (~0.50 delta", text)


# --------------------------------------------------------------------------------------------------------------
class Exits(SignalTestCase):
    def test_bounce_exit_and_the_delete_the_line_instruction(self):
        self.etf("QQQ", high=500.0, low=495.0, last=499.0, prior_high=498.0)       # last above yesterday's high
        self.positions("QQQ,2026-10-30,500.0,2,2026-10-12,1.50")
        text, _, degraded = self.run_report()
        self.assertFalse(degraded)
        self.assertIn("EXIT (bounce): SELL TO CLOSE 2 QQQ 2026-10-30 500.0C, limit at mid, walk toward bid; at 3:58 pm sell at the bid.", text)
        self.assertIn("After you sell, delete that position's line from deliver/positions.csv", text)

    def test_time_exit_on_the_last_session_before_expiry(self):
        self.now = et(2026, 10, 29, 15, 46)                                       # Thursday; expiry Fri 10-30
        for s in sig.ETFS:
            self.etf(s, high=100, low=90, last=95, prior_high=101, day=self.now.date())
        self.positions("SPY,2026-10-30,700.0,1,2026-10-20,3.00")
        text, _, _ = self.run_report()
        self.assertIn("EXIT (time): SELL TO CLOSE 1 SPY 2026-10-30 700.0C today; last session before expiry.", text)
        self.assertIn("delete that position's line", text)

    def test_hold_shows_the_numbers_it_compared(self):
        self.positions("SPY,2026-10-30,700.0,1,2026-10-12,3.00")
        text, _, _ = self.run_report()
        self.assertIn("HOLD SPY 2026-10-30 700.0C (price 95.0 not above prior high 101.0).", text)
        self.assertNotIn("delete that position's line", text)

    def test_exit_needs_price_strictly_above_the_prior_high(self):
        self.positions("QQQ,2026-10-30,500.0,1,2026-10-12,1.50")
        self.etf("QQQ", high=500.0, low=495.0, last=498.0, prior_high=498.0)          # equal -> still a hold
        text, _, _ = self.run_report()
        self.assertIn("HOLD QQQ 2026-10-30 500.0C (price 498.0 not above prior high 498.0)", text)
        self.etf("QQQ", high=500.0, low=495.0, last=498.01, prior_high=498.0)         # a cent above -> exit
        text, _, _ = self.run_report()
        self.assertIn("EXIT (bounce): SELL TO CLOSE 1 QQQ", text)

    def test_position_entered_today_is_not_exited_today(self):
        self.etf("SPY", high=700.0, low=690.0, last=699.0, prior_high=695.0)       # would be a 'bounce' if checked
        self.positions("SPY,2026-10-30,700.0,1,2026-10-14,3.00")
        text, _, _ = self.run_report()
        self.assertIn("HOLD SPY 2026-10-30 700.0C (entered today; the first exit check is the next session).", text)
        self.assertNotIn("EXIT", text)

    def test_expired_line_is_flagged_and_does_not_pretend_to_hold(self):
        self.positions("SPY,2026-10-09,700.0,1,2026-09-28,3.00")
        text, _, _ = self.run_report()
        self.assertIn("STALE LINE in positions.csv: 1 SPY 2026-10-09 700.0C expired on 2026-10-09.", text)
        self.assertIn("delete the line", text)
        self.assertNotIn("HOLD SPY", text)

    def test_malformed_line_is_flagged_without_failing_the_run(self):
        self.positions("SPY,not-a-date,700.0,1,2026-09-28,3.00", "QQQ,2026-10-30", "DIA,2026-10-30,400.0,1,2026-10-12,1.0",
                       "IWM,2026-10-30,276.0,0,2026-10-12,1.0", "IWM,2026-10-30,abc,1,2026-10-12,1.0")
        text, _, degraded = self.run_report()
        self.assertFalse(degraded)                      # this is something to fix, not a broken bot
        self.assertEqual(text.count("BAD LINE in positions.csv"), 5)
        self.assertIn("columns: etf,expiry,strike,contracts,entry_date,entry_price", text)

    def test_lowercase_etf_and_stray_spaces_still_count_as_held(self):
        self.etf("IWM", **SIGNAL_IWM)
        self.chains["IWM"] = chain(AFFORDABLE_030, spot=268.2)
        self.positions("iwm, 2026-10-30,276.0,1,2026-10-12,1.50")
        text, _, _ = self.run_report()
        self.assertIn("you already hold IWM", text)

    def test_held_etf_with_no_prices_says_check_by_hand(self):
        self.yf.set("SPY", minute=ConnectionError("down"))
        self.positions("SPY,2026-10-30,700.0,1,2026-10-12,3.00")
        text, _, degraded = self.run_report()
        self.assertTrue(degraded)
        self.assertIn("CHECK BY HAND: SPY 2026-10-30 700.0C -- no SPY prices today", text)
        self.assertIn("Sell if SPY is trading above yesterday's high (or on Thu 10-29, the last session before expiry).", text)
        self.assertNotIn("HOLD SPY", text)

    def test_time_exit_still_fires_without_prices(self):
        self.now = et(2026, 10, 29, 15, 46)
        for s in sig.ETFS:
            self.etf(s, high=100, low=90, last=95, prior_high=101, day=self.now.date())
        self.yf.set("SPY", minute=ConnectionError("down"))
        self.positions("SPY,2026-10-30,700.0,1,2026-10-20,3.00")
        text, _, _ = self.run_report()
        self.assertIn("EXIT (time): SELL TO CLOSE 1 SPY", text)

    def test_bounce_wins_over_time_and_two_positions_report_separately(self):
        self.etf("QQQ", high=500.0, low=495.0, last=499.0, prior_high=498.0)
        self.positions("QQQ,2026-10-30,500.0,1,2026-10-12,1.50", "SPY,2026-11-20,700.0,1,2026-10-12,3.00")
        text, _, _ = self.run_report(equity=6000, settled=6000)
        self.assertIn("EXIT (bounce): SELL TO CLOSE 1 QQQ", text)
        self.assertIn("HOLD SPY 2026-11-20 700.0C", text)


# --------------------------------------------------------------------------------------------------------------
class DiscordSender(unittest.TestCase):
    URL = "https://discord.com/api/webhooks/1/abc"

    def post(self, responses, **kw):
        sleeps = []
        with mock.patch.object(sig.requests, "post", side_effect=responses) as p:
            try:
                sig.post_discord(self.URL, kw.pop("content", "hello"), sleep=sleeps.append, **kw)
                err = None
            except sig.PostFailed as e:
                err = e
        return p, sleeps, err

    def test_success_is_one_call_with_mentions_disabled(self):
        p, sleeps, err = self.post([FakeResponse(204)])
        self.assertIsNone(err)
        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args.kwargs["json"], {"content": "hello", "allowed_mentions": {"parse": []}})
        self.assertEqual(sleeps, [])

    def test_rate_limit_waits_the_time_discord_asks_for(self):
        p, sleeps, err = self.post([FakeResponse(429, {"retry_after": 2.5}), FakeResponse(204)])
        self.assertIsNone(err)
        self.assertEqual(p.call_count, 2)
        self.assertEqual(sleeps, [2.5])

    def test_server_errors_and_network_errors_are_retried(self):
        p, sleeps, err = self.post([FakeResponse(502), requests.ConnectionError("reset"), FakeResponse(204)])
        self.assertIsNone(err)
        self.assertEqual(p.call_count, 3)
        self.assertEqual(sleeps, [1, 2])

    def test_dead_webhook_fails_at_once_with_the_reason(self):
        p, sleeps, err = self.post([FakeResponse(404, {"message": "Unknown Webhook"})])
        self.assertEqual(p.call_count, 1)
        self.assertIn("HTTP 404", str(err))
        self.assertIn("Unknown Webhook", str(err))

    def test_gives_up_after_the_retries(self):
        p, sleeps, err = self.post([requests.ConnectionError("down")] * 4)
        self.assertEqual(p.call_count, 4)
        self.assertEqual(len(sleeps), 3)
        self.assertIn("gave up after 4 tries", str(err))

    def test_shortened_message_never_exceeds_discord_limit_whatever_the_line_length(self):
        for width in range(20, 140):
            out = sig.fit_discord("\n".join("y" * width for _ in range(200)))
            self.assertLessEqual(len(out), 1980, width)
            self.assertTrue(out.endswith("2000-character limit)"), width)
        self.assertEqual(sig.fit_discord("short"), "short")
        self.assertLessEqual(len(sig.fit_discord("z" * 5000)), 1980)

    def test_overlong_message_is_shortened_not_dropped(self):
        long = "\n".join(f"line {i:03d} " + "x" * 60 for i in range(60))
        self.assertGreater(len(long), 2000)
        p, _, err = self.post([FakeResponse(204)], content=long)
        sent = p.call_args.kwargs["json"]["content"]
        self.assertLessEqual(len(sent), 2000)
        self.assertTrue(sent.endswith("(shortened to fit Discord's 2000-character limit)"))
        self.assertIn("line 000", sent)


class MainEntryPoint(SignalTestCase):
    HOOK = "https://discord.com/api/webhooks/1/abc"

    def go(self, hook=HOOK, env=None, post=None):
        env = dict(env or {})
        sent = []

        def fake_post(webhook, text, **kw):
            sent.append(text)
            if post:
                post()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sig, "ny_now", return_value=self.now), \
                mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(sig, "post_discord", side_effect=fake_post), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if "GITHUB_ACTIONS" not in env:
                os.environ.pop("GITHUB_ACTIONS", None)
            code = sig.main(["--equity", "1500", "--settled", "1500", "--discord", hook])
        return code, sent, out.getvalue(), err.getvalue()

    def test_normal_day_posts_once_and_exits_zero(self):
        code, sent, out, err = self.go()
        self.assertEqual(code, 0)
        self.assertEqual(len(sent), 1)
        self.assertIn("No action today", sent[0])
        self.assertFalse(os.path.exists(self.notified))

    def test_discord_refusal_is_a_failure_with_a_plain_message(self):
        def boom():
            raise sig.PostFailed("HTTP 404: Unknown Webhook")
        code, sent, out, err = self.go(post=boom)
        self.assertEqual(code, 1)
        self.assertIn("Discord refused the message (HTTP 404: Unknown Webhook)", err)
        self.assertIn("did NOT reach you", err)
        self.assertFalse(os.path.exists(self.notified))          # so the generic failure notifier still runs

    def test_missing_webhook_is_a_failure_in_github_actions_only(self):
        code, sent, out, err = self.go(hook="", env={"GITHUB_ACTIONS": "true"})
        self.assertEqual((code, sent), (1, []))
        self.assertIn("no Discord webhook configured", err)
        code, sent, out, err = self.go(hook="")
        self.assertEqual((code, sent), (0, []))                  # a local run just prints

    def test_degraded_run_exits_one_but_marks_the_failure_as_already_explained(self):
        self.yf.set("SPY", minute=ConnectionError("down"))
        code, sent, out, err = self.go()
        self.assertEqual(code, 1)
        self.assertIn("SPY: NO READING", sent[0])
        self.assertTrue(os.path.exists(self.notified))

    def test_degraded_and_undeliverable_does_not_claim_it_was_explained(self):
        self.yf.set("SPY", minute=ConnectionError("down"))

        def boom():
            raise sig.PostFailed("down")
        code, *_ = self.go(post=boom)
        self.assertEqual(code, 1)
        self.assertFalse(os.path.exists(self.notified))

    def test_holiday_posts_nothing_and_exits_zero(self):
        self.now = et(2026, 11, 26, 15, 46)
        code, sent, out, err = self.go()
        self.assertEqual((code, sent), (0, []))
        self.assertIn("Market holiday or weekend", out)

    def test_printed_text_equals_the_posted_text(self):
        code, sent, out, err = self.go()
        self.assertEqual(out.strip(), sent[0].strip())


# --------------------------------------------------------------------------------------------------------------
def original_pick_contract(sym, equity, settled, today, live_last=None):
    """The selection logic exactly as it was before the 2026-10-14 audit (frozen rulebook), kept here as the yardstick."""
    calls, (stamp, chain_spot) = sig.cboe_chain(sym)
    stale = live_last is not None and np.isfinite(chain_spot) and abs(chain_spot / live_last - 1) > 0.005
    exps = sorted(e for e in calls.expiry.unique() if (e - today).days >= sig.DTE_MIN and sig.last_session_before(e) > today)
    if not exps:
        return None, "no expiry >= 10 days"
    e = exps[0]
    g = calls[(calls.expiry == e) & (calls.bid > 0) & (calls.ask > calls.bid)].copy()
    g["mid"] = (g.bid + g.ask) / 2
    g["rel_spread"] = (g.ask - g.bid) / g.mid
    budget = sig.TARGET_F * equity
    for rung in sig.RUNGS:
        c = g.iloc[(g.delta - rung).abs().argsort()[:1]]
        if c.empty:
            continue
        c = c.iloc[0]
        if abs(c.delta - rung) > sig.DELTA_TOL or c.rel_spread > sig.MAX_REL_SPREAD or c.open_interest < sig.MIN_OI:
            continue
        cost = float(c.ask) * 100
        n = math.floor(budget / cost)
        if n < 1 and cost <= sig.ALLOW_MULT * budget:
            n = 1
        if n >= 1 and n * cost <= settled:
            return dict(sym=sym, expiry=str(e), strike=float(c.strike), rung=rung, delta=round(float(c.delta), 3),
                        bid=float(c.bid), ask=float(c.ask), mid=round(float(c.mid), 3),
                        max_limit=round(float(c.mid + 0.25 * (c.ask - c.bid)), 2), contracts=int(n),
                        cost_at_ask=round(n * cost, 2), pct_equity=round(n * cost / equity * 100, 1),
                        quote_time=stamp, stale=stale, chain_spot=chain_spot), None
    return None, "no rung affordable (<= 12% of equity) with spread <= 4% and OI >= 500"


class FrozenRuleUnchanged(SignalTestCase):
    """The audit rewrote how skips are explained; it must not change WHICH contract is chosen."""

    def random_chain(self, rng):
        n = rng.randint(1, 14)
        deltas = sorted((rng.uniform(0.02, 0.98) for _ in range(n)), reverse=True)
        rows = []
        for i, d in enumerate(deltas):
            mid = max(0.05, rng.uniform(0.2, 16) * d)
            rel = rng.choice([0.002, 0.01, 0.03, 0.039, 0.041, 0.06, 0.09])
            rows.append((250 + 2 * i, d, round(mid * (1 - rel / 2), 2), round(mid * (1 + rel / 2), 2),
                         rng.choice([50, 300, 499, 500, 501, 2500, 9000])))
        return chain(rows, spot=rng.choice([268.2, 270.0, 280.0]))

    def test_same_choice_on_a_thousand_random_chains(self):
        rng = random.Random(20261014)
        picked = skipped = 0
        for _ in range(1000):
            self.chains["IWM"] = self.random_chain(rng)
            equity = rng.choice([1500, 2500, 4000, 7500, 20000])
            settled = equity * rng.choice([1, 0.5, 0.2, 0.1])
            want, _ = original_pick_contract("IWM", equity, settled, TODAY, 268.2)
            got, why = sig.pick_contract("IWM", equity, settled, TODAY, 268.2)
            self.assertEqual(got, want)
            if got:
                picked += 1
            else:
                skipped += 1
                self.assertTrue(why)
        self.assertGreater(picked, 100)           # the comparison really exercised both outcomes
        self.assertGreater(skipped, 100)

    def test_constants_are_the_frozen_ones(self):
        self.assertEqual((sig.ETFS, sig.IBS_ENTRY, sig.DTE_MIN, sig.RUNGS, sig.DELTA_TOL, sig.TARGET_F, sig.ALLOW_MULT,
                          sig.MAX_REL_SPREAD, sig.MIN_OI),
                         (["SPY", "QQQ", "IWM"], 0.10, 10, [0.50, 0.40, 0.30], 0.06, 0.08, 1.5, 0.04, 500))
        self.assertEqual([sig.k_slots(e) for e in (1500, 2999, 3000, 4999, 5000)], [1, 1, 2, 2, 3])

    def test_expiry_choice_skips_the_too_near_weekly(self):
        near = chain([(270.0, 0.5, 2.0, 2.05, 5000)], expiry=EXPIRY.replace(day=23))[0]
        far = chain([(270.0, 0.5, 2.0, 2.05, 5000)], expiry=EXPIRY)[0]
        self.chains["IWM"] = (pd.concat([near, far]), ("t", 268.2))
        got, _ = sig.pick_contract("IWM", 4000, 4000, TODAY, 268.2)
        self.assertEqual(got["expiry"], "2026-10-30")           # 10-23 is only 9 days away


class CboeParsing(unittest.TestCase):
    def test_real_payload_shape_is_parsed(self):
        payload = {"timestamp": "2026-10-14 15:40:00", "data": {"current_price": 268.2, "options": [
            {"option": "IWM261030C00276000", "bid": 1.55, "ask": 1.60, "delta": 0.31, "open_interest": 2000},
            {"option": "IWM261030P00276000", "bid": 3.0, "ask": 3.1, "delta": -0.69, "open_interest": 900}]}}
        resp = mock.Mock()
        resp.json.return_value = payload
        with mock.patch.object(sig.requests, "get", return_value=resp):
            calls, (stamp, spot) = sig.cboe_chain("IWM")
        self.assertEqual(len(calls), 1)                          # puts are dropped
        self.assertEqual((float(calls.iloc[0].strike), calls.iloc[0].expiry.isoformat(), stamp, spot),
                         (276.0, "2026-10-30", "2026-10-14 15:40:00", 268.2))


if __name__ == "__main__":
    unittest.main()
