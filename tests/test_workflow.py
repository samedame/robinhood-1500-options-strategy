"""Checks on the workflow files and the failure notifier -- the parts that decide whether a broken run is noticed."""
import importlib.util
import io
import os
import unittest
from unittest import mock

import yaml

from tests.helpers import ROOT

WF = os.path.join(ROOT, ".github", "workflows")


def load(name):
    with open(os.path.join(WF, name)) as f:
        return yaml.safe_load(f)


def steps(wf):
    return {s.get("name", s.get("run", "")[:30]): s for s in wf["jobs"][next(iter(wf["jobs"]))]["steps"]}


class DailyWorkflow(unittest.TestCase):
    def setUp(self):
        self.wf = load("daily-signal.yml")
        self.steps = steps(self.wf)

    def test_triggers_unchanged(self):
        triggers = self.wf.get("on") or self.wf.get(True)
        self.assertEqual(triggers["schedule"], [{"cron": "50 20 * * 1-5"}])   # one backup line behind the Worker
        self.assertIn("workflow_dispatch", triggers)
        self.assertEqual(self.wf["concurrency"], {"group": "ibs-dip-calls", "cancel-in-progress": False})

    def test_guard_step_has_a_plain_english_missed_window_hint(self):
        step = self.steps["Run signal check (guarded)"]
        self.assertIn('--too-late-hint "$TOO_LATE_HINT"', step["run"])
        hint = " ".join(step["env"]["TOO_LATE_HINT"].split())
        for must in ("No 3:45 pm reading was taken", "SPY / QQQ / IWM", "below 0.10", "yesterday's high",
                     "last session before expiry", "Fridays"):
            self.assertIn(must, hint)
        self.assertLess(len(hint), 700)                 # the notice (~200 chars) + hint must fit one Discord message
        self.assertIn("--on-too-late notice", step["run"])
        self.assertIn("--target 15:45 --usable-until 15:58", step["run"])

    def test_log_is_saved_even_when_the_check_failed_and_the_push_is_retried(self):
        step = self.steps["Persist position/signal log + guard state"]
        self.assertEqual(step["if"], "always()")
        self.assertIn("for i in 1 2 3 4 5", step["run"])
        self.assertIn("git pull --rebase --autostash && git push", step["run"])
        self.assertIn("exit 1", step["run"])            # five failed pushes must turn the run red

    def test_a_failed_run_tells_discord(self):
        step = self.steps["Tell Discord if this run failed"]
        self.assertEqual(step["if"], "failure()")
        self.assertIn("python notify_failure.py", step["run"])
        self.assertIn("RUN_URL", step["env"])
        self.assertEqual(self.wf["jobs"]["signal"]["env"]["DISCORD_WEBHOOK_URL"], "${{ secrets.DISCORD_WEBHOOK_URL }}")
        self.assertIn("before 3:58 pm ET, re-run it", step["run"])
        # order matters: save state first, announce last
        names = list(self.steps)
        self.assertLess(names.index("Persist position/signal log + guard state"), names.index("Tell Discord if this run failed"))

    def test_the_script_is_still_run_through_the_guard_with_equity_and_cash(self):
        run = self.steps["Run signal check (guarded)"]["run"]
        self.assertIn("python deliver/ibs_dip_calls_signal.py", run)
        self.assertIn("vars.ACCOUNT_EQUITY", run)
        self.assertIn("vars.SETTLED_CASH", run)

    def test_the_comment_about_fridays_is_there_for_whoever_edits_the_worker(self):
        with open(os.path.join(WF, "daily-signal.yml")) as f:
            text = f.read()
        self.assertIn("MON-FRI", text)
        self.assertIn("Sun-Thu", text)


class TestsWorkflow(unittest.TestCase):
    def test_pull_requests_run_the_suite(self):
        wf = load("tests.yml")
        triggers = wf.get("on") or wf.get(True)
        self.assertIn("pull_request", triggers)
        runs = " ".join(s.get("run", "") for s in wf["jobs"]["test"]["steps"])
        self.assertIn("python -m pytest tests -q", runs)
        self.assertIn("pyyaml", runs)


class NotifyFailure(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("notify_failure", os.path.join(ROOT, "notify_failure.py"))
        self.nf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.nf)

    def test_marker_lives_in_the_repo_root_where_the_signal_script_writes_it(self):
        from tests.helpers import sig
        self.assertEqual(self.nf.MARKER, os.path.join(ROOT, ".notified"))
        self.assertEqual(sig.NOTIFIED_FILE, self.nf.MARKER)

    def test_message_says_the_alert_did_not_arrive(self):
        text = self.nf.message("IBS Dip Calls 3:45 pm check", "Extra.", "https://x/run/1", "workflow_dispatch")
        self.assertIn("this run FAILED", text)
        self.assertIn("did NOT arrive", text)
        self.assertIn("https://x/run/1", text)
        self.assertTrue(text.endswith("Extra."))

    def test_posts_once_and_never_raises(self):
        sent = []

        class R:
            status = 204
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def opener(req, timeout=0):
            sent.append(req.data)
            return R()
        self.assertTrue(self.nf.post("https://hook", "hi", opener=opener, sleep=lambda s: None))
        self.assertEqual(len(sent), 1)

        def boom(req, timeout=0):
            raise OSError("down")
        self.assertFalse(self.nf.post("https://hook", "hi", opener=boom, sleep=lambda s: None))

    def test_stays_quiet_when_the_script_already_explained(self):
        with mock.patch.object(self.nf.os.path, "exists", return_value=True), \
                mock.patch.object(self.nf, "post") as post, mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(self.nf.main(["x"], env={"DISCORD_WEBHOOK_URL": "https://hook"}), 0)
        post.assert_not_called()

    def test_reads_the_webhook_from_the_variable_this_workflow_sets(self):
        with mock.patch.object(self.nf.os.path, "exists", return_value=False), \
                mock.patch.object(self.nf, "post", return_value=True) as post:
            self.nf.main(["IBS Dip Calls 3:45 pm check", "More."], env={"DISCORD_WEBHOOK_URL": "https://hook", "RUN_URL": "u"})
        self.assertEqual(post.call_args.args[0], "https://hook")
        self.assertIn("IBS Dip Calls 3:45 pm check", post.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
