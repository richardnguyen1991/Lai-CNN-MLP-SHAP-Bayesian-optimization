"""The decision to spend another Kaggle session.

workflow_after.py runs on the GitHub runner after every session and emits the
three flags the workflow branches on. Getting them wrong is expensive in both
directions: too eager and a broken run pushes sessions forever, too cautious and
a healthy 100-epoch run stops halfway and waits for a human.

The load-bearing test here is the one about a run that never writes
training_state.json. workflow_state.py also caps sessions, but it counts the
session list inside that file, so a preparation phase that fails every time
never reaches its cap. session_log.json is appended on every session, failed
ones included, which is why the ceiling has to be enforced from the log.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "cicddos2019-main-20260818-abcd1234"
BASE = f"prefix/runs/{RUN_ID}"
STATE_KEY = f"{BASE}/checkpoints/training_state.json"
LOG_KEY = f"{BASE}/session_log.json"


def _load_module():
    path = REPO_ROOT / "scripts" / "workflow_after.py"
    spec = importlib.util.spec_from_file_location("workflow_after", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["workflow_after"] = module
    spec.loader.exec_module(module)
    return module


workflow_after = _load_module()


class FakeS3:
    """Enough of the S3 client for get_object/put_object on JSON blobs."""

    def __init__(self, objects=None) -> None:
        self.objects = {k: json.dumps(v).encode("utf-8")
                        for k, v in (objects or {}).items()}

    def get_object(self, Bucket, Key):          # noqa: N803 - boto3 spelling
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):   # noqa: N803
        self.objects[Key] = Body

    def stored(self, key):
        return json.loads(self.objects[key])


def entry(phase, end_epoch, advanced=True):
    return {"recorded_utc": "2026-08-18T00:00:00+00:00", "github_run": "1",
            "start_epoch": None, "end_epoch": end_epoch, "phase": phase,
            "status": "OK", "advanced": advanced}


@pytest.fixture
def after(monkeypatch, tmp_path):
    """Runs main() against a fake S3 and returns the emitted outputs."""

    def run(s3, max_sessions=None):
        output = tmp_path / "github_output"
        output.write_text("", encoding="utf-8")
        monkeypatch.setenv("S3_BUCKET", "bucket")
        monkeypatch.setenv("S3_PREFIX", "prefix")
        monkeypatch.setenv("RUN_ID", RUN_ID)
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        if max_sessions is None:
            monkeypatch.delenv("MAX_SESSIONS", raising=False)
        else:
            monkeypatch.setenv("MAX_SESSIONS", str(max_sessions))

        import boto3
        monkeypatch.setattr(boto3, "client", lambda service: s3)

        assert workflow_after.main() == 0
        return dict(line.split("=", 1)
                    for line in output.read_text(encoding="utf-8").splitlines()
                    if line)

    return run


def test_a_run_in_progress_asks_for_another_session(after):
    s3 = FakeS3({STATE_KEY: {"current_epoch": 12, "phase": "final_train",
                             "total_epochs": 100, "status": "OK"}})
    out = after(s3)
    assert out["needs_more"] == "true"
    assert out["stalled"] == "false"
    assert out["capped"] == "false"
    assert out["current_epoch"] == "12"


def test_a_finished_run_stops_the_chain(after):
    s3 = FakeS3({STATE_KEY: {"current_epoch": 100, "phase": "done",
                             "total_epochs": 100, "status": "OK"}})
    assert after(s3)["needs_more"] == "false"


def test_epoch_100_still_needs_a_session_while_evaluation_is_pending(after):
    # finished requires both the epoch count and a phase past training.
    s3 = FakeS3({STATE_KEY: {"current_epoch": 100, "phase": "final_train",
                             "total_epochs": 100, "status": "OK"}})
    assert after(s3)["needs_more"] == "true"


def test_two_training_sessions_without_progress_raise_the_stall_flag(after):
    s3 = FakeS3({
        STATE_KEY: {"current_epoch": 40, "phase": "final_train",
                    "total_epochs": 100, "status": "OK"},
        LOG_KEY: [entry("final_train", 40, advanced=True),
                  entry("final_train", 40, advanced=False)],
    })
    # This session also ends at 40, so the last two logged training sessions
    # both failed to advance.
    assert after(s3)["stalled"] == "true"


def test_preparation_sessions_never_count_towards_a_stall(after):
    # prepare legitimately leaves current_epoch at 0; counting it would halt
    # every run before it trained a single epoch.
    s3 = FakeS3({
        STATE_KEY: {"current_epoch": 0, "phase": "prepare",
                    "total_epochs": 100, "status": "OK"},
        LOG_KEY: [entry("prepare", 0, advanced=False),
                  entry("prepare", 0, advanced=False)],
    })
    out = after(s3)
    assert out["stalled"] == "false"
    assert out["needs_more"] == "true"


def test_a_run_that_never_writes_training_state_is_still_capped(after):
    # The regression this module exists for. No training_state.json means
    # workflow_state.py cannot count sessions, and prepare phases are exempt
    # from the stall check, so without the log-based ceiling the chain step
    # would dispatch a failing preparation for ever.
    s3 = FakeS3({LOG_KEY: [entry("prepare", 0, advanced=False)] * 2})
    out = after(s3, max_sessions=3)
    assert out["capped"] == "true"
    assert out["stalled"] == "false"


def test_the_ceiling_does_not_trip_early(after):
    s3 = FakeS3({LOG_KEY: [entry("prepare", 0, advanced=False)]})
    assert after(s3, max_sessions=3)["capped"] == "false"


def test_the_session_is_appended_to_the_log(after):
    s3 = FakeS3({
        STATE_KEY: {"current_epoch": 30, "phase": "final_train",
                    "total_epochs": 100, "status": "OK"},
        LOG_KEY: [entry("final_train", 20)],
    })
    after(s3)
    log = s3.stored(LOG_KEY)
    assert len(log) == 2
    assert log[-1]["start_epoch"] == 20
    assert log[-1]["end_epoch"] == 30
    assert log[-1]["advanced"] is True


def test_a_missing_log_starts_one_rather_than_failing(after):
    s3 = FakeS3({STATE_KEY: {"current_epoch": 1, "phase": "final_train",
                             "total_epochs": 100, "status": "OK"}})
    after(s3)
    log = s3.stored(LOG_KEY)
    assert len(log) == 1
    # No previous session, so there is nothing to compare against and the
    # session counts as progress rather than a stall.
    assert log[0]["start_epoch"] is None
    assert log[0]["advanced"] is True


# --------------------------------------------------------------------------
# Kaggle credentials
# --------------------------------------------------------------------------
# The first real dispatch died here. The workflow wrote the secret straight into
# kaggle.json; the CLI parses that file inside a bare `except: pass`, so a value
# that is not a JSON object was dropped without a word and the run failed three
# steps later with a generic "authentication required".

def _load_credentials_module():
    path = REPO_ROOT / "scripts" / "write_kaggle_credentials.py"
    spec = importlib.util.spec_from_file_location("write_kaggle_credentials", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["write_kaggle_credentials"] = module
    spec.loader.exec_module(module)
    return module


write_credentials = _load_credentials_module()


LEGACY = "0123456789abcdef0123456789abcdef"          # 32 hex, the legacy shape
ACCESS = "kg_live_9f2c4a1e8b7d6503f1a2b3c4d5e6f7a8b"    # what the settings page issues now


def test_a_legacy_api_key_is_paired_with_the_username_secret():
    assert write_credentials.credentials(LEGACY, "richard") == (
        "legacy", {"username": "richard", "key": LEGACY})


def test_a_full_kaggle_json_blob_is_accepted_as_is():
    blob = json.dumps({"username": "embedded", "key": LEGACY})
    assert write_credentials.credentials(blob, "ignored") == (
        "legacy", {"username": "embedded", "key": LEGACY})


def test_a_blob_without_a_username_falls_back_to_the_secret():
    blob = json.dumps({"key": LEGACY})
    assert write_credentials.credentials(blob, "fallback") == (
        "legacy", {"username": "fallback", "key": LEGACY})


def test_a_token_that_is_not_a_legacy_key_is_treated_as_an_access_token():
    # The regression. This token went into kaggle.json's "key" field, which
    # made the CLI pick the legacy path and send it as a legacy key; the server
    # answered 401 and the CLI printed a generic "authentication required" that
    # named neither the file nor the credential kind.
    assert write_credentials.credentials(ACCESS, "richard") == ("access", ACCESS)


def test_an_access_token_needs_no_username():
    # The CLI introspects the token for the username, so an absent
    # KAGGLE_USERNAME must not be treated as an error here.
    assert write_credentials.credentials(ACCESS, "") == ("access", ACCESS)


def test_surrounding_whitespace_does_not_become_part_of_the_credential():
    # A secret pasted with a trailing newline is the classic way to get a
    # credential that looks right and authenticates as nobody.
    trailing = "  " + LEGACY + chr(10)
    assert write_credentials.credentials(trailing, " richard ") == (
        "legacy", {"username": "richard", "key": LEGACY})


@pytest.mark.parametrize("token, username", [
    ("", "richard"),                            # secret not set
    ("   ", "richard"),                         # secret set to whitespace
    ('{"username": "x"}', "richard"),           # blob carrying no key
    ('["not", "an", "object"]', "richard"),
    (LEGACY, ""),                               # legacy key with nowhere to get a username
])
def test_unusable_secrets_fail_loudly(token, username):
    with pytest.raises(SystemExit):
        write_credentials.credentials(token, username)


def test_main_writes_a_legacy_key_where_the_cli_reads_one(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("KAGGLE_API_TOKEN", LEGACY)
    monkeypatch.setenv("KAGGLE_USERNAME", "richard")

    assert write_credentials.main() == 0

    written = json.loads((tmp_path / "cfg" / "kaggle.json").read_text(encoding="utf-8"))
    # Both fields together is the whole point: the legacy path needs username
    # and key, and quietly refuses to authenticate without.
    assert written == {"username": "richard", "key": LEGACY}
    assert not (tmp_path / "cfg" / "access_token").exists()


def test_main_writes_an_access_token_to_its_own_file(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("KAGGLE_API_TOKEN", ACCESS)
    monkeypatch.setenv("KAGGLE_USERNAME", "richard")

    assert write_credentials.main() == 0

    assert (tmp_path / "cfg" / "access_token").read_text(encoding="utf-8") == ACCESS
    # Writing kaggle.json too would put the CLI back on the legacy path, which
    # it prefers to nothing and which cannot work with this credential.
    assert not (tmp_path / "cfg" / "kaggle.json").exists()


# --------------------------------------------------------------------------
# Following the kernel
# --------------------------------------------------------------------------
# `kaggle kernels status` prints its enum through %s, so the status arrives as
# "KernelWorkerStatus.CANCEL_ACKNOWLEDGED" rather than "cancelAcknowledged".
# Reading it naively yields the class name, which matches no known state, and a
# kernel that had already died was polled for twenty minutes because an
# unreadable status was indistinguishable from a running one.

def _load_poll_module():
    path = REPO_ROOT / "scripts" / "poll_kernel.py"
    spec = importlib.util.spec_from_file_location("poll_kernel", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["poll_kernel"] = module
    spec.loader.exec_module(module)
    return module


poll_kernel = _load_poll_module()

KERNEL = "richardnguyen1991/lai-cnn-mlp-shap-bayesian-optimization"


def status_output(enum_name, failure=None):
    """What the CLI actually prints."""
    line = f'{KERNEL} has status "KernelWorkerStatus.{enum_name}"'
    if failure is not None:
        line += f'\nFailure message: "{failure}"'
    return line


@pytest.mark.parametrize("enum_name, expected", [
    ("QUEUED", "queued"),
    ("RUNNING", "running"),
    ("COMPLETE", "complete"),
    ("ERROR", "error"),
    ("CANCEL_REQUESTED", "cancelrequested"),
    ("CANCEL_ACKNOWLEDGED", "cancelacknowledged"),
])
def test_every_kernel_state_parses_to_a_known_token(enum_name, expected):
    status, _ = poll_kernel.parse_status_output(status_output(enum_name))
    assert status == expected


def test_the_class_name_is_not_mistaken_for_the_status():
    # The regression. The previous pattern matched the first word after
    # "status", which is the enum's class name, so nothing ever settled.
    status, _ = poll_kernel.parse_status_output(status_output("ERROR"))
    assert status != "kernelworkerstatus"
    assert status in poll_kernel.SETTLED_BAD


def test_a_cancelled_session_counts_as_a_normal_ending():
    # The path a deliberate cancellation takes, and the one the resume check
    # depends on. The underscore in the enum name has to be normalised away or
    # this reads as an unknown state and the workflow keeps polling.
    status, _ = poll_kernel.parse_status_output(status_output("CANCEL_ACKNOWLEDGED"))
    assert status in poll_kernel.SETTLED_OK


@pytest.mark.parametrize("enum_name", ["QUEUED", "RUNNING", "CANCEL_REQUESTED"])
def test_states_still_in_flight_do_not_settle(enum_name):
    status, _ = poll_kernel.parse_status_output(status_output(enum_name))
    assert status not in poll_kernel.SETTLED_OK
    assert status not in poll_kernel.SETTLED_BAD
    assert status != poll_kernel.UNREADABLE


def test_a_failure_message_is_carried_alongside_the_status():
    status, detail = poll_kernel.parse_status_output(
        status_output("ERROR", failure="FileNotFoundError: no files matching"))
    assert status == "error"
    assert "FileNotFoundError" in detail


def test_output_with_no_status_line_is_its_own_outcome():
    # Not "still running". Conflating the two is what let a dead kernel look
    # alive for the whole timeout.
    status, detail = poll_kernel.parse_status_output(
        "401 Unauthorized\nAuthentication required to call the Kaggle API.")
    assert status == poll_kernel.UNREADABLE
    assert "Authentication required" in detail


def test_the_settled_sets_do_not_overlap():
    assert not (poll_kernel.SETTLED_OK & poll_kernel.SETTLED_BAD)


# --------------------------------------------------------------------------
# The six hour ceiling
# --------------------------------------------------------------------------
# A GitHub-hosted job is killed at 6 hours of execution no matter what
# timeout-minutes says. A killed job never reaches the steps that record the
# session or dispatch the next one, so the run stops without a word. One run
# was cancelled at 6h00m20s with the kernel still going, and the chain died
# there while the configuration claimed a 13 hour budget.

import re  # noqa: E402

import yaml  # noqa: E402

GITHUB_JOB_CEILING_MINUTES = 360
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run-kaggle.yml"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _experiment():
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))


def test_the_actions_job_stays_under_the_ceiling_github_enforces():
    timeout = _workflow()["jobs"]["session"]["timeout-minutes"]
    assert timeout < GITHUB_JOB_CEILING_MINUTES


def test_the_kernel_budget_leaves_the_watcher_room_to_finish():
    # The watcher has to outlive the kernel, or nothing records the session.
    session = _experiment()["session"]
    timeout = _workflow()["jobs"]["session"]["timeout-minutes"]
    assert session["session_time_budget_minutes"] < timeout


def test_the_poll_gives_up_before_the_job_is_killed():
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"poll_kernel\.py[^\n]*--timeout-minutes\s+(\d+)", text)
    assert match, "the poll step no longer passes --timeout-minutes"
    assert int(match.group(1)) < _workflow()["jobs"]["session"]["timeout-minutes"]


def test_the_budget_still_allows_a_useful_amount_of_work():
    # Guarding the ceiling must not be satisfied by shrinking the budget to
    # nothing: a session too short to clear its safety margin would checkpoint
    # immediately and never advance an epoch.
    session = _experiment()["session"]
    assert session["session_time_budget_minutes"] > session["safety_margin_minutes"] * 4
