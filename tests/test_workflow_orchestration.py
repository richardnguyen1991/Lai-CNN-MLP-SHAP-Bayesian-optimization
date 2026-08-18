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


def test_a_raw_api_key_is_paired_with_the_username_secret():
    # The shape that broke the first run: KAGGLE_API_TOKEN holding just the key.
    assert write_credentials.credentials("abc123", "richard") == {
        "username": "richard", "key": "abc123"}


def test_a_full_kaggle_json_blob_is_accepted_as_is():
    blob = json.dumps({"username": "embedded", "key": "k9"})
    assert write_credentials.credentials(blob, "ignored") == {
        "username": "embedded", "key": "k9"}


def test_a_blob_without_a_username_falls_back_to_the_secret():
    assert write_credentials.credentials('{"key": "k9"}', "fallback") == {
        "username": "fallback", "key": "k9"}


def test_surrounding_whitespace_does_not_become_part_of_the_key():
    # A secret pasted with a trailing newline is the classic way to get a key
    # that looks right and authenticates as nobody.
    assert write_credentials.credentials("  abc123\n", " richard ") == {
        "username": "richard", "key": "abc123"}


@pytest.mark.parametrize("token, username", [
    ("", "richard"),                       # secret not set
    ("   ", "richard"),                    # secret set to whitespace
    ('{"username": "x"}', "richard"),      # blob carrying no key
    ('["not", "an", "object"]', "richard"),
    ("abc123", ""),                        # key but nowhere to get a username
])
def test_unusable_secrets_fail_loudly(token, username):
    with pytest.raises(SystemExit):
        write_credentials.credentials(token, username)


def test_main_writes_a_file_the_cli_can_actually_parse(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "abc123")
    monkeypatch.setenv("KAGGLE_USERNAME", "richard")

    assert write_credentials.main() == 0

    written = json.loads((tmp_path / "cfg" / "kaggle.json").read_text(encoding="utf-8"))
    # Both fields present is the whole point: the CLI's legacy path needs
    # username and key together, and quietly refuses to authenticate without.
    assert written == {"username": "richard", "key": "abc123"}
