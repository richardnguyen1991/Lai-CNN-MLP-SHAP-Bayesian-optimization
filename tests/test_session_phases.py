"""Phase progression across Kaggle sessions.

A session prepared 70M rows, uploaded its cache, and the next session prepared
them again -- indefinitely, until the session ceiling stopped it.

The phase is carried by training_state.json. PHASES deliberately starts at
"shap" so that the absence of that file means "prepare". But nothing outside
train.py ever wrote it, and train.py runs only in final_train, which could only
be reached from a state that did not exist yet. The pipeline was a closed loop
at its first step.

Nothing caught it because nothing exercised run_session: the unit tests covered
what each phase does, never the machine that sequences them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import session as session_module  # noqa: E402
from session import PHASE_ORDER, Session, next_phase, run_session  # noqa: E402

RUN_ID = "cicddos2019-main-20260818-d05a4c6d"
PREFIX = f"bucket-prefix/runs/{RUN_ID}"
STATE_KEY = f"{PREFIX}/checkpoints/training_state.json"


class FakeManifest:
    """A snapshot of S3, exactly as the real one is: frozen at session start."""

    def __init__(self, present=()) -> None:
        self.run_id = RUN_ID
        self.entries = {f"{PREFIX}/code_bundle.tar.gz": {"get_url": "https://x"}}
        for key in present:
            self.entries[f"{PREFIX}/{key}"] = {"get_url": "https://x"}

    def exists(self, key: str) -> bool:
        entry = self.entries.get(key)
        return bool(entry and entry.get("get_url"))

    def summary(self) -> str:
        return f"fake manifest: {len(self.entries)} keys"


class FakeIO:
    def __init__(self, manifest, remote=None) -> None:
        self.manifest = manifest
        self.remote = dict(remote or {})
        self.uploaded = {}

    def get_json(self, key):
        return self.remote[key]

    def upload(self, path, key):
        self.uploaded[key] = Path(path).read_text(encoding="utf-8")

    def upload_directory(self, local, key, patterns=("*",)):
        return 0

    def download(self, key, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("shard", encoding="utf-8")

    def download_if_present(self, key, path):
        if key in self.remote:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(self.remote[key]), encoding="utf-8")


def make_session(tmp_path, manifest=None, remote=None):
    manifest = manifest or FakeManifest()
    return Session(FakeIO(manifest, remote), tmp_path / "work", REPO_ROOT,
                   tmp_path / "input", RUN_ID)


# --------------------------------------------------------------------------
# The order itself
# --------------------------------------------------------------------------

def test_next_phase_walks_the_whole_pipeline():
    assert [next_phase(p) for p in PHASE_ORDER[:-1]] == list(PHASE_ORDER[1:])


def test_next_phase_stops_at_the_end():
    assert next_phase("done") == "done"


# --------------------------------------------------------------------------
# Recording the phase
# --------------------------------------------------------------------------

def test_publishing_a_phase_writes_and_uploads_the_state(tmp_path):
    session = make_session(tmp_path)
    session.publish_phase("shap")

    assert STATE_KEY in session.io.uploaded
    assert json.loads(session.io.uploaded[STATE_KEY])["phase"] == "shap"


def test_the_state_file_is_what_lets_the_next_session_skip_preparation(tmp_path):
    # The regression, put as the behaviour that was missing: once prepare is
    # done a state must exist saying "shap", because the absence of that file
    # is the only thing that means "prepare".
    session = make_session(tmp_path)
    assert session.current_phase() == "prepare"

    session.publish_phase("shap")
    published = json.loads(session.io.uploaded[STATE_KEY])

    # A fresh session, seeing that state, starts at shap rather than preparing
    # the whole dataset over again.
    manifest = FakeManifest(present=["checkpoints/training_state.json"])
    resumed = make_session(tmp_path / "second", manifest,
                           remote={STATE_KEY: published})
    assert resumed.current_phase() == "shap"


def test_publishing_a_phase_never_discards_the_epoch_counter(tmp_path):
    # A phase written here must not reset a resumed training run to epoch 0.
    session = make_session(tmp_path)
    session.publish_phase("final_train")

    state_path = session.work / "checkpoints" / "training_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["current_epoch"] = 45
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    session.publish_phase("final_train")
    assert json.loads(state_path.read_text(encoding="utf-8"))["current_epoch"] == 45


@pytest.mark.parametrize("phase", ["shap", "bo", "final_train"])
def test_every_working_phase_can_be_recorded(phase, tmp_path):
    # "prepare" is deliberately absent from PHASES; every other phase has to be
    # writable, or the pipeline stalls wherever the gap is.
    session = make_session(tmp_path)
    session.publish_phase(phase)
    assert json.loads(session.io.uploaded[STATE_KEY])["phase"] == phase


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def test_a_session_advances_instead_of_repeating_one_phase(tmp_path, monkeypatch):
    ran = []

    def recorder(name):
        def handler(self):
            ran.append(name)
        return handler

    monkeypatch.setattr(session_module, "PHASE_HANDLERS", {
        "prepare": recorder("prepare"),
        "shap": recorder("shap"),
        "bo": recorder("bo"),
        "final_train": recorder("final_train"),
    })
    manifest = FakeManifest()
    monkeypatch.setattr(session_module.PresignedManifest, "fetch",
                        staticmethod(lambda url, transport: manifest))
    monkeypatch.setattr(session_module, "PresignedIO", lambda m: FakeIO(m))

    assert run_session("https://manifest", tmp_path / "work", REPO_ROOT,
                       tmp_path / "input", RUN_ID) == 0

    # Four phases, each once. Before the fix this was prepare four times over,
    # because the phase was re-read from a manifest that could not see the
    # upload the previous iteration had just made.
    assert ran == ["prepare", "shap", "bo", "final_train"]


def test_the_loop_stops_once_training_is_complete(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(session_module, "PHASE_HANDLERS",
                        {"final_train": lambda self: ran.append("final_train")})

    finished = {"phase": "final_train", "current_epoch": 100, "total_epochs": 100}
    manifest = FakeManifest(present=["checkpoints/training_state.json"])
    io = FakeIO(manifest, {STATE_KEY: finished})
    monkeypatch.setattr(session_module.PresignedManifest, "fetch",
                        staticmethod(lambda url, transport: manifest))
    monkeypatch.setattr(session_module, "PresignedIO", lambda m: io)

    assert run_session("https://manifest", tmp_path / "work", REPO_ROOT,
                       tmp_path / "input", RUN_ID) == 0
    assert ran == []


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------

def test_a_cache_built_this_session_is_visible_to_the_next_phase(tmp_path):
    # The manifest is minted before the session runs, so it can never show the
    # cache prepare has just uploaded. Consulting it alone made the SHAP phase
    # refuse to start in the very session that produced its input.
    session = make_session(tmp_path)
    shards = session.work / "cache" / "preprocess"
    shards.mkdir(parents=True)
    (shards / "train_000.npz").write_text("data", encoding="utf-8")

    assert session.pull_cache() is True


def test_the_cache_check_does_not_demand_the_output_of_shap(tmp_path):
    # pull_cache used to return True only when selected_features.json existed,
    # which SHAP itself writes -- so the phase required its own result first.
    session = make_session(tmp_path)
    shards = session.work / "cache" / "preprocess"
    shards.mkdir(parents=True)
    (shards / "train_000.npz").write_text("data", encoding="utf-8")

    assert not (session.work / "cache" / "selected_features.json").exists()
    assert session.pull_cache() is True


def test_an_absent_cache_is_still_reported_as_absent(tmp_path):
    session = make_session(tmp_path)
    assert session.pull_cache() is False


def test_a_remote_cache_is_downloaded_when_nothing_is_local(tmp_path):
    manifest = FakeManifest(present=["cache/preprocess/train_000.npz"])
    session = make_session(tmp_path, manifest)

    assert session.pull_cache() is True
    assert (session.work / "cache" / "preprocess" / "train_000.npz").exists()
