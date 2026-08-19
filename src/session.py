"""One Kaggle session: work out the phase, do that work, upload, exit.

The notebook is a four-line bootstrap; everything the session actually does
lives here, inside the code bundle that GitHub Actions built from the
repository. That way the notebook can never drift out of step with the code.

Phase order, driven by training_state.json on S3:

    prepare -> shap -> bo -> final_train -> evaluate -> done

Each phase downloads only what it needs, does its work, uploads its artifacts,
and rewrites training_state.json last. A session that dies partway leaves the
previous phase's state intact, so the next one repeats at most one phase rather
than the whole run.

The dataset itself is read from /kaggle/input, not S3: it is already attached to
the notebook as a Kaggle Dataset, so there is nothing to download.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from checkpoint import (
    CheckpointManager,
    TrainingState,
    atomic_write_json,
    config_hash,
)
from presigned_io import (
    EXIT_MANIFEST_UNUSABLE,
    EXIT_PRESIGNED_EXPIRED,
    ManifestUnusable,
    PresignedExpired,
    PresignedIO,
    PresignedManifest,
)

PHASE_ORDER = ("prepare", "shap", "bo", "final_train", "evaluate", "done")


class Session:
    def __init__(self, io: PresignedIO, work_dir: Path, repo_root: Path,
                 input_root: Path, run_id: str) -> None:
        self.io = io
        self.work = work_dir
        self.repo_root = repo_root
        self.input_root = input_root
        self.run_id = run_id
        self.session_id = uuid.uuid4().hex[:12]
        self.base = f"{io.manifest.run_id or run_id}"
        self.prefix = self._detect_prefix()

        with (repo_root / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
            self.experiment = yaml.safe_load(handle)

    def _detect_prefix(self) -> str:
        """The manifest keys are absolute S3 keys; recover their common base."""
        for key in self.io.manifest.entries:
            if key.endswith("/code_bundle.tar.gz"):
                return key.rsplit("/", 1)[0]
        raise ManifestUnusable("manifest has no code_bundle key to anchor the prefix")

    def key(self, relative: str) -> str:
        return f"{self.prefix}/{relative}"

    # -- state ------------------------------------------------------------
    def load_remote_state(self) -> Optional[Dict[str, Any]]:
        remote = self.key("checkpoints/training_state.json")
        if not self.io.manifest.exists(remote):
            return None
        return self.io.get_json(remote)

    def publish_phase(self, phase: str) -> None:
        """Record that the pipeline has reached `phase`, and upload it.

        The absence of training_state.json is what "prepare" means -- PHASES
        starts at "shap" for exactly that reason -- so this file appearing is
        the only way the pipeline can leave preparation. Nothing wrote it
        except train.py, and train.py is reachable only from final_train, so
        the phase could never advance: every session prepared the data again,
        and the next session started from prepare once more.

        An existing state is edited rather than replaced, so a phase written
        here can never discard the epoch counter a resumed training run
        depends on.
        """
        directory = self.work / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "training_state.json"

        if path.exists():
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["phase"] = phase
            state = TrainingState(**payload)
        else:
            state = TrainingState(
                run_id=self.run_id,
                session_id=self.session_id,
                dataset_name=self.experiment["run"]["dataset"],
                phase=phase,
                total_epochs=self.experiment["training"]["epochs"],
                # No hashes. They belong to the training run and are computed
                # by train.py over {experiment, model, best_params}; a value
                # derived any other way reads to verify_resumable as a changed
                # experiment and refuses the run. Empty means "training has not
                # started", which is exactly true here.
            )

        atomic_write_json(path, state.to_dict())
        self.io.upload(path, self.key("checkpoints/training_state.json"))
        print(f"phase recorded: {phase}")

    def current_phase(self) -> str:
        state = self.load_remote_state()
        if state is None:
            return "prepare"
        phase = state.get("phase", "prepare")
        if phase == "final_train" and state.get("current_epoch", 0) >= state.get(
            "total_epochs", 100
        ):
            return "evaluate"
        return phase

    # -- download ---------------------------------------------------------
    def pull_cache(self) -> bool:
        """Make the preprocessing cache available. False means it is not there yet.

        True means the preprocessing shards are usable, which is what every
        later phase actually needs. It used to be keyed on
        selected_features.json instead -- a file SHAP itself produces -- so
        the SHAP phase demanded its own output before it was allowed to run.

        Local files win over the manifest. The manifest is a snapshot taken
        before the session started, so a cache this same session has just
        built and uploaded is invisible to it.
        """
        cache = self.work / 'cache'
        shard_dir = cache / 'preprocess'
        local = sorted(shard_dir.glob('*')) if shard_dir.is_dir() else []

        if not local:
            shards = [k for k in self.io.manifest.entries
                      if "/cache/preprocess/" in k and self.io.manifest.exists(k)]
            if not shards:
                return False
            for key in shards:
                self.io.download(key, shard_dir / key.rsplit("/", 1)[1])

        for name in ("scaler.joblib", "selected_features.json"):
            self.io.download_if_present(self.key(f"cache/{name}"), cache / name)
        for name in ("preprocessing.json", "split_assignment.npy", "file_offsets.json",
                     "split_manifest.json"):
            self.io.download_if_present(self.key(f"config/{name}"), self.work / "config" / name)

        return bool(local) or any(shard_dir.glob('*'))

    def pull_checkpoints(self) -> None:
        for name in ("model_last.pt", "optimizer_last.pt", "rng_state.pt",
                     "training_state.json", "history.json", "model_best_val.pt"):
            self.io.download_if_present(
                self.key(f"checkpoints/{name}"), self.work / "checkpoints" / name
            )

    # -- upload -----------------------------------------------------------
    def push(self, directory: str, patterns=("*",)) -> int:
        local = self.work / directory
        if not local.exists():
            return 0
        return self.io.upload_directory(local, self.key(directory), patterns)

    def push_file(self, relative: str) -> None:
        path = self.work / relative
        if path.exists():
            self.io.upload(path, self.key(relative))

    # -- phases -----------------------------------------------------------
    def phase_prepare(self) -> None:
        """Audit, split and preprocess, then publish the cache."""
        import data_audit
        import preprocessing
        import split

        dataset = self.experiment["run"]["dataset"]
        pattern = data_audit.load_dataset_config(
            self.repo_root, dataset).get("input_glob", "*.parquet")

        # Resolve once, here, and hand the same root to all three stages. Each
        # of them keys a file by its path relative to this root, so resolving
        # separately per stage would quietly produce three different sets of
        # identities out of one dataset.
        resolved = data_audit.resolve_input_root(self.input_root, pattern)
        if resolved != self.input_root:
            print(f"input root resolved to {resolved}")
            print(f"  (configured {self.input_root}; the mount layout differs)")
        self.input_root = resolved

        config_dir = self.work / "config"
        code = data_audit.audit(dataset, self.input_root, config_dir, self.repo_root)
        if code != 0:
            raise RuntimeError("data_audit failed; see data_profile.json checks")

        if split.run(self.input_root, config_dir, self.repo_root) != 0:
            raise RuntimeError("split failed; see split_manifest.json checks")

        preprocessing.run(self.input_root, config_dir, self.work, self.repo_root)

        self.push("config")
        self.push("cache")

    def phase_shap(self) -> None:
        import shap_selection

        if not self.pull_cache():
            raise RuntimeError("SHAP phase needs the preprocessing cache, which is absent")
        shap_selection.run(self.work / "cache", self.work, self.repo_root)
        self.push("explainability")
        self.push_file("cache/selected_features.json")

    def phase_bo(self) -> None:
        import bayesopt

        self.pull_cache()
        self.io.download_if_present(
            self.key("bayesopt/optuna_study.db"), self.work / "bayesopt" / "optuna_study.db"
        )

        # Sync the study after every trial: a session cancelled mid-search then
        # resumes at the next trial instead of repeating completed ones.
        # A snapshot rather than the live file, which may have writes in flight.
        def after_trial(study, trial) -> None:
            live = self.work / "bayesopt" / "optuna_study.db"
            snapshot = self.work / "bayesopt" / "optuna_study.snapshot.db"
            bayesopt.snapshot_study_db(live, snapshot)
            self.io.upload(snapshot, self.key("bayesopt/optuna_study.db"))

        bayesopt.run(self.work / "cache", self.work, self.repo_root,
                     trial_callback=after_trial)
        self.push("bayesopt")

    def phase_final_train(self) -> None:
        import train

        self.pull_cache()
        self.pull_checkpoints()

        best_params = None
        best_path = self.work / "bayesopt" / "best_params.json"
        self.io.download_if_present(self.key("bayesopt/best_params.json"), best_path)
        if best_path.exists():
            with best_path.open(encoding="utf-8") as handle:
                best_params = json.load(handle)["params"]

        code = train.run(self.work / "cache", self.work, self.repo_root,
                         self.run_id, best_params=best_params)
        self.push("checkpoints")
        self.push("config")
        if code != 0:
            raise RuntimeError(f"training returned {code}")

    def write_run_config(self) -> None:
        payload = {
            "run_id": self.run_id,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": self.experiment,
            "config_hash": config_hash(self.experiment),
            "constraints": {
                "epochs": 100,
                "batch_size": 4096,
                "learning_rate": 0.001,
                "learning_rate_source": "user_constraint",
                # The paper never reports a learning rate; the string "0.001"
                # does not appear in it. See SPEC finding D1.
                "learning_rate_paper_value": "NOT_REPORTED",
                "early_stopping": False,
                "final_model": "model_epoch_100.pt",
            },
            "device": "cpu",
            "versions": _versions(),
        }
        path = self.work / "config" / "run_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        self.push_file("config/run_config.json")


def _versions() -> Dict[str, str]:
    import numpy
    import sklearn
    import torch

    versions = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "cpu_count": str(os.cpu_count()),
    }
    for name in ("optuna", "shap", "lightgbm", "pyarrow"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:                              # noqa: BLE001
            versions[name] = "not installed"
    return versions


def next_phase(phase: str) -> str:
    """The phase that follows, stopping at the end of the order."""
    index = PHASE_ORDER.index(phase)
    return PHASE_ORDER[min(index + 1, len(PHASE_ORDER) - 1)]


PHASE_HANDLERS: Dict[str, Callable[[Session], None]] = {
    "prepare": Session.phase_prepare,
    "shap": Session.phase_shap,
    "bo": Session.phase_bo,
    "final_train": Session.phase_final_train,
}


def run_session(manifest_url: str, work_dir: Path, repo_root: Path,
                input_root: Path, run_id: str, max_phases: int = 4) -> int:
    import requests

    manifest = PresignedManifest.fetch(manifest_url, requests)
    print(manifest.summary())

    io = PresignedIO(manifest)
    session = Session(io, work_dir, repo_root, input_root, run_id)
    session.write_run_config()

    phase = session.current_phase()

    for _ in range(max_phases):
        print(f"\n{'=' * 60}\nphase: {phase}\n{'=' * 60}")

        if phase in ("evaluate", "done"):
            print("training is complete; evaluation runs from evaluate.py")
            return 0

        handler = PHASE_HANDLERS.get(phase)
        if handler is None:
            raise ManifestUnusable(f"no handler for phase {phase!r}")
        handler(session)

        state_path = work_dir / "checkpoints" / "training_state.json"
        if state_path.exists():
            manager = CheckpointManager(work_dir / "checkpoints")
            state = manager.load_state()
            if state and state.status == "RESUME_REQUIRED":
                print("session budget reached; checkpoint uploaded, exiting cleanly")
                return 0

        # Advance locally. current_phase() consults the manifest, which is a
        # snapshot minted before this session began and so cannot see anything
        # uploaded since -- asking it again here would hand back the phase that
        # has only just been finished, and the session would repeat it.
        phase = next_phase(phase)
        if phase in PHASE_HANDLERS:
            session.publish_phase(phase)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest-url", default=os.environ.get("PRESIGNED_MANIFEST_URL"))
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/run"))
    parser.add_argument("--repo-root", type=Path, default=Path("/kaggle/working/code"))
    parser.add_argument("--input-root", type=Path,
                        default=Path("/kaggle/input/cicddos2019-parquet"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if not args.manifest_url:
        print("PRESIGNED_MANIFEST_URL is not set", file=sys.stderr)
        return EXIT_MANIFEST_UNUSABLE

    try:
        return run_session(args.manifest_url, args.work_dir, args.repo_root,
                           args.input_root, args.run_id)
    except PresignedExpired as error:
        # Distinct code: the workflow issues new URLs and resumes. It must not
        # be mistaken for a broken run and must not trigger a fresh start.
        print(f"PRESIGNED EXPIRED: {error}", file=sys.stderr)
        return EXIT_PRESIGNED_EXPIRED
    except ManifestUnusable as error:
        print(f"MANIFEST UNUSABLE: {error}", file=sys.stderr)
        return EXIT_MANIFEST_UNUSABLE


if __name__ == "__main__":
    raise SystemExit(main())
