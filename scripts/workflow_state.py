"""Decide, from S3, whether this invocation resumes a run, starts one, or stops.

Writes run_id, phase and finished to GITHUB_OUTPUT. Runs on the GitHub runner,
which holds the AWS credentials; nothing here goes near Kaggle.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TOTAL_EPOCHS = 100


# Sections that describe how the work is carried out rather than what the
# experiment is. Retiming a session or moving a path changes neither the model
# nor the data, so it must not orphan a run and force hours of completed
# preparation, feature selection and Bayesian search to be repeated.
OPERATIONAL_SECTIONS = {"session", "paths"}


def experimental_config(name: str, path: Path):
    """The part of a config file that decides what the results will be."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if name == "experiment.yaml" and isinstance(payload, dict):
        payload = {k: v for k, v in payload.items() if k not in OPERATIONAL_SECTIONS}
    return payload


def config_fingerprint(variant: str) -> str:
    """Short hash over the configs, so a changed experiment gets a new run_id.

    Hashing the raw bytes would be simpler, but it cannot tell a changed model
    from a changed timeout, and treats a comment as a new experiment.
    """
    digest = hashlib.sha256()
    for name in sorted(os.listdir(REPO_ROOT / "configs")):
        if name.endswith(".yaml"):
            payload = experimental_config(name, REPO_ROOT / "configs" / name)
            digest.update(name.encode("utf-8"))
            digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    digest.update(variant.encode("utf-8"))
    return digest.hexdigest()[:8]


def emit(**outputs) -> None:
    target = os.environ.get("GITHUB_OUTPUT")
    for key, value in outputs.items():
        line = f"{key}={value}"
        print(line)
        if target:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def main() -> int:
    import boto3

    dataset = os.environ.get("DATASET", "cicddos2019")
    variant = os.environ.get("VARIANT", "main")
    force_new = os.environ.get("FORCE_NEW_RUN", "false").lower() == "true"
    override = os.environ.get("RUN_ID_OVERRIDE", "").strip()
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "cnnmlp-shap-bayesopt").rstrip("/")

    fingerprint = config_fingerprint(variant)
    client = boto3.client("s3")

    if override:
        # Adopting a run by name: used when a change that cannot affect the
        # results would otherwise strand hours of finished work under an old id.
        state_key = f"{prefix}/runs/{override}/checkpoints/training_state.json"
        try:
            state = json.loads(client.get_object(Bucket=bucket, Key=state_key)["Body"].read())
        except Exception:                                  # noqa: BLE001
            emit(run_id=override, phase="prepare", finished="false")
            print(f"adopting {override}; no training state yet, preparing")
            return 0
        epoch = state.get("current_epoch", 0)
        phase = state.get("phase", "prepare")
        total = state.get("total_epochs", TOTAL_EPOCHS)
        emit(run_id=override, phase=phase,
             finished=str(epoch >= total and phase in ("evaluate", "done")).lower())
        print(f"adopting {override}: phase={phase} epoch={epoch}/{total}")
        return 0

    # Existing runs for this dataset+variant with the same config fingerprint.
    # A different fingerprint means a different experiment and gets its own id.
    paginator = client.get_paginator("list_objects_v2")
    candidates = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/runs/", Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            name = item["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            if name.startswith(f"{dataset}-{variant}-") and name.endswith(fingerprint):
                candidates.add(name)

    if force_new or not candidates:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        run_id = f"{dataset}-{variant}-{stamp}-{fingerprint}"
        emit(run_id=run_id, phase="prepare", finished="false")
        print(f"starting a new run: {run_id}")
        return 0

    run_id = sorted(candidates)[-1]
    state_key = f"{prefix}/runs/{run_id}/checkpoints/training_state.json"
    try:
        body = client.get_object(Bucket=bucket, Key=state_key)["Body"].read()
        state = json.loads(body.decode("utf-8"))
    except client.exceptions.NoSuchKey:
        emit(run_id=run_id, phase="prepare", finished="false")
        print(f"{run_id}: no training state yet; preparing")
        return 0

    epoch = state.get("current_epoch", 0)
    phase = state.get("phase", "prepare")
    total = state.get("total_epochs", TOTAL_EPOCHS)
    finished = epoch >= total and phase in ("evaluate", "done")

    sessions = state.get("sessions", [])
    if len(sessions) >= int(os.environ.get("MAX_SESSIONS", "20")):
        print(f"{run_id}: reached max_sessions ({len(sessions)}); refusing to push more",
              file=sys.stderr)
        emit(run_id=run_id, phase=phase, finished="true")
        return 0

    emit(run_id=run_id, phase=phase, finished=str(finished).lower())
    print(f"{run_id}: phase={phase} epoch={epoch}/{total} sessions={len(sessions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
