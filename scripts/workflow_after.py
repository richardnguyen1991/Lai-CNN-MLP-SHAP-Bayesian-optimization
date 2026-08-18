"""After a Kaggle session: log it, and decide whether to keep going.

Appends to session_log.json on S3 and sets two outputs:

    needs_more  another session is required
    stalled     two consecutive sessions failed to advance current_epoch

The stall check is what stops the schedule from burning sessions forever on a
run that is not progressing. Without it a silent failure would keep firing every
three hours until max_sessions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

TOTAL_EPOCHS = 100
STALL_LIMIT = 2


def emit(**outputs) -> None:
    target = os.environ.get("GITHUB_OUTPUT")
    for key, value in outputs.items():
        line = f"{key}={value}"
        print(line)
        if target:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def load_json(client, bucket: str, key: str, default):
    try:
        return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:                                  # noqa: BLE001
        return default


def main() -> int:
    import boto3

    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "cnnmlp-shap-bayesopt").rstrip("/")
    run_id = os.environ["RUN_ID"]
    base = f"{prefix}/runs/{run_id}"
    client = boto3.client("s3")

    state = load_json(client, bucket, f"{base}/checkpoints/training_state.json", {})
    epoch = state.get("current_epoch", 0)
    phase = state.get("phase", "prepare")
    total = state.get("total_epochs", TOTAL_EPOCHS)

    log = load_json(client, bucket, f"{base}/session_log.json", [])
    previous_epoch = log[-1]["end_epoch"] if log else None

    entry = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "github_run": os.environ.get("GITHUB_RUN_ID", ""),
        "start_epoch": previous_epoch,
        "end_epoch": epoch,
        "phase": phase,
        "status": state.get("status", "UNKNOWN"),
        "advanced": previous_epoch is None or epoch > previous_epoch,
    }
    log.append(entry)
    client.put_object(
        Bucket=bucket, Key=f"{base}/session_log.json",
        Body=json.dumps(log, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    # Only sessions that were meant to train count towards a stall: a
    # preparation phase legitimately leaves current_epoch at 0.
    recent = [e for e in log[-STALL_LIMIT:] if e["phase"] in ("final_train", "bo")]
    stalled = len(recent) == STALL_LIMIT and not any(e["advanced"] for e in recent)
    finished = epoch >= total and phase in ("evaluate", "done")

    emit(
        needs_more=str(not finished).lower(),
        stalled=str(stalled).lower(),
        current_epoch=epoch,
        phase=phase,
    )
    print(f"{run_id}: epoch {epoch}/{total} phase={phase} "
          f"sessions={len(log)} stalled={stalled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
