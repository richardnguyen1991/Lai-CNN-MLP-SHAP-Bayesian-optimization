"""Mint presigned URLs and publish the bootstrap manifest. Runs on the GitHub
runner, which is the only place that ever sees the AWS credentials.

The notebook receives exactly one URL: a presigned GET for the manifest itself.
Everything else is looked up inside it. Embedding hundreds of URLs in the
notebook would bloat it and make an accidental commit far more damaging.

PUT URLs have to be signed before the objects exist, and at cold start the
number of preprocessing shards is not yet known. Keys are therefore generated
from a deterministic naming scheme up to max_shards, and the unused ones simply
expire. That avoids handing the notebook any credential capable of minting
URLs of its own.

Usage (from the workflow):
    python scripts/make_manifest.py --run-id ... --out bootstrap_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presigned_io import contains_signature, redact  # noqa: E402

MAX_SHARDS = 64
SPLITS = ("train", "val", "test")


def run_prefix(prefix: str, run_id: str) -> str:
    return f"{prefix.rstrip('/')}/runs/{run_id}"


def build_key_list(prefix: str, run_id: str) -> List[str]:
    """Every object the session might read or write, in one flat list."""
    base = run_prefix(prefix, run_id)
    keys = [f"{base}/code_bundle.tar.gz"]

    for name in ("scaler.joblib", "selected_features.json"):
        keys.append(f"{base}/cache/{name}")
    for split in SPLITS:
        keys.append(f"{base}/cache/preprocess/{split}_y.npy")
        # Deterministic pool: signed ahead of knowing how many shards exist.
        keys += [f"{base}/cache/preprocess/{split}_X_shard{i:03d}.npy"
                 for i in range(MAX_SHARDS)]

    for name in ("model_last.pt", "model_epoch_100.pt", "model_best_val.pt",
                 "optimizer_last.pt", "rng_state.pt", "training_state.json",
                 "history.json"):
        keys.append(f"{base}/checkpoints/{name}")

    for name in ("optuna_study.db", "optuna_trials.csv", "best_params.json"):
        keys.append(f"{base}/bayesopt/{name}")

    for name in ("run_config.json", "model_config.json", "preprocessing.json",
                 "column_mapping.json", "data_profile.json", "split_manifest.json",
                 "leakage_audit.json", "label_mapping.json", "split_assignment.npy",
                 "file_offsets.json"):
        keys.append(f"{base}/config/{name}")

    for name in ("history.csv", "test_metrics.json", "summary_metrics.csv",
                 "per_class_metrics.csv", "classification_report.txt",
                 "confusion_matrix_raw.csv", "confusion_matrix_norm.csv",
                 "roc_curve.csv", "pr_curve.csv", "inference_benchmark.json"):
        keys.append(f"{base}/metrics/{name}")

    for name in ("shap_feature_ranking.csv", "shap_waterfall_surrogate.csv",
                 "shap_waterfall_surrogate.json", "shap_beeswarm_sample.npz",
                 "comparison_vs_paper_table2.csv", "permutation_importance.csv"):
        keys.append(f"{base}/explainability/{name}")

    keys += [f"{base}/raw/y_true.npy", f"{base}/raw/y_prob.npy",
             f"{base}/session_log.json"]
    return keys


def build_code_bundle(destination: Path) -> Path:
    """Tar up src/ and configs/ so the notebook runs the repository's code."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for directory in ("src", "configs"):
            for path in sorted((REPO_ROOT / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.add(path, arcname=str(path.relative_to(REPO_ROOT)))
    return destination


def existing_keys(client, bucket: str, prefix: str) -> set:
    keys = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            keys.add(item["Key"])
    return keys


def build_manifest(client, bucket: str, keys: List[str], present: set,
                   ttl_seconds: int, run_id: str, phase: str) -> Dict[str, Any]:
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
    entries: Dict[str, Any] = {}

    for key in keys:
        entries[key] = {
            # Absent objects get a null GET: that is how the notebook detects a
            # cold start, rather than by catching a 404.
            "get_url": client.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": key},
                ExpiresIn=ttl_seconds,
            ) if key in present else None,
            "put_url": client.generate_presigned_url(
                "put_object", Params={"Bucket": bucket, "Key": key},
                ExpiresIn=ttl_seconds,
            ),
            "expires_at": expires_at,
        }

    return {
        "run_id": run_id,
        "phase": phase,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires_at,
        "bucket_redacted": bucket[:3] + "***",
        "entries": entries,
    }


def assert_no_signature_in_log(text: str) -> None:
    if contains_signature(text):
        raise SystemExit("refusing to continue: a signature reached the log output")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", default="unknown")
    parser.add_argument("--out", type=Path, default=Path("bootstrap_manifest.json"))
    parser.add_argument("--ttl-hours", type=float, default=13.0)
    parser.add_argument("--emit-url-to", type=Path, default=None,
                        help="file the workflow reads the manifest URL from")
    args = parser.parse_args()

    import boto3

    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "cnnmlp-shap-bayesopt")
    client = boto3.client("s3")
    ttl_seconds = int(args.ttl_hours * 3600)

    base = run_prefix(prefix, args.run_id)
    keys = build_key_list(prefix, args.run_id)

    with tempfile.TemporaryDirectory() as scratch:
        bundle = build_code_bundle(Path(scratch) / "code_bundle.tar.gz")
        client.upload_file(str(bundle), bucket, f"{base}/code_bundle.tar.gz")
        print(f"code bundle uploaded ({bundle.stat().st_size / 1024:.0f} KiB)")

    present = existing_keys(client, bucket, base)
    print(f"{len(present)} objects already exist under {base}")

    manifest = build_manifest(client, bucket, keys, present, ttl_seconds,
                              args.run_id, args.phase)
    payload = json.dumps(manifest, indent=2).encode("utf-8")

    manifest_key = f"{base}/bootstrap_manifest.json"
    client.put_object(Bucket=bucket, Key=manifest_key, Body=payload,
                      ContentType="application/json")
    manifest_url = client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": manifest_key},
        ExpiresIn=ttl_seconds,
    )

    args.out.write_bytes(payload)
    if args.emit_url_to:
        # Written to a file, never echoed: GitHub masks secrets in logs but not
        # a signature that the job itself printed.
        args.emit_url_to.write_text(manifest_url, encoding="utf-8")

    summary = (f"manifest: {len(keys)} keys, {len(present)} present, "
               f"TTL {args.ttl_hours}h, expires {manifest['expires_at']}\n"
               f"manifest url: {redact(manifest_url)}")
    assert_no_signature_in_log(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
