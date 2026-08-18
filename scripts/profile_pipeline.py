"""Measure the pipeline on a bounded subset, then extrapolate to the real run.

Step 9 of the delivery order. Nothing is committed to a 100-epoch run until the
numbers here say it fits: the hand estimate was 115 hours per run, which would
be roughly ten Kaggle sessions for one of seven variants.

Bounded on purpose. Running the full prepare phase to find out how long the full
prepare phase takes costs exactly as much as the thing being measured, and if it
does not fit in a session there is no checkpoint to resume from. So it runs on a
few capture files, measures a rate, and scales.

What is measured, each with its own peak RSS:
    audit -> split -> preprocess -> SHAP selection -> N training epochs

What is extrapolated:
    the prepare phases, by rows
    training, by rows and by the 100 epochs plus 20 x 10 Bayesian trials
    sessions needed, against session_time_budget_minutes

Usage:
    python scripts/profile_pipeline.py --input-root /kaggle/input/... \
        --work-dir /kaggle/working/profile --max-files 3 --profile-epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def peak_rss_mb() -> Optional[float]:
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except ImportError:
        try:
            import psutil

            return round(psutil.Process().memory_info().rss / 2 ** 20, 1)
        except ImportError:
            return None


class Profiler:
    def __init__(self) -> None:
        self.stages: List[Dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str):
        print(f"\n--- {name} ---", flush=True)
        started = time.perf_counter()
        before = peak_rss_mb()
        record: Dict[str, Any] = {"stage": name}
        self.stages.append(record)
        try:
            yield record
        finally:
            record["seconds"] = round(time.perf_counter() - started, 2)
            # getrusage reports the high-water mark for the WHOLE process, so a
            # stage's raw figure includes every earlier stage. Reporting it as
            # "this stage used 2.2 GB" would be wrong; the rise is the honest
            # per-stage number.
            after = peak_rss_mb()
            record["process_peak_rss_mb"] = after
            record["rss_rise_mb"] = (round(after - before, 1)
                                     if after is not None and before is not None
                                     else None)
            print(f"    {name}: {record['seconds']:.1f}s  "
                  f"process peak {after} MB (rose {record['rss_rise_mb']} MB)",
                  flush=True)

    def get(self, name: str) -> Dict[str, Any]:
        return next(s for s in self.stages if s["stage"] == name)


def count_rows(files: List[Path]) -> int:
    """Row counts come from the Parquet footer; no column data is read."""
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def count_benign(files: List[Path], label_candidates: List[str]) -> int:
    """BENIGN rows across the given files, reading only the label column.

    Needed because training cost does NOT scale with total rows. Under decision
    B2 the training split is (all BENIGN) x (1 + attack_per_benign) x train_ratio,
    so it scales with the BENIGN count. The first three capture files hold 7,020
    of the 113,828 BENIGN rows but 11.8M of the 70.4M total: scaling epoch time
    by total rows would understate the real cost by a factor of about 2.7.
    """
    from schema import find_label_column

    total = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        label = find_label_column(list(parquet.schema_arrow.names), label_candidates)
        for group in range(parquet.num_row_groups):
            column = parquet.read_row_group(group, columns=[label]).column(label)
            total += sum(1 for v in column.to_pylist()
                         if str(v).strip().upper() in ("BENIGN", "NORMAL"))
    return total


def build_subset(input_root: Path, destination: Path, max_files: int) -> List[Path]:
    """A directory holding the first N shards, preserving the capture-day layout.

    Symlinks where the platform allows, copies otherwise. The directory layout
    matters: split.py reads the capture day out of the relative path.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    everything = sorted(input_root.rglob("*.parquet"))
    if not everything:
        raise SystemExit(f"no parquet under {input_root}")

    chosen = everything[:max_files] if max_files > 0 else everything
    for source in chosen:
        target = destination / source.relative_to(input_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(source)
        except (OSError, NotImplementedError):
            shutil.copy2(source, target)
    return chosen


def load_configs(experiment_config: Optional[Path]) -> Dict[str, Any]:
    """Config trio, with an override hook.

    The override exists so a smoke test can run the whole chain in a minute on
    a tiny fixture, and so a variant can be profiled without editing the repo's
    configs. The real profile run passes nothing and reads what the pipeline
    itself will read.
    """
    path = experiment_config or (REPO_ROOT / "configs" / "experiment.yaml")
    with path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    with (REPO_ROOT / "configs" / "model.yaml").open(encoding="utf-8") as handle:
        model_yaml = yaml.safe_load(handle)
    with (REPO_ROOT / "configs" / "bayesopt.yaml").open(encoding="utf-8") as handle:
        bo_cfg = yaml.safe_load(handle)
    return {"experiment": experiment, "model": model_yaml, "bayesopt": bo_cfg,
            "path": path}


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    import data_audit
    import preprocessing
    import shap_selection
    import split
    import torch
    import torch.nn as nn
    from dataset import SelectedFeatures, load_split
    from model import build_model, model_config_from_yaml
    from train import build_optimizer, configure_cpu, evaluate_split, train_one_epoch

    configs = load_configs(args.experiment_config)
    experiment, model_yaml, bo_cfg = (configs["experiment"], configs["model"],
                                      configs["bayesopt"])

    threads = configure_cpu(experiment["device"].get("cpu_threads", "auto"))
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    with (REPO_ROOT / "configs" /
          f"dataset_{experiment['run']['dataset']}.yaml").open(encoding="utf-8") as handle:
        dataset_cfg = yaml.safe_load(handle)

    all_files = sorted(args.input_root.rglob("*.parquet"))
    rows_full = count_rows(all_files)
    subset_files = build_subset(args.input_root, work / "subset", args.max_files)
    rows_subset = count_rows(subset_files)

    # Two different scale factors, because two different things scale.
    row_scale = rows_full / rows_subset
    print("counting BENIGN across the full dataset (label column only)...", flush=True)
    candidates = dataset_cfg["label_column_candidates"]
    benign_full = count_benign(all_files, candidates)
    benign_subset = count_benign(subset_files, candidates)
    benign_scale = benign_full / max(benign_subset, 1)

    print(f"CPU threads      : {threads}")
    print(f"full dataset     : {len(all_files)} files, {rows_full:,} rows, "
          f"{benign_full:,} BENIGN")
    print(f"profiling subset : {len(subset_files)} files, {rows_subset:,} rows, "
          f"{benign_subset:,} BENIGN")
    print(f"scale, prepare   : {row_scale:.2f}x  (by total rows)")
    print(f"scale, training  : {benign_scale:.2f}x  (by BENIGN, which sets train size)")
    if benign_subset == 0:
        raise SystemExit("the subset holds no BENIGN rows; raise --max-files")

    profiler = Profiler()
    config_dir = work / "config"
    subset_root = work / "subset"

    with profiler.stage("audit") as record:
        code = data_audit.audit(experiment["run"]["dataset"], subset_root,
                                config_dir, REPO_ROOT)
        record["rows"] = rows_subset
        record["exit_code"] = code

    with profiler.stage("split") as record:
        record["exit_code"] = split.run(subset_root, config_dir, REPO_ROOT,
                                        configs["path"])
        with (config_dir / "split_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        record["summary"] = manifest["summary"]
        record["rows"] = rows_subset

    train_rows = manifest["summary"]["train"]["rows"]
    if train_rows == 0:
        raise SystemExit("the subset produced an empty training split; raise --max-files")

    with profiler.stage("preprocess") as record:
        preprocessing.run(subset_root, config_dir, work, REPO_ROOT, configs["path"])
        with (work / "config" / "preprocessing.json").open(encoding="utf-8") as handle:
            record["n_kept_features"] = json.load(handle)["n_kept"]
        record["rows"] = rows_subset

    with profiler.stage("shap_selection") as record:
        shap_selection.run(work / "cache", work, REPO_ROOT, configs["path"])
        record["rows"] = train_rows

    # --- training ---------------------------------------------------------
    features = SelectedFeatures(work / "cache")
    X_train, y_train = load_split(work / "cache", "train", features)
    X_val, y_val = load_split(work / "cache", "val", features)

    training_cfg = experiment["training"]
    torch.manual_seed(experiment["run"]["seed"])
    model = build_model(model_config_from_yaml(model_yaml, len(features)),
                        seed=experiment["run"]["seed"])
    optimizer = build_optimizer(model, training_cfg)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(experiment["run"]["seed"])

    epoch_seconds: List[float] = []
    with profiler.stage("train_epochs") as record:
        for epoch in range(1, args.profile_epochs + 1):
            started = time.perf_counter()
            train_one_epoch(model, optimizer, criterion, X_train, y_train,
                            training_cfg["batch_size"], generator)
            evaluate_split(model, X_val, y_val, training_cfg["batch_size"],
                           training_cfg["decision_threshold"])
            elapsed = time.perf_counter() - started
            epoch_seconds.append(elapsed)
            print(f"    epoch {epoch}: {elapsed:.1f}s", flush=True)

        record["epochs"] = args.profile_epochs
        record["train_rows"] = int(len(X_train))
        record["val_rows"] = int(len(X_val))
        record["n_features"] = len(features)
        record["parameters"] = sum(p.numel() for p in model.parameters())
        # The first epoch carries allocator warm-up; the median is the honest
        # figure for the other 99.
        record["median_epoch_seconds"] = round(float(np.median(epoch_seconds)), 2)
        record["epoch_seconds"] = [round(s, 2) for s in epoch_seconds]

    return build_report(profiler, experiment, bo_cfg, row_scale, benign_scale,
                        rows_full, rows_subset, benign_full, benign_subset,
                        train_rows, threads, args)


def build_report(profiler, experiment, bo_cfg, row_scale, benign_scale,
                 rows_full, rows_subset, benign_full, benign_subset,
                 train_rows_subset, threads, args) -> Dict[str, Any]:
    session_cfg = experiment["session"]
    usable_minutes = (session_cfg["session_time_budget_minutes"]
                      - session_cfg["safety_margin_minutes"])

    prepare_subset = sum(profiler.get(name)["seconds"]
                         for name in ("audit", "split", "preprocess"))
    prepare_full_minutes = prepare_subset * row_scale / 60

    train_stage = profiler.get("train_epochs")
    # Training scales with the BENIGN count, not with total rows: B2 keeps every
    # BENIGN row and thins attacks to a multiple of it. SHAP selection scales
    # with neither -- it runs on a fixed-size sample.
    epoch_minutes_full = train_stage["median_epoch_seconds"] * benign_scale / 60
    shap_minutes = profiler.get("shap_selection")["seconds"] / 60

    epochs_final = experiment["training"]["epochs"]
    bo_epochs = bo_cfg["n_trials"] * bo_cfg["bo_epochs"]
    train_minutes = epoch_minutes_full * epochs_final
    bo_minutes = epoch_minutes_full * bo_epochs

    one_run_minutes = prepare_full_minutes + shap_minutes + bo_minutes + train_minutes
    # Six of the seven variants reuse the cache and the search, so they cost
    # only their own 100 epochs.
    reuse_minutes = train_minutes
    all_variants_minutes = one_run_minutes + 6 * reuse_minutes

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "cpu_threads": threads,
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "peak_rss_mb_overall": max(
                (s.get("process_peak_rss_mb") or 0) for s in profiler.stages
            ),
            "ram_budget_gb": experiment["device"]["ram_budget_gb"],
            "projected_peak_gb": round(_projected_peak_gb(rows_full, benign_full), 2),
            "ram_note": ("Projected analytically, not by scaling the observed RSS. "
                         "Most of the measured footprint is per-row-group buffers "
                         "that do not grow with the dataset; what does grow is "
                         "split's per-row index and its dedup workspace."),
        },
        "subset": {
            "files": args.max_files,
            "rows": rows_subset,
            "benign": benign_subset,
            "train_rows": train_rows_subset,
            "row_scale_to_full": round(row_scale, 2),
            "benign_scale_to_full": round(benign_scale, 2),
            "scale_note": ("prepare scales by total rows; training scales by BENIGN, "
                           "since B2 sets train size from the BENIGN count"),
        },
        "full_dataset_rows": rows_full,
        "full_dataset_benign": benign_full,
        "projected_train_rows": int(train_rows_subset * benign_scale),
        "stages": profiler.stages,
        "extrapolation_minutes": {
            "prepare_once": round(prepare_full_minutes, 1),
            "shap_once": round(shap_minutes, 1),
            "per_epoch": round(epoch_minutes_full, 3),
            "bayesian_search": round(bo_minutes, 1),
            "final_training_100_epochs": round(train_minutes, 1),
            "first_run_total": round(one_run_minutes, 1),
            "each_reused_variant": round(reuse_minutes, 1),
            "all_seven_variants": round(all_variants_minutes, 1),
        },
        "sessions": {
            "usable_minutes_per_session": usable_minutes,
            "first_run": _ceil_div(one_run_minutes, usable_minutes),
            "all_seven_variants": _ceil_div(all_variants_minutes, usable_minutes),
            "max_sessions_configured": session_cfg["max_sessions"],
        },
        "verdict": _verdict(all_variants_minutes, usable_minutes,
                            session_cfg["max_sessions"],
                            _projected_peak_gb(rows_full, benign_full),
                            experiment["device"]["ram_budget_gb"]),
    }


def _projected_peak_gb(rows_full: int, benign_full: int, n_features: int = 68,
                       attack_per_benign: int = 30, train_ratio: float = 0.70) -> float:
    """Largest resident footprint at full scale.

    split.py dominates: 19 bytes per row for the index (file id, sort key, label,
    raw label, row hash) plus the stable argsort workspace that deduplication
    needs. preprocessing holds the training matrix, which is far smaller because
    B2 sizes it from the BENIGN count.
    """
    index_bytes = rows_full * (1 + 8 + 1 + 1 + 8)
    dedup_bytes = rows_full * (8 + 8 + 1)
    train_rows = benign_full * (1 + attack_per_benign) * train_ratio
    train_bytes = train_rows * n_features * 4
    return max(index_bytes + dedup_bytes, train_bytes) / 2 ** 30


def _ceil_div(minutes: float, per_session: float) -> int:
    return int(-(-minutes // per_session))


def _verdict(total_minutes: float, usable: float, max_sessions: int,
             projected_peak_gb: float, ram_budget_gb: float) -> Dict[str, Any]:
    needed = _ceil_div(total_minutes, usable)
    fits = needed <= max_sessions
    ram_fits = projected_peak_gb <= ram_budget_gb
    return {
        "fits_within_max_sessions": fits,
        "sessions_needed": needed,
        "fits_within_ram_budget": ram_fits,
        "projected_peak_gb": round(projected_peak_gb, 2),
        "ram_message": (
            f"projected peak {projected_peak_gb:.2f} GiB against a "
            f"{ram_budget_gb} GiB budget"
            + ("" if ram_fits else " -- OVER. Shard the split index or raise the budget.")
        ),
        "message": (
            f"{needed} sessions needed against max_sessions={max_sessions}. "
            + ("Within budget." if fits else
               "OVER BUDGET. Options, in the order section 15 requires them to be "
               "offered: reduce n_trials, lower max_train_rows with a stratified "
               "subsample, or raise max_sessions. Do not cut epochs.")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/profile"))
    parser.add_argument("--max-files", type=int, default=3,
                        help="capture files to profile on; 0 means all")
    parser.add_argument("--profile-epochs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--experiment-config", type=Path, default=None,
                        help="override experiment.yaml, e.g. to profile a variant")
    args = parser.parse_args()

    report = profile(args)
    out = args.out or (args.work_dir / "profile_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    extrapolation = report["extrapolation_minutes"]
    print("\n" + "=" * 62)
    print("EXTRAPOLATION TO THE FULL DATASET")
    print("=" * 62)
    print(f"  prepare (once)          {extrapolation['prepare_once']:>10,.1f} min")
    print(f"  SHAP selection (once)   {extrapolation['shap_once']:>10,.1f} min")
    print(f"  per epoch               {extrapolation['per_epoch']:>10,.3f} min")
    print(f"  Bayesian search         {extrapolation['bayesian_search']:>10,.1f} min")
    print(f"  final training x100     {extrapolation['final_training_100_epochs']:>10,.1f} min")
    print(f"  first run total         {extrapolation['first_run_total']:>10,.1f} min "
          f"({extrapolation['first_run_total'] / 60:.1f} h)")
    print(f"  all seven variants      {extrapolation['all_seven_variants']:>10,.1f} min "
          f"({extrapolation['all_seven_variants'] / 60:.1f} h)")
    print(f"\n  peak RSS observed       {report['environment']['peak_rss_mb_overall']:,.0f} MB "
          f"(budget {report['environment']['ram_budget_gb']} GB)")
    print(f"\n  {report['verdict']['message']}")

    print("\n===== PASTE THIS BACK =====")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
