"""Resolve a variant into a complete experiment config.

Each file in configs/variants/ holds only what differs from experiment.yaml, so
a reader can see at a glance what one run changes. This merges the two into a
single resolved file, which every module then reads through its existing
--experiment-config argument. No module needs to know variants exist.

Keys starting with an underscore are metadata about the variant rather than
configuration, and are carried through into run_config.json so the resolved file
still says which variant produced it.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive merge; a scalar or list in the override replaces the base value.

    Lists are replaced rather than concatenated: a variant that narrows
    subsample.apply_to means the shorter list, not the union.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def available_variants(repo_root: Path) -> List[str]:
    return sorted(p.stem for p in (repo_root / "configs" / "variants").glob("*.yaml"))


def resolve(repo_root: Path, variant: str) -> Dict[str, Any]:
    base_path = repo_root / "configs" / "experiment.yaml"
    variant_path = repo_root / "configs" / "variants" / f"{variant}.yaml"
    if not variant_path.exists():
        raise FileNotFoundError(
            f"no variant {variant!r}; available: {available_variants(repo_root)}"
        )

    with base_path.open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with variant_path.open(encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}

    merged = deep_merge(base, override)
    merged["run"] = dict(merged.get("run", {}))
    merged["run"]["variant"] = variant
    merged["_variant_metadata"] = {
        key: value for key, value in override.items() if key.startswith("_")
    }
    _validate(merged, variant)
    return merged


def _validate(config: Dict[str, Any], variant: str) -> None:
    """Guard the constraints no variant is allowed to relax."""
    training = config["training"]
    problems = []
    if training["epochs"] != 100:
        problems.append(f"epochs is {training['epochs']}, must stay 100")
    if training["batch_size"] != 4096:
        problems.append(f"batch_size is {training['batch_size']}, must stay 4096")
    if training["learning_rate"] != 0.001:
        problems.append(f"learning_rate is {training['learning_rate']}, must stay 0.001")
    if training["early_stopping"]:
        problems.append("early_stopping must stay false")

    # Only paperlike may leak, and only because measuring the leak is its job.
    leaks = config.get("leakage", {}).get("upsample_before_split", False)
    if leaks and variant != "paperlike":
        problems.append(
            f"variant {variant!r} sets upsample_before_split; only paperlike may"
        )

    if problems:
        raise ValueError(
            f"variant {variant!r} violates a fixed constraint:\n  "
            + "\n  ".join(problems)
        )


def write_resolved(repo_root: Path, variant: str, destination: Path) -> Path:
    resolved = resolve(repo_root, variant)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    path = write_resolved(args.repo_root, args.variant, args.out)
    resolved = resolve(args.repo_root, args.variant)
    metadata = resolved["_variant_metadata"]
    print(f"resolved {args.variant!r} -> {path}")
    print(f"  {metadata.get('_description', '').strip()}")
    print(f"  runs Bayesian search: {metadata.get('_runs_bayesian_search', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
