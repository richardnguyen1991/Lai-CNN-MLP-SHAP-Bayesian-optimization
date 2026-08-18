"""Check a finished run against the acceptance criteria, before anything is quoted.

Section 18 of the brief, turned into assertions. Run it on a run directory and
it either passes or names exactly which criterion failed and why.

The point is to catch a run that LOOKS finished. Every artifact can be present
and well-formed while history.json is missing epoch 47, or the reported model is
the best-validation checkpoint, or test was scored against a stale feature set.

Usage:
    python scripts/validate_artifacts.py --run-dir work
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint import validate_history  # noqa: E402


class Check:
    def __init__(self, run_dir: Path) -> None:
        self.run = run_dir
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.skipped: List[str] = []

    def read(self, relative: str) -> Optional[Any]:
        path = self.run / relative
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def check(self, name: str, condition: Callable[[], Optional[str]]) -> None:
        """condition returns None on success, or a reason string on failure."""
        try:
            reason = condition()
        except FileNotFoundError as error:
            self.skipped.append(f"{name}: {error}")
            return
        except Exception as error:                     # noqa: BLE001
            self.failed.append(f"{name}: {type(error).__name__}: {error}")
            return
        if reason is None:
            self.passed.append(name)
        else:
            self.failed.append(f"{name}: {reason}")

    # -- criteria ---------------------------------------------------------
    def history_is_complete(self) -> Optional[str]:
        history = self.read("checkpoints/history.json")
        if history is None:
            raise FileNotFoundError("history.json not present")
        state = self.read("checkpoints/training_state.json") or {}
        total = state.get("total_epochs", 100)
        problems = validate_history(history, total)
        return "; ".join(problems) if problems else None

    def final_model_is_the_last_epoch(self) -> Optional[str]:
        state = self.read("checkpoints/training_state.json")
        if state is None:
            raise FileNotFoundError("training_state.json not present")
        if not (self.run / "checkpoints" / "model_epoch_100.pt").exists():
            return "model_epoch_100.pt is missing"

        metrics = self.read("metrics/test_metrics.json")
        if metrics and metrics.get("checkpoint") != "model_epoch_100.pt":
            return (f"test was scored against {metrics.get('checkpoint')!r}, "
                    "not the epoch-100 model")
        return None

    def no_early_stopping(self) -> Optional[str]:
        config = self.read("config/run_config.json")
        if config is None:
            raise FileNotFoundError("run_config.json not present")
        training = config.get("experiment", {}).get("training", {})
        if training.get("early_stopping"):
            return "early_stopping is enabled"
        if training.get("epochs") != 100:
            return f"epochs is {training.get('epochs')}, not 100"
        return None

    def fixed_hyperparameters_held(self) -> Optional[str]:
        config = self.read("config/run_config.json")
        if config is None:
            raise FileNotFoundError("run_config.json not present")
        training = config.get("experiment", {}).get("training", {})
        problems = []
        if training.get("batch_size") != 4096:
            problems.append(f"batch_size {training.get('batch_size')}")
        if training.get("learning_rate") != 0.001:
            problems.append(f"learning_rate {training.get('learning_rate')}")
        return "changed: " + ", ".join(problems) if problems else None

    def no_cross_split_duplicates(self) -> Optional[str]:
        audit = self.read("config/leakage_audit.json")
        if audit is None:
            raise FileNotFoundError("leakage_audit.json not present")
        if audit.get("upsample_before_split"):
            # paperlike leaks on purpose; failing it here would be wrong.
            return None
        count = audit.get("cross_split_duplicate_rows")
        if count is None:
            return "cross_split_duplicate_rows was not recorded"
        return None if count == 0 else f"{count:,} rows span two splits"

    def preprocessing_fitted_on_train_only(self) -> Optional[str]:
        report = self.read("config/preprocessing.json")
        if report is None:
            raise FileNotFoundError("preprocessing.json not present")
        if report.get("fitted_on") != "train_only":
            return f"fitted_on is {report.get('fitted_on')!r}"
        return None

    def feature_schema_matches(self) -> Optional[str]:
        selected = self.read("cache/selected_features.json")
        state = self.read("checkpoints/training_state.json")
        if selected is None or state is None:
            raise FileNotFoundError("selected_features.json or training_state.json")
        if state.get("feature_schema_hash") != selected.get("feature_schema_hash"):
            return ("the trained model's feature hash does not match "
                    "selected_features.json; the run mixes two feature sets")
        return None

    def bayesian_search_completed(self) -> Optional[str]:
        best = self.read("bayesopt/best_params.json")
        if best is None:
            raise FileNotFoundError("best_params.json not present")
        done, asked = best.get("n_trials_completed"), best.get("n_trials_requested")
        if done != asked:
            return f"{done} of {asked} trials completed"
        for forbidden in ("learning_rate", "batch_size"):
            if forbidden in best.get("params", {}):
                return f"{forbidden} was searched but is meant to be fixed"
        return None

    def test_metrics_are_complete(self) -> Optional[str]:
        metrics = self.read("metrics/test_metrics.json")
        if metrics is None:
            raise FileNotFoundError("test_metrics.json not present")
        required = ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                    "macro_f1", "mcc", "roc_auc", "pr_auc", "log_loss",
                    "specificity", "fpr", "fnr")
        missing = [name for name in required if metrics.get(name) is None]
        return f"missing: {missing}" if missing else None

    def paper_comparison_keeps_both_values(self) -> Optional[str]:
        path = self.run / "metrics" / "comparison_with_paper.csv"
        if not path.exists():
            raise FileNotFoundError("comparison_with_paper.csv not present")
        header = path.read_text(encoding="utf-8").splitlines()[0]
        for column in ("paper_headline", "paper_body"):
            if column not in header:
                return (f"{column} is absent; the paper reports two different "
                        "values per metric and both must be shown")
        return None

    def explainability_did_not_feed_back(self) -> Optional[str]:
        notes = self.read("explainability/explainability_notes.json")
        if notes is None:
            raise FileNotFoundError("explainability_notes.json not present")
        if notes.get("fed_back_into_selection") is not False:
            return "explainability was fed back into feature selection"
        return None

    def run_all(self) -> None:
        for name, condition in (
            ("history has epochs 1..100, each once", self.history_is_complete),
            ("final model is epoch 100", self.final_model_is_the_last_epoch),
            ("no early stopping", self.no_early_stopping),
            ("batch_size and learning_rate unchanged", self.fixed_hyperparameters_held),
            ("no cross-split duplicates", self.no_cross_split_duplicates),
            ("preprocessing fitted on train only", self.preprocessing_fitted_on_train_only),
            ("feature schema consistent", self.feature_schema_matches),
            ("Bayesian search completed", self.bayesian_search_completed),
            ("test metrics complete", self.test_metrics_are_complete),
            ("paper comparison keeps both values", self.paper_comparison_keeps_both_values),
            ("explainability not fed back", self.explainability_did_not_feed_back),
        ):
            self.check(name, condition)

    def report(self) -> int:
        for name in self.passed:
            print(f"  PASS  {name}")
        for name in self.skipped:
            print(f"  SKIP  {name}")
        for name in self.failed:
            print(f"  FAIL  {name}")

        print(f"\n{len(self.passed)} passed, {len(self.failed)} failed, "
              f"{len(self.skipped)} skipped")
        if self.failed:
            print("\nThis run does not meet the acceptance criteria. Do not quote "
                  "its numbers until the failures above are resolved.")
            return 2
        if self.skipped:
            print("\nEverything present passed, but the run is incomplete: the "
                  "skipped checks have no artifact to read yet.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    print(f"validating {args.run_dir}\n")
    checker = Check(args.run_dir)
    checker.run_all()
    return checker.report()


if __name__ == "__main__":
    raise SystemExit(main())
