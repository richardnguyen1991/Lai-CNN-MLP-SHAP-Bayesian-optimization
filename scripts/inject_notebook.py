"""Write the manifest URL into a throwaway copy of the notebook.

Only the cell tagged `injected-parameters` is rewritten, so the rest of the
notebook stays byte-identical to the committed version.

The output goes to a path the workflow deletes and that .gitignore blocks. A
presigned URL in a committed notebook would be a live, signed credential in
public history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presigned_io import contains_signature, redact  # noqa: E402

TAG = "injected-parameters"


def inject(notebook_path: Path, manifest_url: str, run_id: str,
           input_root: str, destination: Path) -> Path:
    with notebook_path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    target = [
        cell for cell in notebook["cells"]
        if TAG in cell.get("metadata", {}).get("tags", [])
    ]
    if len(target) != 1:
        raise SystemExit(
            f"expected exactly one cell tagged {TAG!r}, found {len(target)}"
        )

    target[0]["source"] = [
        "# Injected by GitHub Actions. This file is never committed.\n",
        f'PRESIGNED_MANIFEST_URL = "{manifest_url}"\n',
        f'RUN_ID = "{run_id}"\n',
        f'INPUT_ROOT = "{input_root}"',
    ]

    # Outputs can carry anything the previous run printed; strip them so a
    # stale value cannot ride along into the pushed kernel.
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1)
    return destination


def assert_committed_notebook_is_clean(notebook_path: Path) -> None:
    """The version in git must never contain signing material."""
    text = notebook_path.read_text(encoding="utf-8")
    if contains_signature(text):
        raise SystemExit(f"{notebook_path} contains a signature; refusing to proceed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebook", type=Path, default=REPO_ROOT / "kaggle_notebook.ipynb")
    parser.add_argument("--manifest-url-file", type=Path, required=True,
                        help="file holding the URL; never passed on the command line")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-root", default="/kaggle/input/cicddos2019-parquet")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    assert_committed_notebook_is_clean(args.notebook)
    manifest_url = args.manifest_url_file.read_text(encoding="utf-8").strip()

    # Read from a file rather than argv: command lines show up in process
    # listings and in the workflow's own step log.
    written = inject(args.notebook, manifest_url, args.run_id, args.input_root, args.out)
    print(f"injected into {written} (url {redact(manifest_url)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
