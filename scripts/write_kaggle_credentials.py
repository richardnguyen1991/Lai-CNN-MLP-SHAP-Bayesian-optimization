"""Write ~/.kaggle/kaggle.json from the repository secrets.

The Kaggle CLI authenticates with a legacy API key only when its configuration
carries both `username` and `key`. Those come from kaggle.json or from the
KAGGLE_USERNAME / KAGGLE_KEY environment pair -- and the workflow sets
KAGGLE_USERNAME but no KAGGLE_KEY, so the file is the only source of the key.

Writing the secret straight into kaggle.json is what failed before: the CLI
parses that file inside a bare `except: pass`, so a value that is not a JSON
object is discarded without a word, and the run dies several steps later with a
generic "authentication required". Hence this script: it accepts either shape
the secret can plausibly hold, and fails loudly and immediately if it holds
neither.

KAGGLE_API_TOKEN may be:
  * a full kaggle.json blob, `{"username": ..., "key": ...}`
  * the raw API key on its own, in which case KAGGLE_USERNAME supplies the name
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def config_dir() -> Path:
    """Where the CLI will look, following its own resolution order.

    On Linux the CLI falls back to ~/.config/kaggle when ~/.kaggle does not
    exist, so the directory has to be created, not merely named.
    """
    override = os.environ.get("KAGGLE_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".kaggle"


def credentials(token: str, username: str) -> dict:
    token = token.strip()
    if not token:
        raise SystemExit("KAGGLE_API_TOKEN is empty; set it in the repository secrets")

    try:
        blob = json.loads(token)
    except json.JSONDecodeError:
        blob = {"key": token}

    if not isinstance(blob, dict):
        raise SystemExit("KAGGLE_API_TOKEN parsed as JSON but is not an object")

    key = str(blob.get("key", "")).strip()
    if not key:
        raise SystemExit("KAGGLE_API_TOKEN holds no API key")

    name = str(blob.get("username", "")).strip() or username.strip()
    if not name:
        raise SystemExit("no Kaggle username: set KAGGLE_USERNAME or embed one in the token")

    return {"username": name, "key": key}


def main() -> int:
    creds = credentials(os.environ.get("KAGGLE_API_TOKEN", ""),
                        os.environ.get("KAGGLE_USERNAME", ""))

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "kaggle.json"
    path.write_text(json.dumps(creds), encoding="utf-8")
    path.chmod(0o600)

    # Never print the key. Its length is enough to tell a truncated secret from
    # a plausible one, and GitHub masks the username anyway.
    print(f"wrote {path} for user {creds['username']} (key length {len(creds['key'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
