"""Put the Kaggle secret where the CLI will actually look for it.

The CLI accepts two unrelated kinds of credential, and putting one where the
other belongs fails in a way that reads like a missing secret:

  legacy API key   32 hex characters, supplied as {"username", "key"} in
                   ~/.kaggle/kaggle.json, or as KAGGLE_USERNAME / KAGGLE_KEY
  access token     the opaque string the settings page issues today, supplied
                   as KAGGLE_API_TOKEN or in ~/.kaggle/access_token

This matters because authenticate() selects the legacy path on the mere
presence of a username and key. An access token written into the "key" field is
therefore accepted locally, sent as a legacy key, and rejected by the server --
and the CLI answers a 401 by printing its generic "authentication required"
help, which names neither the file it read nor the credential it sent. That
cost a full diagnosis cycle; the shape check below is what stops it recurring.

KAGGLE_API_TOKEN may hold a full kaggle.json blob, a bare legacy key, or an
access token. Each is routed to the file that matches it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

LEGACY_KEY = re.compile("[0-9a-f]{32}")


def config_dir() -> Path:
    """Where the CLI will look, following its own resolution order.

    On Linux the CLI falls back to ~/.config/kaggle when ~/.kaggle does not
    exist, so the directory has to be created, not merely named.
    """
    override = os.environ.get("KAGGLE_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".kaggle"


def credentials(token: str, username: str):
    """Classify the secret. Returns ("legacy", {...}) or ("access", token)."""
    token = token.strip()
    if not token:
        raise SystemExit("KAGGLE_API_TOKEN is empty; set it in the repository secrets")

    try:
        blob = json.loads(token)
    except json.JSONDecodeError:
        blob = None

    if blob is not None:
        if not isinstance(blob, dict):
            raise SystemExit("KAGGLE_API_TOKEN parsed as JSON but is not an object")
        key = str(blob.get("key", "")).strip()
        if not key:
            raise SystemExit("KAGGLE_API_TOKEN is a JSON object with no 'key' field")
        name = str(blob.get("username", "")).strip() or username.strip()
        if not name:
            raise SystemExit("no Kaggle username: set KAGGLE_USERNAME or embed one in the token")
        return "legacy", {"username": name, "key": key}

    if LEGACY_KEY.fullmatch(token):
        name = username.strip()
        if not name:
            raise SystemExit("no Kaggle username: set KAGGLE_USERNAME alongside the API key")
        return "legacy", {"username": name, "key": token}

    # Anything else is treated as an access token. The CLI introspects it for
    # the username, so none is needed here.
    return "access", token


def main() -> int:
    kind, payload = credentials(os.environ.get("KAGGLE_API_TOKEN", ""),
                                os.environ.get("KAGGLE_USERNAME", ""))

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)

    # Never print a credential. Kind and length are enough to tell a plausible
    # secret from the wrong sort of string, and GitHub masks the username.
    if kind == "legacy":
        path = directory / "kaggle.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o600)
        print(f"wrote {path} as a legacy API key for {payload['username']}")
    else:
        path = directory / "access_token"
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o600)
        print(f"wrote {path} as an access token (length {len(payload)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
