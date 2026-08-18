"""S3 access from Kaggle, through presigned URLs only.

The Kaggle notebook holds no AWS credentials and never calls boto3. GitHub
Actions is the only place that reads the repository secrets; it mints short-
lived presigned GET and PUT URLs, writes them into a manifest, and hands the
notebook a single URL pointing at that manifest.

Three rules, each enforced here rather than left to discipline:

  A signature never reaches a log. Query strings carry X-Amz-Signature, so
  every URL is redacted before it can be printed, and __repr__ is overridden so
  an accidental print of the object cannot leak one either.

  An expired URL fails loudly, with its own exit code. Silently starting a
  fresh run instead would discard a checkpoint that is still perfectly good and
  restart 100 epochs from zero.

  A missing key is cold start, not an error. GitHub Actions writes null for
  objects that do not exist yet, which is how the first session tells that
  there is nothing to resume.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol

# Distinct exit codes so the workflow can tell "give me fresh URLs" apart from
# "this run is broken".
EXIT_PRESIGNED_EXPIRED = 75
EXIT_MANIFEST_UNUSABLE = 76

_QUERY = re.compile(r"\?.*$", re.DOTALL)

# A signing parameter only matters when it carries a value. Matching the bare
# name would flag documentation and comments that merely mention it, and a guard
# that cries wolf on its own source gets switched off.
_SIGNATURE = re.compile(
    r"(?:X-Amz-Signature|X-Amz-Credential|X-Amz-Security-Token|AWSAccessKeyId"
    r"|(?<![\w-])Signature)=[^&\s\"'<>]{8,}",
    re.IGNORECASE,
)


class PresignedExpired(RuntimeError):
    """A URL outlived its TTL. The workflow must mint new ones."""


class ManifestUnusable(RuntimeError):
    """The manifest is missing, malformed, or lacks a key that is required."""


class Transport(Protocol):
    def get(self, url: str, timeout: int = ...) -> Any: ...
    def put(self, url: str, data: bytes, timeout: int = ...) -> Any: ...


def redact(url: str) -> str:
    """A URL safe to print: path kept, entire query string dropped."""
    if not url:
        return ""
    return _QUERY.sub("?<redacted>", url)


def contains_signature(text: str) -> bool:
    """True if a string carries live signing material.

    Requires a parameter WITH a value, so prose naming X-Amz-Signature does not
    trip the guard while a real presigned URL always does.
    """
    return _SIGNATURE.search(text) is not None


class PresignedManifest:
    """The mapping from S3 key to short-lived URLs, as written by GitHub Actions.

    Entry shape: {"get_url": str|None, "put_url": str|None, "expires_at": iso8601}
    A null entry means the object does not exist yet.
    """

    def __init__(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict) or "entries" not in payload:
            raise ManifestUnusable("manifest has no 'entries' object")
        self.run_id: str = payload.get("run_id", "")
        self.phase: Optional[str] = payload.get("phase")
        self.generated_utc: str = payload.get("generated_utc", "")
        self.entries: Dict[str, Dict[str, Any]] = payload["entries"]

    def __repr__(self) -> str:      # never expose URLs through a stray print
        return (f"PresignedManifest(run_id={self.run_id!r}, "
                f"entries={len(self.entries)}, phase={self.phase!r})")

    @classmethod
    def fetch(cls, manifest_url: str, transport: Transport) -> "PresignedManifest":
        response = transport.get(manifest_url, timeout=60)
        _raise_for_expiry(response, manifest_url)
        if response.status_code != 200:
            raise ManifestUnusable(
                f"manifest fetch returned {response.status_code} for {redact(manifest_url)}"
            )
        try:
            return cls(json.loads(response.content.decode("utf-8")))
        except (ValueError, UnicodeDecodeError) as error:
            raise ManifestUnusable(f"manifest is not valid JSON: {error}") from error

    def exists(self, key: str) -> bool:
        entry = self.entries.get(key)
        return bool(entry and entry.get("get_url"))

    def url(self, key: str, direction: str) -> str:
        entry = self.entries.get(key)
        if entry is None:
            raise ManifestUnusable(
                f"key {key!r} is not in the manifest; add it to make_manifest.py"
            )
        url = entry.get(f"{direction}_url")
        if not url:
            raise ManifestUnusable(f"key {key!r} has no {direction} URL")
        self._check_expiry(key, entry)
        return url

    def _check_expiry(self, key: str, entry: Dict[str, Any]) -> None:
        stamp = entry.get("expires_at")
        if not stamp:
            return
        expires = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if expires <= datetime.now(timezone.utc):
            raise PresignedExpired(
                f"the URL for {key!r} expired at {stamp}; the workflow must issue "
                "new ones. Not starting a fresh run: the checkpoint is still valid."
            )

    def summary(self) -> str:
        present = sum(1 for e in self.entries.values() if e and e.get("get_url"))
        return (f"manifest {self.run_id}: {len(self.entries)} keys, {present} present, "
                f"{len(self.entries) - present} absent (cold start)")


class PresignedIO:
    """Reads and writes S3 objects using only the manifest's URLs."""

    def __init__(self, manifest: PresignedManifest, transport: Optional[Transport] = None,
                 timeout: int = 900) -> None:
        if transport is None:
            import requests

            transport = requests
        self.manifest = manifest
        self.transport = transport
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"PresignedIO({self.manifest!r})"

    # -- read -------------------------------------------------------------
    def get_bytes(self, key: str) -> bytes:
        url = self.manifest.url(key, "get")
        response = self.transport.get(url, timeout=self.timeout)
        _raise_for_expiry(response, url)
        if response.status_code != 200:
            raise ManifestUnusable(
                f"GET {key} returned {response.status_code} ({redact(url)})"
            )
        return response.content

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key).decode("utf-8"))

    def download(self, key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.get_bytes(key))
        print(f"  <- {key} ({destination.stat().st_size / 2**20:.1f} MiB)")
        return destination

    def download_if_present(self, key: str, destination: Path) -> Optional[Path]:
        """Cold start returns None instead of raising."""
        if not self.manifest.exists(key):
            return None
        return self.download(key, destination)

    # -- write ------------------------------------------------------------
    def put_bytes(self, key: str, payload: bytes) -> None:
        url = self.manifest.url(key, "put")
        response = self.transport.put(url, data=payload, timeout=self.timeout)
        _raise_for_expiry(response, url)
        if response.status_code not in (200, 201, 204):
            raise ManifestUnusable(
                f"PUT {key} returned {response.status_code} ({redact(url)})"
            )
        print(f"  -> {key} ({len(payload) / 2**20:.1f} MiB)")

    def upload(self, path: Path, key: str) -> None:
        self.put_bytes(key, path.read_bytes())

    def upload_directory(self, directory: Path, prefix: str,
                         patterns: Iterable[str] = ("*",)) -> int:
        """Upload matching files under a directory, keyed by relative path."""
        uploaded = 0
        for pattern in patterns:
            for path in sorted(directory.rglob(pattern)):
                if not path.is_file():
                    continue
                key = f"{prefix}/{path.relative_to(directory).as_posix()}"
                if key in self.manifest.entries:
                    self.upload(path, key)
                    uploaded += 1
        return uploaded

    # -- code bundle ------------------------------------------------------
    def extract_code_bundle(self, key: str, destination: Path) -> Path:
        """Unpack the src/ and configs/ tarball the workflow built from the repo.

        The notebook stays free of pipeline code this way, so it never drifts
        out of step with the repository it was generated from.
        """
        destination.mkdir(parents=True, exist_ok=True)
        payload = self.get_bytes(key)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            _safe_extract(archive, destination)
        print(f"  <- {key}: code unpacked into {destination}")
        return destination


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Reject members that would escape the destination directory."""
    root = destination.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if not str(target).startswith(str(root)):
            raise ManifestUnusable(f"archive member escapes destination: {member.name}")
    # filter="data" also strips absolute paths, links and device nodes. It
    # became available in 3.12 and is the default from 3.14; the explicit
    # argument keeps behaviour identical across the versions Kaggle ships.
    try:
        archive.extractall(destination, filter="data")
    except TypeError:
        archive.extractall(destination)


def _raise_for_expiry(response: Any, url: str) -> None:
    """Turn S3's expiry response into our own error rather than a bare 403."""
    if getattr(response, "status_code", None) != 403:
        return
    body = b""
    try:
        body = response.content or b""
    except Exception:                                  # noqa: BLE001
        pass
    text = body.decode("utf-8", errors="replace")
    if "Request has expired" in text or "AccessDenied" in text:
        raise PresignedExpired(
            f"presigned URL rejected as expired or invalid: {redact(url)}. "
            "Re-run the workflow to mint new URLs; do not start a new run."
        )


def clear_directory(path: Path) -> None:
    """Used between phases so a stale artifact cannot be re-uploaded as fresh."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
