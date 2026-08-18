"""Presigned S3 access: redaction, expiry, cold start, notebook injection.

The redaction tests are the load-bearing ones. A presigned URL is a live signed
credential; one printed into a Kaggle output or committed into a notebook is a
leak that survives in public history.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from presigned_io import (  # noqa: E402
    EXIT_PRESIGNED_EXPIRED,
    ManifestUnusable,
    PresignedExpired,
    PresignedIO,
    PresignedManifest,
    contains_signature,
    redact,
)

SIGNED = ("https://bucket.s3.amazonaws.com/runs/x/model.pt"
          "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAEXAMPLE"
          "&X-Amz-Signature=deadbeefcafe1234")


def _future(hours: float = 5) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _past() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content


class FakeTransport:
    """Records calls so tests can assert which URL was used."""

    def __init__(self, responses=None) -> None:
        self.responses = responses or {}
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        return self.responses.get(url, FakeResponse(200, b"payload"))

    def put(self, url, data=None, timeout=None):
        self.calls.append(("PUT", url, data))
        return self.responses.get(url, FakeResponse(200))


def _manifest(entries=None) -> PresignedManifest:
    return PresignedManifest({
        "run_id": "cicddos2019-main-20260817-abcd1234",
        "phase": "final_train",
        "entries": entries if entries is not None else {
            "p/runs/r/code_bundle.tar.gz": {"get_url": SIGNED, "put_url": SIGNED,
                                            "expires_at": _future()},
            "p/runs/r/checkpoints/model_last.pt": {"get_url": SIGNED, "put_url": SIGNED,
                                                   "expires_at": _future()},
            "p/runs/r/checkpoints/history.json": {"get_url": None, "put_url": SIGNED,
                                                  "expires_at": _future()},
        },
    })


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------

def test_redact_removes_the_entire_query_string():
    safe = redact(SIGNED)
    assert safe == "https://bucket.s3.amazonaws.com/runs/x/model.pt?<redacted>"
    assert not contains_signature(safe)


def test_contains_signature_detects_every_signing_parameter():
    for parameter in ("X-Amz-Signature=deadbeefcafe1234",
                      "X-Amz-Credential=AKIAEXAMPLE%2F20260817",
                      "X-Amz-Security-Token=FwoGZXIvYXdzEBYa",
                      "AWSAccessKeyId=AKIAIOSFODNN7EXAMPLE",
                      "Signature=aGVsbG93b3JsZA%3D%3D"):
        assert contains_signature(f"https://host/key?{parameter}")
    assert not contains_signature("https://host/key")


def test_prose_naming_a_signing_parameter_is_not_a_leak():
    """The guard runs over source and logs, which legitimately discuss these
    parameter names. Flagging the mention would make it useless."""
    assert not contains_signature("a presigned query string carries X-Amz-Signature")
    assert not contains_signature("# never log X-Amz-Credential or the token")
    assert not contains_signature("https://host/key?<redacted>")


def test_repr_never_exposes_a_url():
    """A stray print of the object must not leak a signed URL."""
    io_client = PresignedIO(_manifest(), FakeTransport())
    for text in (repr(io_client), repr(io_client.manifest), str(io_client.manifest)):
        assert not contains_signature(text)
        assert "deadbeef" not in text


def test_error_messages_are_redacted():
    manifest = _manifest()
    transport = FakeTransport({SIGNED: FakeResponse(500, b"boom")})
    with pytest.raises(ManifestUnusable) as error:
        PresignedIO(manifest, transport).get_bytes("p/runs/r/code_bundle.tar.gz")
    assert not contains_signature(str(error.value))


# --------------------------------------------------------------------------
# Expiry
# --------------------------------------------------------------------------

def test_expired_entry_is_refused_before_any_request():
    manifest = _manifest({
        "k": {"get_url": SIGNED, "put_url": SIGNED, "expires_at": _past()},
    })
    transport = FakeTransport()
    with pytest.raises(PresignedExpired, match="expired"):
        PresignedIO(manifest, transport).get_bytes("k")
    assert transport.calls == [], "an expired URL should not be requested at all"


def test_s3_expiry_response_becomes_our_error():
    """S3 answers an outdated signature with 403; a bare 403 would read as a
    permissions problem and invite starting over."""
    manifest = _manifest()
    transport = FakeTransport({
        SIGNED: FakeResponse(403, b"<Error><Code>AccessDenied</Code>"
                                  b"<Message>Request has expired</Message></Error>")
    })
    with pytest.raises(PresignedExpired, match="do not start a new run"):
        PresignedIO(manifest, transport).get_bytes("p/runs/r/code_bundle.tar.gz")


def test_expiry_has_its_own_exit_code():
    assert EXIT_PRESIGNED_EXPIRED == 75


# --------------------------------------------------------------------------
# Cold start and missing keys
# --------------------------------------------------------------------------

def test_null_get_url_means_cold_start_not_failure(tmp_path):
    """GitHub Actions writes null for objects that do not exist yet."""
    io_client = PresignedIO(_manifest(), FakeTransport())
    assert io_client.manifest.exists("p/runs/r/checkpoints/history.json") is False
    assert io_client.download_if_present(
        "p/runs/r/checkpoints/history.json", tmp_path / "history.json"
    ) is None
    assert not (tmp_path / "history.json").exists()


def test_a_key_outside_the_manifest_is_an_error_not_a_silent_skip():
    io_client = PresignedIO(_manifest(), FakeTransport())
    with pytest.raises(ManifestUnusable, match="not in the manifest"):
        io_client.get_bytes("p/runs/r/never/declared.npy")


def test_manifest_without_entries_is_rejected():
    with pytest.raises(ManifestUnusable, match="no 'entries'"):
        PresignedManifest({"run_id": "x"})


def test_summary_counts_present_and_absent_keys():
    summary = _manifest().summary()
    assert "3 keys" in summary and "2 present" in summary
    assert not contains_signature(summary)


# --------------------------------------------------------------------------
# Code bundle
# --------------------------------------------------------------------------

def _tarball(members: dict) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_code_bundle_extracts(tmp_path):
    payload = _tarball({"src/model.py": b"# model", "configs/model.yaml": b"a: 1"})
    manifest = _manifest({"bundle": {"get_url": SIGNED, "put_url": None,
                                     "expires_at": _future()}})
    transport = FakeTransport({SIGNED: FakeResponse(200, payload)})

    destination = PresignedIO(manifest, transport).extract_code_bundle("bundle", tmp_path)
    assert (destination / "src" / "model.py").read_bytes() == b"# model"


def test_path_traversal_in_the_bundle_is_rejected(tmp_path):
    payload = _tarball({"../escaped.py": b"malicious"})
    manifest = _manifest({"bundle": {"get_url": SIGNED, "put_url": None,
                                     "expires_at": _future()}})
    transport = FakeTransport({SIGNED: FakeResponse(200, payload)})

    with pytest.raises(ManifestUnusable, match="escapes destination"):
        PresignedIO(manifest, transport).extract_code_bundle("bundle", tmp_path / "out")
    assert not (tmp_path / "escaped.py").exists()


# --------------------------------------------------------------------------
# Notebook injection
# --------------------------------------------------------------------------

def test_committed_notebook_holds_no_url_and_no_credential():
    text = (REPO_ROOT / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    assert not contains_signature(text)
    for forbidden in ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "aws_secret",
                      "boto3", "amazonaws.com"):
        assert forbidden not in text, forbidden

    # Parse rather than string-match: the placeholder is JSON-escaped on disk.
    notebook = json.loads(text)
    tagged = [c for c in notebook["cells"]
              if "injected-parameters" in c.get("metadata", {}).get("tags", [])]
    assert len(tagged) == 1, "exactly one cell must be the injection target"
    source = "".join(tagged[0]["source"])
    assert 'PRESIGNED_MANIFEST_URL = ""' in source
    assert 'RUN_ID = ""' in source


def test_kernel_metadata_disables_gpu_and_stays_private():
    metadata = json.loads((REPO_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["dungnguyen28101991/cicddos2019-parquet"]


def test_injection_writes_the_url_only_into_the_tagged_cell(tmp_path):
    url_file = tmp_path / "url.txt"
    url_file.write_text(SIGNED, encoding="utf-8")
    destination = tmp_path / "build" / "kaggle_notebook.ipynb"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "inject_notebook.py"),
         "--manifest-url-file", str(url_file), "--run-id", "run-x",
         "--out", str(destination)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # The URL must never appear in the step log, only in the throwaway file.
    assert not contains_signature(result.stdout)

    notebook = json.loads(destination.read_text(encoding="utf-8"))
    tagged = [c for c in notebook["cells"]
              if "injected-parameters" in c.get("metadata", {}).get("tags", [])]
    assert len(tagged) == 1
    assert SIGNED in "".join(tagged[0]["source"])
    assert "run-x" in "".join(tagged[0]["source"])

    others = [c for c in notebook["cells"] if c is not tagged[0]]
    assert not contains_signature(json.dumps(others))
    assert all(c.get("outputs") == [] for c in notebook["cells"]
               if c.get("cell_type") == "code")


def test_injected_notebook_is_git_ignored():
    """The injected copy carries a live credential and must never be commitable."""
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "bootstrap_manifest.json" in ignore
    assert "kaggle.json" in ignore


def test_workflow_never_echoes_the_manifest_url():
    workflow = (REPO_ROOT / ".github" / "workflows" / "run-kaggle.yml").read_text(
        encoding="utf-8"
    )
    assert "--emit-url-to manifest_url.txt" in workflow
    assert "--manifest-url-file manifest_url.txt" in workflow
    # The URL is passed by file, never interpolated into a shell command.
    assert "--manifest-url " not in workflow
    assert "rm -f ~/.aws/credentials ~/.kaggle/kaggle.json manifest_url.txt" in workflow
