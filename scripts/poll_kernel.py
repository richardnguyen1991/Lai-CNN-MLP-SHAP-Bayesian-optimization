"""Follow a pushed Kaggle kernel until it settles.

`cancelAcknowledged` is treated as success. A CPU session hitting the 12 hour
limit is the expected way most sessions end, and the pipeline is built for it:
the checkpoint is uploaded before the cut-off and the next session resumes.
Failing the workflow here would turn a normal outcome into a red build every
few hours.

The status only reaches us as prose. `kaggle kernels status` has no
machine-readable output, and it prints its enum through %s, so the line reads

    owner/slug has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"

Two things follow, and both were wrong here before. The enum's class name has
to be stripped, or the parsed token is "KernelWorkerStatus" and matches nothing
at all -- a kernel that had already failed was polled for twenty minutes.
And the names carry underscores, so a token compared against a spelling like
"cancelacknowledged" has to be normalised rather than matched literally; that
spelling is exactly the one a cancelled session produces.

An unreadable status is therefore treated as its own outcome rather than as
"still running". Being unable to tell the two apart is what let a dead kernel
look alive for the whole timeout.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

STATUS_LINE = re.compile(r'has status\s+"([^"]+)"', re.IGNORECASE)
FAILURE_LINE = re.compile(r'failure message:\s+"([^"]*)"', re.IGNORECASE)

SETTLED_OK = frozenset({"complete", "cancelacknowledged", "cancelled"})
SETTLED_BAD = frozenset({"error", "failed"})
UNREADABLE = "unreadable"

# Roughly ten minutes of backed-off polling. Long enough to ride out a blip in
# the Kaggle API, short enough that a parse that has stopped working is caught
# in the same session rather than after the timeout.
MAX_UNREADABLE = 10


def normalize_status(raw: str) -> str:
    """"KernelWorkerStatus.CANCEL_ACKNOWLEDGED" -> "cancelacknowledged"."""
    token = raw.strip().rsplit(".", 1)[-1]
    return re.sub("[^a-z]", "", token.lower())


def parse_status_output(text: str):
    """Returns (status, detail). status is UNREADABLE when no line matched."""
    match = STATUS_LINE.search(text)
    if not match:
        return UNREADABLE, " ".join(text.split())[-200:]
    failure = FAILURE_LINE.search(text)
    return normalize_status(match.group(1)), failure.group(1) if failure else ""


def read_status(kernel: str):
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        capture_output=True, text=True, check=False,
    )
    return parse_status_output(result.stdout + result.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--timeout-minutes", type=float, default=330)
    parser.add_argument("--initial-interval", type=float, default=30)
    parser.add_argument("--max-interval", type=float, default=300)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_minutes * 60
    interval = args.initial_interval
    last = None
    unreadable = 0

    while time.monotonic() < deadline:
        status, detail = read_status(args.kernel)

        if status == UNREADABLE:
            unreadable += 1
            if unreadable >= MAX_UNREADABLE:
                print(f"could not read the kernel status {unreadable} times in "
                      f"a row; last output: {detail!r}", file=sys.stderr)
                return 1
        else:
            unreadable = 0

        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] {status}", flush=True)
            last = status

        if status in SETTLED_OK:
            print(f"kernel settled: {status}")
            return 0
        if status in SETTLED_BAD:
            print(f"kernel failed: {status}", file=sys.stderr)
            if detail:
                print(f"failure message: {detail}", file=sys.stderr)
            return 1

        time.sleep(interval)
        # Back off: a session runs for hours, so polling every 30 s the whole
        # time is just noise against the API.
        interval = min(interval * 1.5, args.max_interval)

    print(f"timed out after {args.timeout_minutes} min with status {last!r}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
