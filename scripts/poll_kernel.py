"""Follow a pushed Kaggle kernel until it settles.

`cancelAcknowledged` is treated as success. A CPU session hitting the 12 hour
limit is the expected way most sessions end, and the pipeline is built for it:
the checkpoint is uploaded before the cut-off and the next session resumes.
Failing the workflow here would turn a normal outcome into a red build every
few hours.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

SETTLED_OK = ("complete", "cancelacknowledged", "cancelled")
SETTLED_BAD = ("error", "failed")


def kernel_status(kernel: str) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        capture_output=True, text=True, check=False,
    )
    text = (result.stdout + result.stderr).strip().lower()
    match = re.search(r'status\s+"?([a-z]+)"?', text)
    return match.group(1) if match else text.split("\n")[-1][:60]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--timeout-minutes", type=float, default=730)
    parser.add_argument("--initial-interval", type=float, default=30)
    parser.add_argument("--max-interval", type=float, default=300)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_minutes * 60
    interval = args.initial_interval
    last = None

    while time.monotonic() < deadline:
        status = kernel_status(args.kernel)
        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] {status}", flush=True)
            last = status

        if any(token in status for token in SETTLED_OK):
            print(f"kernel settled: {status}")
            return 0
        if any(token in status for token in SETTLED_BAD):
            print(f"kernel failed: {status}", file=sys.stderr)
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
