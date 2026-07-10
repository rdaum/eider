#!/usr/bin/env python3
"""Wait for an OpenAI-compatible server to become ready."""

import argparse
import http.client
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    args = parse_args()
    deadline = time.monotonic() + args.timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return 0
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as err:
            last_error = f"HTTP {err.code}: {err.read().decode('utf-8', errors='replace')}"
        except urllib.error.URLError as err:
            last_error = str(err)
        except (ConnectionError, OSError, http.client.HTTPException) as err:
            last_error = str(err)
        time.sleep(args.interval)

    print(f"timed out waiting for {args.url}: {last_error}", file=sys.stderr)
    if args.container:
        print(f"recent logs for {args.container}:", file=sys.stderr)
        subprocess.run(
            ["docker", "logs", "--tail", str(args.log_lines), args.container],
            check=False,
        )
    return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--container")
    parser.add_argument("--log-lines", type=int, default=200)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.interval <= 0:
        parser.error("--interval must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
