#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible vLLM completions endpoint.

The output intentionally mirrors the high-signal lines from qwen-bench:
prefill here is time-to-first-token, and decode is total time after first token.
That makes it a server-surface comparison against Eider's local runner.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    args = parse_args()
    runs = []
    for repeat in range(args.repeats):
        if repeat == 0 and args.warmup:
            run_once(args)
        run = run_once(args)
        print(
            "  repeat {repeat}: ttft_ms={ttft_ms:.3f} decode_ms={decode_ms:.3f} "
            "total_ms={total_ms:.3f} completion_tokens={completion_tokens}".format(
                repeat=repeat,
                **run,
            ),
            file=sys.stderr,
        )
        runs.append(run)

    ttft_ms = median(run["ttft_ms"] for run in runs)
    decode_ms = median(run["decode_ms"] for run in runs)
    total_ms = median(run["total_ms"] for run in runs)
    completion_tokens = median(run["completion_tokens"] for run in runs)
    decode_tps = tokens_per_second(completion_tokens, decode_ms)

    print(
        f"vllm_model={args.model} requested_decode_tokens={args.decode_tokens} repeats={args.repeats}"
    )
    print(f"vllm_ttft_ms={ttft_ms:.3f}")
    print(
        "vllm_decode_tokens={tokens} vllm_decode_ms={decode_ms:.3f} "
        "vllm_decode_tps={decode_tps:.3f}".format(
            tokens=completion_tokens,
            decode_ms=decode_ms,
            decode_tps=decode_tps,
        )
    )
    print(f"vllm_total_ms={total_ms:.3f}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/completions",
        help="OpenAI-compatible completions URL",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--decode-tokens", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one unreported request before measuring",
    )
    parser.add_argument(
        "--allow-eos",
        action="store_true",
        help="Allow generation to stop before --decode-tokens",
    )
    args = parser.parse_args()
    if args.decode_tokens <= 0:
        parser.error("--decode-tokens must be > 0")
    if args.repeats <= 0:
        parser.error("--repeats must be > 0")
    return args


def run_once(args):
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.decode_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not args.allow_eos:
        payload["ignore_eos"] = True

    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token_at = None
    completion_tokens = None
    generated_fragments = 0
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                for choice in event.get("choices", []):
                    text = choice.get("text") or ""
                    if text:
                        generated_fragments += 1
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{args.url} returned HTTP {err.code}: {detail}") from err
    except urllib.error.URLError as err:
        raise SystemExit(f"failed to connect to {args.url}: {err}") from err

    end = time.perf_counter()
    if first_token_at is None:
        first_token_at = end
    if completion_tokens is None:
        completion_tokens = args.decode_tokens if generated_fragments else 0

    ttft_ms = (first_token_at - start) * 1000.0
    total_ms = (end - start) * 1000.0
    return {
        "ttft_ms": ttft_ms,
        "decode_ms": max(total_ms - ttft_ms, 0.0),
        "total_ms": total_ms,
        "completion_tokens": completion_tokens,
    }


def median(values):
    return statistics.median(list(values))


def tokens_per_second(tokens, ms):
    if tokens <= 0 or ms <= 0.0:
        return 0.0
    return tokens / (ms / 1000.0)


if __name__ == "__main__":
    raise SystemExit(main())
