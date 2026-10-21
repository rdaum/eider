#!/usr/bin/env python3
"""Compare cold/warm Flint requests, API adapters, sampling, and cancellation.

Start a fresh Flint server with prefix retention enabled before this check.
Timing includes HTTP, prompt processing, graph capture, sampling, and streaming.
"""
import argparse
import concurrent.futures
import json
import time
import urllib.request


def stream(base, api, prompt, **policy):
    body = {"model": "eider-flint", "stream": True, "temperature": 0, **policy}
    if api == "chat/completions":
        body.update(messages=[{"role": "user", "content": prompt}], max_tokens=64,
                    stream_options={"include_usage": True})
    else:
        body.update(input=prompt, max_output_tokens=64)
    request = urllib.request.Request(base + "/v1/" + api, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    output = {"text": "", "reasoning": ""}
    usage = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event or event.get("type") == "error":
                raise AssertionError(event)
            if api == "chat/completions":
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    for key, source in [("text", "content"), ("reasoning", "reasoning_content")]:
                        if delta.get(source):
                            first = first or time.perf_counter()
                            output[key] += delta[source]
                usage = event.get("usage") or usage
            else:
                key = {"response.output_text.delta": "text",
                       "response.reasoning_summary_text.delta": "reasoning"}.get(event.get("type"))
                if key and event.get("delta"):
                    first = first or time.perf_counter()
                    output[key] += event["delta"]
                if event.get("type") in ("response.completed", "response.incomplete"):
                    usage = event["response"]["usage"]
    assert usage and first, (output, usage)
    elapsed = time.perf_counter() - start
    if api == "chat/completions":
        tokens = usage["completion_tokens"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        prompt_tokens = usage["prompt_tokens"]
    else:
        tokens = usage["output_tokens"]
        cached = usage["input_tokens_details"]["cached_tokens"]
        prompt_tokens = usage["input_tokens"]
    return {"output": output, "prompt_tokens": prompt_tokens, "cached_tokens": cached,
            "completion_tokens": tokens, "ttft_seconds": first-start, "seconds": elapsed,
            "decode_tokens_per_second": (tokens-1) / max(1e-9, time.perf_counter()-first)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18080")
    args = parser.parse_args()
    prompt = ("Read these inventory records.\n" + "\n".join(
        f"Record {i}: warehouse north, item bolts, count {i+10}." for i in range(128))
        + "\nExplain briefly why an inventory ledger helps a warehouse manager.")
    cold = stream(args.base, "chat/completions", prompt)
    warm = stream(args.base, "chat/completions", prompt)
    responses = stream(args.base, "responses", prompt)
    assert cold["cached_tokens"] == 0, "Start with a fresh server for cold timing"
    assert warm["cached_tokens"] > 0
    for result in [warm, responses]:
        assert result["output"] == cold["output"], "prefix/API output mismatch"
        assert result["completion_tokens"] == cold["completion_tokens"]
        assert result["prompt_tokens"] == cold["prompt_tokens"]
    policy = {"temperature": 0.8, "seed": 123, "top_k": 20,
              "presence_penalty": 0.5, "frequency_penalty": 0.1}
    isolated_sampled = stream(args.base, "chat/completions", prompt, **policy)
    # Both requests share KV but keep RNG, penalties, and output parsing private.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(stream, args.base, api, prompt, **policy)
                   for api in ["chat/completions", "responses"]]
        sampled = [future.result() for future in futures]
    assert all(r["output"] == isolated_sampled["output"] for r in sampled), "packed/isolated sampling differs"
    assert all(result["cached_tokens"] > 0 for result in sampled)
    # Cancel one member of a live pair, then verify the surviving request.
    request = urllib.request.Request(args.base + "/v1/chat/completions", json.dumps({
        "model": "eider-flint", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 1024, "stream": True,
    }).encode(), {"Content-Type": "application/json"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        survivor = pool.submit(stream, args.base, "responses", prompt)
        with urllib.request.urlopen(request, timeout=180) as response:
            deltas = 0
            for line in response:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    event = json.loads(line[6:])
                    if any(c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")
                           for c in event.get("choices", [])):
                        deltas += 1
                        if deltas == 8:
                            break
        during_cancel = survivor.result()
    assert during_cancel["output"] == cold["output"], "cancellation affected the surviving request"
    after_cancel = stream(args.base, "responses", prompt)
    assert after_cancel["output"] == cold["output"]
    assert after_cancel["cached_tokens"] > 0
    results = {"cold": cold, "warm": warm, "responses": responses,
               "isolated_sampled": isolated_sampled, "concurrent_sampled": sampled,
               "during_cancel": during_cancel, "after_cancel": after_cancel}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
