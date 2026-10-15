#!/usr/bin/env python3
"""Time cold and repeated LFM2 prompts through a freshly started Eider server.

Start the server separately. Requires only Python's standard library. Output
includes generated text so runs can be compared for both correctness and time.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=70,
                        help='prompt paragraph repetitions (220 gives about 5K tokens)')
    parser.add_argument('--max-tokens', type=int, default=96)
    args = parser.parse_args()
    text = (
        'A kestrel watches the meadow from an oak branch. '
        'The stream flows east past the stone bridge.\n'
    ) * args.repetitions + '\nWhat kind of bird watches the meadow? Answer in one sentence.'
    payload = {
        'model': 'eider-lfm2.5-24b-a2b',
        'messages': [{'role': 'user', 'content': text}],
        'temperature': 0,
        'max_tokens': args.max_tokens,
        'stream': True,
        'stream_options': {'include_usage': True},
    }
    results = []
    for name in ['cold', 'repeat']:
        start = time.perf_counter()
        first = None
        output = reasoning = ''
        usage = {}
        request = urllib.request.Request(
            args.url.rstrip('/') + '/v1/chat/completions',
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    break
                event = json.loads(raw)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if first is None and (delta.get('content') or delta.get('reasoning_content')):
                        first = time.perf_counter()
                    output += delta.get('content') or ''
                    reasoning += delta.get('reasoning_content') or ''
        if first is None or not usage:
            raise RuntimeError('missing output or usage in the SSE response')
        result = dict(name=name, ttft=first - start, elapsed=time.perf_counter() - start,
                      output=output, reasoning=reasoning, usage=usage)
        print(json.dumps(result), flush=True)
        results.append(result)
    assert results[0]['output'] == results[1]['output']
    assert results[0]['reasoning'] == results[1]['reasoning']
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
