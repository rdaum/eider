#!/usr/bin/env python3
"""Compare full distributions from the lfm2-logits example with forced tokens."""

import argparse
import json
import math
from pathlib import Path


def log_softmax(values):
    maximum = max(values)
    normalizer = maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))
    return [x - normalizer for x in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    if (reference['prompt_tokens'] != candidate['prompt_tokens']
            or reference['continuation'] != candidate['continuation']
            or len(reference['logits']) != len(candidate['logits'])):
        parser.error('the prompt lengths and forced continuation tokens must match')
    rows = []
    for step, (expected, actual) in enumerate(zip(reference['logits'], candidate['logits'])):
        if not expected or len(expected) != len(actual):
            parser.error(f'logit dimensions differ at step {step}')
        if not all(math.isfinite(x) for x in expected + actual):
            parser.error(f'nonfinite logits at step {step}')
        p, q = log_softmax(expected), log_softmax(actual)
        order = sorted(range(len(expected)), key=expected.__getitem__, reverse=True)
        selected = max(range(len(actual)), key=actual.__getitem__)
        row = dict(step=step,
                   max_absolute_error=max(abs(x - y) for x, y in zip(expected, actual)),
                   kl_reference_to_candidate=max(0.0, math.fsum(math.exp(x) * (x - y) for x, y in zip(p, q))),
                   reference_argmax=order[0], candidate_argmax=selected,
                   reference_margin=expected[order[0]] - expected[order[1]])
        if step < len(reference['continuation']):
            token = reference['continuation'][step]
            row['forced_token_nll_change'] = p[token] - q[token]
        rows.append(row)
    if not rows:
        parser.error('the files contain no logits')
    changes = [r['forced_token_nll_change'] for r in rows if 'forced_token_nll_change' in r]
    print(json.dumps(dict(
        steps=rows,
        mean_kl=math.fsum(r['kl_reference_to_candidate'] for r in rows) / len(rows),
        max_kl=max(r['kl_reference_to_candidate'] for r in rows),
        argmax_matches=sum(r['reference_argmax'] == r['candidate_argmax'] for r in rows),
        mean_forced_token_nll_change=math.fsum(changes) / len(changes) if changes else None,
    ), indent=2))


if __name__ == '__main__':
    main()
