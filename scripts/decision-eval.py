#!/usr/bin/env python3
"""Evaluate and calibrate structured decision endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMPT_FORMAT = "eider-decision-v1"
EPSILON = 1e-12


@dataclass(frozen=True)
class Target:
    name: str
    url: str
    model: str
    api_key: str | None


def parse_mapping(values: list[str], flag: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} requires NAME=VALUE, got {value!r}")
        name, mapped = value.split("=", 1)
        if not name or not mapped:
            raise ValueError(f"{flag} requires non-empty NAME=VALUE")
        result[name] = mapped
    return result


def load_targets(args: argparse.Namespace) -> list[Target]:
    urls = parse_mapping(args.target, "--target")
    models = parse_mapping(args.model, "--model")
    key_envs = parse_mapping(args.api_key_env, "--api-key-env")
    unknown = (models.keys() | key_envs.keys()) - urls.keys()
    if unknown:
        raise ValueError(f"target settings reference unknown targets: {sorted(unknown)}")
    targets = []
    for name, url in urls.items():
        key_env = key_envs.get(name)
        api_key = os.environ.get(key_env) if key_env else None
        if key_env and not api_key:
            raise ValueError(f"environment variable {key_env!r} is empty")
        targets.append(
            Target(
                name=name,
                url=url,
                model=models.get(name, "decision-latest"),
                api_key=api_key,
            )
        )
    return targets


def load_dataset(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("dataset must be one JSON object")
    if document.get("prompt_format") != PROMPT_FORMAT:
        raise ValueError(f"dataset prompt_format must be {PROMPT_FORMAT!r}")
    examples = document.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("dataset examples must be a non-empty array")
    seen: set[str] = set()
    for example in examples:
        if not isinstance(example, dict) or not isinstance(example.get("id"), str):
            raise ValueError("each example must have a string id")
        if example["id"] in seen:
            raise ValueError(f"duplicate example id {example['id']!r}")
        seen.add(example["id"])
        questions = example.get("questions")
        expected = example.get("expected")
        if not isinstance(questions, dict) or not questions:
            raise ValueError(f"example {example['id']!r} has no questions")
        if not isinstance(expected, dict) or set(expected) != set(questions):
            raise ValueError(
                f"example {example['id']!r} expected keys must match question keys"
            )
        for question_id, question in questions.items():
            validate_expected(example["id"], question_id, question, expected[question_id])
        if example.get("split", "evaluation") not in {"calibration", "evaluation"}:
            raise ValueError(f"example {example['id']!r} has an invalid split")
    return document, hashlib.sha256(raw).hexdigest()


def option_keys(question: dict[str, Any]) -> list[str]:
    kind = question.get("type")
    if kind == "noul":
        return ["true", "false"]
    criteria = question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict):
        return list(criteria)
    if kind == "score" and isinstance(criteria, list):
        return [str(index) for index in range(len(criteria))]
    raise ValueError(f"invalid question shape for type {kind!r}")


def normalise_expected(expected: Any) -> str:
    if isinstance(expected, bool):
        return str(expected).lower()
    if isinstance(expected, int):
        return str(expected)
    if isinstance(expected, str):
        return expected
    raise ValueError(f"expected answer must be a string, integer, or boolean, got {expected!r}")


def validate_expected(
    example_id: str, question_id: str, question: dict[str, Any], expected: Any
) -> None:
    expected_key = normalise_expected(expected)
    keys = option_keys(question)
    if expected_key not in keys:
        raise ValueError(
            f"{example_id}/{question_id} expected {expected_key!r} is not in {keys!r}"
        )


def request_json(
    target: Target,
    payload: dict[str, Any],
    timeout: float,
    attempts: int,
) -> tuple[dict, float]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    headers = {"content-type": "application/json"}
    if target.api_key:
        headers["authorization"] = f"Bearer {target.api_key}"
    started = time.perf_counter()
    for attempt in range(attempts):
        request = urllib.request.Request(target.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code not in {429, 503, 524, 529} or attempt + 1 == attempts:
                raise RuntimeError(
                    f"{target.name} returned HTTP {error.code}: {detail}"
                ) from error
            retry_after = error.headers.get("retry-after")
            delay = float(retry_after) if retry_after else min(8.0, 0.5 * 2**attempt)
            time.sleep(max(0.0, delay))
        except urllib.error.URLError as error:
            if attempt + 1 == attempts:
                raise RuntimeError(f"{target.name} request failed: {error}") from error
            time.sleep(min(8.0, 0.5 * 2**attempt))
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise RuntimeError(f"{target.name} returned an invalid decision response")
    return result, elapsed_ms


def probabilities(question: dict[str, Any], answer: dict[str, Any]) -> dict[str, float]:
    keys = option_keys(question)
    if question["type"] == "noul":
        value = float(answer["noul"])
        result = {"true": value, "false": 1.0 - value}
    else:
        supplied = answer.get("probabilities")
        if not isinstance(supplied, dict):
            raise ValueError("response omitted probabilities")
        result = {key: float(supplied[key]) for key in keys}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError(f"invalid probabilities: {result!r}")
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError("probabilities sum to zero")
    return {key: value / total for key, value in result.items()}


def prediction_record(
    target: Target,
    response_model: str,
    example: dict[str, Any],
    question_id: str,
    answer: dict[str, Any],
    mode: str,
    latency_ms: float,
    usage: Any,
) -> dict[str, Any]:
    question = example["questions"][question_id]
    probs = probabilities(question, answer)
    predicted = max(probs, key=probs.__getitem__)
    expected = normalise_expected(example["expected"][question_id])
    return {
        "target": target.name,
        "model": response_model,
        "example_id": example["id"],
        "question_id": question_id,
        "question_type": question["type"],
        "option_count": len(probs),
        "split": example.get("split", "evaluation"),
        "strata": example.get("strata", {}),
        "mode": mode,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "probabilities": probs,
        "concentration": answer.get("confidence"),
        "latency_ms": latency_ms,
        "usage": usage,
    }


def decision_payload(
    target: Target, state: Any, questions: dict[str, Any]
) -> dict[str, Any]:
    return {"model": target.model, "state": state, "questions": questions}


def permute_choice(question: dict[str, Any], seed: int) -> dict[str, Any]:
    result = dict(question)
    items = list(question["criteria"].items())
    if items:
        offset = seed % len(items)
        items = items[offset:] + items[:offset]
    result["criteria"] = dict(items)
    return result


def max_probability_delta(left: dict[str, float], right: dict[str, float]) -> float:
    if set(left) != set(right):
        return math.inf
    return max(abs(left[key] - right[key]) for key in left)


def evaluate_target(
    target: Target,
    dataset: dict[str, Any],
    timeout: float,
    permutation_count: int,
    attempts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for example in dataset["examples"]:
        response, latency_ms = request_json(
            target,
            decision_payload(target, example["state"], example["questions"]),
            timeout,
            attempts,
        )
        if set(response["answers"]) != set(example["questions"]):
            raise RuntimeError(
                f"{target.name} returned answer keys that do not match example {example['id']!r}"
            )
        multi: dict[str, dict[str, Any]] = {}
        for question_id, answer in response["answers"].items():
            record = prediction_record(
                target,
                str(response.get("model", target.model)),
                example,
                question_id,
                answer,
                "multi",
                latency_ms,
                response.get("usage"),
            )
            records.append(record)
            multi[question_id] = record

        for question_id, question in example["questions"].items():
            response, single_latency = request_json(
                target,
                decision_payload(target, example["state"], {question_id: question}),
                timeout,
                attempts,
            )
            single = prediction_record(
                target,
                str(response.get("model", target.model)),
                example,
                question_id,
                response["answers"][question_id],
                "single",
                single_latency,
                response.get("usage"),
            )
            checks.append(
                {
                    "kind": "single_multi",
                    "target": target.name,
                    "example_id": example["id"],
                    "question_id": question_id,
                    "top_match": single["predicted"] == multi[question_id]["predicted"],
                    "max_probability_delta": max_probability_delta(
                        single["probabilities"], multi[question_id]["probabilities"]
                    ),
                    "multi_usage": multi[question_id]["usage"],
                    "single_usage": single["usage"],
                }
            )
            if question["type"] == "choice":
                for permutation in range(permutation_count):
                    permuted = permute_choice(question, permutation + 1)
                    response, permutation_latency = request_json(
                        target,
                        decision_payload(
                            target, example["state"], {question_id: permuted}
                        ),
                        timeout,
                        attempts,
                    )
                    changed = prediction_record(
                        target,
                        str(response.get("model", target.model)),
                        example,
                        question_id,
                        response["answers"][question_id],
                        "permutation",
                        permutation_latency,
                        response.get("usage"),
                    )
                    checks.append(
                        {
                            "kind": "option_order",
                            "target": target.name,
                            "example_id": example["id"],
                            "question_id": question_id,
                            "permutation": permutation,
                            "top_match": changed["predicted"] == single["predicted"],
                            "max_probability_delta": max_probability_delta(
                                changed["probabilities"], single["probabilities"]
                            ),
                        }
                    )

        for variant_index, state in enumerate(example.get("injection_variants", [])):
            response, variant_latency = request_json(
                target,
                decision_payload(target, state, example["questions"]),
                timeout,
                attempts,
            )
            for question_id, answer in response["answers"].items():
                variant = prediction_record(
                    target,
                    str(response.get("model", target.model)),
                    example,
                    question_id,
                    answer,
                    "injection",
                    variant_latency,
                    response.get("usage"),
                )
                checks.append(
                    {
                        "kind": "prompt_injection",
                        "target": target.name,
                        "example_id": example["id"],
                        "question_id": question_id,
                        "variant": variant_index,
                        "correct": variant["correct"],
                        "predicted": variant["predicted"],
                        "top_match": variant["predicted"]
                        == multi[question_id]["predicted"],
                        "max_probability_delta": max_probability_delta(
                            variant["probabilities"],
                            multi[question_id]["probabilities"],
                        ),
                    }
                )
    return records, checks


def record_losses(record: dict[str, Any]) -> tuple[float, float, float | None]:
    probs = record["probabilities"]
    expected = record["expected"]
    nll = -math.log(max(float(probs[expected]), EPSILON))
    brier = sum(
        (probability - (1.0 if key == expected else 0.0)) ** 2
        for key, probability in probs.items()
    )
    rps = None
    if record["question_type"] == "score":
        keys = sorted(probs, key=int)
        expected_index = keys.index(expected)
        cumulative = 0.0
        total = 0.0
        for index, key in enumerate(keys[:-1]):
            cumulative += probs[key]
            observed = 1.0 if index >= expected_index else 0.0
            total += (cumulative - observed) ** 2
        rps = total / max(1, len(keys) - 1)
    return nll, brier, rps


def reliability(records: list[dict[str, Any]], bins: int = 10) -> tuple[list[dict], float]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for record in records:
        confidence = max(record["probabilities"].values())
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append((confidence, record["correct"]))
    result = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_confidence = sum(value for value, _ in bucket) / len(bucket)
        accuracy = sum(correct for _, correct in bucket) / len(bucket)
        ece += len(bucket) / len(records) * abs(mean_confidence - accuracy)
        result.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(bucket),
                "mean_probability": mean_confidence,
                "empirical_accuracy": accuracy,
            }
        )
    return result, ece


def coverage_risk(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    ranked = sorted(records, key=lambda item: max(item["probabilities"].values()), reverse=True)
    result = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        count = max(1, math.ceil(len(ranked) * fraction))
        selected = ranked[:count]
        result.append(
            {
                "coverage": count / len(ranked),
                "risk": 1.0 - sum(item["correct"] for item in selected) / count,
                "minimum_probability": min(
                    max(item["probabilities"].values()) for item in selected
                ),
            }
        )
    return result


def summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    losses = [record_losses(record) for record in records]
    diagram, ece = reliability(records)
    score_losses = [rps for _, _, rps in losses if rps is not None]
    return {
        "count": len(records),
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "nll": sum(item[0] for item in losses) / len(losses),
        "brier": sum(item[1] for item in losses) / len(losses),
        "ranked_probability_score": (
            sum(score_losses) / len(score_losses) if score_losses else None
        ),
        "ece": ece,
        "reliability": diagram,
        "coverage_risk": coverage_risk(records),
        "mean_latency_ms": sum(record["latency_ms"] for record in records)
        / len(records),
    }


def grouped_summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "question_type": defaultdict(list),
        "option_count": defaultdict(list),
        "question": defaultdict(list),
        "strata": defaultdict(list),
    }
    for record in records:
        groups["question_type"][record["question_type"]].append(record)
        groups["option_count"][str(record["option_count"])].append(record)
        groups["question"][record["question_id"]].append(record)
        for name, value in record["strata"].items():
            groups["strata"][f"{name}={value}"].append(record)
    return {
        name: {key: summary(items) for key, items in values.items()}
        for name, values in groups.items()
    }


def target_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_case[(record["example_id"], record["question_id"])][record["target"]] = record
    comparisons = []
    for (example_id, question_id), targets in by_case.items():
        names = sorted(targets)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                left = targets[left_name]
                right = targets[right_name]
                comparisons.append(
                    {
                        "example_id": example_id,
                        "question_id": question_id,
                        "left": left_name,
                        "right": right_name,
                        "top_match": left["predicted"] == right["predicted"],
                        "max_probability_delta": max_probability_delta(
                            left["probabilities"], right["probabilities"]
                        ),
                    }
                )
    return comparisons


def temperature_probabilities(probs: dict[str, float], temperature: float) -> dict[str, float]:
    scaled = {
        key: math.log(max(value, EPSILON)) / temperature for key, value in probs.items()
    }
    maximum = max(scaled.values())
    weights = {key: math.exp(value - maximum) for key, value in scaled.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def calibration_objective(
    records: list[dict[str, Any]], temperature: float, ordinal: bool
) -> float:
    adjusted = [
        {**record, "probabilities": temperature_probabilities(record["probabilities"], temperature)}
        for record in records
    ]
    losses = [record_losses(record) for record in adjusted]
    if ordinal:
        return sum(item[2] for item in losses if item[2] is not None) / len(losses)
    return sum(item[0] for item in losses) / len(losses)


def fit_temperature(records: list[dict[str, Any]], ordinal: bool) -> tuple[float, float]:
    low = math.log(0.05)
    high = math.log(20.0)
    for _ in range(80):
        left = low + (high - low) / 3.0
        right = high - (high - low) / 3.0
        if calibration_objective(records, math.exp(left), ordinal) < calibration_objective(
            records, math.exp(right), ordinal
        ):
            high = right
        else:
            low = left
    temperature = math.exp((low + high) / 2.0)
    return temperature, calibration_objective(records, temperature, ordinal)


def fit_calibration(
    dataset: dict[str, Any], dataset_hash: str, target: Target, records: list[dict[str, Any]]
) -> dict[str, Any]:
    if dataset.get("validated") is not True:
        raise ValueError("calibration fitting requires dataset.validated=true")
    calibration = [record for record in records if record["split"] == "calibration"]
    evaluation = [record for record in records if record["split"] == "evaluation"]
    if not calibration or not evaluation:
        raise ValueError("calibration fitting requires calibration and evaluation splits")
    model_revisions = {record["model"] for record in records}
    if len(model_revisions) != 1:
        raise ValueError(
            f"calibration fitting requires one returned model revision, got {sorted(model_revisions)}"
        )
    model_revision = next(iter(model_revisions))
    maps = {}
    for question_type in ("noul", "choice", "score"):
        selected = [record for record in calibration if record["question_type"] == question_type]
        if not selected:
            continue
        ordinal = question_type == "score"
        temperature, objective = fit_temperature(selected, ordinal)
        maps[question_type] = {
            "method": "ordinal_temperature_rps" if ordinal else "temperature_nll",
            "temperature": temperature,
            "fit_objective": objective,
            "sample_count": len(selected),
        }
    calibrated_evaluation = []
    for record in evaluation:
        fitted = maps.get(record["question_type"])
        if fitted is None:
            continue
        calibrated_evaluation.append(
            {
                **record,
                "probabilities": temperature_probabilities(
                    record["probabilities"], fitted["temperature"]
                ),
            }
        )
    return {
        "schema": "eider-decision-calibration-v1",
        "validated_dataset": True,
        "model": model_revision,
        "target": target.name,
        "deployment": {
            "url": target.url,
            "requested_model": target.model,
            "returned_model": model_revision,
        },
        "prompt_format": PROMPT_FORMAT,
        "dataset": dataset["name"],
        "dataset_version": dataset["version"],
        "dataset_sha256": dataset_hash,
        "maps": maps,
        "evaluation": {
            "uncalibrated": summary(evaluation),
            "calibrated": summary(calibrated_evaluation),
        },
    }


def self_test() -> None:
    records = [
        {
            "question_type": "choice",
            "probabilities": {"a": 0.8, "b": 0.2},
            "expected": "a",
            "predicted": "a",
            "correct": True,
            "latency_ms": 1.0,
            "option_count": 2,
            "question_id": "q",
            "strata": {},
        },
        {
            "question_type": "choice",
            "probabilities": {"a": 0.7, "b": 0.3},
            "expected": "b",
            "predicted": "a",
            "correct": False,
            "latency_ms": 2.0,
            "option_count": 2,
            "question_id": "q",
            "strata": {},
        },
    ]
    result = summary(records)
    assert result["count"] == 2
    assert result["accuracy"] == 0.5
    assert math.isfinite(result["nll"])
    temperature, objective = fit_temperature(records, False)
    assert 0.05 <= temperature <= 20.0
    assert math.isfinite(objective)
    assert temperature_probabilities({"a": 0.5, "b": 0.5}, 2.0) == {
        "a": 0.5,
        "b": 0.5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="decision endpoint; repeat to compare Eider deployments",
    )
    parser.add_argument("--model", action="append", default=[], metavar="NAME=MODEL")
    parser.add_argument(
        "--api-key-env", action="append", default=[], metavar="NAME=ENVIRONMENT_VARIABLE"
    )
    parser.add_argument("--output", type=Path, default=Path("decision-evaluation.json"))
    parser.add_argument("--fit-calibration", type=Path)
    parser.add_argument("--calibration-target")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("decision evaluation self-test passed")
        return 0
    if args.dataset is None or not args.target:
        raise ValueError("--dataset and at least one --target are required")
    if args.permutations < 0:
        raise ValueError("--permutations must not be negative")
    if args.attempts < 1:
        raise ValueError("--attempts must be positive")
    dataset, dataset_hash = load_dataset(args.dataset)
    targets = load_targets(args)
    all_records = []
    all_checks = []
    target_reports = {}
    for target in targets:
        records, checks = evaluate_target(
            target, dataset, args.timeout, args.permutations, args.attempts
        )
        all_records.extend(records)
        all_checks.extend(checks)
        target_reports[target.name] = {
            "endpoint": target.url,
            "requested_model": target.model,
            "returned_models": sorted({record["model"] for record in records}),
            "overall": summary(records),
            "groups": grouped_summaries(records),
        }
    report = {
        "schema": "eider-decision-evaluation-v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_format": PROMPT_FORMAT,
        "dataset": dataset["name"],
        "dataset_version": dataset["version"],
        "dataset_sha256": dataset_hash,
        "targets": target_reports,
        "target_comparisons": target_comparisons(all_records),
        "invariance_checks": all_checks,
        "predictions": all_records,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.fit_calibration:
        target_name = args.calibration_target or targets[0].name
        target = next((item for item in targets if item.name == target_name), None)
        if target is None:
            raise ValueError(f"unknown calibration target {target_name!r}")
        selected = [record for record in all_records if record["target"] == target_name]
        artifact = fit_calibration(dataset, dataset_hash, target, selected)
        args.fit_calibration.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
