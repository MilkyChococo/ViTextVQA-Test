#!/usr/bin/env python
"""Evaluate ViTextVQA predictions with normalized EM and token-level F1."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from spatial_graph.io_utils import fix_mojibake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ViTextVQA EM/F1 from prediction file and ground-truth file."
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Prediction file path. Supports ViTextVQA-style JSON or JSONL with question_id/prediction.",
    )
    parser.add_argument(
        "--gt",
        default="vitextvqa/ViTextVQA_images/ViTextVQA_test_gt.json",
        help="Ground-truth ViTextVQA JSON path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save evaluation details as JSON.",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Ignore samples without predictions instead of counting them as zero.",
    )
    return parser.parse_args()


def resolve_repo_relative_path(repo_root: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_first_answer(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return extract_first_answer(value[0])
    if value is None:
        return ""
    return str(value)


def raw_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def canonical_answer(text: Any) -> str:
    normalized = fix_mojibake(raw_text(text))
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.lower()
    normalized = "".join(
        ch if not unicodedata.category(ch).startswith(("P", "S")) else " " for ch in normalized
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: Any) -> list[str]:
    canonical = canonical_answer(text)
    if not canonical:
        return []
    return canonical.split()


def exact_match_score(prediction: Any, ground_truth: Any) -> float:
    return float(canonical_answer(prediction) == canonical_answer(ground_truth))


def f1_score(prediction: Any, ground_truth: Any) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gt_tokens)
    num_correct = sum(overlap.values())
    if num_correct == 0:
        return 0.0

    precision = num_correct / len(pred_tokens)
    recall = num_correct / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, truth) for truth in ground_truths)


def load_ground_truth_map(gt_path: Path) -> dict[int, dict[str, Any]]:
    payload = load_json(gt_path)
    annotations = payload.get("annotations", []) if isinstance(payload, dict) else []
    mapping: dict[int, dict[str, Any]] = {}
    for ann in annotations:
        ann_id = int(ann["id"])
        answers = [raw_text(item) for item in ann.get("answers", [])]
        mapping[ann_id] = {
            "image_id": ann.get("image_id"),
            "question": raw_text(ann.get("question", "")),
            "answers": answers,
        }
    return mapping


def load_prediction_map(pred_path: Path) -> dict[int, str]:
    def assign_if_non_empty(mapping: dict[int, str], key: int, value: Any) -> None:
        answer = raw_text(extract_first_answer(value))
        if not canonical_answer(answer):
            return
        if answer:
            mapping[key] = answer

    suffix = pred_path.suffix.lower()
    if suffix == ".jsonl":
        rows = load_jsonl(pred_path)
        mapping: dict[int, str] = {}
        for row in rows:
            if "question_id" in row:
                key = int(row["question_id"])
            elif "id" in row:
                key = int(row["id"])
            else:
                continue

            if "prediction" in row:
                assign_if_non_empty(mapping, key, row["prediction"])
            elif "answer" in row:
                assign_if_non_empty(mapping, key, row["answer"])
            elif "answers" in row:
                assign_if_non_empty(mapping, key, row["answers"])
        return mapping

    payload = load_json(pred_path)
    mapping: dict[int, str] = {}

    if isinstance(payload, dict) and "annotations" in payload:
        for ann in payload.get("annotations", []):
            ann_id = int(ann["id"])
            if "prediction" in ann:
                assign_if_non_empty(mapping, ann_id, ann["prediction"])
            elif "answer" in ann:
                assign_if_non_empty(mapping, ann_id, ann["answer"])
            else:
                assign_if_non_empty(mapping, ann_id, ann.get("answers", []))
        return mapping

    if isinstance(payload, list):
        for item in payload:
            if "question_id" in item:
                key = int(item["question_id"])
            elif "id" in item:
                key = int(item["id"])
            else:
                continue

            if "prediction" in item:
                assign_if_non_empty(mapping, key, item["prediction"])
            elif "answer" in item:
                assign_if_non_empty(mapping, key, item["answer"])
            elif "answers" in item:
                assign_if_non_empty(mapping, key, item["answers"])
        return mapping

    raise ValueError(f"Unsupported prediction format: {pred_path}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    pred_path = resolve_repo_relative_path(repo_root, args.pred)
    gt_path = resolve_repo_relative_path(repo_root, args.gt)
    output_path = resolve_repo_relative_path(repo_root, args.output)

    if pred_path is None or gt_path is None:
        raise ValueError("Failed to resolve input paths.")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    ground_truth = load_ground_truth_map(gt_path)
    predictions = load_prediction_map(pred_path)

    total_questions = len(ground_truth)
    evaluated_questions = 0
    matched_predictions = 0
    missing_predictions = 0
    skipped_missing_predictions = 0
    exact_matches = 0
    em_sum = 0.0
    f1_sum = 0.0
    details: list[dict[str, Any]] = []

    for ann_id, gt_item in ground_truth.items():
        has_prediction = ann_id in predictions
        pred_answer = predictions.get(ann_id, "")
        gt_answers = gt_item["answers"]

        if has_prediction:
            matched_predictions += 1
        else:
            missing_predictions += 1
            if args.ignore_missing:
                skipped_missing_predictions += 1
                details.append(
                    {
                        "id": ann_id,
                        "image_id": gt_item["image_id"],
                        "question": gt_item["question"],
                        "prediction": pred_answer,
                        "answers": gt_answers,
                        "normalized_prediction": canonical_answer(pred_answer),
                        "normalized_answers": [canonical_answer(item) for item in gt_answers],
                        "exact_match": None,
                        "f1": None,
                        "evaluated": False,
                    }
                )
                continue

        em = metric_max_over_ground_truths(exact_match_score, pred_answer, gt_answers)
        f1 = metric_max_over_ground_truths(f1_score, pred_answer, gt_answers)

        exact_matches += int(em == 1.0)
        em_sum += em
        f1_sum += f1
        evaluated_questions += 1

        details.append(
            {
                "id": ann_id,
                "image_id": gt_item["image_id"],
                "question": gt_item["question"],
                "prediction": pred_answer,
                "answers": gt_answers,
                "normalized_prediction": canonical_answer(pred_answer),
                "normalized_answers": [canonical_answer(item) for item in gt_answers],
                "prediction_tokens": tokenize(pred_answer),
                "answers_tokens": [tokenize(item) for item in gt_answers],
                "exact_match": round(em, 6),
                "f1": round(f1, 6),
                "evaluated": True,
            }
        )

    denominator = evaluated_questions if args.ignore_missing else total_questions
    result = {
        "prediction_path": str(pred_path),
        "ground_truth_path": str(gt_path),
        "total_questions": total_questions,
        "evaluated_questions": evaluated_questions,
        "matched_predictions": matched_predictions,
        "missing_predictions": missing_predictions,
        "skipped_missing_predictions": skipped_missing_predictions,
        "ignore_missing": bool(args.ignore_missing),
        "num_exact_matches": exact_matches,
        "exact_match": round(exact_matches / denominator if denominator else 0.0, 6),
        "f1": round(f1_sum / denominator if denominator else 0.0, 6),
    }

    print(f"prediction_path={pred_path}")
    print(f"ground_truth_path={gt_path}")
    print(f"total_questions={result['total_questions']}")
    print(f"evaluated_questions={result['evaluated_questions']}")
    #print(f"matched_predictions={result['matched_predictions']}")
    #print(f"missing_predictions={result['missing_predictions']}")
    #print(f"skipped_missing_predictions={result['skipped_missing_predictions']}")
    print(f"ignore_missing={result['ignore_missing']}")
    print(f"num_exact_matches={result['num_exact_matches']}")
    print(f"exact_match (EM)={result['exact_match']:.6f}")
    print(f"f1 score (F1-Score)={result['f1']:.6f}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_payload = {
            "summary": result,
            "details": details,
        }
        output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output={output_path}")


if __name__ == "__main__":
    main()
