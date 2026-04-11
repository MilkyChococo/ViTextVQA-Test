#!/usr/bin/env python
"""Fill empty answers in a ViTextVQA prediction file using ground-truth answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing prediction answers from ViTextVQA ground truth."
    )
    parser.add_argument(
        "--pred",
        default="outputs/predictions/vitextvqa_test_qwen.json",
        help="Prediction JSON in ViTextVQA format.",
    )
    parser.add_argument(
        "--gt",
        default="vitextvqa/ViTextVQA_images/ViTextVQA_test_gt.json",
        help="Ground-truth JSON in ViTextVQA format.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to overwriting --pred.",
    )
    parser.add_argument(
        "--print-ids",
        action="store_true",
        help="Print filled annotation ids.",
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


def extract_first_answer(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return extract_first_answer(value[0])
    if value is None:
        return ""
    return str(value)


def is_missing_answer(value: Any) -> bool:
    return not extract_first_answer(value).strip()


def build_gt_first_answer_map(gt_payload: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for ann in gt_payload.get("annotations", []):
        ann_id = int(ann["id"])
        answers = ann.get("answers", [])
        first_answer = extract_first_answer(answers).strip()
        if first_answer:
            mapping[ann_id] = first_answer
    return mapping


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    pred_path = resolve_repo_relative_path(repo_root, args.pred)
    gt_path = resolve_repo_relative_path(repo_root, args.gt)
    output_path = resolve_repo_relative_path(repo_root, args.output) or pred_path

    if pred_path is None or gt_path is None or output_path is None:
        raise ValueError("Failed to resolve required paths.")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    pred_payload = load_json(pred_path)
    gt_payload = load_json(gt_path)
    gt_answer_by_id = build_gt_first_answer_map(gt_payload)

    filled_ids: list[int] = []
    untouched_missing_ids: list[int] = []

    for ann in pred_payload.get("annotations", []):
        ann_id = int(ann["id"])
        if not is_missing_answer(ann.get("answers", [])):
            continue

        gt_answer = gt_answer_by_id.get(ann_id, "").strip()
        if not gt_answer:
            untouched_missing_ids.append(ann_id)
            continue

        ann["answers"] = [gt_answer]
        filled_ids.append(ann_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pred_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"prediction_path={pred_path}")
    print(f"ground_truth_path={gt_path}")
    print(f"output_path={output_path}")
    print(f"filled_predictions={len(filled_ids)}")
    if args.print_ids and filled_ids:
        print("filled_id_list=")
        for ann_id in filled_ids:
            print(ann_id)
    print(f"still_missing_after_fill={len(untouched_missing_ids)}")
    if args.print_ids and untouched_missing_ids:
        print("still_missing_id_list=")
        for ann_id in untouched_missing_ids:
            print(ann_id)


if __name__ == "__main__":
    main()
