#!/usr/bin/env python
"""Detect OCR boxes for ViTextVQA test images and export per-image JSONL files.

The script:
- reads image ids from a ViTextVQA split JSON, defaulting to test
- loads the corresponding files from the image directory
- runs PaddleX text detection only
- writes one folder per image, each containing `bboxes.jsonl`

No text recognition is performed here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from paddlex import create_predictor
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Paddle OCR bounding boxes per ViTextVQA test image.")
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("vitextvqa") / "ViTextVQA_images" / "ViTextVQA_test.json",
        help="ViTextVQA split JSON used to choose which images to process.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("vitextvqa") / "ViTextVQA_images" / "st_images",
        help="Directory that contains the image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "test_bboxes",
        help="Root output directory. One folder per image will be created here.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images from the split.")
    parser.add_argument("--batch-size", type=int, default=1, help="Detection batch size.")
    parser.add_argument(
        "--det-model-name",
        type=str,
        default="PP-OCRv5_server_det",
        help="PaddleX detection model name.",
    )
    parser.add_argument("--device", type=str, default="gpu", help="PaddleX device, for example gpu or cpu.")
    parser.add_argument("--unclip-ratio", type=float, default=2.5, help="Detection unclip ratio.")
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Also save a visualization image with bounding boxes inside each image folder.",
    )
    return parser.parse_args()


def load_split_image_paths(split_json: Path, image_dir: Path, limit: int | None = None) -> list[Path]:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    image_paths: list[Path] = []
    for image in data["images"]:
        image_path = image_dir / image["filename"]
        if image_path.exists():
            image_paths.append(image_path)
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def batched(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_images(image_paths: Sequence[Path]) -> tuple[list[Path], list[np.ndarray]]:
    valid_paths: list[Path] = []
    images: list[np.ndarray] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        valid_paths.append(path)
        images.append(image)
    return valid_paths, images


def normalize_polygons(dt_polys: object) -> list[list[list[float]]]:
    if dt_polys is None:
        return []

    polygons = np.array(dt_polys, dtype=np.float32)
    if polygons.size == 0:
        return []
    if polygons.ndim == 2:
        polygons = polygons[np.newaxis, :, :]

    normalized: list[list[list[float]]] = []
    for polygon in polygons:
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            continue
        normalized.append(polygon.tolist())
    return normalized


def polygon_to_box(polygon: list[list[float]]) -> list[float]:
    points = np.array(polygon, dtype=np.float32)
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    return [x_min, y_min, x_max, y_max]


def clip_box(box: list[float], width: int, height: int) -> list[int]:
    x_min = max(0, min(int(round(box[0])), width - 1))
    y_min = max(0, min(int(round(box[1])), height - 1))
    x_max = max(0, min(int(round(box[2])), width))
    y_max = max(0, min(int(round(box[3])), height))
    return [x_min, y_min, x_max, y_max]


def clip_polygon(polygon: list[list[float]], width: int, height: int) -> list[list[int]]:
    clipped: list[list[int]] = []
    for x, y in polygon:
        clipped_x = max(0, min(int(round(x)), width - 1))
        clipped_y = max(0, min(int(round(y)), height - 1))
        clipped.append([clipped_x, clipped_y])
    return clipped


def normalize_scores(raw_scores: object, expected_size: int) -> list[float | None]:
    if raw_scores is None:
        return [None] * expected_size

    scores = list(raw_scores)
    if len(scores) < expected_size:
        scores.extend([None] * (expected_size - len(scores)))
    return [None if score is None else float(score) for score in scores[:expected_size]]


def compute_reading_order(boxes: Sequence[list[int]]) -> list[int]:
    if not boxes:
        return []

    heights = [max(1, box[3] - box[1]) for box in boxes]
    line_threshold = max(10.0, float(np.median(heights)) * 0.6)
    sorted_indices = sorted(range(len(boxes)), key=lambda idx: (boxes[idx][1], boxes[idx][0]))

    lines: list[dict[str, object]] = []
    for idx in sorted_indices:
        box = boxes[idx]
        center_y = (box[1] + box[3]) / 2.0
        placed = False
        for line in lines:
            if abs(center_y - float(line["center_y"])) <= line_threshold:
                members = line["indices"]
                assert isinstance(members, list)
                members.append(idx)
                line["center_y"] = (float(line["center_y"]) * (len(members) - 1) + center_y) / len(members)
                placed = True
                break
        if not placed:
            lines.append({"center_y": center_y, "indices": [idx]})

    ordered_indices: list[int] = []
    for line in sorted(lines, key=lambda item: float(item["center_y"])):
        members = line["indices"]
        assert isinstance(members, list)
        ordered_indices.extend(sorted(members, key=lambda idx: (boxes[idx][0], boxes[idx][1])))
    return ordered_indices


def build_bbox_rows(image_path: Path, prediction: dict, source_image: np.ndarray) -> list[dict]:
    height, width = source_image.shape[:2]
    polygons = normalize_polygons(prediction.get("dt_polys"))
    scores = normalize_scores(prediction.get("dt_scores"), len(polygons))
    boxes = [clip_box(polygon_to_box(polygon), width, height) for polygon in polygons]
    polygons = [clip_polygon(polygon, width, height) for polygon in polygons]

    ordered_indices = compute_reading_order(boxes)
    image_id = image_path.stem

    rows: list[dict] = []
    for order, idx in enumerate(ordered_indices, start=1):
        rows.append(
            {
                "bbox_id": f"img_{image_id}_bbox_{order:04d}",
                "image_id": image_id,
                "image_name": image_path.name,
                "bbox": boxes[idx],
                "polygon": polygons[idx],
                "order": order,
                "det_score": scores[idx],
                "source": "paddle_det",
            }
        )
    return rows


def draw_boxes(image: np.ndarray, rows: Sequence[dict]) -> np.ndarray:
    visualized = image.copy()
    for row in rows:
        x_min, y_min, x_max, y_max = row["bbox"]
        label = str(row["order"])
        cv2.rectangle(visualized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(y_min - 4, text_height + 6)
        cv2.rectangle(
            visualized,
            (x_min, text_y - text_height - 6),
            (x_min + text_width + 6, text_y + 2),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            visualized,
            label,
            (x_min + 3, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return visualized


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_image_meta(path: Path, image_path: Path, rows: Sequence[dict]) -> None:
    payload = {
        "image_id": image_path.stem,
        "image_name": image_path.name,
        "bbox_count": len(rows),
        "jsonl_file": "bboxes.jsonl",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_paths = load_split_image_paths(args.split_json, args.image_dir, args.limit)
    if not image_paths:
        raise SystemExit(f"No images found from split {args.split_json} in {args.image_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector = create_predictor(
        model_name=args.det_model_name,
        device=args.device,
        unclip_ratio=args.unclip_ratio,
    )

    total_images = 0
    total_boxes = 0
    total_batches = math.ceil(len(image_paths) / args.batch_size)

    for batch_paths in tqdm(batched(image_paths, args.batch_size), total=total_batches, desc="Detect"):
        valid_paths, images = load_images(batch_paths)
        if not images:
            continue

        predictions = detector.predict(images, batch_size=len(images))
        for image_path, image, prediction in zip(valid_paths, images, predictions):
            source_image = prediction.get("input_img", image)
            rows = build_bbox_rows(image_path, prediction, source_image)

            image_output_dir = args.output_dir / image_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(image_output_dir / "bboxes.jsonl", rows)
            write_image_meta(image_output_dir / "meta.json", image_path, rows)

            if args.save_vis:
                visualized = draw_boxes(source_image, rows)
                cv2.imwrite(str(image_output_dir / f"{image_path.stem}_boxes.jpg"), visualized)

            total_images += 1
            total_boxes += len(rows)

    print(f"Processed images: {total_images}")
    print(f"Detected boxes: {total_boxes}")
    print(f"Saved per-image outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
