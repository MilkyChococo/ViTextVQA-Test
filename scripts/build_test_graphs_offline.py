#!/usr/bin/env python
"""Build offline spatial graphs for the whole ViTextVQA test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from encode.embeddings import init_embedding_models, resolve_device
from spatial_graph.pipeline import run_pipeline
from utils.config import GraphConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline graphs for the whole ViTextVQA test split.")
    parser.add_argument("--split-json", default=None, help="Path to ViTextVQA_test.json. Defaults to repo dataset file.")
    parser.add_argument("--image-root", default=None, help="Path to st_images directory.")
    parser.add_argument("--ocr-root", default=None, help="Path to OCR root directory, e.g. outputs/OCR_img.")
    parser.add_argument("--output-root", default=None, help="Path to offline graph output root. Defaults to outputs/graph_test.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of test images to process.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset within the test image list.")
    parser.add_argument("--resume", action="store_true", help="Skip images that already have graph_enriched.json.")
    parser.add_argument("--no-embeddings", action="store_true", help="Build graphs without text/crop embeddings.")
    parser.add_argument("--save-node-crops", action="store_true", help="Save per-node crop files.")
    parser.add_argument("--save-visuals", action="store_true", help="Save overlay visualization images.")
    parser.add_argument("--device", default="cuda", help="Preferred device for embedding models.")
    parser.add_argument("--batch-report", default=None, help="Optional path to final batch summary JSON.")
    return parser.parse_args()


def load_test_images(split_json_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"Unexpected test split format in: {split_json_path}")
    return images


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    split_json_path = Path(args.split_json) if args.split_json else repo_root / "vitextvqa" / "ViTextVQA_images" / "ViTextVQA_test.json"
    image_root = Path(args.image_root) if args.image_root else repo_root / "vitextvqa" / "ViTextVQA_images" / "st_images"
    ocr_root = Path(args.ocr_root) if args.ocr_root else repo_root / "outputs" / "OCR_img"
    output_root = Path(args.output_root) if args.output_root else repo_root / "outputs" / "graph_test"

    if not split_json_path.exists():
        raise FileNotFoundError(f"Split JSON not found: {split_json_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if not ocr_root.exists():
        raise FileNotFoundError(f"OCR root not found: {ocr_root}")

    images = load_test_images(split_json_path)
    images = images[args.start_index :]
    if args.limit is not None:
        images = images[: args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    batch_dir = output_root / "_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    template_config = GraphConfig(
        repo_root=repo_root,
        image_id="0",
        image_filename="0.jpg",
        image_root=image_root,
        ocr_root=ocr_root,
        output_root=output_root,
        build_embeddings=not args.no_embeddings,
        preferred_device=args.device,
        save_node_crops=args.save_node_crops,
        save_visuals=args.save_visuals,
    )

    summary_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failed = 0
    preloaded_models = None

    if not args.no_embeddings:
        runtime_device = resolve_device(args.device)
        preloaded_models = init_embedding_models(
            text_model_name=template_config.text_embedding_model,
            image_model_name=template_config.image_embedding_model,
            device=runtime_device,
        )

    for image_info in tqdm(images, desc="BuildTestGraphs"):
        image_id = str(image_info["id"])
        image_filename = str(image_info.get("filename") or f"{image_id}.jpg")
        config = GraphConfig(
            repo_root=repo_root,
            image_id=image_id,
            image_filename=image_filename,
            image_root=image_root,
            ocr_root=ocr_root,
            output_root=output_root,
            build_embeddings=not args.no_embeddings,
            preferred_device=args.device,
            save_node_crops=args.save_node_crops,
            save_visuals=args.save_visuals,
        )

        if args.resume and config.graph_enriched_path.exists():
            skipped += 1
            summary_rows.append(
                {
                    "image_id": image_id,
                    "status": "skipped",
                    "reason": "graph_enriched_exists",
                    "output_dir": str(config.output_dir),
                }
            )
            continue

        if not config.ocr_jsonl_path.exists():
            failed += 1
            row = {
                "image_id": image_id,
                "status": "failed",
                "reason": "missing_ocr_jsonl",
                "ocr_jsonl_path": str(config.ocr_jsonl_path),
            }
            summary_rows.append(row)
            error_rows.append(row)
            continue

        if not config.image_path.exists():
            failed += 1
            row = {
                "image_id": image_id,
                "status": "failed",
                "reason": "missing_image",
                "image_path": str(config.image_path),
            }
            summary_rows.append(row)
            error_rows.append(row)
            continue

        try:
            stats = run_pipeline(config, preloaded_models=preloaded_models)
            processed += 1
            summary_rows.append(
                {
                    "image_id": image_id,
                    "status": "ok",
                    "image_filename": image_filename,
                    "raw_rows": stats["raw_rows"],
                    "merged_nodes": stats["merged_nodes"],
                    "spatial_edges": stats["spatial_edges"],
                    "output_dir": stats["output_dir"],
                    "graph_enriched_json": stats["graph_enriched_json"],
                }
            )
        except Exception as exc:
            failed += 1
            row = {
                "image_id": image_id,
                "status": "failed",
                "reason": type(exc).__name__,
                "error": str(exc),
                "image_filename": image_filename,
            }
            summary_rows.append(row)
            error_rows.append(row)

    summary = {
        "split_json": str(split_json_path),
        "image_root": str(image_root),
        "ocr_root": str(ocr_root),
        "output_root": str(output_root),
        "requested_images": len(images),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "build_embeddings": not args.no_embeddings,
        "save_node_crops": args.save_node_crops,
        "save_visuals": args.save_visuals,
        "device": args.device,
    }

    summary_path = Path(args.batch_report) if args.batch_report else batch_dir / "build_test_graphs_summary.json"
    details_path = batch_dir / "build_test_graphs_details.json"
    errors_path = batch_dir / "build_test_graphs_errors.json"
    save_json(summary_path, summary)
    save_json(details_path, summary_rows)
    save_json(errors_path, error_rows)

    print(f"requested_images={summary['requested_images']}")
    print(f"processed={processed}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    print(f"output_root={output_root}")
    print(f"summary_json={summary_path}")
    print(f"details_json={details_path}")
    print(f"errors_json={errors_path}")


if __name__ == "__main__":
    main()
