#!/usr/bin/env python
"""Run full ViTextVQA test workflow with Qwen and export predictions in test JSON format."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from algo.cse_query import run_text_first_cse
from encode.embeddings import init_clip_model, init_text_model, resolve_device
from model import (
    collect_context_nodes,
    collect_crop_paths,
    load_env_file,
    resolve_repo_relative_path,
)
from model_qwen import DEFAULT_QWEN_MODEL_NAME, generate_with_qwen, init_qwen_model
from process.text_preprocess import preprocess_query_text
from utils.config import GraphConfig
from utils.prompts import build_vlm_prompt

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ViTextVQA test workflow with Qwen2.5-VL-7B.")
    parser.add_argument("--test-json", default="vitextvqa/ViTextVQA_images/ViTextVQA_test.json", help="Path to ViTextVQA_test.json.")
    parser.add_argument("--graph-root", default="outputs/graph_test", help="Root directory of prebuilt graph stores.")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL_NAME, help="Qwen2.5-VL model name or local path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of text seed nodes for CSE.")
    parser.add_argument("--hops", type=int, default=3, help="Expansion hops for CSE.")
    parser.add_argument("--top-m", type=int, default=5, help="Top outgoing edges kept per frontier node.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Minimum CSE edge score.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for offline edge confidence.")
    parser.add_argument("--lambda-hub", type=float, default=0.05, help="Hub penalty.")
    parser.add_argument("--max-nodes", type=int, default=100, help="Maximum nodes in each subgraph.")
    parser.add_argument("--max-edges", type=int, default=200, help="Maximum edges in each subgraph.")
    parser.add_argument("--max-subgraphs", type=int, default=2, help="Maximum retrieved subgraphs to include.")
    parser.add_argument("--max-nodes-per-subgraph", type=int, default=3, help="Maximum nodes per subgraph.")
    parser.add_argument("--max-crops", type=int, default=4, help="Maximum node crop images to attach. Use 0 to attach all selected node crops.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum newly generated tokens.")
    parser.add_argument("--output", default="outputs/predictions/vitextvqa_test_qwen.json", help="Output JSON path.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset inside annotations.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of annotations to run.")
    parser.add_argument("--max-graph-cache-size", type=int, default=1, help="Maximum number of per-image graph bundles kept in RAM.")
    parser.add_argument("--text-retrieval-device", default="cuda", help="Device for BGE-M3 query embedding.")
    parser.add_argument("--clip-retrieval-device", default="cpu", help="Device for CLIP query embedding.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output JSON if it exists.")
    return parser.parse_args()


def load_test_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "images" not in payload or "annotations" not in payload:
        raise ValueError(f"Unexpected ViTextVQA test format: {path}")
    return payload


def build_image_filename_lookup(images: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in images:
        image_id = str(item.get("id"))
        filename = str(item.get("filename") or f"{image_id}.jpg")
        lookup[image_id] = filename
    return lookup


def load_graph_bundle(config: GraphConfig) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    graph_enriched = json.loads(config.graph_enriched_path.read_text(encoding="utf-8"))
    text_embeddings = np.load(config.text_embeddings_path)
    crop_embeddings = np.load(config.crop_embeddings_path)
    return graph_enriched, text_embeddings, crop_embeddings


def build_prediction_payload(
    base_payload: dict[str, Any],
    answer_by_ann_id: dict[int, str],
) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = []
    for ann in base_payload.get("annotations", []):
        ann_id = int(ann["id"])
        new_ann = dict(ann)
        new_ann["answers"] = [answer_by_ann_id.get(ann_id, "")]
        annotations.append(new_ann)
    return {
        "images": base_payload.get("images", []),
        "annotations": annotations,
    }


def save_prediction_payload(path: Path, base_payload: dict[str, Any], answer_by_ann_id: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_prediction_payload(base_payload, answer_by_ann_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cleanup_runtime_crops(crop_paths: list[tuple[str, Path]], runtime_crops_dir: Path) -> None:
    runtime_root = runtime_crops_dir.resolve()
    for _, crop_path in crop_paths:
        try:
            resolved = crop_path.resolve()
        except Exception:
            continue
        if resolved.parent == runtime_root and resolved.exists():
            try:
                resolved.unlink()
            except OSError:
                pass


def clear_runtime_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    load_env_file(repo_root / ".env")

    test_json_path = resolve_repo_relative_path(repo_root, args.test_json)
    graph_root = resolve_repo_relative_path(repo_root, args.graph_root)
    output_path = resolve_repo_relative_path(repo_root, args.output)
    if test_json_path is None or graph_root is None or output_path is None:
        raise ValueError("Failed to resolve required paths.")

    payload = load_test_payload(test_json_path)
    annotations = list(payload.get("annotations", []))
    image_filename_lookup = build_image_filename_lookup(payload.get("images", []))

    answer_by_ann_id: dict[int, str] = {}
    if args.resume and output_path.exists():
        existing = load_test_payload(output_path)
        for ann in existing.get("annotations", []):
            ann_id = int(ann["id"])
            answers = ann.get("answers", [])
            if isinstance(answers, list) and answers:
                answer = str(answers[0]).strip()
                if answer:
                    answer_by_ann_id[ann_id] = answer

    annotations = annotations[args.start_index :]
    if args.limit is not None:
        annotations = annotations[: args.limit]

    runtime_device = resolve_device("cuda")
    text_retrieval_device = resolve_device(args.text_retrieval_device)
    clip_retrieval_device = resolve_device(args.clip_retrieval_device)
    text_embedder = init_text_model("BAAI/bge-m3", text_retrieval_device)
    clip_processor, clip_model = init_clip_model("openai/clip-vit-base-patch32", clip_retrieval_device)
    qwen_model, qwen_processor = init_qwen_model(args.model)

    graph_cache: OrderedDict[str, tuple[GraphConfig, dict[str, Any], np.ndarray, np.ndarray]] = OrderedDict()

    processed = 0
    skipped = 0
    failed = 0
    progress_bar = tqdm(annotations, desc="RunTestQwen", unit="question")
    for ann in progress_bar:
        ann_id = int(ann["id"])
        image_id = str(ann["image_id"])
        question = str(ann["question"])

        if args.resume and ann_id in answer_by_ann_id:
            skipped += 1
            progress_bar.set_postfix(processed=processed, skipped=skipped, failed=failed)
            continue

        if image_id not in graph_cache:
            image_filename = image_filename_lookup.get(image_id, f"{image_id}.jpg")
            config = GraphConfig(
                repo_root=repo_root,
                image_id=image_id,
                image_filename=image_filename,
                output_root=graph_root,
            )
            graph_enriched, text_embeddings, crop_embeddings = load_graph_bundle(config)
            graph_cache[image_id] = (config, graph_enriched, text_embeddings, crop_embeddings)
            if args.max_graph_cache_size > 0:
                while len(graph_cache) > args.max_graph_cache_size:
                    graph_cache.popitem(last=False)
        else:
            graph_cache.move_to_end(image_id)

        config, graph_enriched, text_embeddings, crop_embeddings = graph_cache[image_id]
        runtime_crops_dir = config.output_dir / "vlm_qwen_runtime_crops"
        crop_paths: list[tuple[str, Path]] = []
        query_for_embedding: str | None = None
        cse_payload: dict[str, Any] | None = None
        context_nodes: list[dict[str, Any]] = []
        prompt: str | None = None
        answer: str | None = None

        try:
            query_for_embedding = preprocess_query_text(config, question)
            cse_payload = run_text_first_cse(
                graph_enriched=graph_enriched,
                text_embeddings=text_embeddings,
                crop_embeddings=crop_embeddings,
                query=question,
                query_for_embedding=query_for_embedding,
                text_model_name=config.text_embedding_model,
                image_model_name=config.image_embedding_model,
                device=text_retrieval_device,
                text_device=text_retrieval_device,
                image_device=clip_retrieval_device,
                top_k=args.top_k,
                hops=args.hops,
                top_m=args.top_m,
                threshold=args.threshold,
                alpha=args.alpha,
                lambda_hub=args.lambda_hub,
                max_nodes=args.max_nodes,
                max_edges=args.max_edges,
                rel_text_weight=config.rel_text_weight,
                rel_image_weight=config.rel_image_weight,
                preloaded_text_embedder=text_embedder,
                preloaded_clip=(clip_processor, clip_model),
            )
            context_nodes = collect_context_nodes(
                cse_payload=cse_payload,
                max_subgraphs=args.max_subgraphs,
                max_nodes_per_subgraph=args.max_nodes_per_subgraph,
            )
            crop_paths = collect_crop_paths(
                context_nodes,
                image_path=config.image_path,
                runtime_crops_dir=runtime_crops_dir,
                max_crops=args.max_crops,
            )
            prompt = build_vlm_prompt(question, context_nodes=context_nodes, crop_paths=crop_paths)
            answer = generate_with_qwen(
                model=qwen_model,
                processor=qwen_processor,
                prompt=prompt,
                image_path=config.image_path,
                crop_paths=crop_paths,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            answer_by_ann_id[ann_id] = answer
            processed += 1
        except Exception as exc:
            answer_by_ann_id[ann_id] = ""
            failed += 1
            print(f"[FAIL] ann_id={ann_id} image_id={image_id} error={type(exc).__name__}: {exc}")
        finally:
            cleanup_runtime_crops(crop_paths, runtime_crops_dir)
            crop_paths = []
            query_for_embedding = None
            cse_payload = None
            context_nodes = []
            prompt = None
            answer = None
            clear_runtime_memory()

        save_prediction_payload(output_path, payload, answer_by_ann_id)
        progress_bar.set_postfix(processed=processed, skipped=skipped, failed=failed, cache=len(graph_cache))

    save_prediction_payload(output_path, payload, answer_by_ann_id)

    print(f"processed={processed}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
