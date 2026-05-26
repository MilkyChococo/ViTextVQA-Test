#!/usr/bin/env python
"""Run ViTextVQA ablation modes with Qwen2.5-VL.

Modes:
- model_only
- model_ocr
- spatial_graph_conf
- cse_no_hub
- cse_with_hub
"""

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

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from algo.cse_query import (  # noqa: E402
    build_query_relevance_scores,
    run_text_first_cse,
    select_top_k_seed_nodes,
)
from encode.embeddings import init_clip_model, init_text_model, resolve_device  # noqa: E402
from model import (  # noqa: E402
    collect_context_nodes,
    collect_crop_paths,
    load_env_file,
    resolve_repo_relative_path,
    safe_text,
)
from model_qwen import DEFAULT_QWEN_MODEL_NAME, generate_with_qwen, init_qwen_model  # noqa: E402
from process.text_preprocess import preprocess_query_text  # noqa: E402
from spatial_graph.io_utils import load_rows  # noqa: E402
from utils.config import GraphConfig  # noqa: E402
from utils.prompts import build_image_only_prompt, build_ocr_prompt, build_vlm_prompt  # noqa: E402

from cse_no_hub import MODE as CSE_NO_HUB_MODE  # noqa: E402
from cse_with_hub import MODE as CSE_WITH_HUB_MODE  # noqa: E402
from model_ocr import MODE as MODEL_OCR_MODE  # noqa: E402
from model_only import MODE as MODEL_ONLY_MODE  # noqa: E402
from spatial_graph_conf import MODE as SPATIAL_GRAPH_CONF_MODE  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODES = {
    item["name"]: item
    for item in (
        MODEL_ONLY_MODE,
        MODEL_OCR_MODE,
        SPATIAL_GRAPH_CONF_MODE,
        CSE_NO_HUB_MODE,
        CSE_WITH_HUB_MODE,
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ViTextVQA ablation mode with Qwen2.5-VL.")
    parser.add_argument("--mode", choices=sorted(MODES), required=True, help="Ablation mode to run.")
    parser.add_argument("--test-json", default="vitextvqa/ViTextVQA_images/ViTextVQA_test.json")
    parser.add_argument("--graph-root", default="outputs/graph_test")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda-hub", type=float, default=0.05)
    parser.add_argument("--max-nodes", type=int, default=100)
    parser.add_argument("--max-edges", type=int, default=200)
    parser.add_argument("--max-subgraphs", type=int, default=2)
    parser.add_argument("--max-nodes-per-subgraph", type=int, default=4)
    parser.add_argument("--max-crops", type=int, default=4)
    parser.add_argument("--max-ocr-lines", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-graph-cache-size", type=int, default=1)
    parser.add_argument("--text-retrieval-device", default="cuda")
    parser.add_argument("--clip-retrieval-device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_test_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "images" not in payload or "annotations" not in payload:
        raise ValueError(f"Unexpected ViTextVQA test format: {path}")
    return payload


def build_image_filename_lookup(images: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(item.get("id")): str(item.get("filename") or f"{item.get('id')}.jpg")
        for item in images
    }


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
    return {"images": base_payload.get("images", []), "annotations": annotations}


def save_prediction_payload(path: Path, base_payload: dict[str, Any], answer_by_ann_id: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_prediction_payload(base_payload, answer_by_ann_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_runtime_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


def load_graph_bundle(config: GraphConfig) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    graph_enriched = json.loads(config.graph_enriched_path.read_text(encoding="utf-8"))
    text_embeddings = np.load(config.text_embeddings_path)
    crop_embeddings = np.load(config.crop_embeddings_path)
    return graph_enriched, text_embeddings, crop_embeddings


def graph_node_to_context(node: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    return {
        "subgraph_rank": rank,
        "subgraph_score": score,
        "node_id": str(node.get("node_id", "")),
        "text": safe_text(node.get("text", "")),
        "bbox": node.get("bbox"),
        "crop_bbox": node.get("crop_bbox"),
        "final_score": score,
        "rel": float(node.get("rel", score) or 0.0),
        "crop_path": node.get("crop_path"),
    }


def build_graph_conf_context_nodes(
    *,
    graph_enriched: dict[str, Any],
    text_embeddings: np.ndarray,
    crop_embeddings: np.ndarray,
    query: str,
    query_for_embedding: str,
    config: GraphConfig,
    args: argparse.Namespace,
    text_embedder: Any,
    clip_bundle: tuple[Any, Any],
    text_device: str,
    image_device: str,
) -> list[dict[str, Any]]:
    del query

    rel_scores = build_query_relevance_scores(
        query_for_embedding=query_for_embedding,
        text_embeddings=text_embeddings,
        crop_embeddings=crop_embeddings,
        graph=graph_enriched,
        text_model_name=config.text_embedding_model,
        image_model_name=config.image_embedding_model,
        device=text_device,
        text_device=text_device,
        image_device=image_device,
        rel_text_weight=config.rel_text_weight,
        rel_image_weight=config.rel_image_weight,
        preloaded_text_embedder=text_embedder,
        preloaded_clip=clip_bundle,
    )
    seeds = select_top_k_seed_nodes(
        graph=graph_enriched,
        rel_scores=rel_scores,
        top_k=args.top_k,
        allowed_node_types=("text",),
    )
    node_index = {str(node.get("node_id", "")): node for node in graph_enriched.get("nodes", [])}

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for seed_rank, seed in enumerate(seeds, start=1):
        seed_node = dict(node_index.get(seed.node_id, {}))
        if not seed_node or seed.node_id in seen:
            continue
        seed_node["rel"] = seed.rel
        selected.append(graph_node_to_context(seed_node, seed_rank, seed.rel))
        seen.add(seed.node_id)
    return selected


def build_context_for_mode(
    *,
    mode: dict[str, Any],
    config: GraphConfig,
    question: str,
    args: argparse.Namespace,
    graph_bundle: tuple[dict[str, Any], np.ndarray, np.ndarray] | None,
    text_embedder: Any | None,
    clip_bundle: tuple[Any, Any] | None,
    text_device: str,
    image_device: str,
) -> tuple[str, list[dict[str, Any]], list[tuple[str, Path]]]:
    strategy = str(mode["strategy"])
    crop_paths: list[tuple[str, Path]] = []

    if strategy == "image_only":
        return build_image_only_prompt(question), [], []

    if strategy == "ocr_text":
        rows = load_rows(config.ocr_jsonl_path) if config.ocr_jsonl_path.exists() else []
        return build_ocr_prompt(question, rows, max_ocr_lines=args.max_ocr_lines), [], []

    if graph_bundle is None:
        raise ValueError(f"Mode {mode['name']} requires a prebuilt graph bundle.")
    graph_enriched, text_embeddings, crop_embeddings = graph_bundle
    query_for_embedding = preprocess_query_text(config, question)

    if strategy == "graph_conf":
        if text_embedder is None or clip_bundle is None:
            raise ValueError("graph_conf requires retrieval models.")
        context_nodes = build_graph_conf_context_nodes(
            graph_enriched=graph_enriched,
            text_embeddings=text_embeddings,
            crop_embeddings=crop_embeddings,
            query=question,
            query_for_embedding=query_for_embedding,
            config=config,
            args=args,
            text_embedder=text_embedder,
            clip_bundle=clip_bundle,
            text_device=text_device,
            image_device=image_device,
        )
    elif strategy == "cse":
        if text_embedder is None or clip_bundle is None:
            raise ValueError("cse requires retrieval models.")
        lambda_hub = mode.get("lambda_hub_override")
        if lambda_hub is None:
            lambda_hub = args.lambda_hub
        cse_payload = run_text_first_cse(
            graph_enriched=graph_enriched,
            text_embeddings=text_embeddings,
            crop_embeddings=crop_embeddings,
            query=question,
            query_for_embedding=query_for_embedding,
            text_model_name=config.text_embedding_model,
            image_model_name=config.image_embedding_model,
            device=text_device,
            text_device=text_device,
            image_device=image_device,
            top_k=args.top_k,
            hops=args.hops,
            top_m=args.top_m,
            threshold=args.threshold,
            alpha=args.alpha,
            lambda_hub=float(lambda_hub),
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            rel_text_weight=config.rel_text_weight,
            rel_image_weight=config.rel_image_weight,
            preloaded_text_embedder=text_embedder,
            preloaded_clip=clip_bundle,
        )
        context_nodes = collect_context_nodes(
            cse_payload=cse_payload,
            max_subgraphs=args.max_subgraphs,
            max_nodes_per_subgraph=args.max_nodes_per_subgraph,
        )
    else:
        raise ValueError(f"Unknown ablation strategy: {strategy}")

    runtime_crops_dir = config.output_dir / f"ablation_{mode['name']}_runtime_crops"
    crop_paths = collect_crop_paths(
        context_nodes,
        image_path=config.image_path,
        runtime_crops_dir=runtime_crops_dir,
        max_crops=args.max_crops,
    )
    prompt = build_vlm_prompt(question, context_nodes=context_nodes, crop_paths=crop_paths)
    return prompt, context_nodes, crop_paths


def default_output_path(repo_root: Path, mode_name: str) -> Path:
    return repo_root / "outputs" / "predictions" / f"vitextvqa_test_qwen_ablation_{mode_name}.json"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    mode = MODES[args.mode]
    repo_root = SCRIPT_ROOT.parent
    load_env_file(repo_root / ".env")

    test_json_path = resolve_repo_relative_path(repo_root, args.test_json)
    graph_root = resolve_repo_relative_path(repo_root, args.graph_root)
    output_path = resolve_repo_relative_path(repo_root, args.output) if args.output else default_output_path(repo_root, args.mode)
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
            if isinstance(answers, list) and answers and str(answers[0]).strip():
                answer_by_ann_id[ann_id] = str(answers[0]).strip()

    annotations = annotations[args.start_index :]
    if args.limit is not None:
        annotations = annotations[: args.limit]

    text_device = resolve_device(args.text_retrieval_device)
    image_device = resolve_device(args.clip_retrieval_device)
    text_embedder = None
    clip_bundle = None
    if mode.get("requires_retrieval_models"):
        text_embedder = init_text_model("BAAI/bge-m3", text_device)
        clip_bundle = init_clip_model("openai/clip-vit-base-patch32", image_device)

    qwen_model, qwen_processor = init_qwen_model(args.model)
    graph_cache: OrderedDict[str, tuple[dict[str, Any], np.ndarray, np.ndarray]] = OrderedDict()

    processed = 0
    skipped = 0
    failed = 0
    progress_bar = tqdm(annotations, desc=f"Ablation:{args.mode}", unit="question")
    for ann in progress_bar:
        ann_id = int(ann["id"])
        image_id = str(ann["image_id"])
        question = str(ann["question"])

        if args.resume and ann_id in answer_by_ann_id:
            skipped += 1
            progress_bar.set_postfix(processed=processed, skipped=skipped, failed=failed)
            continue

        image_filename = image_filename_lookup.get(image_id, f"{image_id}.jpg")
        config = GraphConfig(
            repo_root=repo_root,
            image_id=image_id,
            image_filename=image_filename,
            output_root=graph_root,
        )
        graph_bundle = None
        crop_paths: list[tuple[str, Path]] = []
        runtime_crops_dir = config.output_dir / f"ablation_{mode['name']}_runtime_crops"

        try:
            if mode.get("requires_graph"):
                if image_id not in graph_cache:
                    graph_cache[image_id] = load_graph_bundle(config)
                    if args.max_graph_cache_size > 0:
                        while len(graph_cache) > args.max_graph_cache_size:
                            graph_cache.popitem(last=False)
                else:
                    graph_cache.move_to_end(image_id)
                graph_bundle = graph_cache[image_id]

            prompt, _, crop_paths = build_context_for_mode(
                mode=mode,
                config=config,
                question=question,
                args=args,
                graph_bundle=graph_bundle,
                text_embedder=text_embedder,
                clip_bundle=clip_bundle,
                text_device=text_device,
                image_device=image_device,
            )
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
            print(f"[FAIL] ann_id={ann_id} image_id={image_id} mode={args.mode} {type(exc).__name__}: {exc}")
        finally:
            cleanup_runtime_crops(crop_paths, runtime_crops_dir)
            clear_runtime_memory()

        save_prediction_payload(output_path, payload, answer_by_ann_id)
        progress_bar.set_postfix(
            processed=processed,
            skipped=skipped,
            failed=failed,
            cache=len(graph_cache),
        )

    save_prediction_payload(output_path, payload, answer_by_ann_id)
    print(f"mode={args.mode}")
    print(f"description={mode['description']}")
    print(f"processed={processed}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
