#!/usr/bin/env python
"""Run full CSE -> VLM flow with local Qwen2.5-VL-7B-Instruct."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from model import (
    build_cse_payload,
    collect_context_nodes,
    collect_crop_paths,
    load_cse_payload,
    load_env_file,
    resolve_repo_relative_path,
    sanitize_model_name,
)
from utils.config import GraphConfig
from utils.prompts import build_vlm_prompt


DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feed image + OCR CSE context into Qwen2.5-VL-7B.")
    parser.add_argument("query", help="Vietnamese question to answer.")
    parser.add_argument("--image-id", default="1003", help="Image id / graph store id.")
    parser.add_argument("--graph-root", default=None, help="Optional root directory of prebuilt graph stores, e.g. outputs/graph_test.")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL_NAME, help="Qwen2.5-VL model name or local path.")
    parser.add_argument("--cse-json", default=None, help="Optional precomputed CSE JSON. If omitted, CSE is computed automatically.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of text seed nodes for text-first CSE.")
    parser.add_argument("--hops", type=int, default=3, help="Expansion hops for CSE.")
    parser.add_argument("--top-m", type=int, default=5, help="Top outgoing edges kept per frontier node.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Minimum CSE edge score.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for offline edge confidence.")
    parser.add_argument("--lambda-hub", type=float, default=0.05, help="Hub penalty.")
    parser.add_argument("--max-nodes", type=int, default=100, help="Maximum nodes in each subgraph.")
    parser.add_argument("--max-edges", type=int, default=200, help="Maximum edges in each subgraph.")
    parser.add_argument("--max-subgraphs", type=int, default=3, help="Maximum retrieved subgraphs to include.")
    parser.add_argument("--max-nodes-per-subgraph", type=int, default=5, help="Maximum nodes per subgraph.")
    parser.add_argument("--max-crops", type=int, default=0, help="Maximum node crop images to attach. Use 0 to attach all selected node crops.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum newly generated tokens.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def default_output_path(config: GraphConfig, model_name: str, query: str) -> Path:
    digest = hashlib.md5(query.encode("utf-8")).hexdigest()[:10]
    return config.output_dir / "vlm_qwen" / f"{sanitize_model_name(model_name)}_{digest}.json"


def init_qwen_model(model_name: str) -> tuple[object, object]:
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies for Qwen2.5-VL. Install torch, transformers, and accelerate."
        ) from exc

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def build_qwen_messages(prompt: str, image_path: Path, crop_paths: list[tuple[str, Path]]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    content.append({"type": "image", "image": str(image_path.resolve())})
    for _, crop_path in crop_paths:
        content.append({"type": "image", "image": str(crop_path.resolve())})
    return [{"role": "user", "content": content}]


def generate_with_qwen(
    model: object,
    processor: object,
    prompt: str,
    image_path: Path,
    crop_paths: list[tuple[str, Path]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    messages = build_qwen_messages(prompt=prompt, image_path=image_path, crop_paths=crop_paths)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
    }
    generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return str(output_text).strip()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    load_env_file(repo_root / ".env")
    graph_root = resolve_repo_relative_path(repo_root, args.graph_root)
    config = GraphConfig(repo_root=repo_root, image_id=args.image_id, output_root=graph_root)

    if not config.image_path.exists():
        raise FileNotFoundError(f"Image not found: {config.image_path}")

    if args.cse_json:
        cse_json_path = Path(args.cse_json)
        if not cse_json_path.exists():
            raise FileNotFoundError(f"CSE JSON not found: {cse_json_path}")
        cse_payload = load_cse_payload(cse_json_path)
    else:
        cse_payload, cse_json_path = build_cse_payload(config, args.query, args)

    context_nodes = collect_context_nodes(
        cse_payload=cse_payload,
        max_subgraphs=args.max_subgraphs,
        max_nodes_per_subgraph=args.max_nodes_per_subgraph,
    )
    crop_paths = collect_crop_paths(
        context_nodes,
        image_path=config.image_path,
        runtime_crops_dir=config.output_dir / "vlm_qwen_runtime_crops",
        max_crops=args.max_crops,
    )
    prompt = build_vlm_prompt(args.query, context_nodes=context_nodes, crop_paths=crop_paths)

    model, processor = init_qwen_model(args.model)
    answer = generate_with_qwen(
        model=model,
        processor=processor,
        prompt=prompt,
        image_path=config.image_path,
        crop_paths=crop_paths,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    output_path = Path(args.output) if args.output else default_output_path(config, args.model, args.query)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": args.query,
        "model": args.model,
        "image_id": config.image_id,
        "image_path": str(config.image_path),
        "cse_json_path": str(cse_json_path),
        "cse_params": {
            "top_k": args.top_k,
            "hops": args.hops,
            "top_m": args.top_m,
            "threshold": args.threshold,
            "alpha": args.alpha,
            "lambda_hub": args.lambda_hub,
            "max_nodes": args.max_nodes,
            "max_edges": args.max_edges,
        },
        "max_subgraphs": args.max_subgraphs,
        "max_nodes_per_subgraph": args.max_nodes_per_subgraph,
        "max_crops": args.max_crops,
        "max_new_tokens": args.max_new_tokens,
        "used_crop_paths": [str(path) for _, path in crop_paths],
        "used_node_ids": [node["node_id"] for node in context_nodes],
        "answer": answer,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"query={args.query}")
    print(f"model={args.model}")
    print(f"image_path={config.image_path}")
    print(f"cse_json={cse_json_path}")
    print(f"used_nodes={len(context_nodes)}")
    print(f"used_crops={len(crop_paths)}")
    print()
    print(answer)
    print()
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
