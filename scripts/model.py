#!/usr/bin/env python
"""Call a VLM with the original image and retrieved OCR subgraph context."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from algo.cse_query import run_text_first_cse, save_cse_subgraph
from encode.embeddings import resolve_device
from process.text_preprocess import preprocess_query_text
from spatial_graph.io_utils import fix_mojibake
from utils.config import GraphConfig
from utils.prompts import build_vlm_prompt


DEFAULT_GEMINI_MODEL_NAME = "gemini-2.5-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feed image + OCR CSE context into a VLM.")
    parser.add_argument("query", help="Vietnamese question to answer.")
    parser.add_argument("--image-id", default="1003", help="Image id / graph store id.")
    parser.add_argument("--graph-root", default=None, help="Optional root directory of prebuilt graph stores, e.g. outputs/graph_test.")
    parser.add_argument("--backend", choices=("gemini", "openai"), default="openai", help="Model backend.")
    parser.add_argument("--model", default=None, help="Model name. If omitted, use backend-specific default/env.")
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
    parser.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="Maximum node crop images to attach. Use 0 to attach all selected node crops.",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_repo_relative_path(repo_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def safe_text(text: str, max_len: int = 180) -> str:
    cleaned = fix_mojibake(" ".join(str(text or "").split()))
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        return mime_type
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def build_image_part(path: Path) -> Any:
    from google.genai import types

    return types.Part.from_bytes(
        data=path.read_bytes(),
        mime_type=guess_mime_type(path),
    )


def path_to_data_url(path: Path) -> str:
    mime_type = guess_mime_type(path)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_pil_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                text_attr = getattr(item, "text", None)
                if text_attr:
                    parts.append(str(text_attr))
        return "\n".join(part.strip() for part in parts if str(part).strip())

    text_attr = getattr(content, "text", None)
    if text_attr:
        return str(text_attr).strip()
    return str(content).strip()


def load_cse_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_cse_payload(config: GraphConfig, query: str, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    query_for_embedding = preprocess_query_text(config, query)
    graph_enriched = json.loads(config.graph_enriched_path.read_text(encoding="utf-8"))
    text_embeddings = np.load(config.text_embeddings_path)
    crop_embeddings = np.load(config.crop_embeddings_path)
    payload = run_text_first_cse(
        graph_enriched=graph_enriched,
        text_embeddings=text_embeddings,
        crop_embeddings=crop_embeddings,
        query=query,
        query_for_embedding=query_for_embedding,
        text_model_name=config.text_embedding_model,
        image_model_name=config.image_embedding_model,
        device=resolve_device(config.preferred_device),
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
    )
    output_path = config.cse_dir / f"text_first_cse_{config.image_id}.json"
    save_cse_subgraph(payload, output_path)
    return payload, output_path


def collect_context_nodes(
    cse_payload: dict[str, Any],
    max_subgraphs: int,
    max_nodes_per_subgraph: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()

    for subgraph in cse_payload.get("subgraphs", [])[:max_subgraphs]:
        subgraph_rank = int(subgraph.get("rank", 0) or 0)
        subgraph_score = float(subgraph.get("subgraph_score", 0.0) or 0.0)
        for node in subgraph.get("nodes", [])[:max_nodes_per_subgraph]:
            node_id = str(node.get("node_id", "")).strip()
            if not node_id or node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            selected.append(
                {
                    "subgraph_rank": subgraph_rank,
                    "subgraph_score": subgraph_score,
                    "node_id": node_id,
                    "text": safe_text(node.get("text", "")),
                    "bbox": node.get("bbox"),
                    "crop_bbox": node.get("crop_bbox"),
                    "final_score": float(node.get("final_score", 0.0) or 0.0),
                    "rel": float(node.get("rel", 0.0) or 0.0),
                    "crop_path": node.get("crop_path"),
                }
            )
    return selected


def clamp_bbox_to_image(bbox: list[int] | tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def collect_crop_paths(
    context_nodes: list[dict[str, Any]],
    image_path: Path,
    runtime_crops_dir: Path,
    max_crops: int,
) -> list[tuple[str, Path]]:
    crop_paths: list[tuple[str, Path]] = []
    seen_keys: set[str] = set()
    max_items = max_crops if max_crops and max_crops > 0 else None
    runtime_crops_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    try:
        for node in context_nodes:
            node_id = str(node.get("node_id", "")).strip()
            if not node_id:
                continue

            crop_path = str(node.get("crop_path", "") or "").strip()
            if crop_path:
                candidate = Path(crop_path)
                if candidate.exists():
                    key = f"path::{candidate.resolve()}"
                    if key not in seen_keys:
                        crop_paths.append((node_id, candidate))
                        seen_keys.add(key)
                    if max_items is not None and len(crop_paths) >= max_items:
                        break
                    continue

            raw_bbox = node.get("crop_bbox") or node.get("bbox")
            clamped_bbox = clamp_bbox_to_image(raw_bbox, width=image_width, height=image_height)
            if clamped_bbox is None:
                continue
            key = f"bbox::{node_id}::{clamped_bbox}"
            if key in seen_keys:
                continue
            x1, y1, x2, y2 = clamped_bbox
            crop_output_path = runtime_crops_dir / f"{node_id}.png"
            image.crop((x1, y1, x2, y2)).save(crop_output_path)
            crop_paths.append((node_id, crop_output_path))
            seen_keys.add(key)
            if max_items is not None and len(crop_paths) >= max_items:
                break
    finally:
        image.close()

    return crop_paths


def sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")
    return sanitized or "gemini_model"


def default_output_path(config: GraphConfig, model_name: str, query: str) -> Path:
    digest = hashlib.md5(query.encode("utf-8")).hexdigest()[:10]
    return config.output_dir / "vlm" / f"{sanitize_model_name(model_name)}_{digest}.json"


def resolve_model_name(backend: str, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    if backend == "gemini":
        return os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL_NAME
    model_name = os.getenv("VLM_MODEL") or os.getenv("OPENAI_VLM_MODEL")
    if not model_name:
        raise EnvironmentError("Missing VLM_MODEL/OPENAI_VLM_MODEL for openai backend, or pass --model.")
    return model_name


def init_model_client(backend: str) -> tuple[str, Any]:
    if backend == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
        try:
            import google.genai as modern_genai

            return "google.genai", modern_genai.Client(api_key=api_key)
        except Exception:
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=api_key)
            return "google.generativeai", legacy_genai

    api_key = os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    base_url = os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise EnvironmentError("Missing VLM_BASE_URL/OPENAI_BASE_URL for openai backend.")
    from openai import OpenAI

    return "openai", OpenAI(api_key=api_key, base_url=base_url)


def generate_with_model(
    backend: str,
    client_or_module: Any,
    model_name: str,
    prompt: str,
    image_path: Path,
    crop_paths: list[tuple[str, Path]],
    temperature: float,
) -> Any:
    if backend == "google.genai":
        from google.genai import types

        contents: list[Any] = [types.Part.from_text(text=prompt), build_image_part(image_path)]
        contents.extend(build_image_part(path) for _, path in crop_paths)
        return client_or_module.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )

    if backend == "google.generativeai":
        model = client_or_module.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature},
        )
        contents: list[Any] = [prompt, build_pil_image(image_path)]
        contents.extend(build_pil_image(path) for _, path in crop_paths)
        return model.generate_content(contents)

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(image_path)}})
    for _, path in crop_paths:
        content.append({"type": "image_url", "image_url": {"url": path_to_data_url(path)}})
    return client_or_module.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )


def extract_response_text(response: Any) -> str:
    if hasattr(response, "choices") and response.choices:
        return normalize_message_content(response.choices[0].message.content)
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    return ""


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
    model_name = resolve_model_name(args.backend, args.model)

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
        runtime_crops_dir=config.output_dir / "vlm_runtime_crops",
        max_crops=args.max_crops,
    )
    prompt = build_vlm_prompt(args.query, context_nodes=context_nodes, crop_paths=crop_paths)

    backend, client_or_module = init_model_client(args.backend)
    response = generate_with_model(
        backend=backend,
        client_or_module=client_or_module,
        model_name=model_name,
        prompt=prompt,
        image_path=config.image_path,
        crop_paths=crop_paths,
        temperature=args.temperature,
    )

    answer = extract_response_text(response)
    output_path = Path(args.output) if args.output else default_output_path(config, model_name, args.query)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": args.query,
        "requested_backend": args.backend,
        "model": model_name,
        "backend": backend,
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
        "used_crop_paths": [str(path) for _, path in crop_paths],
        "used_node_ids": [node["node_id"] for node in context_nodes],
        "answer": answer,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"query={args.query}")
    print(f"backend={backend}")
    print(f"model={model_name}")
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
