#!/usr/bin/env python
"""Call Gemini 2.5 Flash with only the original image and the user query."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from model import (
    extract_response_text,
    generate_with_model,
    init_model_client,
    load_env_file,
    resolve_model_name,
    sanitize_model_name,
)
from utils.config import GraphConfig
from utils.prompts import build_image_only_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feed only the original image and query into a VLM.")
    parser.add_argument("query", help="Vietnamese question to answer.")
    parser.add_argument("--image-id", default="1003", help="Image id / graph store id.")
    parser.add_argument("--backend", choices=("gemini", "openai"), default="openai", help="Model backend.")
    parser.add_argument("--model", default=None, help="Model name. If omitted, use backend-specific default/env.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()

def default_output_path(config: GraphConfig, model_name: str, query: str) -> Path:
    digest = hashlib.md5(query.encode("utf-8")).hexdigest()[:10]
    return config.output_dir / "vlm_only" / f"{sanitize_model_name(model_name)}_{digest}.json"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    load_env_file(repo_root / ".env")
    config = GraphConfig(repo_root=repo_root, image_id=args.image_id)
    model_name = resolve_model_name(args.backend, args.model)

    if not config.image_path.exists():
        raise FileNotFoundError(f"Image not found: {config.image_path}")

    backend, client_or_module = init_model_client(args.backend)
    response = generate_with_model(
        backend=backend,
        client_or_module=client_or_module,
        model_name=model_name,
        prompt=build_image_only_prompt(args.query),
        image_path=config.image_path,
        crop_paths=[],
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
        "answer": answer,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"query={args.query}")
    print(f"backend={backend}")
    print(f"model={model_name}")
    print(f"image_path={config.image_path}")
    print()
    print(answer)
    print()
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
