#!/usr/bin/env python
"""Run text-first top-k seed retrieval and CSE expansion on one enriched graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from algo.cse_query import run_text_first_cse, save_cse_subgraph
from process.text_preprocess import preprocess_query_text
from utils.config import GraphConfig
from encode.embeddings import resolve_device
from model import resolve_repo_relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-first CSE on one graph store.")
    parser.add_argument("query", help="Vietnamese user query used to retrieve top-k text seeds.")
    parser.add_argument("--image-id", default="1003", help="Image id / graph store id.")
    parser.add_argument("--graph-root", default=None, help="Optional root directory of prebuilt graph stores, e.g. outputs/graph_test.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of text seed nodes.")
    parser.add_argument("--hops", type=int, default=3, help="Expansion hops.")
    parser.add_argument("--top-m", type=int, default=5, help="Top outgoing edges kept per frontier node.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Minimum CSE edge score.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for offline edge confidence.")
    parser.add_argument("--lambda-hub", type=float, default=0.05, help="Hub penalty.")
    parser.add_argument("--max-nodes", type=int, default=100, help="Maximum nodes in each subgraph.")
    parser.add_argument("--max-edges", type=int, default=200, help="Maximum edges in each subgraph.")
    parser.add_argument("--print-top-nodes", type=int, default=5, help="How many top nodes to print per subgraph.")
    return parser.parse_args()


def safe_text(value: str, max_len: int = 120) -> str:
    text = " ".join(str(value).split())
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def build_node_lookup(graph_enriched: dict) -> dict[str, dict]:
    return {str(node.get("node_id", "")): node for node in graph_enriched.get("nodes", [])}


def print_payload_summary(payload: dict, node_lookup: dict[str, dict], print_top_nodes: int) -> None:
    print("\n=== Seeds ===")
    for seed in payload.get("seed_nodes", []):
        node_id = str(seed.get("node_id", ""))
        rel = float(seed.get("rel", 0.0) or 0.0)
        rank = int(seed.get("rank", 0) or 0)
        node_text = safe_text(node_lookup.get(node_id, {}).get("text", ""))
        print(f"[Seed {rank}] {node_id} | rel={rel:.4f} | text={node_text}")

    print("\n=== Subgraphs ===")
    for subgraph in payload.get("subgraphs", []):
        subgraph_id = str(subgraph.get("subgraph_id", ""))
        rank = int(subgraph.get("rank", 0) or 0)
        subgraph_score = float(subgraph.get("subgraph_score", 0.0) or 0.0)
        seed_node = subgraph.get("seed_node", {})
        seed_id = str(seed_node.get("node_id", ""))
        seed_text = safe_text(node_lookup.get(seed_id, {}).get("text", ""))
        print(f"\n[Subgraph {rank}] {subgraph_id} | score={subgraph_score:.4f}")
        print(f"seed={seed_id} | text={seed_text}")
        for index, node in enumerate(subgraph.get("nodes", [])[:print_top_nodes], start=1):
            node_id = str(node.get("node_id", ""))
            final_score = float(node.get("final_score", node.get("rel", 0.0)) or 0.0)
            rel = float(node.get("rel", 0.0) or 0.0)
            text = safe_text(node.get("text", ""))
            print(f"  {index}. {node_id} | final={final_score:.4f} | rel={rel:.4f} | text={text}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    graph_root = resolve_repo_relative_path(repo_root, args.graph_root)
    config = GraphConfig(repo_root=repo_root, image_id=args.image_id, output_root=graph_root)
    query_for_embedding = preprocess_query_text(config, args.query)

    graph_enriched = json.loads(config.graph_enriched_path.read_text(encoding="utf-8"))
    text_embeddings = np.load(config.text_embeddings_path)
    crop_embeddings = np.load(config.crop_embeddings_path)
    payload = run_text_first_cse(
        graph_enriched=graph_enriched,
        text_embeddings=text_embeddings,
        crop_embeddings=crop_embeddings,
        query=args.query,
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
    node_lookup = build_node_lookup(graph_enriched)
    print_payload_summary(payload, node_lookup=node_lookup, print_top_nodes=args.print_top_nodes)
    print()
    print(f"query={args.query}")
    print(f"graph_enriched={config.graph_enriched_path}")
    print(f"text_embeddings={config.text_embeddings_path}")
    print(f"crop_embeddings={config.crop_embeddings_path}")
    print(f"cse_output={output_path}")


if __name__ == "__main__":
    main()
