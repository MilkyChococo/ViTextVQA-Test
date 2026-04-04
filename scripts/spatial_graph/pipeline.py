from __future__ import annotations

import numpy as np

from algo.cse_indexing import enrich_graph_for_cse, save_enriched_graph
from process.text_preprocess import preprocess_node_texts
from .edges import build_edges
from .io_utils import load_rows, save_json, save_npy
from .merge import merge_horizontally, merge_vertically, rows_to_base_nodes
from .visualize import draw_overlay
from encode.embeddings import attach_embeddings, build_nodes_for_json, resolve_device
from utils.config import GraphConfig


def build_graph_payload(config: GraphConfig, nodes: list[dict], edges: list[dict], device: str) -> dict:
    return {
        "meta": {
            "image_id": config.image_id,
            "image_path": str(config.image_path),
            "ocr_jsonl_path": str(config.ocr_jsonl_path),
            "graph_type": "spatial_text_graph",
            "build_embeddings": config.build_embeddings,
            "text_embedding_model": config.text_embedding_model,
            "image_embedding_model": config.image_embedding_model,
            "enable_text_preprocessing": config.enable_text_preprocessing,
            "text_preprocess_backend": config.text_preprocess_backend,
            "device": device,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "artifacts_dir": str(config.artifacts_dir),
            "text_embeddings_path": str(config.text_embeddings_path),
            "crop_embeddings_path": str(config.crop_embeddings_path),
            "graph_enriched_path": str(config.graph_enriched_path),
            "overlay_path": str(config.overlay_path),
        },
        "nodes": nodes,
        "edges": edges,
    }


def run_pipeline(config: GraphConfig, preloaded_models: tuple | None = None) -> dict:
    if not config.image_path.exists():
        raise FileNotFoundError(f"Image not found: {config.image_path}")
    if not config.ocr_jsonl_path.exists():
        raise FileNotFoundError(f"OCR results not found: {config.ocr_jsonl_path}")

    raw_rows = load_rows(config.ocr_jsonl_path)
    base_nodes = rows_to_base_nodes(raw_rows)
    horizontal_nodes = merge_horizontally(config, base_nodes)
    merged_nodes = merge_vertically(config, horizontal_nodes)
    edges = build_edges(config, merged_nodes)
    node_texts_for_embedding = preprocess_node_texts(config, merged_nodes)

    runtime_device = resolve_device(config.preferred_device)
    if config.build_embeddings:
        graph_nodes, text_embeddings, crop_embeddings = attach_embeddings(
            config,
            merged_nodes,
            texts_for_embedding=node_texts_for_embedding,
            image_path=config.image_path,
            device=runtime_device,
            preloaded_models=preloaded_models,
        )
    else:
        graph_nodes = merged_nodes
        text_embeddings = np.zeros((len(graph_nodes), 0), dtype=np.float32)
        crop_embeddings = np.zeros((len(graph_nodes), 0), dtype=np.float32)

    json_nodes = build_nodes_for_json(graph_nodes)
    graph_payload = build_graph_payload(config, json_nodes, edges, device=runtime_device)
    enriched_graph = enrich_graph_for_cse(
        graph=graph_payload,
        text_embeddings=text_embeddings,
        crop_embeddings=crop_embeddings,
        lambda_hub=0.1,
        conf_text_weight=config.conf_text_weight,
        conf_image_weight=config.conf_image_weight,
    )

    save_json(config.artifacts_dir / "raw_rows.json", raw_rows)
    save_json(config.artifacts_dir / "base_nodes.json", base_nodes)
    save_json(config.artifacts_dir / "horizontal_nodes.json", horizontal_nodes)
    save_json(config.artifacts_dir / "merged_nodes.json", json_nodes)
    save_json(config.artifacts_dir / "spatial_edges.json", edges)
    save_json(config.artifacts_dir / "edges.json", edges)
    save_npy(config.text_embeddings_path, text_embeddings)
    save_npy(config.crop_embeddings_path, crop_embeddings)
    save_json(config.graph_json_path, graph_payload)
    save_enriched_graph(enriched_graph, config.graph_enriched_path)
    if config.save_visuals:
        draw_overlay(config.image_path, raw_rows, graph_nodes, edges, config.overlay_path)

    return {
        "raw_rows": len(raw_rows),
        "base_nodes": len(base_nodes),
        "horizontal_nodes": len(horizontal_nodes),
        "merged_nodes": len(merged_nodes),
        "spatial_edges": len(edges),
        "graph_json": str(config.graph_json_path),
        "graph_enriched_json": str(config.graph_enriched_path),
        "text_embeddings_npy": str(config.text_embeddings_path),
        "crop_embeddings_npy": str(config.crop_embeddings_path),
        "output_dir": str(config.output_dir),
    }
