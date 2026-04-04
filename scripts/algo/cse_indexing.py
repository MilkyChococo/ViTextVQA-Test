from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def normalized_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return (1.0 + cosine_similarity(vector_a, vector_b)) / 2.0


def _empty_neighbor_summary() -> dict[str, Any]:
    return {
        "incoming": [],
        "outgoing": [],
        "neighbor_ids": [],
    }


def build_text_node_to_row(graph: dict[str, Any]) -> dict[str, int]:
    node_to_row: dict[str, int] = {}
    for node in graph.get("nodes", []):
        node_id = str(node.get("node_id", "")).strip()
        row = node.get("text_embedding_index")
        if node_id and isinstance(row, int) and row >= 0:
            node_to_row[node_id] = row
    return node_to_row


def build_crop_node_to_row(graph: dict[str, Any]) -> dict[str, int]:
    node_to_row: dict[str, int] = {}
    for node in graph.get("nodes", []):
        node_id = str(node.get("node_id", "")).strip()
        row = node.get("crop_embedding_index")
        if node_id and isinstance(row, int) and row >= 0:
            node_to_row[node_id] = row
    return node_to_row


def enrich_graph_for_cse(
    graph: dict[str, Any],
    text_embeddings: np.ndarray,
    crop_embeddings: np.ndarray,
    lambda_hub: float = 0.1,
    conf_text_weight: float = 0.4,
    conf_image_weight: float = 0.6,
) -> dict[str, Any]:
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    text_node_to_row = build_text_node_to_row(graph)
    crop_node_to_row = build_crop_node_to_row(graph)

    node_index: dict[str, dict[str, Any]] = {}
    neighbor_summary: dict[str, dict[str, Any]] = {}
    deg_in: dict[str, int] = {}
    deg_out: dict[str, int] = {}

    for node in nodes:
        node_id = str(node.get("node_id", ""))
        node_index[node_id] = node
        neighbor_summary[node_id] = _empty_neighbor_summary()
        deg_in[node_id] = 0
        deg_out[node_id] = 0

    enriched_edges: list[dict[str, Any]] = []
    for edge in edges:
        source_id = str(edge.get("source", ""))
        target_id = str(edge.get("target", ""))
        relation = str(edge.get("type", ""))
        text_source_row = text_node_to_row.get(source_id)
        text_target_row = text_node_to_row.get(target_id)
        crop_source_row = crop_node_to_row.get(source_id)
        crop_target_row = crop_node_to_row.get(target_id)

        text_conf = None
        if (
            text_source_row is not None
            and text_target_row is not None
            and text_embeddings.ndim == 2
            and text_embeddings.shape[1] > 0
            and 0 <= text_source_row < text_embeddings.shape[0]
            and 0 <= text_target_row < text_embeddings.shape[0]
        ):
            text_conf = normalized_cosine_similarity(
                text_embeddings[text_source_row],
                text_embeddings[text_target_row],
            )

        crop_conf = None
        if (
            crop_source_row is not None
            and crop_target_row is not None
            and crop_embeddings.ndim == 2
            and crop_embeddings.shape[1] > 0
            and 0 <= crop_source_row < crop_embeddings.shape[0]
            and 0 <= crop_target_row < crop_embeddings.shape[0]
        ):
            crop_conf = normalized_cosine_similarity(
                crop_embeddings[crop_source_row],
                crop_embeddings[crop_target_row],
            )

        conf_parts: list[float] = []
        total_weight = 0.0
        if text_conf is not None:
            conf_parts.append(conf_text_weight * text_conf)
            total_weight += conf_text_weight
        if crop_conf is not None:
            conf_parts.append(conf_image_weight * crop_conf)
            total_weight += conf_image_weight
        conf_off = (sum(conf_parts) / total_weight) if total_weight > 0.0 else None

        enriched_edge = dict(edge)
        if conf_off is not None:
            enriched_edge["conf_off"] = round(conf_off, 6)
        if text_conf is not None:
            enriched_edge["conf_off_text"] = round(text_conf, 6)
        if crop_conf is not None:
            enriched_edge["conf_off_image"] = round(crop_conf, 6)
        enriched_edges.append(enriched_edge)

        if source_id in deg_out:
            deg_out[source_id] += 1
        if target_id in deg_in:
            deg_in[target_id] += 1

        if source_id in neighbor_summary:
            neighbor_summary[source_id]["outgoing"].append(
                {
                    "target_id": target_id,
                    "relation": relation,
                    "conf_off": round(conf_off, 6) if conf_off is not None else None,
                    "conf_off_text": round(text_conf, 6) if text_conf is not None else None,
                    "conf_off_image": round(crop_conf, 6) if crop_conf is not None else None,
                }
            )
        if target_id in neighbor_summary:
            neighbor_summary[target_id]["incoming"].append(
                {
                    "source_id": source_id,
                    "relation": relation,
                    "conf_off": round(conf_off, 6) if conf_off is not None else None,
                    "conf_off_text": round(text_conf, 6) if text_conf is not None else None,
                    "conf_off_image": round(crop_conf, 6) if crop_conf is not None else None,
                }
            )

    enriched_nodes: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        node_copy = dict(node)

        embedding_row = text_node_to_row.get(node_id)
        total_deg = deg_in.get(node_id, 0) + deg_out.get(node_id, 0)
        hub = math.log(1.0 + total_deg)

        summary = neighbor_summary.get(node_id, _empty_neighbor_summary())
        summary["neighbor_ids"] = sorted(
            {
                *(item["target_id"] for item in summary["outgoing"]),
                *(item["source_id"] for item in summary["incoming"]),
            }
        )

        node_copy["embedding_row"] = embedding_row
        node_copy["deg_in"] = deg_in.get(node_id, 0)
        node_copy["deg_out"] = deg_out.get(node_id, 0)
        node_copy["deg"] = total_deg
        node_copy["hub"] = round(hub, 6)
        node_copy["lambda_hub"] = lambda_hub
        node_copy["neighbors"] = summary
        enriched_nodes.append(node_copy)

    enriched_graph = dict(graph)
    enriched_meta = dict(graph.get("meta", {}))
    enriched_meta["graph_type"] = "spatial_text_graph_enriched"
    enriched_graph["meta"] = enriched_meta
    enriched_graph["nodes"] = enriched_nodes
    enriched_graph["edges"] = enriched_edges
    enriched_graph["cse_offline"] = {
        "lambda_hub": lambda_hub,
        "embedding_source": "text_embeddings+crop_embeddings",
        "conf_text_weight": conf_text_weight,
        "conf_image_weight": conf_image_weight,
        "num_text_embedded_nodes": len(text_node_to_row),
        "num_crop_embedded_nodes": len(crop_node_to_row),
        "num_edges_with_conf_off": sum(1 for edge in enriched_edges if "conf_off" in edge),
    }
    return enriched_graph


def save_enriched_graph(enriched_graph: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(enriched_graph, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
