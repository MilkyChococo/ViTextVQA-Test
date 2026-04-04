from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from encode.embeddings import encode_clip_texts, init_clip_model, init_text_model

from .cse_indexing import normalized_cosine_similarity


@dataclass(slots=True)
class SeedCandidate:
    node_id: str
    node_type: str
    rel: float
    rank: int

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExpansionCandidate:
    source_id: str
    target_id: str
    relation: str
    conf_off: float
    rel: float
    hub: float
    score: float
    hop: int

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def load_graph(graph_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(graph_path).read_text(encoding="utf-8"))


def build_query_relevance_scores(
    query_for_embedding: str,
    text_embeddings: np.ndarray,
    crop_embeddings: np.ndarray,
    graph: dict[str, Any],
    text_model_name: str,
    image_model_name: str,
    device: str = "cpu",
    batch_size: int = 8,
    query_instruction: str = "Represent this Vietnamese question for retrieving relevant OCR graph nodes.",
    rel_text_weight: float = 0.4,
    rel_image_weight: float = 0.6,
    preloaded_text_embedder: Any | None = None,
    preloaded_clip: tuple[Any, Any] | None = None,
) -> dict[str, float]:
    if (
        (text_embeddings.ndim != 2 or text_embeddings.shape[0] == 0 or text_embeddings.shape[1] == 0)
        and (crop_embeddings.ndim != 2 or crop_embeddings.shape[0] == 0 or crop_embeddings.shape[1] == 0)
    ):
        raise ValueError("Both text and crop embeddings are empty. Build embeddings before running multimodal CSE.")

    text_embedder = preloaded_text_embedder or init_text_model(text_model_name, device)
    query_text_embedding = text_embedder.encode(
        [query_for_embedding],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        prompt_name=None,
    )
    query_text_vector = query_text_embedding[0]

    if preloaded_clip is None:
        clip_processor, clip_model = init_clip_model(image_model_name, device)
    else:
        clip_processor, clip_model = preloaded_clip
    query_image_embedding = np.asarray(
        encode_clip_texts(
            processor=clip_processor,
            model=clip_model,
            texts=[query_for_embedding],
            device=device,
            batch_size=1,
        ),
        dtype=np.float32,
    )
    query_image_vector = query_image_embedding[0]

    scores: dict[str, float] = {}
    for node in graph.get("nodes", []):
        node_id = str(node.get("node_id", "")).strip()
        if not node_id:
            continue

        weighted_parts: list[float] = []
        total_weight = 0.0

        text_row = int(node.get("text_embedding_index", -1))
        if (
            rel_text_weight > 0.0
            and text_embeddings.ndim == 2
            and text_embeddings.shape[1] > 0
            and 0 <= text_row < text_embeddings.shape[0]
        ):
            text_rel = normalized_cosine_similarity(query_text_vector, text_embeddings[text_row])
            weighted_parts.append(rel_text_weight * text_rel)
            total_weight += rel_text_weight

        crop_row = int(node.get("crop_embedding_index", -1))
        if (
            rel_image_weight > 0.0
            and crop_embeddings.ndim == 2
            and crop_embeddings.shape[1] > 0
            and 0 <= crop_row < crop_embeddings.shape[0]
        ):
            image_rel = normalized_cosine_similarity(query_image_vector, crop_embeddings[crop_row])
            weighted_parts.append(rel_image_weight * image_rel)
            total_weight += rel_image_weight

        if total_weight <= 0.0:
            continue

        scores[node_id] = round(sum(weighted_parts) / total_weight, 6)
    return scores


def select_top_k_seed_nodes(
    graph: dict[str, Any],
    rel_scores: dict[str, float],
    top_k: int = 5,
    allowed_node_types: tuple[str, ...] | None = None,
) -> list[SeedCandidate]:
    allowed = None
    if allowed_node_types:
        allowed = {item.strip().lower() for item in allowed_node_types}

    ranked_nodes: list[tuple[str, str, float]] = []
    for node in graph.get("nodes", []):
        node_id = str(node.get("node_id", ""))
        node_type = str(node.get("node_type", "")).strip().lower()
        if allowed is not None and node_type not in allowed:
            continue
        if node_id not in rel_scores:
            continue
        ranked_nodes.append((node_id, node_type, rel_scores[node_id]))

    ranked_nodes.sort(key=lambda item: item[2], reverse=True)
    return [
        SeedCandidate(node_id=node_id, node_type=node_type, rel=score, rank=index + 1)
        for index, (node_id, node_type, score) in enumerate(ranked_nodes[:top_k])
    ]


def _build_node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node.get("node_id", "")): node for node in graph.get("nodes", [])}


def _build_outgoing_edge_index(
    graph: dict[str, Any],
    allowed_relations: tuple[str, ...] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    allowed = None
    if allowed_relations:
        allowed = {item.strip() for item in allowed_relations}

    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get("edges", []):
        relation = str(edge.get("type", ""))
        if allowed is not None and relation not in allowed:
            continue
        source_id = str(edge.get("source", ""))
        outgoing.setdefault(source_id, []).append(edge)
    return outgoing


def compute_cse_edge_score(
    edge: dict[str, Any],
    target_node: dict[str, Any],
    rel_scores: dict[str, float],
    alpha: float = 0.5,
    lambda_hub: float = 0.1,
) -> tuple[float, float, float, float]:
    target_id = str(target_node.get("node_id", ""))
    conf_off = float(edge.get("conf_off", 0.0) or 0.0)
    rel = float(rel_scores.get(target_id, 0.0))
    hub = float(target_node.get("hub", 0.0) or 0.0)
    score = alpha * conf_off + rel - lambda_hub * hub
    return round(score, 6), round(conf_off, 6), round(rel, 6), round(hub, 6)


def run_basic_cse(
    graph: dict[str, Any],
    rel_scores: dict[str, float],
    seed_candidates: list[SeedCandidate],
    hops: int = 2,
    top_m: int = 5,
    threshold: float = 0.35,
    alpha: float = 0.5,
    lambda_hub: float = 0.1,
    max_nodes: int = 100,
    max_edges: int = 200,
    allowed_relations: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    node_index = _build_node_index(graph)
    outgoing_edge_index = _build_outgoing_edge_index(graph, allowed_relations=allowed_relations)

    seed_ids = [item.node_id for item in seed_candidates]
    selected_node_ids: set[str] = set(seed_ids)
    selected_edges: list[dict[str, Any]] = []
    next_frontier: list[str] = list(seed_ids)
    visited_edge_keys: set[tuple[str, str, str]] = set()
    expansion_trace: list[dict[str, Any]] = []

    for hop in range(1, hops + 1):
        frontier = list(next_frontier)
        next_frontier = []
        if not frontier:
            break

        for source_id in frontier:
            candidates: list[ExpansionCandidate] = []
            for edge in outgoing_edge_index.get(source_id, []):
                target_id = str(edge.get("target", ""))
                target_node = node_index.get(target_id)
                if target_node is None:
                    continue
                score, conf_off, rel, hub = compute_cse_edge_score(
                    edge=edge,
                    target_node=target_node,
                    rel_scores=rel_scores,
                    alpha=alpha,
                    lambda_hub=lambda_hub,
                )
                if score < threshold:
                    continue
                candidates.append(
                    ExpansionCandidate(
                        source_id=source_id,
                        target_id=target_id,
                        relation=str(edge.get("type", "")),
                        conf_off=conf_off,
                        rel=rel,
                        hub=hub,
                        score=score,
                        hop=hop,
                    )
                )

            candidates.sort(key=lambda item: item.score, reverse=True)
            selected = candidates[:top_m]
            expansion_trace.extend(item.to_payload() for item in selected)

            for item in selected:
                edge_key = (item.source_id, item.target_id, item.relation)
                if edge_key in visited_edge_keys:
                    continue
                visited_edge_keys.add(edge_key)
                selected_edges.append(
                    {
                        "source": item.source_id,
                        "target": item.target_id,
                        "type": item.relation,
                        "score": item.score,
                        "conf_off": item.conf_off,
                        "rel": item.rel,
                        "hub": item.hub,
                        "hop": item.hop,
                    }
                )
                if item.target_id not in selected_node_ids:
                    selected_node_ids.add(item.target_id)
                    next_frontier.append(item.target_id)

                if len(selected_node_ids) >= max_nodes or len(selected_edges) >= max_edges:
                    return {
                        "seed_nodes": [item.to_payload() for item in seed_candidates],
                        "selected_node_ids": sorted(selected_node_ids),
                        "selected_edges": selected_edges,
                        "expansion_trace": expansion_trace,
                        "stopped_early": True,
                    }

    return {
        "seed_nodes": [item.to_payload() for item in seed_candidates],
        "selected_node_ids": sorted(selected_node_ids),
        "selected_edges": selected_edges,
        "expansion_trace": expansion_trace,
        "stopped_early": False,
    }


def build_cse_subgraph_payload_for_seed(
    enriched_graph: dict[str, Any],
    seed_candidate: SeedCandidate,
    cse_result: dict[str, Any],
    rel_scores: dict[str, float],
) -> dict[str, Any]:
    node_index = _build_node_index(enriched_graph)
    selected_node_ids = set(cse_result.get("selected_node_ids", []))
    final_node_scores: dict[str, float] = {seed_candidate.node_id: float(seed_candidate.rel)}

    for edge in cse_result.get("selected_edges", []):
        target_id = str(edge.get("target", ""))
        edge_score = float(edge.get("score", 0.0) or 0.0)
        if not target_id:
            continue
        final_node_scores[target_id] = max(
            final_node_scores.get(target_id, float("-inf")),
            edge_score,
        )

    selected_nodes: list[dict[str, Any]] = []
    for node_id in selected_node_ids:
        node = node_index.get(node_id)
        if node is None:
            continue
        node_payload = dict(node)
        node_payload["rel"] = float(rel_scores.get(node_id, 0.0))
        node_payload["final_score"] = round(
            float(final_node_scores.get(node_id, node_payload["rel"])),
            6,
        )
        selected_nodes.append(node_payload)

    selected_nodes.sort(
        key=lambda item: float(item.get("final_score", item.get("rel", 0.0))),
        reverse=True,
    )
    top_scores = [float(item.get("final_score", item.get("rel", 0.0))) for item in selected_nodes[:3]]
    subgraph_score = sum(top_scores) / len(top_scores) if top_scores else float(seed_candidate.rel)

    return {
        "subgraph_id": f"subgraph_{seed_candidate.rank:03d}_{seed_candidate.node_id}",
        "rank": seed_candidate.rank,
        "seed_node": seed_candidate.to_payload(),
        "subgraph_score": round(float(subgraph_score), 6),
        "stats": {
            "num_selected_nodes": len(selected_nodes),
            "num_selected_edges": len(cse_result.get("selected_edges", [])),
            "stopped_early": bool(cse_result.get("stopped_early", False)),
        },
        "nodes": selected_nodes,
        "edges": cse_result.get("selected_edges", []),
        "expansion_trace": cse_result.get("expansion_trace", []),
    }


def run_text_first_cse(
    graph_enriched: dict[str, Any],
    text_embeddings: np.ndarray,
    crop_embeddings: np.ndarray,
    query: str,
    query_for_embedding: str,
    text_model_name: str,
    image_model_name: str,
    device: str = "cpu",
    top_k: int = 5,
    hops: int = 2,
    top_m: int = 5,
    threshold: float = 0.35,
    alpha: float = 0.5,
    lambda_hub: float = 0.1,
    max_nodes: int = 100,
    max_edges: int = 200,
    allowed_seed_node_types: tuple[str, ...] | None = ("text",),
    allowed_relations: tuple[str, ...] | None = None,
    rel_text_weight: float = 0.4,
    rel_image_weight: float = 0.6,
    preloaded_text_embedder: Any | None = None,
    preloaded_clip: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    rel_scores = build_query_relevance_scores(
        query_for_embedding=query_for_embedding,
        text_embeddings=text_embeddings,
        crop_embeddings=crop_embeddings,
        graph=graph_enriched,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        device=device,
        rel_text_weight=rel_text_weight,
        rel_image_weight=rel_image_weight,
        preloaded_text_embedder=preloaded_text_embedder,
        preloaded_clip=preloaded_clip,
    )
    seeds = select_top_k_seed_nodes(
        graph=graph_enriched,
        rel_scores=rel_scores,
        top_k=top_k,
        allowed_node_types=allowed_seed_node_types,
    )
    expanded_subgraphs: list[dict[str, Any]] = []
    for seed in seeds:
        cse_result = run_basic_cse(
            graph=graph_enriched,
            rel_scores=rel_scores,
            seed_candidates=[seed],
            hops=hops,
            top_m=top_m,
            threshold=threshold,
            alpha=alpha,
            lambda_hub=lambda_hub,
            max_nodes=max_nodes,
            max_edges=max_edges,
            allowed_relations=allowed_relations,
        )
        expanded_subgraphs.append(
            build_cse_subgraph_payload_for_seed(
                enriched_graph=graph_enriched,
                seed_candidate=seed,
                cse_result=cse_result,
                rel_scores=rel_scores,
            )
        )

    ordered_subgraphs = sorted(
        expanded_subgraphs,
        key=lambda item: (
            -float(item.get("subgraph_score", 0.0)),
            int(item.get("rank", 10**9)),
        ),
    )
    return {
        "query": query,
        "params": {
            "top_k": top_k,
            "hops": hops,
            "top_m": top_m,
            "threshold": threshold,
            "alpha": alpha,
            "lambda_hub": lambda_hub,
            "rel_text_weight": rel_text_weight,
            "rel_image_weight": rel_image_weight,
        },
        "seed_nodes": [item.to_payload() for item in seeds],
        "stats": {
            "num_subgraphs": len(ordered_subgraphs),
            "num_seed_nodes": len(seeds),
            "num_total_nodes": sum(len(item.get("nodes", [])) for item in ordered_subgraphs),
            "num_total_edges": sum(len(item.get("edges", [])) for item in ordered_subgraphs),
        },
        "subgraphs": ordered_subgraphs,
    }


def save_cse_subgraph(payload: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
