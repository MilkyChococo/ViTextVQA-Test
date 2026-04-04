from __future__ import annotations

from utils.config import GraphConfig
from .geometry import horizontal_gap, vertical_gap, x_overlap_ratio, y_overlap_ratio


def find_right_neighbor(config: GraphConfig, node: dict, nodes: list[dict]) -> dict | None:
    candidates: list[tuple[int, float, int, dict]] = []
    for other in nodes:
        if other["node_id"] == node["node_id"]:
            continue
        if other["x1"] <= node["x1"]:
            continue
        overlap = y_overlap_ratio(node["bbox"], other["bbox"])
        if overlap < config.y_overlap_for_right_edge:
            continue
        gap = horizontal_gap(node["bbox"], other["bbox"])
        if gap > config.max_right_gap:
            continue
        candidates.append((gap, -overlap, other["y1"], other))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def find_down_neighbor(config: GraphConfig, node: dict, nodes: list[dict]) -> dict | None:
    candidates: list[tuple[int, float, int, dict]] = []
    for other in nodes:
        if other["node_id"] == node["node_id"]:
            continue
        if other["y1"] <= node["y1"]:
            continue
        overlap = x_overlap_ratio(node["bbox"], other["bbox"])
        if overlap < config.x_overlap_for_down_edge:
            continue
        gap = vertical_gap(node["bbox"], other["bbox"])
        if gap > config.max_down_gap:
            continue
        candidates.append((gap, -overlap, other["x1"], other))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def build_edges(config: GraphConfig, nodes: list[dict]) -> list[dict]:
    edges: list[dict] = []
    for node in nodes:
        right_neighbor = find_right_neighbor(config, node, nodes)
        if right_neighbor is not None:
            edges.append(
                {
                    "source": node["node_id"],
                    "target": right_neighbor["node_id"],
                    "type": "RIGHT_NEIGHBOR",
                }
            )
        down_neighbor = find_down_neighbor(config, node, nodes)
        if down_neighbor is not None:
            edges.append(
                {
                    "source": node["node_id"],
                    "target": down_neighbor["node_id"],
                    "type": "DOWN_NEIGHBOR",
                }
            )
    return edges
