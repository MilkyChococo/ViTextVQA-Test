from __future__ import annotations

from utils.config import GraphConfig
from .geometry import (
    box_metrics,
    horizontal_gap,
    vertical_gap,
    x_overlap_ratio,
    x_overlap_ratio_on_max,
    y_overlap_ratio,
)
from .io_utils import fix_mojibake


def should_merge_horizontally(config: GraphConfig, current_box: list[int], next_box: list[int]) -> bool:
    if next_box[0] < current_box[0]:
        return False
    overlap = y_overlap_ratio(current_box, next_box)
    gap = horizontal_gap(current_box, next_box)
    current_h = max(1, current_box[3] - current_box[1])
    next_h = max(1, next_box[3] - next_box[1])
    height_ratio = max(current_h, next_h) / max(1, min(current_h, next_h))
    return (
        overlap >= config.y_overlap_for_horizontal_merge
        and gap <= config.max_horizontal_gap
        and height_ratio <= config.max_height_ratio
    )


def should_merge_vertically(config: GraphConfig, current_box: list[int], next_box: list[int]) -> bool:
    if next_box[1] < current_box[1]:
        return False
    overlap = x_overlap_ratio(current_box, next_box)
    overlap_on_max = x_overlap_ratio_on_max(current_box, next_box)
    gap = vertical_gap(current_box, next_box)
    current_w = max(1, current_box[2] - current_box[0])
    next_w = max(1, next_box[2] - next_box[0])
    width_ratio = max(current_w, next_w) / max(1, min(current_w, next_w))
    current_cx = (current_box[0] + current_box[2]) / 2.0
    next_cx = (next_box[0] + next_box[2]) / 2.0
    center_x_diff = abs(current_cx - next_cx)
    allowed_center_x_diff = min(current_w, next_w) * config.max_center_x_diff_ratio
    return (
        overlap >= config.x_overlap_for_vertical_merge
        and overlap_on_max >= config.x_overlap_for_vertical_merge_on_max
        and gap <= config.max_vertical_gap
        and width_ratio <= config.max_width_ratio
        and center_x_diff <= allowed_center_x_diff
    )


def flatten_member_bbox_ids(item: dict) -> list[str]:
    if "member_bbox_ids" in item:
        return list(item["member_bbox_ids"])
    return [item["bbox_id"]]


def flatten_member_orders(item: dict) -> list[int]:
    if "member_orders" in item:
        return list(item["member_orders"])
    return [item["order"]]


def merge_items(group_items: list[dict], merged_index: int, text_joiner: str) -> dict:
    boxes = [item["bbox"] for item in group_items]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    merged_box = [x1, y1, x2, y2]
    merged_text_parts = []
    member_bbox_ids: list[str] = []
    member_orders: list[int] = []
    for item in group_items:
        text = fix_mojibake(str(item.get("text", "")).strip())
        if text:
            merged_text_parts.append(text)
        member_bbox_ids.extend(flatten_member_bbox_ids(item))
        member_orders.extend(flatten_member_orders(item))
    merged_text = text_joiner.join(merged_text_parts)
    metrics = box_metrics(merged_box)
    return {
        "node_id": f"img_{group_items[0]['image_id']}_node_{merged_index:03d}",
        "node_type": "text",
        "image_id": group_items[0]["image_id"],
        "image_name": group_items[0]["image_name"],
        "member_bbox_ids": member_bbox_ids,
        "member_orders": member_orders,
        "text": merged_text,
        "bbox": merged_box,
        "x1": metrics["x1"],
        "y1": metrics["y1"],
        "x2": metrics["x2"],
        "y2": metrics["y2"],
        "cx": metrics["cx"],
        "cy": metrics["cy"],
        "width": metrics["width"],
        "height": metrics["height"],
        "member_count": len(member_bbox_ids),
        "source_count": len(group_items),
    }


def rows_to_base_nodes(rows: list[dict]) -> list[dict]:
    base_nodes = [merge_items([row], idx, text_joiner=" ") for idx, row in enumerate(rows, start=1)]
    base_nodes.sort(key=lambda node: (node["y1"], node["x1"]))
    return base_nodes


def merge_horizontally(config: GraphConfig, nodes: list[dict]) -> list[dict]:
    remaining = sorted(nodes, key=lambda node: (node["y1"], node["x1"]))
    used = [False] * len(remaining)
    merged_nodes: list[dict] = []

    for i, node in enumerate(remaining):
        if used[i]:
            continue
        group = [node]
        used[i] = True
        current_box = node["bbox"]
        current_right = current_box[2]

        while True:
            best_j = None
            best_gap = None
            best_box = None
            for j, candidate in enumerate(remaining):
                if used[j]:
                    continue
                candidate_box = candidate["bbox"]
                if candidate_box[0] < current_right:
                    continue
                if not should_merge_horizontally(config, current_box, candidate_box):
                    continue
                gap = horizontal_gap(current_box, candidate_box)
                if best_gap is None or gap < best_gap:
                    best_j = j
                    best_gap = gap
                    best_box = candidate_box
            if best_j is None or best_box is None:
                break

            group.append(remaining[best_j])
            used[best_j] = True
            current_box = [
                min(current_box[0], best_box[0]),
                min(current_box[1], best_box[1]),
                max(current_box[2], best_box[2]),
                max(current_box[3], best_box[3]),
            ]
            current_right = current_box[2]

        merged_nodes.append(merge_items(group, len(merged_nodes) + 1, text_joiner=" "))

    merged_nodes.sort(key=lambda node: (node["y1"], node["x1"]))
    return merged_nodes


def merge_vertically(config: GraphConfig, nodes: list[dict]) -> list[dict]:
    remaining = sorted(nodes, key=lambda node: (node["y1"], node["x1"]))
    used = [False] * len(remaining)
    merged_nodes: list[dict] = []

    for i, node in enumerate(remaining):
        if used[i]:
            continue
        group = [node]
        used[i] = True
        current_box = node["bbox"]
        current_bottom = current_box[3]

        while True:
            best_j = None
            best_gap = None
            best_box = None
            for j, candidate in enumerate(remaining):
                if used[j]:
                    continue
                candidate_box = candidate["bbox"]
                if candidate_box[1] < current_bottom:
                    continue
                if not should_merge_vertically(config, current_box, candidate_box):
                    continue
                gap = vertical_gap(current_box, candidate_box)
                if best_gap is None or gap < best_gap:
                    best_j = j
                    best_gap = gap
                    best_box = candidate_box
            if best_j is None or best_box is None:
                break

            group.append(remaining[best_j])
            used[best_j] = True
            current_box = [
                min(current_box[0], best_box[0]),
                min(current_box[1], best_box[1]),
                max(current_box[2], best_box[2]),
                max(current_box[3], best_box[3]),
            ]
            current_bottom = current_box[3]

        merged_nodes.append(merge_items(group, len(merged_nodes) + 1, text_joiner="\n"))

    merged_nodes.sort(key=lambda node: (node["y1"], node["x1"]))
    return merged_nodes
