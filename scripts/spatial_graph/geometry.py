from __future__ import annotations


def box_metrics(box: list[int]) -> dict[str, float]:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": width,
        "height": height,
        "cx": x1 + width / 2.0,
        "cy": y1 + height / 2.0,
    }


def overlap_1d(a1: int, a2: int, b1: int, b2: int) -> int:
    return max(0, min(a2, b2) - max(a1, b1))


def y_overlap_ratio(box_a: list[int], box_b: list[int]) -> float:
    overlap = overlap_1d(box_a[1], box_a[3], box_b[1], box_b[3])
    min_height = max(1, min(box_a[3] - box_a[1], box_b[3] - box_b[1]))
    return overlap / min_height


def x_overlap_ratio(box_a: list[int], box_b: list[int]) -> float:
    overlap = overlap_1d(box_a[0], box_a[2], box_b[0], box_b[2])
    min_width = max(1, min(box_a[2] - box_a[0], box_b[2] - box_b[0]))
    return overlap / min_width


def x_overlap_ratio_on_max(box_a: list[int], box_b: list[int]) -> float:
    overlap = overlap_1d(box_a[0], box_a[2], box_b[0], box_b[2])
    max_width = max(1, max(box_a[2] - box_a[0], box_b[2] - box_b[0]))
    return overlap / max_width


def horizontal_gap(box_a: list[int], box_b: list[int]) -> int:
    return max(0, box_b[0] - box_a[2])


def vertical_gap(box_a: list[int], box_b: list[int]) -> int:
    return max(0, box_b[1] - box_a[3])


def clamp_bbox(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(1, min(int(x2), width))
    y2 = max(1, min(int(y2), height))
    return [x1, y1, x2, y2]


def pad_bbox(box: list[int], width: int, height: int, padding: int) -> list[int]:
    x1, y1, x2, y2 = box
    return clamp_bbox([x1 - padding, y1 - padding, x2 + padding, y2 + padding], width, height)
