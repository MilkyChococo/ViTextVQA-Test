from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_arrow(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], thickness: int = 2) -> None:
    cv2.arrowedLine(image, start, end, color, thickness, tipLength=0.18)


def draw_overlay(image_path: Path, raw_rows: list[dict], nodes: list[dict], edges: list[dict], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for row in raw_rows:
        x1, y1, x2, y2 = row["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (90, 180, 90), 1)

    node_lookup = {node["node_id"]: node for node in nodes}

    for idx, node in enumerate(nodes, start=1):
        x1, y1, x2, y2 = node["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"N{idx}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        text_y = max(y1 - 4, th + 6)
        cv2.rectangle(image, (x1, text_y - th - 6), (x1 + tw + 6, text_y + 2), (0, 200, 255), -1)
        cv2.putText(image, label, (x1 + 3, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    for edge in edges:
        source = node_lookup[edge["source"]]
        target = node_lookup[edge["target"]]
        start = (int(source["cx"]), int(source["cy"]))
        end = (int(target["cx"]), int(target["cy"]))
        if edge["type"] == "RIGHT_NEIGHBOR":
            draw_arrow(image, start, end, (255, 80, 80), 2)
        else:
            draw_arrow(image, start, end, (80, 80, 255), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
