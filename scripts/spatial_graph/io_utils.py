from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import pad_bbox


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda row: row.get("order", 10**9))
    return rows


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def fix_mojibake(text: str) -> str:
    if not text:
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except Exception:
        return text
    weird_score_before = text.count("Ãƒ") + text.count("Ã„") + text.count("Ã¡Âº")
    weird_score_after = repaired.count("Ãƒ") + repaired.count("Ã„") + repaired.count("Ã¡Âº")
    return repaired if weird_score_after < weird_score_before else text


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def crop_node_images(
    nodes: list[dict],
    image_path: Path,
    crops_dir: Path,
    save_crops: bool,
    padding: int,
) -> list[Image.Image]:
    rgb_image = load_rgb_image(image_path)
    width, height = rgb_image.size
    crops: list[Image.Image] = []
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for node in nodes:
        crop_bbox = pad_bbox(node["bbox"], width=width, height=height, padding=padding)
        crop = rgb_image.crop(tuple(crop_bbox))
        crops.append(crop)
        node["crop_bbox"] = crop_bbox
        if save_crops:
            crop_path = crops_dir / f"{node['node_id']}.png"
            crop.save(crop_path)
            node["crop_path"] = str(crop_path)
        else:
            node["crop_path"] = None
    return crops
