#!/usr/bin/env python
"""Recognize text from per-image bbox folders using a Vintern API endpoint.

Expected input layout:
- bbox-root/
  - 2145/
    - bboxes.jsonl
  - 5566/
    - bboxes.jsonl

For each image folder, the script:
- finds the source image from image-root
- crops each bbox
- sends the crop to a Vintern OpenAI-compatible endpoint
- saves ocr_results.json and ocr_results.jsonl into the same folder
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from tqdm import tqdm


DEFAULT_PROMPT = (
    "You are an OCR-only text recognition system. Recognize the text in this crop as accurately as possible. "
    "Preserve exact spelling, punctuation, capitalization, and spacing. "
    "Do not explain. Do not infer. If there is no readable text, return <EMPTY>."
)


# =========================
# CAU HINH
# =========================
# Doi lai cac path nay theo Colab/Kaggle cua ban.
BBOX_ROOT = Path("/content/test_bboxes")
IMAGE_ROOT = Path("/content/st_images")

BASE_URL = "http://localhost:8000/v1"
API_KEY = "vintern-local"
MODEL_NAME = None  # Neu None, script se lay model dau tien tren server.

PROMPT = DEFAULT_PROMPT
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 128
MAX_SIDE = 768

SAVE_CROPS = False
RESUME = True
LIMIT = None  # Vi du 100 neu chi muon chay thu 100 folder dau.


def load_bboxes(jsonl_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    items.sort(key=lambda item: (item.get("order", 10**9), item.get("bbox_id", "")))
    return items


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return [x1, y1, x2, y2]


def crop_bbox(image: Image.Image, bbox: list[int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))


def pil_to_data_url(img: Image.Image, max_side: int) -> str:
    formatted = img.copy()
    if max_side > 0:
        formatted.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    formatted.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                text_attr = getattr(item, "text", None)
                if text_attr:
                    parts.append(str(text_attr))
        return "\n".join(part.strip() for part in parts if str(part).strip())

    text_attr = getattr(content, "text", None)
    if text_attr:
        return str(text_attr).strip()
    return str(content).strip()


def normalize_ocr_text(text: str) -> str:
    normalized = text.strip()
    normalized = re.sub(r"^```(?:text)?\s*", "", normalized)
    normalized = re.sub(r"\s*```$", "", normalized)
    normalized = normalized.strip().strip('"').strip("'").strip()
    if normalized.upper() == "<EMPTY>":
        return ""
    return normalized


def init_client(base_url: str, api_key: str, model_name: str | None) -> tuple[OpenAI, str]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    if model_name:
        return client, model_name
    resolved_model = client.models.list().data[0].id
    return client, resolved_model


def call_ocr_on_crop(
    client: OpenAI,
    model_name: str,
    crop_img: Image.Image,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_side: int,
) -> dict[str, Any]:
    data_url = pil_to_data_url(crop_img, max_side=max_side)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    raw_content = response.choices[0].message.content
    text = normalize_ocr_text(normalize_message_content(raw_content))
    return {"text": text}


def resolve_image_path(image_root: Path, image_name: str | None, image_id: str | None, folder_name: str) -> Path:
    candidates: list[Path] = []
    if image_name:
        candidates.append(image_root / image_name)
    if image_id:
        candidates.append(image_root / f"{image_id}.jpg")
        candidates.append(image_root / f"{image_id}.png")
    candidates.append(image_root / f"{folder_name}.jpg")
    candidates.append(image_root / f"{folder_name}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find source image for folder: {folder_name}")


def save_results(folder_path: Path, results: list[dict[str, Any]]) -> None:
    output_json_path = folder_path / "ocr_results.json"
    output_jsonl_path = folder_path / "ocr_results.jsonl"

    output_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_folder(
    folder_path: Path,
    image_root: Path,
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_side: int,
    save_crops: bool,
    resume: bool,
) -> int:
    bbox_path = folder_path / "bboxes.jsonl"
    if not bbox_path.exists():
        return 0

    output_jsonl_path = folder_path / "ocr_results.jsonl"
    if resume and output_jsonl_path.exists():
        return 0

    bboxes = load_bboxes(bbox_path)
    if not bboxes:
        save_results(folder_path, [])
        return 0

    image_path = resolve_image_path(
        image_root=image_root,
        image_name=bboxes[0].get("image_name"),
        image_id=bboxes[0].get("image_id"),
        folder_name=folder_path.name,
    )
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    crops_dir = folder_path / "crops"
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for idx, item in enumerate(bboxes, start=1):
        bbox_id = item.get("bbox_id", f"bbox_{idx:04d}")
        bbox = clamp_bbox(item["bbox"], width=width, height=height)
        crop_img = crop_bbox(image, bbox)

        crop_path = None
        if save_crops:
            crop_name = f"{idx:04d}_{bbox_id}.png"
            crop_path = crops_dir / crop_name
            crop_img.save(crop_path)

        ocr_result = call_ocr_on_crop(
            client=client,
            model_name=model_name,
            crop_img=crop_img,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_side=max_side,
        )

        results.append(
            {
                "index": idx,
                "bbox_id": bbox_id,
                "image_id": item.get("image_id"),
                "image_name": item.get("image_name"),
                "order": item.get("order"),
                "det_score": item.get("det_score"),
                "source": item.get("source"),
                "bbox": bbox,
                "polygon": item.get("polygon"),
                "crop_path": str(crop_path) if crop_path else None,
                "text": ocr_result["text"],
            }
        )

    save_results(folder_path, results)
    return len(results)


def main() -> None:
    if not BBOX_ROOT.exists():
        raise FileNotFoundError(f"bbox root not found: {BBOX_ROOT}")
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"image root not found: {IMAGE_ROOT}")

    client, model_name = init_client(BASE_URL, API_KEY, MODEL_NAME)
    folder_paths = sorted(path for path in BBOX_ROOT.iterdir() if path.is_dir())
    if LIMIT is not None:
        folder_paths = folder_paths[:LIMIT]

    processed_folders = 0
    processed_bboxes = 0
    for folder_path in tqdm(folder_paths, desc="Recognize"):
        bbox_count = process_folder(
            folder_path=folder_path,
            image_root=IMAGE_ROOT,
            client=client,
            model_name=model_name,
            prompt=PROMPT,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            max_side=MAX_SIDE,
            save_crops=SAVE_CROPS,
            resume=RESUME,
        )
        if bbox_count > 0:
            processed_folders += 1
            processed_bboxes += bbox_count

    print(f"Processed folders: {processed_folders}")
    print(f"Processed bboxes: {processed_bboxes}")
    print(f"Saved OCR results under: {BBOX_ROOT}")


if __name__ == "__main__":
    main()
