#!/usr/bin/env python
"""Run full CSE -> Vintern HF flow with a single composite image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from model import (
    build_cse_payload,
    collect_context_nodes,
    collect_crop_paths,
    load_cse_payload,
    load_env_file,
    resolve_repo_relative_path,
    sanitize_model_name,
)
from utils.config import GraphConfig
from utils.prompts import build_vlm_prompt


DEFAULT_VINTERN_MODEL_NAME = "5CD-AI/Vintern-1B-v3_5"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feed image + OCR CSE context into Vintern via Hugging Face.")
    parser.add_argument("query", help="Vietnamese question to answer.")
    parser.add_argument("--image-id", default="1003", help="Image id / graph store id.")
    parser.add_argument("--graph-root", default=None, help="Optional root directory of prebuilt graph stores, e.g. outputs/graph_test.")
    parser.add_argument("--model", default=DEFAULT_VINTERN_MODEL_NAME, help="Vintern HF model name or local path.")
    parser.add_argument("--cse-json", default=None, help="Optional precomputed CSE JSON. If omitted, CSE is computed automatically.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of text seed nodes for text-first CSE.")
    parser.add_argument("--hops", type=int, default=3, help="Expansion hops for CSE.")
    parser.add_argument("--top-m", type=int, default=5, help="Top outgoing edges kept per frontier node.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Minimum CSE edge score.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for offline edge confidence.")
    parser.add_argument("--lambda-hub", type=float, default=0.05, help="Hub penalty.")
    parser.add_argument("--max-nodes", type=int, default=100, help="Maximum nodes in each subgraph.")
    parser.add_argument("--max-edges", type=int, default=200, help="Maximum edges in each subgraph.")
    parser.add_argument("--max-subgraphs", type=int, default=3, help="Maximum retrieved subgraphs to include.")
    parser.add_argument("--max-nodes-per-subgraph", type=int, default=5, help="Maximum nodes per subgraph.")
    parser.add_argument("--max-crops", type=int, default=0, help="Maximum node crop images to attach. Use 0 to attach all selected node crops.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum newly generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature. Use 0 for deterministic decoding.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p used when temperature > 0.")
    parser.add_argument("--num-beams", type=int, default=3, help="Beam count for generation.")
    parser.add_argument("--repetition-penalty", type=float, default=2.5, help="Repetition penalty.")
    parser.add_argument("--input-size", type=int, default=448, help="Tile size used by Vintern preprocessing.")
    parser.add_argument("--max-num", type=int, default=12, help="Maximum dynamic image tiles.")
    parser.add_argument("--only-composite", action="store_true", help="Only build composite image then exit.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def default_output_path(config: GraphConfig, model_name: str, query: str) -> Path:
    digest = hashlib.md5(query.encode("utf-8")).hexdigest()[:10]
    return config.output_dir / "vlm_vintern_hf" / f"{sanitize_model_name(model_name)}_{digest}.json"


def init_vintern_model(model_name: str) -> tuple[object, object]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers import modeling_utils
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies for Vintern HF. Install torch and transformers."
        ) from exc

    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Newer transformers versions instantiate many models on `meta` by default.
    # Vintern's custom vision code calls `.item()` during `__init__`, which breaks on meta tensors.
    original_get_init_context = modeling_utils.PreTrainedModel.get_init_context
    original_mark_tied_weights_as_initialized = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized
    original_move_missing_keys_from_meta_to_device = modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device
    original_tie_weights = modeling_utils.PreTrainedModel.tie_weights

    @classmethod
    def no_meta_get_init_context(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels):
        init_contexts = [
            modeling_utils.local_torch_dtype(dtype, cls.__name__),
            modeling_utils.init.no_tie_weights(),
            modeling_utils.apply_patches(),
        ]
        if allow_all_kernels:
            init_contexts.append(modeling_utils.allow_all_hub_kernels())
        return init_contexts

    def get_compat_tied_keys(model) -> dict:
        tied = getattr(model, "all_tied_weights_keys", None)
        if isinstance(tied, dict):
            return tied
        legacy_tied = getattr(model, "_tied_weights_keys", None)
        if isinstance(legacy_tied, dict):
            return dict(legacy_tied)
        if isinstance(legacy_tied, (list, tuple, set)):
            return {str(key): str(key) for key in legacy_tied}
        return {}

    def with_compat_tied_keys(fn):
        def wrapper(self, *args, **kwargs):
            injected = False
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = get_compat_tied_keys(self)
                injected = True
            try:
                return fn(self, *args, **kwargs)
            finally:
                if injected:
                    delattr(self, "all_tied_weights_keys")

        return wrapper

    def safe_mark_tied_weights_as_initialized(self, loading_info):
        return with_compat_tied_keys(original_mark_tied_weights_as_initialized)(self, loading_info)

    modeling_utils.PreTrainedModel.get_init_context = no_meta_get_init_context
    modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = safe_mark_tied_weights_as_initialized
    modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = with_compat_tied_keys(
        original_move_missing_keys_from_meta_to_device
    )
    modeling_utils.PreTrainedModel.tie_weights = with_compat_tied_keys(original_tie_weights)
    try:
        model = AutoModel.from_pretrained(
            model_name,
            dtype=torch_dtype,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval()
    finally:
        modeling_utils.PreTrainedModel.get_init_context = original_get_init_context
        modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = original_mark_tied_weights_as_initialized
        modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = (
            original_move_missing_keys_from_meta_to_device
        )
        modeling_utils.PreTrainedModel.tie_weights = original_tie_weights

    if torch.cuda.is_available():
        model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def build_transform(input_size: int):
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
    except ImportError as exc:
        raise ImportError("Missing torchvision for Vintern preprocessing.") from exc

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: Path, input_size: int = 448, max_num: int = 12):
    import torch

    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


def fit_image(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    copy = img.copy()
    copy.thumbnail((max_width, max_height))
    return copy


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "arial.ttf",
        "tahoma.ttf",
        "DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def measure_text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return int(right - left)


def ellipsize_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> str:
    value = text.strip()
    if not value:
        return ""
    if measure_text_width(draw, value, font) <= max_width:
        return value
    while value and measure_text_width(draw, f"{value}...", font) > max_width:
        value = value[:-1].rstrip()
    return f"{value}..." if value else "..."


def wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_lines: int,
) -> list[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ["(no ocr text)"]

    words = normalized.split(" ")
    lines: list[str] = []
    current = ""
    consumed = 0
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if measure_text_width(draw, candidate, font) <= max_width:
            current = candidate
            consumed += 1
            continue

        if current:
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
        else:
            lines.append(ellipsize_to_width(draw, word, max_width, font))
            consumed += 1
            current = ""
            if len(lines) >= max_lines:
                break

    if len(lines) < max_lines and current:
        lines.append(current)

    remaining_words = len(words) - consumed
    if remaining_words > 0 and lines:
        lines[-1] = ellipsize_to_width(draw, lines[-1], max_width, font)
    return lines[:max_lines]


def build_composite_image(
    image_path: Path,
    crop_paths: list[tuple[str, Path]],
    node_ocr_texts: dict[str, str],
    output_path: Path,
) -> Path:
    base_image = Image.open(image_path).convert("RGB")
    base_panel = fit_image(base_image, max_width=1200, max_height=900)
    margin = 16
    label_h = 24
    inner_pad = 8
    ocr_max_lines = 3
    crop_box_w = 260
    crop_box_h = 280
    label_font = load_font(14)
    ocr_font = load_font(16)
    measure_draw = ImageDraw.Draw(Image.new("RGB", (32, 32), color=(255, 255, 255)))
    line_top, line_bottom = ocr_font.getbbox("Ag")[1], ocr_font.getbbox("Ag")[3]
    ocr_line_h = max(16, int(line_bottom - line_top) + 2)
    ocr_area_h = ocr_line_h * ocr_max_lines + 6
    image_slot_h = crop_box_h - label_h - ocr_area_h - inner_pad * 3
    image_slot_w = crop_box_w - inner_pad * 2

    crop_panels: list[tuple[str, Image.Image, list[str]]] = []
    for index, (node_id, crop_path) in enumerate(crop_paths, start=1):
        crop_image = Image.open(crop_path).convert("RGB")
        ocr_text = str(node_ocr_texts.get(node_id, "") or "").strip()
        ocr_lines = wrap_text_lines(
            measure_draw,
            text=ocr_text,
            max_width=image_slot_w,
            font=ocr_font,
            max_lines=ocr_max_lines,
        )
        crop_panels.append((f"crop_ref={index} | {node_id}", fit_image(crop_image, image_slot_w, image_slot_h), ocr_lines))

    cols = 3
    crop_rows = math.ceil(len(crop_panels) / cols) if crop_panels else 0
    crop_grid_height = crop_rows * crop_box_h + max(0, crop_rows - 1) * margin
    canvas_width = max(base_panel.width, cols * crop_box_w + (cols - 1) * margin) + margin * 2
    canvas_height = margin * 3 + base_panel.height + crop_grid_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    base_x = (canvas_width - base_panel.width) // 2
    canvas.paste(base_panel, (base_x, margin))
    draw.text((base_x, margin + base_panel.height + 4), "image_ref=0 | original_image", fill=(0, 0, 0))

    crop_start_y = margin * 2 + base_panel.height + label_h
    for idx, (label, crop_img, ocr_lines) in enumerate(crop_panels):
        row = idx // cols
        col = idx % cols
        x = margin + col * (crop_box_w + margin)
        y = crop_start_y + row * (crop_box_h + margin)
        draw.rectangle((x, y, x + crop_box_w, y + crop_box_h), outline=(180, 180, 180), width=1)
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0), font=label_font)
        paste_x = x + (crop_box_w - crop_img.width) // 2
        image_slot_top = y + label_h + inner_pad
        paste_y = image_slot_top + (image_slot_h - crop_img.height) // 2
        canvas.paste(crop_img, (paste_x, paste_y))
        ocr_y = y + crop_box_h - ocr_area_h - inner_pad + 2
        for line in ocr_lines:
            draw.text((x + inner_pad, ocr_y), line, fill=(0, 0, 0), font=ocr_font)
            ocr_y += ocr_line_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def generate_with_vintern(
    model: object,
    tokenizer: object,
    prompt: str,
    image_path: Path,
    input_size: int,
    max_num: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
    repetition_penalty: float,
) -> str:
    pixel_values = load_image(image_path, input_size=input_size, max_num=max_num).to(model.dtype).cuda()
    do_sample = temperature > 0
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": 1 if do_sample else num_beams,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        generation_config["temperature"] = temperature
        generation_config["top_p"] = top_p
    question = f"<image>\n{prompt}"
    response = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=False)
    if isinstance(response, tuple):
        return str(response[0]).strip()
    return str(response).strip()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    load_env_file(repo_root / ".env")
    graph_root = resolve_repo_relative_path(repo_root, args.graph_root)
    config = GraphConfig(repo_root=repo_root, image_id=args.image_id, output_root=graph_root)

    if not config.image_path.exists():
        raise FileNotFoundError(f"Image not found: {config.image_path}")

    if args.cse_json:
        cse_json_path = Path(args.cse_json)
        if not cse_json_path.exists():
            raise FileNotFoundError(f"CSE JSON not found: {cse_json_path}")
        cse_payload = load_cse_payload(cse_json_path)
    else:
        cse_payload, cse_json_path = build_cse_payload(config, args.query, args)

    context_nodes = collect_context_nodes(
        cse_payload=cse_payload,
        max_subgraphs=args.max_subgraphs,
        max_nodes_per_subgraph=args.max_nodes_per_subgraph,
    )
    crop_paths = collect_crop_paths(
        context_nodes,
        image_path=config.image_path,
        runtime_crops_dir=config.output_dir / "vlm_vintern_runtime_crops",
        max_crops=args.max_crops,
    )
    node_ocr_texts = {str(node.get("node_id", "")).strip(): str(node.get("text", "") or "").strip() for node in context_nodes}
    prompt = build_vlm_prompt(args.query, context_nodes=context_nodes, crop_paths=crop_paths)

    composite_path = build_composite_image(
        image_path=config.image_path,
        crop_paths=crop_paths,
        node_ocr_texts=node_ocr_texts,
        output_path=config.output_dir / "vlm_vintern_runtime_crops" / f"composite_{config.image_id}.png",
    )

    if args.only_composite:
        print(f"query={args.query}")
        print(f"image_path={config.image_path}")
        print(f"composite_image={composite_path}")
        print(f"cse_json={cse_json_path}")
        print(f"used_nodes={len(context_nodes)}")
        print(f"used_crops={len(crop_paths)}")
        return

    model, tokenizer = init_vintern_model(args.model)
    answer = generate_with_vintern(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        image_path=composite_path,
        input_size=args.input_size,
        max_num=args.max_num,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )

    output_path = Path(args.output) if args.output else default_output_path(config, args.model, args.query)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": args.query,
        "model": args.model,
        "image_id": config.image_id,
        "image_path": str(config.image_path),
        "composite_image_path": str(composite_path),
        "cse_json_path": str(cse_json_path),
        "cse_params": {
            "top_k": args.top_k,
            "hops": args.hops,
            "top_m": args.top_m,
            "threshold": args.threshold,
            "alpha": args.alpha,
            "lambda_hub": args.lambda_hub,
            "max_nodes": args.max_nodes,
            "max_edges": args.max_edges,
        },
        "max_subgraphs": args.max_subgraphs,
        "max_nodes_per_subgraph": args.max_nodes_per_subgraph,
        "max_crops": args.max_crops,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "used_crop_paths": [str(path) for _, path in crop_paths],
        "used_node_ids": [node["node_id"] for node in context_nodes],
        "answer": answer,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"query={args.query}")
    print(f"model={args.model}")
    print(f"image_path={config.image_path}")
    print(f"composite_image={composite_path}")
    print(f"cse_json={cse_json_path}")
    print(f"used_nodes={len(context_nodes)}")
    print(f"used_crops={len(crop_paths)}")
    print()
    print(answer)
    print()
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
