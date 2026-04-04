from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from utils.config import GraphConfig
from spatial_graph.io_utils import crop_node_images


def resolve_device(preferred_device: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if preferred_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def init_text_model(model_name: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Missing dependency 'sentence-transformers'. Install it before building text embeddings.") from exc
    return SentenceTransformer(model_name, device=device)


def init_clip_model(model_name: str, device: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, CLIPModel
    except ImportError as exc:
        raise ImportError("Missing dependency 'transformers'. Install it before building CLIP embeddings.") from exc
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def init_embedding_models(
    text_model_name: str,
    image_model_name: str,
    device: str,
) -> tuple[Any, Any, Any]:
    text_model = init_text_model(text_model_name, device)
    clip_processor, clip_model = init_clip_model(image_model_name, device)
    return text_model, clip_processor, clip_model


def encode_clip_texts(
    processor: Any,
    model: Any,
    texts: list[str],
    device: str,
    batch_size: int,
) -> list[list[float]]:
    import torch

    def coerce_to_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            hidden = output.last_hidden_state
            if isinstance(hidden, torch.Tensor):
                if hidden.ndim == 3:
                    return hidden.mean(dim=1)
                return hidden
        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                if first.ndim == 3:
                    return first.mean(dim=1)
                return first
        raise TypeError(f"Unsupported CLIP text output type: {type(output)!r}")

    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(model, "get_text_features"):
                output = model.get_text_features(**inputs)
            else:
                output = model(**inputs)
            embeddings = coerce_to_tensor(output)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        all_embeddings.extend(embeddings.cpu().numpy().astype(np.float32).tolist())
    return all_embeddings


def encode_text_embeddings(model: Any, texts: list[str], batch_size: int) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32).tolist()


def encode_image_embeddings(
    processor: Any,
    model: Any,
    images: list[Image.Image],
    device: str,
    batch_size: int,
) -> list[list[float]]:
    import torch

    def coerce_to_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            hidden = output.last_hidden_state
            if isinstance(hidden, torch.Tensor):
                if hidden.ndim == 3:
                    return hidden.mean(dim=1)
                return hidden
        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                if first.ndim == 3:
                    return first.mean(dim=1)
                return first
        raise TypeError(f"Unsupported image embedding output type: {type(output)!r}")

    all_embeddings: list[list[float]] = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                output = model.get_image_features(**inputs)
            else:
                output = model(**inputs)
            embeddings = coerce_to_tensor(output)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        all_embeddings.extend(embeddings.cpu().numpy().astype(np.float32).tolist())
    return all_embeddings


def attach_embeddings(
    config: GraphConfig,
    nodes: list[dict],
    texts_for_embedding: list[str],
    image_path,
    device: str,
    preloaded_models: tuple[Any, Any, Any] | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    if preloaded_models is None:
        text_model = init_text_model(config.text_embedding_model, device)
        clip_processor, clip_model = init_clip_model(config.image_embedding_model, device)
    else:
        text_model, clip_processor, clip_model = preloaded_models

    crops = crop_node_images(
        nodes,
        image_path=image_path,
        crops_dir=config.node_crops_dir,
        save_crops=config.save_node_crops,
        padding=config.crop_padding,
    )

    text_embeddings = np.asarray(
        encode_text_embeddings(text_model, texts_for_embedding, batch_size=config.embed_batch_size),
        dtype=np.float32,
    )
    image_embeddings = np.asarray(
        encode_image_embeddings(
            processor=clip_processor,
            model=clip_model,
            images=crops,
            device=device,
            batch_size=config.embed_batch_size,
        ),
        dtype=np.float32,
    )

    for idx, node in enumerate(nodes):
        node["text_embedding_index"] = idx
        node["crop_embedding_index"] = idx
        node["text_embedding_dim"] = int(text_embeddings.shape[1])
        node["crop_embedding_dim"] = int(image_embeddings.shape[1])
    return nodes, text_embeddings, image_embeddings


def build_nodes_for_json(nodes: list[dict]) -> list[dict]:
    return [dict(node) for node in nodes]
