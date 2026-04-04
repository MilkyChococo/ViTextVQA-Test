from __future__ import annotations

from utils.config import GraphConfig
from utils.text_processing import build_text_preprocessor


def preprocess_node_texts(config: GraphConfig, nodes: list[dict]) -> list[str]:
    preprocessor = build_text_preprocessor(
        backend=config.text_preprocess_backend,
        enable_word_segmentation=config.enable_text_preprocessing,
    )
    texts = [str(node.get("text", "") or "") for node in nodes]
    return preprocessor.preprocess_batch(texts)


def preprocess_query_text(config: GraphConfig, query: str) -> str:
    preprocessor = build_text_preprocessor(
        backend=config.text_preprocess_backend,
        enable_word_segmentation=config.enable_text_preprocessing,
    )
    return preprocessor.preprocess(query)
