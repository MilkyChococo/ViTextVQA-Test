from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _resolve_segmenter(backend: str) -> tuple[str, Callable[[str], str] | None]:
    candidates = [backend] if backend != "auto" else ["underthesea", "pyvi"]

    for candidate in candidates:
        if candidate == "underthesea":
            try:
                from underthesea import word_tokenize
            except ImportError:
                continue
            return "underthesea", lambda text: word_tokenize(text, format="text")
        if candidate == "pyvi":
            try:
                from pyvi import ViTokenizer
            except ImportError:
                continue
            return "pyvi", lambda text: ViTokenizer.tokenize(text)

    return "none", None


@dataclass
class VietnameseTextPreprocessor:
    backend: str = "auto"
    enable_word_segmentation: bool = True

    backend_used: str = "none"
    _segment: Callable[[str], str] | None = None

    def __post_init__(self) -> None:
        if self.enable_word_segmentation:
            self.backend_used, self._segment = _resolve_segmenter(self.backend)
        else:
            self.backend_used, self._segment = "disabled", None

    def preprocess(self, text: str) -> str:
        normalized = normalize_text(text)
        if not normalized:
            return ""
        if self._segment is None:
            return normalized
        segmented = normalize_text(self._segment(normalized))
        return segmented or normalized

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        return [self.preprocess(text) for text in texts]


def build_text_preprocessor(
    backend: str = "auto",
    enable_word_segmentation: bool = True,
) -> VietnameseTextPreprocessor:
    return VietnameseTextPreprocessor(
        backend=backend,
        enable_word_segmentation=enable_word_segmentation,
    )
