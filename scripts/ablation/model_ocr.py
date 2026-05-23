from __future__ import annotations


MODE = {
    "name": "model_ocr",
    "description": "VLM with raw OCR text appended to the prompt.",
    "requires_graph": False,
    "requires_retrieval_models": False,
    "strategy": "ocr_text",
}
