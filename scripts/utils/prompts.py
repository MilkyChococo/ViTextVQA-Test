from __future__ import annotations

from pathlib import Path
from typing import Any

from spatial_graph.io_utils import fix_mojibake


def build_vlm_prompt(query: str, context_nodes: list[dict[str, Any]], crop_paths: list[tuple[str, Path]]) -> str:
    lines: list[str] = []

    # 🔒 Strong system instruction
    lines.append("You are a strict Visual Question Answering (VQA) assistant.")
    lines.append("Your task is to answer the question using ONLY the provided image and OCR subgraph context.")
    lines.append("")
    lines.append("CRITICAL RULES:")
    lines.append("- Do NOT use any external knowledge, prior knowledge, or assumptions.")
    lines.append("- Do NOT guess or infer beyond the visible evidence.")
    lines.append("- If the answer is not clearly supported, say: 'Không đủ thông tin'.")
    lines.append("- Prefer evidence from the original image; use OCR only as supporting information.")
    lines.append("- If OCR and image conflict, trust the image more.")
    lines.append("- Combine multiple OCR nodes ONLY if they clearly refer to the same entity.")
    lines.append("- Be precise, concise, and factual.")
    lines.append("- Answer in Vietnamese only.")
    lines.append("- EXCEPTION: If the answer is explicitly shown in the image/OCR as English text, copy it exactly without translation.")
    lines.append("- Do NOT translate, paraphrase, or modify extracted text.")
    lines.append("")
    lines.append("ANSWERING STRATEGY:")
    lines.append("1. Locate relevant regions in the image.")
    lines.append("2. Match them with OCR nodes if applicable.")
    lines.append("3. Verify consistency between image and OCR.")
    lines.append("4. Produce a short final answer (no explanation).")
    lines.append("")
    lines.append("EXAMPLES:")
    lines.append("Example 1:")
    lines.append("Question: Biển ghi gì?")
    lines.append("OCR: 'Cá cắn câu'")
    lines.append("Answer: Cá cắn câu")
    lines.append("")

    # Example 2: avoid over-generation
    lines.append("Example 2:")
    lines.append("Question: Nội dung chính là gì?")
    lines.append("OCR: 'dù lâu vẫn đợi Cá cắn câu'")
    lines.append("Answer: Cá cắn câu")
    lines.append("")

# Example 3: English text
    lines.append("Example 3:")
    lines.append("Question: Tên thương hiệu là gì?")
    lines.append("OCR: 'Highlands Coffee'")
    lines.append("Answer: Highlands Coffee")
    lines.append("")

# Example 4: insufficient info
    lines.append("Example 4:")
    lines.append("Question: Người trong ảnh tên gì?")
    lines.append("OCR: 'Hello world'")
    lines.append("Answer: Không đủ thông tin")
    lines.append("")
    lines.append("")
    lines.append(f"Question: {fix_mojibake(query)}")
    lines.append("")
    lines.append("ATTACHED INFORMATION:")
    if crop_paths:
        lines.append("- The following images are OCR crops corresponding to nodes:")
        for crop_index, (node_id, _) in enumerate(crop_paths, start=1):
            lines.append(f"  crop_ref={crop_index} <-> node_id={node_id}")
    else:
        lines.append("- No OCR crops provided. Only the original image is available.")
    lines.append("")
    lines.append("OCR SUBGRAPH CONTEXT:")
    if not context_nodes:
        lines.append("- No OCR nodes retrieved.")
    else:
        for index, node in enumerate(context_nodes, start=1):
            lines.append(
                f"{index}. node_id={node['node_id']} | subgraph={node['subgraph_rank']} | "
                f"score={node['final_score']:.4f} | rel={node['rel']:.4f} | bbox={node['bbox']} | "
                f"text={node['text']}"
            )

    lines.append("")
    lines.append("FINAL ANSWER (concise, no explanation):")

    return "\n".join(lines)


def build_image_only_prompt(query: str) -> str:
    lines: list[str] = []
    lines.append("Ban la tro ly Visual Question Answering cho tieng Viet.")
    lines.append("Hay tra loi ngan gon, dung trong tam, chi dua tren anh duoc cung cap.")
    lines.append("Chi tra loi dap an cuoi cung bang tieng Viet, khong giai thich.")
    lines.append("")
    lines.append(f"Cau hoi: {fix_mojibake(query)}")
    return "\n".join(lines)
