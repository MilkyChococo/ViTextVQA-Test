from __future__ import annotations

from pathlib import Path
from typing import Any

from spatial_graph.io_utils import fix_mojibake


def build_vlm_prompt(query: str, context_nodes: list[dict[str, Any]], crop_paths: list[tuple[str, Path]]) -> str:
    lines: list[str] = []
    lines.append("Bạn là trợ lý Visual Question Answering cho Tiếng Việt.")
    lines.append("Hãy trả lời ngắn gọn, đúng trọng tâm, chỉ dựa trên ảnh gốc va OCR subgraph đã truy hồi.")
    lines.append("Được quyền ghép nối thông tin từ các node OCR và crop OCR liên quan, nhưng đừng thêm bất kỳ kiến thức nền nào khác.")
    lines.append("Đừng trả lời dựa trên kiến thức bên ngoài, chỉ đưa ra đáp án bằng Tiếng Việt dựa trên thông tin được cung cấp.")
    lines.append("Trả lời thông tin chủ yếu qua ảnh gốc và các subgraph OCR chỉ dùng để hỗ trợ thông tin OCR bổ sung ngữ nghĩa")
    lines.append("Nếu câu hỏi liên quan đến nguồn thì ưu tiên dùng thông tin từ OCR subgraph sau đó đối chứng với hình ảnh và đưa ra câu trả lời cuối cùng.")
    lines.append("")
    lines.append(f"Câu hỏi: {fix_mojibake(query)}")
    lines.append("")
    lines.append("Thông tin đính kèm:")
    if crop_paths:
        lines.append("- Các ảnh tiếp theo là crop của các node OCR liên quan, theo đúng thứ tự dưới đây.")
        for crop_index, (node_id, _) in enumerate(crop_paths, start=1):
            lines.append(f"  crop_ref={crop_index} <-> node_id={node_id}")
    else:
        lines.append("- Không có crop OCR bổ sung, chỉ có ảnh gốc.")
    lines.append("")
    lines.append("OCR subgraph context:")

    if not context_nodes:
        lines.append("- Không có node OCR nào được truy hồi.")
    else:
        for index, node in enumerate(context_nodes, start=1):
            lines.append(
                f"{index}. node_id={node['node_id']} | subgraph={node['subgraph_rank']} | "
                f"score={node['final_score']:.4f} | rel={node['rel']:.4f} | bbox={node['bbox']} | "
                f"text={node['text']}"
            )
    return "\n".join(lines)


def build_image_only_prompt(query: str) -> str:
    lines: list[str] = []
    lines.append("Ban la tro ly Visual Question Answering cho tieng Viet.")
    lines.append("Hay tra loi ngan gon, dung trong tam, chi dua tren anh duoc cung cap.")
    lines.append("Chi tra loi dap an cuoi cung bang tieng Viet, khong giai thich.")
    lines.append("")
    lines.append(f"Cau hoi: {fix_mojibake(query)}")
    return "\n".join(lines)
