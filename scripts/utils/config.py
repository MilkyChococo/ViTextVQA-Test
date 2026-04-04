from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GraphConfig:
    repo_root: Path
    image_id: str = "1003"
    image_filename: str | None = None

    y_overlap_for_horizontal_merge: float = 0.55
    max_horizontal_gap: int = 28
    max_height_ratio: float = 2.2

    x_overlap_for_vertical_merge: float = 0.60
    x_overlap_for_vertical_merge_on_max: float = 0.45
    max_vertical_gap: int = 10
    max_width_ratio: float = 2.2
    max_center_x_diff_ratio: float = 0.30

    y_overlap_for_right_edge: float = 0.30
    x_overlap_for_down_edge: float = 0.30
    max_right_gap: int = 180
    max_down_gap: int = 160

    text_embedding_model: str = "BAAI/bge-m3"
    image_embedding_model: str = "openai/clip-vit-base-patch32"
    build_embeddings: bool = True
    preferred_device: str = "cuda"
    embed_batch_size: int = 8
    save_node_crops: bool = True
    crop_padding: int = 4
    rel_text_weight: float = 0.4
    rel_image_weight: float = 0.6
    conf_text_weight: float = 0.4
    conf_image_weight: float = 0.6
    enable_text_preprocessing: bool = True
    text_preprocess_backend: str = "auto"
    save_visuals: bool = True

    image_root: Path | None = None
    ocr_root: Path | None = None
    output_root: Path | None = None

    image_path: Path = field(init=False)
    ocr_jsonl_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)
    visuals_dir: Path = field(init=False)
    node_crops_dir: Path = field(init=False)
    graph_json_path: Path = field(init=False)
    graph_enriched_path: Path = field(init=False)
    text_embeddings_path: Path = field(init=False)
    crop_embeddings_path: Path = field(init=False)
    overlay_path: Path = field(init=False)
    cse_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        resolved_image_root = Path(self.image_root) if self.image_root is not None else (
            self.repo_root / "vitextvqa" / "ViTextVQA_images" / "st_images"
        )
        resolved_ocr_root = Path(self.ocr_root) if self.ocr_root is not None else (
            self.repo_root / "outputs" / "test_bboxes"
        )
        resolved_output_root = Path(self.output_root) if self.output_root is not None else (
            self.repo_root / "outputs" / "graph_prototypes"
        )

        image_filename = self.image_filename or f"{self.image_id}.jpg"
        self.image_path = resolved_image_root / image_filename
        self.ocr_jsonl_path = resolved_ocr_root / self.image_id / "ocr_results.jsonl"
        self.output_dir = resolved_output_root / self.image_id
        self.artifacts_dir = self.output_dir / "artifacts"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.visuals_dir = self.output_dir / "visuals"
        self.cse_dir = self.output_dir / "cse"
        self.node_crops_dir = self.visuals_dir / "node_crops"
        self.graph_json_path = self.output_dir / "graph.json"
        self.graph_enriched_path = self.output_dir / "graph_enriched.json"
        self.text_embeddings_path = self.embeddings_dir / "text_embeddings.npy"
        self.crop_embeddings_path = self.embeddings_dir / "crop_embeddings.npy"
        self.overlay_path = self.visuals_dir / "spatial_graph_overlay.png"
