#!/usr/bin/env python
"""Entry script for building a spatial graph prototype for one image."""

from __future__ import annotations

from pathlib import Path

from spatial_graph.pipeline import run_pipeline
from utils.config import GraphConfig


def main() -> None:
    config = GraphConfig(repo_root=Path(__file__).resolve().parent.parent, image_id="9")
    stats = run_pipeline(config)

    for key in [
        "raw_rows",
        "base_nodes",
        "horizontal_nodes",
        "merged_nodes",
        "spatial_edges",
        "graph_enriched_json",
        "text_embeddings_npy",
        "crop_embeddings_npy",
        "graph_json",
        "output_dir",
    ]:
        print(f"{key}={stats[key]}")


if __name__ == "__main__":
    main()
