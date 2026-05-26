from __future__ import annotations


MODE = {
    "name": "spatial_graph_conf",
    "description": "Feed top-k spatial graph nodes selected by query relevance with the image and query.",
    "requires_graph": True,
    "requires_retrieval_models": True,
    "strategy": "graph_conf",
}
