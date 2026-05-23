from __future__ import annotations


MODE = {
    "name": "spatial_graph_conf",
    "description": "Query seeds + spatial neighbors ranked by offline edge confidence.",
    "requires_graph": True,
    "requires_retrieval_models": True,
    "strategy": "graph_conf",
}
