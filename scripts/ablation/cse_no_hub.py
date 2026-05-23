from __future__ import annotations


MODE = {
    "name": "cse_no_hub",
    "description": "CSE expansion with lambda_hub forced to 0.",
    "requires_graph": True,
    "requires_retrieval_models": True,
    "strategy": "cse",
    "lambda_hub_override": 0.0,
}
