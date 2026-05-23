from __future__ import annotations


MODE = {
    "name": "cse_with_hub",
    "description": "Full CSE expansion with hub penalty.",
    "requires_graph": True,
    "requires_retrieval_models": True,
    "strategy": "cse",
    "lambda_hub_override": None,
}
