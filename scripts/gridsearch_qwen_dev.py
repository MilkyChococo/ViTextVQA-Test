#!/usr/bin/env python
"""Run GridSearchCV over the ViTextVQA dev split using the existing Qwen workflow."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from algo.cse_query import run_text_first_cse
from encode.embeddings import init_clip_model, init_text_model, resolve_device
from evaluate_vitextvqa import exact_match_score, f1_score, metric_max_over_ground_truths, raw_text
from model import (
    collect_context_nodes,
    collect_crop_paths,
    load_env_file,
    resolve_repo_relative_path,
)
from model_qwen import DEFAULT_QWEN_MODEL_NAME, generate_with_qwen, init_qwen_model
from process.text_preprocess import preprocess_query_text
from utils.config import GraphConfig
from utils.prompts import build_vlm_prompt

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "hops": [1, 2, 3, 4, 5],
    "threshold": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "alpha": [0.3, 0.4, 0.5],
    "lambda_hub": [0.01, 0.03, 0.05, 0.07, 0.10],
}

_TEXT_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_CLIP_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_QWEN_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_GRAPH_BUNDLE_CACHE: OrderedDict[tuple[str, str], tuple[GraphConfig, dict[str, Any], np.ndarray, np.ndarray]] = OrderedDict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune ViTextVQA hyperparameters on any labeled split with GroupKFold + GridSearchCV."
    )
    parser.add_argument(
        "--dev-json",
        default="vitextvqa/ViTextVQA_images/ViTextVQA_dev.json",
        help="Labeled split JSON path. Can be dev JSON or test_gt JSON.",
    )
    parser.add_argument(
        "--gt-json",
        default=None,
        help="Optional ground-truth JSON path if --dev-json does not contain answers.",
    )
    parser.add_argument(
        "--graph-root",
        default="outputs/graph_test",
        help="Root directory of prebuilt graph stores.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_QWEN_MODEL_NAME,
        help="Qwen2.5-VL model name or local path.",
    )
    parser.add_argument(
        "--param-grid-json",
        default=None,
        help="Optional JSON string for GridSearchCV param_grid. Defaults to a small retrieval grid.",
    )
    parser.add_argument("--cv", type=int, default=3, help="Number of GroupKFold splits.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of annotations to evaluate after sampling.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Optional fraction of annotations to keep, e.g. 0.5 for 50%%.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle annotations before applying --limit.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for optional shuffling.")
    parser.add_argument("--top-m", type=int, default=5, help="Top outgoing edges kept per frontier node.")
    parser.add_argument("--lambda-hub", type=float, default=0.05, help="Hub penalty.")
    parser.add_argument("--max-nodes", type=int, default=100, help="Maximum nodes in each subgraph.")
    parser.add_argument("--max-edges", type=int, default=200, help="Maximum edges in each subgraph.")
    parser.add_argument("--max-subgraphs", type=int, default=2, help="Maximum retrieved subgraphs to include.")
    parser.add_argument(
        "--max-nodes-per-subgraph",
        type=int,
        default=3,
        help="Maximum nodes per subgraph passed to the prompt.",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=4,
        help="Maximum OCR crop images attached to Qwen.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument(
        "--text-retrieval-device",
        default="cuda",
        help="Device for BGE-M3 query embedding.",
    )
    parser.add_argument(
        "--clip-retrieval-device",
        default="cpu",
        help="Device for CLIP query embedding.",
    )
    parser.add_argument(
        "--max-graph-cache-size",
        type=int,
        default=1,
        help="Maximum number of per-image graph bundles kept in RAM.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="GridSearchCV n_jobs. Keep 1 when using a GPU model.")
    parser.add_argument("--verbose", type=int, default=2, help="GridSearchCV verbosity.")
    parser.add_argument(
        "--output",
        default="outputs/search/gridsearch_qwen_dev_results.json",
        help="Output JSON path for GridSearchCV results.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_payload(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict) or "images" not in payload or "annotations" not in payload:
        raise ValueError(f"Unexpected ViTextVQA format: {path}")
    return payload


def build_image_filename_lookup(images: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in images:
        image_id = str(item.get("id"))
        filename = str(item.get("filename") or f"{image_id}.jpg")
        lookup[image_id] = filename
    return lookup


def load_graph_bundle(config: GraphConfig) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    graph_enriched = json.loads(config.graph_enriched_path.read_text(encoding="utf-8"))
    text_embeddings = np.load(config.text_embeddings_path)
    crop_embeddings = np.load(config.crop_embeddings_path)
    return graph_enriched, text_embeddings, crop_embeddings


def cleanup_runtime_crops(crop_paths: list[tuple[str, Path]], runtime_crops_dir: Path) -> None:
    runtime_root = runtime_crops_dir.resolve()
    for _, crop_path in crop_paths:
        try:
            resolved = crop_path.resolve()
        except Exception:
            continue
        if resolved.parent == runtime_root and resolved.exists():
            try:
                resolved.unlink()
            except OSError:
                pass


def clear_runtime_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_annotations_with_answers(
    split_payload: dict[str, Any],
    gt_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    image_lookup = build_image_filename_lookup(split_payload.get("images", []))
    gt_answer_by_id: dict[int, list[str]] = {}
    if gt_payload is not None:
        for ann in gt_payload.get("annotations", []):
            ann_id = int(ann["id"])
            gt_answer_by_id[ann_id] = [raw_text(item) for item in ann.get("answers", [])]

    records: list[dict[str, Any]] = []
    for ann in split_payload.get("annotations", []):
        ann_id = int(ann["id"])
        image_id = str(ann["image_id"])
        answers = [raw_text(item) for item in ann.get("answers", [])]
        if not answers and ann_id in gt_answer_by_id:
            answers = gt_answer_by_id[ann_id]
        records.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "image_filename": image_lookup.get(image_id, f"{image_id}.jpg"),
                "question": raw_text(ann.get("question", "")),
                "answers": answers,
            }
        )
    return records


def parse_param_grid(param_grid_json: str | None) -> dict[str, list[Any]] | list[dict[str, list[Any]]]:
    if not param_grid_json:
        return dict(DEFAULT_PARAM_GRID)
    parsed = json.loads(param_grid_json)
    if isinstance(parsed, dict):
        return {str(key): list(value) for key, value in parsed.items()}
    if isinstance(parsed, list):
        grids: list[dict[str, list[Any]]] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("Each item in param-grid-json list must be an object.")
            grids.append({str(key): list(value) for key, value in item.items()})
        return grids
    raise ValueError("param-grid-json must be a JSON object or a list of JSON objects.")


def average_metric(
    predictions: list[str] | np.ndarray,
    annotations: list[dict[str, Any]] | np.ndarray,
    metric_fn,
) -> float:
    total = 0.0
    count = 0
    for prediction, ann in zip(predictions, annotations):
        answers = [raw_text(item) for item in ann.get("answers", [])]
        total += metric_max_over_ground_truths(metric_fn, raw_text(prediction), answers)
        count += 1
    return total / count if count else 0.0


class ViTextVQAQwenEstimator:
    def __init__(
        self,
        *,
        repo_root: str,
        graph_root: str,
        model_name: str = DEFAULT_QWEN_MODEL_NAME,
        top_k: int = 5,
        hops: int = 3,
        top_m: int = 5,
        threshold: float = 0.35,
        alpha: float = 0.5,
        lambda_hub: float = 0.05,
        max_nodes: int = 100,
        max_edges: int = 200,
        max_subgraphs: int = 2,
        max_nodes_per_subgraph: int = 3,
        max_crops: int = 4,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        text_retrieval_device: str = "cuda",
        clip_retrieval_device: str = "cpu",
        max_graph_cache_size: int = 1,
    ) -> None:
        self.repo_root = repo_root
        self.graph_root = graph_root
        self.model_name = model_name
        self.top_k = top_k
        self.hops = hops
        self.top_m = top_m
        self.threshold = threshold
        self.alpha = alpha
        self.lambda_hub = lambda_hub
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_subgraphs = max_subgraphs
        self.max_nodes_per_subgraph = max_nodes_per_subgraph
        self.max_crops = max_crops
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.text_retrieval_device = text_retrieval_device
        self.clip_retrieval_device = clip_retrieval_device
        self.max_graph_cache_size = max_graph_cache_size
        self._prediction_cache: dict[tuple[int, ...], np.ndarray] = {}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "graph_root": self.graph_root,
            "model_name": self.model_name,
            "top_k": self.top_k,
            "hops": self.hops,
            "top_m": self.top_m,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "lambda_hub": self.lambda_hub,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_subgraphs": self.max_subgraphs,
            "max_nodes_per_subgraph": self.max_nodes_per_subgraph,
            "max_crops": self.max_crops,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "text_retrieval_device": self.text_retrieval_device,
            "clip_retrieval_device": self.clip_retrieval_device,
            "max_graph_cache_size": self.max_graph_cache_size,
        }

    def set_params(self, **params: Any) -> "ViTextVQAQwenEstimator":
        for key, value in params.items():
            setattr(self, key, value)
        self._prediction_cache = {}
        return self

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ViTextVQAQwenEstimator":
        self._prediction_cache = {}
        load_env_file(Path(self.repo_root) / ".env")
        self._ensure_runtime()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        cache_key = tuple(int(item["id"]) for item in X)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key].copy()

        predictions: list[str] = []
        iterator = tqdm(
            X,
            desc=self._progress_desc(len(X)),
            unit="question",
            leave=False,
            dynamic_ncols=True,
            disable=len(X) <= 1,
        )
        for ann in iterator:
            predictions.append(self._predict_one(ann))
        output = np.asarray(predictions, dtype=object)
        self._prediction_cache[cache_key] = output
        return output.copy()

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        predictions = self.predict(X)
        return average_metric(predictions, X, f1_score)

    def _progress_desc(self, num_questions: int) -> str:
        return (
            f"GridSearchQwen "
            f"h={self.hops} "
            f"th={self.threshold:.2f} "
            f"a={self.alpha:.2f} "
            f"l={self.lambda_hub:.2f} "
            f"n={num_questions}"
        )

    def _ensure_runtime(self) -> tuple[Any, tuple[Any, Any], tuple[Any, Any]]:
        text_key = ("BAAI/bge-m3", self.text_retrieval_device)
        if text_key not in _TEXT_MODEL_CACHE:
            _TEXT_MODEL_CACHE[text_key] = init_text_model("BAAI/bge-m3", self.text_retrieval_device)

        clip_key = ("openai/clip-vit-base-patch32", self.clip_retrieval_device)
        if clip_key not in _CLIP_MODEL_CACHE:
            _CLIP_MODEL_CACHE[clip_key] = init_clip_model("openai/clip-vit-base-patch32", self.clip_retrieval_device)

        if self.model_name not in _QWEN_MODEL_CACHE:
            _QWEN_MODEL_CACHE[self.model_name] = init_qwen_model(self.model_name)

        return (
            _TEXT_MODEL_CACHE[text_key],
            _CLIP_MODEL_CACHE[clip_key],
            _QWEN_MODEL_CACHE[self.model_name],
        )

    def _get_graph_bundle(
        self,
        image_id: str,
        image_filename: str,
    ) -> tuple[GraphConfig, dict[str, Any], np.ndarray, np.ndarray]:
        cache_key = (self.graph_root, image_id)
        if cache_key in _GRAPH_BUNDLE_CACHE:
            _GRAPH_BUNDLE_CACHE.move_to_end(cache_key)
            return _GRAPH_BUNDLE_CACHE[cache_key]

        config = GraphConfig(
            repo_root=Path(self.repo_root),
            image_id=image_id,
            image_filename=image_filename,
            output_root=Path(self.graph_root),
        )
        bundle = (config, *load_graph_bundle(config))
        _GRAPH_BUNDLE_CACHE[cache_key] = bundle
        if self.max_graph_cache_size > 0:
            while len(_GRAPH_BUNDLE_CACHE) > self.max_graph_cache_size:
                _GRAPH_BUNDLE_CACHE.popitem(last=False)
        return bundle

    def _predict_one(self, ann: dict[str, Any]) -> str:
        text_embedder, clip_runtime, qwen_runtime = self._ensure_runtime()
        clip_processor, clip_model = clip_runtime
        qwen_model, qwen_processor = qwen_runtime

        image_id = str(ann["image_id"])
        image_filename = str(ann["image_filename"])
        question = raw_text(ann["question"])

        config, graph_enriched, text_embeddings, crop_embeddings = self._get_graph_bundle(image_id, image_filename)
        runtime_crops_dir = config.output_dir / "vlm_qwen_runtime_crops"
        crop_paths: list[tuple[str, Path]] = []
        query_for_embedding: str | None = None
        cse_payload: dict[str, Any] | None = None
        context_nodes: list[dict[str, Any]] = []
        prompt: str | None = None

        try:
            query_for_embedding = preprocess_query_text(config, question)
            cse_payload = run_text_first_cse(
                graph_enriched=graph_enriched,
                text_embeddings=text_embeddings,
                crop_embeddings=crop_embeddings,
                query=question,
                query_for_embedding=query_for_embedding,
                text_model_name=config.text_embedding_model,
                image_model_name=config.image_embedding_model,
                device=self.text_retrieval_device,
                text_device=self.text_retrieval_device,
                image_device=self.clip_retrieval_device,
                top_k=self.top_k,
                hops=self.hops,
                top_m=self.top_m,
                threshold=self.threshold,
                alpha=self.alpha,
                lambda_hub=self.lambda_hub,
                max_nodes=self.max_nodes,
                max_edges=self.max_edges,
                rel_text_weight=config.rel_text_weight,
                rel_image_weight=config.rel_image_weight,
                preloaded_text_embedder=text_embedder,
                preloaded_clip=(clip_processor, clip_model),
            )
            context_nodes = collect_context_nodes(
                cse_payload=cse_payload,
                max_subgraphs=self.max_subgraphs,
                max_nodes_per_subgraph=self.max_nodes_per_subgraph,
            )
            crop_paths = collect_crop_paths(
                context_nodes,
                image_path=config.image_path,
                runtime_crops_dir=runtime_crops_dir,
                max_crops=self.max_crops,
            )
            prompt = build_vlm_prompt(question, context_nodes=context_nodes, crop_paths=crop_paths)
            return generate_with_qwen(
                model=qwen_model,
                processor=qwen_processor,
                prompt=prompt,
                image_path=config.image_path,
                crop_paths=crop_paths,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            print(f"[WARN] ann_id={ann['id']} image_id={image_id} error={type(exc).__name__}: {exc}")
            return ""
        finally:
            cleanup_runtime_crops(crop_paths, runtime_crops_dir)
            crop_paths = []
            query_for_embedding = None
            cse_payload = None
            context_nodes = []
            prompt = None
            clear_runtime_memory()


def f1_scorer(estimator: ViTextVQAQwenEstimator, X: np.ndarray, y: np.ndarray | None = None) -> float:
    predictions = estimator.predict(X)
    return average_metric(predictions, X, f1_score)


def em_scorer(estimator: ViTextVQAQwenEstimator, X: np.ndarray, y: np.ndarray | None = None) -> float:
    predictions = estimator.predict(X)
    return average_metric(predictions, X, exact_match_score)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'scikit-learn'. Install it with: pip install scikit-learn"
        ) from exc

    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    load_env_file(repo_root / ".env")

    dev_json_path = resolve_repo_relative_path(repo_root, args.dev_json)
    gt_json_path = resolve_repo_relative_path(repo_root, args.gt_json)
    graph_root = resolve_repo_relative_path(repo_root, args.graph_root)
    output_path = resolve_repo_relative_path(repo_root, args.output)

    if dev_json_path is None or graph_root is None or output_path is None:
        raise ValueError("Failed to resolve required paths.")
    if not dev_json_path.exists():
        raise FileNotFoundError(f"Dev JSON not found: {dev_json_path}")
    if gt_json_path is not None and not gt_json_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_json_path}")

    dev_payload = load_payload(dev_json_path)
    gt_payload = load_payload(gt_json_path) if gt_json_path is not None else None
    annotations = build_annotations_with_answers(dev_payload, gt_payload)
    annotations = [ann for ann in annotations if ann["answers"]]

    if args.shuffle:
        rng = np.random.default_rng(args.random_state)
        rng.shuffle(annotations)

    if args.sample_ratio is not None:
        if not (0.0 < args.sample_ratio <= 1.0):
            raise ValueError("--sample-ratio must be in the interval (0, 1].")
        sample_size = max(1, int(round(len(annotations) * args.sample_ratio)))
        annotations = annotations[:sample_size]

    if args.limit is not None:
        annotations = annotations[: args.limit]

    if not annotations:
        raise ValueError("No dev annotations with answers are available for tuning.")

    groups = np.asarray([ann["image_id"] for ann in annotations], dtype=object)
    unique_groups = {str(item) for item in groups.tolist()}
    if len(unique_groups) < args.cv:
        raise ValueError(
            f"GroupKFold with cv={args.cv} requires at least {args.cv} unique image_id groups, found {len(unique_groups)}."
        )

    X = np.asarray(annotations, dtype=object)
    y = np.zeros(len(annotations), dtype=np.int32)

    param_grid = parse_param_grid(args.param_grid_json)
    total_candidates = len(list(ParameterGrid(param_grid)))

    estimator = ViTextVQAQwenEstimator(
        repo_root=str(repo_root),
        graph_root=str(graph_root),
        model_name=args.model,
        top_m=args.top_m,
        lambda_hub=args.lambda_hub,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_subgraphs=args.max_subgraphs,
        max_nodes_per_subgraph=args.max_nodes_per_subgraph,
        max_crops=args.max_crops,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        text_retrieval_device=resolve_device(args.text_retrieval_device),
        clip_retrieval_device=resolve_device(args.clip_retrieval_device),
        max_graph_cache_size=args.max_graph_cache_size,
    )

    splitter = GroupKFold(n_splits=args.cv)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring={"f1": f1_scorer, "em": em_scorer},
        refit="f1",
        cv=splitter,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        return_train_score=False,
    )

    print(f"split_json={dev_json_path}")
    print(f"graph_root={graph_root}")
    print(f"samples={len(annotations)}")
    print(f"unique_image_groups={len(unique_groups)}")
    print(f"cv={args.cv}")
    print(f"grid_candidates={total_candidates}")
    print(f"total_fits={total_candidates * args.cv}")

    search.fit(X, y, groups=groups)

    best_index = int(search.best_index_)
    best_mean_f1 = float(search.cv_results_["mean_test_f1"][best_index])
    best_mean_em = float(search.cv_results_["mean_test_em"][best_index])

    ranked_indices = sorted(
        range(len(search.cv_results_["params"])),
        key=lambda idx: float(search.cv_results_["mean_test_f1"][idx]),
        reverse=True,
    )
    top_results: list[dict[str, Any]] = []
    for idx in ranked_indices[: min(10, len(ranked_indices))]:
        top_results.append(
            {
                "rank": len(top_results) + 1,
                "params": search.cv_results_["params"][idx],
                "mean_test_f1": round(float(search.cv_results_["mean_test_f1"][idx]), 6),
                "std_test_f1": round(float(search.cv_results_["std_test_f1"][idx]), 6),
                "mean_test_em": round(float(search.cv_results_["mean_test_em"][idx]), 6),
                "std_test_em": round(float(search.cv_results_["std_test_em"][idx]), 6),
            }
        )

    result_payload = {
        "split_json": str(dev_json_path),
        "gt_json": str(gt_json_path) if gt_json_path is not None else None,
        "graph_root": str(graph_root),
        "num_annotations": len(annotations),
        "num_unique_image_groups": len(unique_groups),
        "sample_ratio": args.sample_ratio,
        "cv": args.cv,
        "n_jobs": args.n_jobs,
        "refit_metric": "f1",
        "param_grid": param_grid,
        "grid_candidates": total_candidates,
        "total_fits": total_candidates * args.cv,
        "best_index": best_index,
        "best_params": search.best_params_,
        "best_mean_test_f1": round(best_mean_f1, 6),
        "best_mean_test_em": round(best_mean_em, 6),
        "top_results": top_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"best_params={search.best_params_}")
    print(f"best_mean_test_f1={best_mean_f1:.6f}")
    print(f"best_mean_test_em={best_mean_em:.6f}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
