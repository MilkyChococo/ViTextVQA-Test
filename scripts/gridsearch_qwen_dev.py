#!/usr/bin/env python
"""Run a checkpointed grid search over a labeled ViTextVQA split using the existing Qwen workflow."""

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
    "hops": [2, 3, 4, 5],
    "threshold": [0.25, 0.35, 0.45, 0.55],
    "alpha": [0.3, 0.4, 0.5, 0.6],
    "lambda_hub": [0.03, 0.05, 0.07, 0.10],
}

_TEXT_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_CLIP_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_QWEN_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_GRAPH_BUNDLE_CACHE: OrderedDict[tuple[str, str], tuple[GraphConfig, dict[str, Any], np.ndarray, np.ndarray]] = OrderedDict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune ViTextVQA hyperparameters on any labeled split with GroupKFold and checkpointed grid search."
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
        help="Optional JSON string for the parameter grid. Defaults to a small retrieval grid.",
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
    parser.add_argument("--n-jobs", type=int, default=1, help="Reserved for compatibility. The search runs sequentially.")
    parser.add_argument("--verbose", type=int, default=2, help="Search verbosity.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint if available.")
    parser.add_argument(
        "--checkpoint-json",
        default=None,
        help="Optional checkpoint JSON path. Defaults next to --output.",
    )
    parser.add_argument(
        "--progress-jsonl",
        default=None,
        help="Optional JSONL log path for per-fold progress. Defaults next to --output.",
    )
    parser.add_argument(
        "--question-log-jsonl",
        default=None,
        help="Optional base JSONL path for per-question predictions. Each candidate is saved as a separate JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/search/gridsearch_qwen_dev_results.json",
        help="Output JSON path for final grid search results.",
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


def build_candidate_key(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_candidate_question_log_dir(base_path: Path) -> Path:
    return base_path.with_suffix("")


def build_candidate_question_log_path(base_path: Path, candidate_index: int) -> Path:
    log_dir = build_candidate_question_log_dir(base_path)
    return log_dir / f"candidate_{candidate_index:04d}.jsonl"


def summarize_candidate_result(
    candidate_index: int,
    candidate_key: str,
    params: dict[str, Any],
    fold_results: list[dict[str, Any]],
    total_folds: int,
    question_log_path: Path | None = None,
) -> dict[str, Any]:
    ordered_folds = sorted(fold_results, key=lambda item: int(item["fold_index"]))
    completed_folds = len(ordered_folds)
    mean_f1 = float(np.mean([item["f1"] for item in ordered_folds])) if ordered_folds else 0.0
    std_f1 = float(np.std([item["f1"] for item in ordered_folds])) if ordered_folds else 0.0
    mean_em = float(np.mean([item["em"] for item in ordered_folds])) if ordered_folds else 0.0
    std_em = float(np.std([item["em"] for item in ordered_folds])) if ordered_folds else 0.0
    return {
        "candidate_index": candidate_index,
        "candidate_key": candidate_key,
        "params": params,
        "status": "completed" if completed_folds == total_folds else "partial",
        "completed_folds": completed_folds,
        "total_folds": total_folds,
        "mean_test_f1": round(mean_f1, 6),
        "std_test_f1": round(std_f1, 6),
        "mean_test_em": round(mean_em, 6),
        "std_test_em": round(std_em, 6),
        "question_log_jsonl": str(question_log_path) if question_log_path is not None else None,
        "fold_results": ordered_folds,
    }


def save_search_checkpoint(
    path: Path,
    *,
    metadata: dict[str, Any],
    candidate_results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = [item for item in candidate_results if item.get("status") == "completed"]
    best_completed = max(
        completed,
        key=lambda item: (
            float(item.get("mean_test_f1", 0.0)),
            float(item.get("mean_test_em", 0.0)),
            -int(item.get("candidate_index", 10**9)),
        ),
        default=None,
    )
    payload = dict(metadata)
    payload.update(
        {
            "completed_candidates": len(completed),
            "total_recorded_candidates": len(candidate_results),
            "candidate_results": candidate_results,
            "best_completed_candidate": best_completed,
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        self._question_log_path: Path | None = None
        self._question_log_context: dict[str, Any] | None = None

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

    def set_question_log(self, path: Path | None, context: dict[str, Any] | None) -> None:
        self._question_log_path = path
        self._question_log_context = dict(context) if context is not None else None

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
            prediction = self._predict_one(ann)
            predictions.append(prediction)
            self._append_question_log(ann, prediction)
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

    def _append_question_log(self, ann: dict[str, Any], prediction: str) -> None:
        if self._question_log_path is None or self._question_log_context is None:
            return

        prediction_text = raw_text(prediction)
        answers = [raw_text(item) for item in ann.get("answers", [])]
        append_jsonl_record(
            self._question_log_path,
            {
                **self._question_log_context,
                "ann_id": int(ann["id"]),
                "image_id": str(ann["image_id"]),
                "image_filename": str(ann["image_filename"]),
                "question": raw_text(ann["question"]),
                "answers": answers,
                "prediction": prediction_text,
                "has_prediction": bool(prediction_text.strip()),
                "exact_match": round(
                    float(metric_max_over_ground_truths(exact_match_score, prediction_text, answers)),
                    6,
                ),
                "f1": round(
                    float(metric_max_over_ground_truths(f1_score, prediction_text, answers)),
                    6,
                ),
            },
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


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        from sklearn.model_selection import GroupKFold, ParameterGrid
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
    checkpoint_json_path = resolve_repo_relative_path(repo_root, args.checkpoint_json)
    progress_jsonl_path = resolve_repo_relative_path(repo_root, args.progress_jsonl)
    question_log_jsonl_path = resolve_repo_relative_path(repo_root, args.question_log_jsonl)

    if dev_json_path is None or graph_root is None or output_path is None:
        raise ValueError("Failed to resolve required paths.")
    if not dev_json_path.exists():
        raise FileNotFoundError(f"Dev JSON not found: {dev_json_path}")
    if gt_json_path is not None and not gt_json_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_json_path}")
    if checkpoint_json_path is None:
        checkpoint_json_path = output_path.with_name(f"{output_path.stem}_checkpoint.json")
    if progress_jsonl_path is None:
        progress_jsonl_path = output_path.with_name(f"{output_path.stem}_progress.jsonl")
    if question_log_jsonl_path is None:
        question_log_jsonl_path = output_path.with_name(f"{output_path.stem}_questions.jsonl")
    question_log_dir = build_candidate_question_log_dir(question_log_jsonl_path)

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
    candidate_params_list = list(ParameterGrid(param_grid))
    total_candidates = len(candidate_params_list)
    splitter = GroupKFold(n_splits=args.cv)
    splits = list(splitter.split(X, y, groups))

    base_estimator_params = {
        "repo_root": str(repo_root),
        "graph_root": str(graph_root),
        "model_name": args.model,
        "top_m": args.top_m,
        "lambda_hub": args.lambda_hub,
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
        "max_subgraphs": args.max_subgraphs,
        "max_nodes_per_subgraph": args.max_nodes_per_subgraph,
        "max_crops": args.max_crops,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "text_retrieval_device": resolve_device(args.text_retrieval_device),
        "clip_retrieval_device": resolve_device(args.clip_retrieval_device),
        "max_graph_cache_size": args.max_graph_cache_size,
    }

    if args.n_jobs != 1:
        print("[INFO] --n-jobs is ignored by the checkpointed manual search; running sequentially.")

    if not args.resume and progress_jsonl_path.exists():
        progress_jsonl_path.write_text("", encoding="utf-8")
    question_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"split_json={dev_json_path}")
    print(f"graph_root={graph_root}")
    print(f"samples={len(annotations)}")
    print(f"unique_image_groups={len(unique_groups)}")
    print(f"cv={args.cv}")
    print(f"grid_candidates={total_candidates}")
    print(f"total_fits={total_candidates * args.cv}")

    metadata = {
        "split_json": str(dev_json_path),
        "gt_json": str(gt_json_path) if gt_json_path is not None else None,
        "graph_root": str(graph_root),
        "num_annotations": len(annotations),
        "num_unique_image_groups": len(unique_groups),
        "sample_ratio": args.sample_ratio,
        "limit": args.limit,
        "cv": args.cv,
        "n_jobs": args.n_jobs,
        "search_backend": "manual_parameter_grid",
        "refit_metric": "f1",
        "param_grid": param_grid,
        "grid_candidates": total_candidates,
        "total_fits": total_candidates * args.cv,
        "question_log_base": str(question_log_jsonl_path),
        "question_log_dir": str(question_log_dir),
    }

    recorded_results: list[dict[str, Any]] = []
    if args.resume and checkpoint_json_path.exists():
        checkpoint_payload = load_json(checkpoint_json_path)
        recorded_results = list(checkpoint_payload.get("candidate_results", []))
        print(f"resume_checkpoint={checkpoint_json_path}")
        print(f"loaded_recorded_candidates={len(recorded_results)}")

    recorded_by_key = {
        str(item.get("candidate_key")): item
        for item in recorded_results
        if item.get("candidate_key")
    }

    candidate_iterator = tqdm(
        list(enumerate(candidate_params_list, start=1)),
        total=total_candidates,
        desc="GridCandidates",
        unit="candidate",
        dynamic_ncols=True,
    )
    for candidate_index, params in candidate_iterator:
        candidate_key = build_candidate_key(params)
        existing = recorded_by_key.get(candidate_key)
        if existing is not None and existing.get("status") == "completed":
            candidate_iterator.set_postfix(status="resume-skip", idx=candidate_index)
            continue

        candidate_question_log_path = build_candidate_question_log_path(question_log_jsonl_path, candidate_index)
        if not args.resume and candidate_question_log_path.exists():
            candidate_question_log_path.unlink()

        estimator = ViTextVQAQwenEstimator(**base_estimator_params)
        estimator.set_params(**params)

        fold_results = list(existing.get("fold_results", [])) if existing is not None else []
        done_folds = {int(item["fold_index"]) for item in fold_results}

        for fold_index, (train_idx, val_idx) in enumerate(splits, start=1):
            if fold_index in done_folds:
                continue

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]

            if args.verbose:
                print(
                    f"[CAND {candidate_index}/{total_candidates}] "
                    f"fold={fold_index}/{args.cv} params={params}"
                )

            estimator.fit(X_train, y_train)
            estimator.set_question_log(
                candidate_question_log_path,
                {
                    "candidate_index": candidate_index,
                    "candidate_key": candidate_key,
                    "params": params,
                    "fold_index": fold_index,
                },
            )
            predictions = estimator.predict(X_val)
            estimator.set_question_log(None, None)
            fold_f1 = average_metric(predictions, X_val, f1_score)
            fold_em = average_metric(predictions, X_val, exact_match_score)

            fold_record = {
                "fold_index": fold_index,
                "num_train": int(len(train_idx)),
                "num_val": int(len(val_idx)),
                "f1": round(float(fold_f1), 6),
                "em": round(float(fold_em), 6),
            }
            fold_results.append(fold_record)
            fold_results.sort(key=lambda item: int(item["fold_index"]))

            append_jsonl_record(
                progress_jsonl_path,
                {
                    "candidate_index": candidate_index,
                    "candidate_key": candidate_key,
                    "params": params,
                    "fold_result": fold_record,
                },
            )

            candidate_summary = summarize_candidate_result(
                candidate_index=candidate_index,
                candidate_key=candidate_key,
                params=params,
                fold_results=fold_results,
                total_folds=args.cv,
                question_log_path=candidate_question_log_path,
            )
            recorded_by_key[candidate_key] = candidate_summary

            current_results = sorted(
                recorded_by_key.values(),
                key=lambda item: int(item.get("candidate_index", 10**9)),
            )
            save_search_checkpoint(
                checkpoint_json_path,
                metadata=metadata,
                candidate_results=current_results,
            )
            candidate_iterator.set_postfix(
                idx=candidate_index,
                fold=f"{fold_index}/{args.cv}",
                f1=f"{fold_f1:.4f}",
                em=f"{fold_em:.4f}",
            )

        candidate_summary = summarize_candidate_result(
            candidate_index=candidate_index,
            candidate_key=candidate_key,
            params=params,
            fold_results=fold_results,
            total_folds=args.cv,
            question_log_path=candidate_question_log_path,
        )
        recorded_by_key[candidate_key] = candidate_summary
        current_results = sorted(
            recorded_by_key.values(),
            key=lambda item: int(item.get("candidate_index", 10**9)),
        )
        save_search_checkpoint(
            checkpoint_json_path,
            metadata=metadata,
            candidate_results=current_results,
        )

    final_candidate_results = sorted(
        recorded_by_key.values(),
        key=lambda item: int(item.get("candidate_index", 10**9)),
    )
    completed_candidates = [item for item in final_candidate_results if item.get("status") == "completed"]
    ranked_candidates = sorted(
        completed_candidates,
        key=lambda item: (
            -float(item.get("mean_test_f1", 0.0)),
            -float(item.get("mean_test_em", 0.0)),
            int(item.get("candidate_index", 10**9)),
        ),
    )
    top_results: list[dict[str, Any]] = []
    for item in ranked_candidates[: min(10, len(ranked_candidates))]:
        top_results.append(
            {
                "rank": len(top_results) + 1,
                "params": item["params"],
                "mean_test_f1": item["mean_test_f1"],
                "std_test_f1": item["std_test_f1"],
                "mean_test_em": item["mean_test_em"],
                "std_test_em": item["std_test_em"],
            }
        )

    best_candidate = ranked_candidates[0] if ranked_candidates else None
    result_payload = dict(metadata)
    result_payload.update(
        {
            "completed_candidates": len(completed_candidates),
            "candidate_results": final_candidate_results,
            "best_index": int(best_candidate["candidate_index"]) if best_candidate is not None else None,
            "best_params": best_candidate["params"] if best_candidate is not None else None,
            "best_mean_test_f1": best_candidate["mean_test_f1"] if best_candidate is not None else None,
            "best_mean_test_em": best_candidate["mean_test_em"] if best_candidate is not None else None,
            "top_results": top_results,
            "checkpoint_json": str(checkpoint_json_path),
            "progress_jsonl": str(progress_jsonl_path),
            "question_log_base": str(question_log_jsonl_path),
            "question_log_dir": str(question_log_dir),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"checkpoint_json={checkpoint_json_path}")
    print(f"progress_jsonl={progress_jsonl_path}")
    print(f"question_log_base={question_log_jsonl_path}")
    print(f"question_log_dir={question_log_dir}")
    if best_candidate is not None:
        print(f"best_params={best_candidate['params']}")
        print(f"best_mean_test_f1={float(best_candidate['mean_test_f1']):.6f}")
        print(f"best_mean_test_em={float(best_candidate['mean_test_em']):.6f}")
    else:
        print("best_params=None")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
