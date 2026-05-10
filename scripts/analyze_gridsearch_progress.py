#!/usr/bin/env python
"""Analyze checkpointed grid-search progress JSONL and generate summaries/plots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gridsearch progress JSONL.")
    parser.add_argument(
        "--progress-jsonl",
        required=True,
        help="Path to gridsearch progress JSONL (per-fold records).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/search/analysis",
        help="Directory to save summary files and plots.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top candidates to print/export.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc
            if "candidate_index" not in payload or "fold_result" not in payload:
                continue
            records.append(payload)
    if not records:
        raise ValueError(f"No valid records found in {path}")
    return records


def aggregate_by_candidate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for rec in records:
        idx = int(rec["candidate_index"])
        params = rec.get("params", {})
        fold_result = rec.get("fold_result", {})
        entry = grouped.setdefault(
            idx,
            {
                "candidate_index": idx,
                "params": params,
                "fold_results": [],
            },
        )
        entry["fold_results"].append(
            {
                "fold_index": int(fold_result.get("fold_index", -1)),
                "f1": float(fold_result.get("f1", 0.0)),
                "em": float(fold_result.get("em", 0.0)),
                "num_train": int(fold_result.get("num_train", 0)),
                "num_val": int(fold_result.get("num_val", 0)),
            }
        )

    rows: list[dict[str, Any]] = []
    for idx, entry in grouped.items():
        folds = sorted(entry["fold_results"], key=lambda x: x["fold_index"])
        f1s = np.array([f["f1"] for f in folds], dtype=float)
        ems = np.array([f["em"] for f in folds], dtype=float)
        rows.append(
            {
                "candidate_index": idx,
                "params": entry["params"],
                "num_folds": len(folds),
                "mean_f1": float(np.mean(f1s)) if len(f1s) else 0.0,
                "std_f1": float(np.std(f1s)) if len(f1s) else 0.0,
                "mean_em": float(np.mean(ems)) if len(ems) else 0.0,
                "std_em": float(np.std(ems)) if len(ems) else 0.0,
                "fold_results": folds,
            }
        )
    rows.sort(key=lambda r: (-r["mean_f1"], -r["mean_em"], r["candidate_index"]))
    return rows


def aggregate_param_effect(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        params = rec.get("params", {})
        fold_result = rec.get("fold_result", {})
        f1 = float(fold_result.get("f1", 0.0))
        for key, value in params.items():
            buckets[key][value].append(f1)

    output: dict[str, list[dict[str, Any]]] = {}
    for key, value_map in buckets.items():
        rows: list[dict[str, Any]] = []
        for value, f1_list in value_map.items():
            arr = np.array(f1_list, dtype=float)
            rows.append(
                {
                    "value": value,
                    "count": int(len(arr)),
                    "mean_f1": float(np.mean(arr)) if len(arr) else 0.0,
                    "std_f1": float(np.std(arr)) if len(arr) else 0.0,
                }
            )
        rows.sort(key=lambda r: r["value"])
        output[key] = rows
    return output


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_txt(path: Path, ranked: list[dict[str, Any]], top_k: int) -> None:
    lines: list[str] = []
    lines.append("Grid Search Analysis Summary")
    lines.append("=" * 40)
    lines.append(f"total_candidates={len(ranked)}")
    lines.append("")
    lines.append(f"Top {min(top_k, len(ranked))} by mean_f1:")
    for i, row in enumerate(ranked[:top_k], start=1):
        lines.append(
            f"{i:02d}. idx={row['candidate_index']} mean_f1={row['mean_f1']:.6f} "
            f"std_f1={row['std_f1']:.6f} mean_em={row['mean_em']:.6f} params={row['params']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_all(
    out_dir: Path,
    ranked: list[dict[str, Any]],
    records: list[dict[str, Any]],
    param_effect: dict[str, list[dict[str, Any]]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib is not installed; skipped plot generation."]

    notes: list[str] = []

    # 1) Mean F1 by candidate rank.
    plt.figure(figsize=(10, 4.8))
    xs = list(range(1, len(ranked) + 1))
    ys = [r["mean_f1"] for r in ranked]
    plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=3)
    plt.title("Mean F1 by Candidate Rank")
    plt.xlabel("Rank (sorted by mean F1)")
    plt.ylabel("Mean F1")
    plt.grid(alpha=0.3)
    path1 = out_dir / "rank_vs_mean_f1.png"
    plt.tight_layout()
    plt.savefig(path1, dpi=180)
    plt.close()

    # 2) Fold trend per candidate index (raw fold rows).
    by_fold: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for rec in records:
        cidx = int(rec["candidate_index"])
        fold = int(rec["fold_result"].get("fold_index", -1))
        f1 = float(rec["fold_result"].get("f1", 0.0))
        by_fold[fold].append((cidx, f1))
    plt.figure(figsize=(10, 4.8))
    for fold, pairs in sorted(by_fold.items()):
        pairs.sort(key=lambda x: x[0])
        plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker=".", linewidth=1.2, label=f"fold {fold}")
    plt.title("F1 Trend Across Candidate Index by Fold")
    plt.xlabel("Candidate Index")
    plt.ylabel("F1")
    plt.grid(alpha=0.3)
    plt.legend()
    path2 = out_dir / "fold_trends_by_candidate.png"
    plt.tight_layout()
    plt.savefig(path2, dpi=180)
    plt.close()

    # 3) Parameter value vs mean F1.
    for param_name, rows in param_effect.items():
        plt.figure(figsize=(7, 4.2))
        xs = [r["value"] for r in rows]
        ys = [r["mean_f1"] for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.6)
        plt.title(f"Mean F1 by {param_name}")
        plt.xlabel(param_name)
        plt.ylabel("Mean F1 (over fold records)")
        plt.grid(alpha=0.3)
        plot_path = out_dir / f"param_{param_name}_mean_f1.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

    notes.append(f"Saved plots to {out_dir}")
    return notes


def main() -> None:
    args = parse_args()
    input_path = Path(args.progress_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    ranked = aggregate_by_candidate(records)
    param_effect = aggregate_param_effect(records)

    best = ranked[0]
    payload = {
        "input_progress_jsonl": str(input_path),
        "num_records": len(records),
        "num_candidates": len(ranked),
        "best_candidate_index": best["candidate_index"],
        "best_params": best["params"],
        "best_mean_f1": round(best["mean_f1"], 6),
        "best_mean_em": round(best["mean_em"], 6),
        "ranked_candidates": ranked,
        "param_effect_f1": param_effect,
    }

    save_json(output_dir / "analysis_summary.json", payload)
    write_summary_txt(output_dir / "analysis_summary.txt", ranked, args.top_k)
    notes = plot_all(output_dir, ranked, records, param_effect)

    print(f"input={input_path}")
    print(f"records={len(records)}")
    print(f"candidates={len(ranked)}")
    print(f"best_candidate_index={best['candidate_index']}")
    print(f"best_mean_f1={best['mean_f1']:.6f}")
    print(f"best_mean_em={best['mean_em']:.6f}")
    print(f"output_dir={output_dir}")
    for note in notes:
        print(f"note={note}")


if __name__ == "__main__":
    main()
