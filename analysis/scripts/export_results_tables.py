#!/usr/bin/env python3
"""Export experiment results grouped by keypoint set and by descriptor.

This helper joins ``experiments``/``results``/``keypoint_sets`` and produces
readable tables for quick inspection in the terminal or notebooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys
from typing import Dict, Iterable, List, Sequence

import math
import pandas as pd

DEFAULT_METRICS: Sequence[str] = (
    "true_map_macro",
    "true_map_micro",
    "true_map_macro_with_zeros",
    "true_map_micro_with_zeros",
    "keypoint_verification_ap",
    "keypoint_retrieval_ap",
    "image_retrieval_map",
    "precision_at_1",
    "precision_at_5",
    "recall_at_1",
    "recall_at_5",
    "mean_average_precision",
    "processing_time_ms",
    "total_matches",
    "total_keypoints",
)


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "build" / "experiments.db"
    return Path(db_path).expanduser().resolve()


def _load_dataframe(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Experiment database not found: {db_path}")

    query = """
        SELECT
            e.id AS experiment_id,
            e.descriptor_type AS descriptor_name,
            e.pooling_strategy,
            e.dataset_name,
            e.parameters,
            e.keypoint_set_id,
            ks.name AS keypoint_set,
            ks.generation_method AS keypoint_generation_method,
            ks.tolerance_px AS keypoint_tolerance_px,
            r.true_map_macro,
            r.true_map_micro,
            r.true_map_macro_with_zeros,
            r.true_map_micro_with_zeros,
            r.keypoint_verification_ap,
            r.verification_viewpoint_ap,
            r.verification_illumination_ap,
            r.keypoint_retrieval_ap,
            r.retrieval_viewpoint_ap,
            r.retrieval_illumination_ap,
            r.image_retrieval_map,
            r.mean_average_precision,
            r.legacy_mean_precision,
            r.precision_at_1,
            r.precision_at_5,
            r.recall_at_1,
            r.recall_at_5,
            r.total_matches,
            r.total_keypoints,
            r.processing_time_ms,
            r.timestamp
        FROM experiments AS e
        JOIN results AS r ON e.id = r.experiment_id
        LEFT JOIN keypoint_sets AS ks ON e.keypoint_set_id = ks.id
        ORDER BY keypoint_set, descriptor_type, experiment_id
    """

    with sqlite3.connect(db_path) as conn:
        # Some legacy rows may have non-UTF8 timestamps/metadata; ignore errors.
        conn.text_factory = lambda b: b.decode(errors="ignore")
        df = pd.read_sql_query(query, conn)

    df["keypoint_set"] = df["keypoint_set"].fillna("unknown")
    df["pooling_strategy"] = df["pooling_strategy"].fillna("unknown")
    df = _enrich_parameters(df)
    return df


def _parse_kv(raw: str | None) -> Dict[str, str]:
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _enrich_parameters(df: pd.DataFrame) -> pd.DataFrame:
    params = df["parameters"].apply(_parse_kv)
    df["experiment_name"] = params.apply(lambda d: d.get("experiment_name", ""))
    df["base_descriptor_type"] = params.apply(lambda d: d.get("descriptor_type", ""))
    df["pooling_strategy"] = df["pooling_strategy"].fillna("unknown")
    return df


def _select_columns(df: pd.DataFrame, leading: Iterable[str], metrics: Sequence[str]) -> pd.DataFrame:
    columns: List[str] = list(leading) + [col for col in metrics if col in df.columns]
    return df[columns]


def _render_table(df: pd.DataFrame, fmt: str) -> str:
    if fmt == "markdown":
        return _render_markdown_table(df)
    return df.to_string(index=False)


def _render_markdown_table(df: pd.DataFrame) -> str:
    """Render a markdown table without external dependencies."""

    def _format_value(val: object) -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return ""
        if isinstance(val, float):
            return f"{val:.6f}".rstrip("0").rstrip(".")
        return str(val)

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in df.iterrows():
        cells = [_format_value(row[col]) for col in columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _write_or_print(content: str, path: Path | None) -> None:
    if path:
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path}")
    else:
        print(content)
        print()


def generate_by_experiment(df: pd.DataFrame, metrics: Sequence[str], fmt: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for exp_name, group in df.groupby("experiment_name"):
        group_sorted = group.sort_values(["keypoint_set", "descriptor_name", "experiment_id"])
        table = _select_columns(
            group_sorted,
            leading=(
                "descriptor_name",
                "base_descriptor_type",
                "keypoint_set",
                "experiment_id",
                "pooling_strategy",
            ),
            metrics=metrics,
        )
        title = exp_name if exp_name else "unknown_experiment"
        content = f"## Experiment: {title}\n\n{_render_table(table, fmt)}\n"
        sections.append((f"experiment_{title}", content))
    return sections


def generate_by_keypoint_set(df: pd.DataFrame, metrics: Sequence[str], fmt: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for keypoint_set, group in df.groupby("keypoint_set"):
        group_sorted = group.sort_values(["descriptor_name", "experiment_id"])
        table = _select_columns(
            group_sorted,
            leading=(
                "descriptor_name",
                "base_descriptor_type",
                "experiment_name",
                "experiment_id",
                "dataset_name",
                "pooling_strategy",
            ),
            metrics=metrics,
        )
        content = f"## Keypoint set: {keypoint_set}\n\n{_render_table(table, fmt)}\n"
        sections.append((f"keypoint_set_{keypoint_set}", content))
    return sections


def generate_by_descriptor(df: pd.DataFrame, metrics: Sequence[str], fmt: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for descriptor, group in df.groupby("descriptor_name"):
        group_sorted = group.sort_values(["keypoint_set", "experiment_id"])
        table = _select_columns(
            group_sorted,
            leading=(
                "keypoint_set",
                "base_descriptor_type",
                "experiment_name",
                "experiment_id",
                "dataset_name",
                "pooling_strategy",
            ),
            metrics=metrics,
        )
        content = f"## Descriptor: {descriptor}\n\n{_render_table(table, fmt)}\n"
        sections.append((f"descriptor_{descriptor}", content))
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export experiment results grouped by keypoint set and by descriptor."
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Path to experiments.db (default: build/experiments.db).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns to include.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "plain"],
        default="markdown",
        help="Table format (default: markdown).",
    )
    parser.add_argument(
        "--single-output",
        dest="single_output",
        type=Path,
        help="Write all tables into one document (markdown or plain). Overrides stdout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to write one file per table. If omitted, prints to stdout.",
    )

    args = parser.parse_args()
    db_path = _resolve_db_path(args.db_path)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(db_path)
    metrics = tuple(args.metrics)
    fmt = args.format
    by_experiment = generate_by_experiment(df, metrics, fmt)
    by_set = generate_by_keypoint_set(df, metrics, fmt)
    by_desc = generate_by_descriptor(df, metrics, fmt)

    if args.output_dir:
        for name, content in by_experiment:
            _write_or_print(content, args.output_dir / f"{name}.md")
        for name, content in by_set:
            _write_or_print(content, args.output_dir / f"{name}.md")
        for name, content in by_desc:
            _write_or_print(content, args.output_dir / f"{name}.md")
        return

    combined_parts: list[str] = ["# Results grouped by experiment\n"]
    combined_parts.extend([content for _, content in by_experiment])
    combined_parts.append("# Results grouped by keypoint set\n")
    combined_parts.extend([content for _, content in by_set])
    combined_parts.append("# Results grouped by descriptor\n")
    combined_parts.extend([content for _, content in by_desc])
    combined = "\n".join(combined_parts)

    _write_or_print(combined, args.single_output)


if __name__ == "__main__":
    main()
