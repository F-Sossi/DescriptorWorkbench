"""Database summary utilities for experiment analysis.

These helpers provide a thin, dependency-free wrapper (pandas only) around
``experiments.db`` so notebooks and scripts can consume metrics without
hard-coding experiment names or descriptor assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd
import sqlite3

# Columns we expose in the default summary table.
DEFAULT_SUMMARY_COLUMNS: Sequence[str] = (
    "experiment_name",
    "experiment_id",
    "descriptor_type",
    "pooling_strategy",
    "dataset_name",
    "mean_average_precision",
    "true_map_macro",
    "true_map_macro_with_zeros",
    "true_map_micro",
    "true_map_micro_with_zeros",
    "image_retrieval_map",
    "precision_at_1",
    "precision_at_5",
    "recall_at_1",
    "recall_at_5",
    "processing_time_ms",
    "processing_time_s",
)


@dataclass(frozen=True)
class ExperimentFilters:
    """Optional filters when loading experiment data."""

    experiment_names: Optional[Iterable[str]] = None
    descriptor_types: Optional[Iterable[str]] = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df
        if self.experiment_names:
            names = {name for name in self.experiment_names}
            filtered = filtered[filtered["experiment_name"].isin(names)]
        if self.descriptor_types:
            types = {name for name in self.descriptor_types}
            filtered = filtered[filtered["descriptor_type"].isin(types)]
        return filtered


def _resolve_db_path(db_path: Optional[str | Path]) -> Path:
    """Resolve the experiments.db path, defaulting to ``build/experiments.db``."""

    if db_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "build" / "experiments.db"
    return Path(db_path).expanduser().resolve()


def _parse_kv_string(raw: Optional[str]) -> Dict[str, str]:
    """Parse ``key=value`` pairs separated by semicolons into a dict."""

    if not raw:
        return {}

    entries: Dict[str, str] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            # Some legacy metadata stores JSON-ish blobs; skip for now.
            continue
        key, value = chunk.split("=", 1)
        entries[key.strip()] = value.strip()
    return entries


def load_experiment_results(
    db_path: Optional[str | Path] = None,
    filters: Optional[ExperimentFilters] = None,
) -> pd.DataFrame:
    """Load experiment + result metrics into a DataFrame.

    Parameters
    ----------
    db_path:
        Optional custom path to ``experiments.db``. Defaults to ``build/experiments.db``.
    filters:
        Optional :class:`ExperimentFilters` to limit the rows returned.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing joined experiment and result records, enriched with
        parsed parameter/metadata dictionaries and convenience columns (e.g.,
        ``experiment_name`` and ``processing_time_s``).
    """

    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        raise FileNotFoundError(f"Experiment database not found: {database_path}")

    query = """
        SELECT
            e.id AS experiment_id,
            e.descriptor_type,
            e.pooling_strategy,
            e.dataset_name,
            e.similarity_threshold,
            e.max_features,
            e.parameters,
            r.mean_average_precision,
            r.true_map_macro,
            r.true_map_micro,
            r.true_map_macro_with_zeros,
            r.true_map_micro_with_zeros,
            r.image_retrieval_map,
            r.legacy_mean_precision,
            r.precision_at_1,
            r.precision_at_5,
            r.recall_at_1,
            r.recall_at_5,
            r.total_matches,
            r.total_keypoints,
            r.processing_time_ms,
            r.metadata
        FROM experiments AS e
        JOIN results AS r ON e.id = r.experiment_id
        ORDER BY e.id ASC
    """

    with sqlite3.connect(database_path) as conn:
        df = pd.read_sql_query(query, conn)

    # Parse parameters / metadata into explicit columns.
    param_dicts = df["parameters"].apply(_parse_kv_string)
    df["experiment_name"] = param_dicts.apply(lambda d: d.get("experiment_name", ""))
    df["pooling_strategy"] = df["pooling_strategy"].fillna("unknown")

    metadata_dicts = df["metadata"].apply(_parse_kv_string)
    for field in ["total_images", "total_keypoints", "match_time_ms", "detect_time_ms", "compute_time_ms"]:
        df[field] = metadata_dicts.apply(
            lambda d: _safe_float(d.get(field)) if d.get(field) is not None else pd.NA
        )

    df["processing_time_ms"] = df["processing_time_ms"].astype(float)
    df["processing_time_s"] = df["processing_time_ms"] / 1000.0

    # Provide consistent MAP aliases in case downstream notebooks expect them.
    df["macro_map"] = df["true_map_macro"].fillna(df["mean_average_precision"])
    df["micro_map"] = df["true_map_micro"].fillna(df["mean_average_precision"])

    if filters is not None:
        df = filters.apply(df)

    return df


def build_summary_table(
    df: pd.DataFrame,
    columns: Sequence[str] = DEFAULT_SUMMARY_COLUMNS,
    sort_by: Optional[Sequence[str]] = ("experiment_name", "descriptor_type"),
) -> pd.DataFrame:
    """Return a tidy table of the most relevant metrics."""

    available_cols = [col for col in columns if col in df.columns]
    summary = df[available_cols].copy()
    if sort_by:
        sort_columns = [col for col in sort_by if col in summary.columns]
        if sort_columns:
            summary = summary.sort_values(sort_columns)
    return summary.reset_index(drop=True)


def _safe_float(value: Optional[str]) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


__all__ = [
    "DEFAULT_SUMMARY_COLUMNS",
    "ExperimentFilters",
    "build_summary_table",
    "load_experiment_results",
]
