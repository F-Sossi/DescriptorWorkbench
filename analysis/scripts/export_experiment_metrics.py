#!/usr/bin/env python3
"""Command-line helper to dump experiment metrics from experiments.db.

The script is deliberately lightweight so it can be used from notebooks or CI
jobs to inspect current results without hard-coding experiment names.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional

import pandas as pd

# Ensure the repository root is on sys.path so we can import ``analysis`` as a package
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.utils.db_summary import (
    DEFAULT_SUMMARY_COLUMNS,
    ExperimentFilters,
    build_summary_table,
    load_experiment_results,
)


def _parse_experiment_list(values: Optional[Iterable[str]]) -> Optional[Iterable[str]]:
    if not values:
        return None
    # Support comma-separated lists passed as single argument.
    if len(values) == 1 and "," in next(iter(values)):
        return [v.strip() for v in next(iter(values)).split(",") if v.strip()]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiment metrics from experiments.db")
    parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Optional path to experiments.db (defaults to build/experiments.db)",
    )
    parser.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        help="Filter to specific experiment name(s). Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--descriptor",
        dest="descriptors",
        action="append",
        help="Filter to specific descriptor type(s).",
    )
    parser.add_argument(
        "--columns",
        dest="columns",
        nargs="+",
        default=list(DEFAULT_SUMMARY_COLUMNS),
        help="Columns to include in the output table.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        help="Optional output file (CSV or JSON). If omitted, prints to stdout.",
    )

    args = parser.parse_args()

    experiments = _parse_experiment_list(args.experiments)
    descriptors = _parse_experiment_list(args.descriptors)

    filters = ExperimentFilters(experiment_names=experiments, descriptor_types=descriptors)
    df = load_experiment_results(db_path=args.db_path, filters=filters)
    summary = build_summary_table(df, columns=args.columns)

    if args.format == "csv":
        if not args.output:
            raise SystemExit("--output must be specified when using --format csv")
        summary.to_csv(args.output, index=False)
        print(f"Wrote {len(summary)} rows to {args.output}")
        return

    if args.format == "json":
        if args.output:
            summary.to_json(args.output, orient="records", indent=2)
            print(f"Wrote {len(summary)} rows to {args.output}")
        else:
            print(summary.to_json(orient="records", indent=2))
        return

    # Default: pretty-printed table.
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
