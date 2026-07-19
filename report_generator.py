"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Evidence Report and Graph Generator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Converts one completed benchmark run directory into:

- evidence-backed PNG graphs;
- a Markdown benchmark report;
- a report manifest with file hashes.

This module does not:
- call a model provider;
- re-run evaluators;
- alter raw benchmark evidence;
- fabricate missing scores or observations.

Default usage
-------------
python report_generator.py \
  --run results/<run-id>

Self-test
---------
python report_generator.py --self-test
==========================================================
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

except ImportError as error:
    raise SystemExit(
        "Matplotlib is required to generate graphs. "
        "Install it with: python -m pip install matplotlib"
    ) from error


# ----------------------------------------------------------
# Data Models
# ----------------------------------------------------------

@dataclass(frozen=True)
class EvaluatorSummary:
    evaluator: str
    records: int
    passed: int
    failed: int
    pass_rate: float
    mean_score: float | None


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    records: int
    passed: int
    failed: int
    pass_rate: float


@dataclass(frozen=True)
class ReportOutputs:
    report_path: Path
    manifest_path: Path
    chart_paths: tuple[Path, ...]
    skipped_charts: tuple[str, ...]


# ----------------------------------------------------------
# General Helpers
# ----------------------------------------------------------

def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} must contain a JSON object."
        )

    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read an append-only JSONL evidence file."""

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, raw_line in enumerate(
            input_file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at "
                    f"line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path} line {line_number} "
                    "must contain a JSON object."
                )

            records.append(record)

    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read the score CSV."""

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        return list(
            csv.DictReader(input_file)
        )


def safe_float(value: Any) -> float | None:
    """Convert a value to a finite float."""

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(number):
        return None

    return number


def safe_int(value: Any) -> int | None:
    """Convert a value to an integer where possible."""

    number = safe_float(value)

    if number is None:
        return None

    return int(number)


def parse_bool(value: Any) -> bool | None:
    """Parse boolean values from JSON or CSV."""

    if isinstance(value, bool):
        return value

    if value is None:
        return None

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    return None


def mean_or_none(
    values: Iterable[float],
) -> float | None:
    """Return the arithmetic mean or None."""

    materialised = list(values)

    if not materialised:
        return None

    return statistics.fmean(materialised)


def percentage_text(
    value: float | None,
) -> str:
    """Format a percentage-point value."""

    if value is None:
        return "n/a"

    return f"{value:.1f}%"


def number_text(
    value: float | None,
    decimals: int = 2,
) -> str:
    """Format a numeric value."""

    if value is None:
        return "n/a"

    return f"{value:.{decimals}f}"


def escape_markdown(value: Any) -> str:
    """Escape table-breaking Markdown characters."""

    return str(value).replace(
        "|",
        r"\|",
    ).replace(
        "\n",
        " ",
    )


def sha256_file(path: Path) -> str:
    """Return one output file's SHA-256 hash."""

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def nested_value(
    payload: Any,
    keys: Sequence[str],
) -> Any:
    """Find the first matching key in nested evidence."""

    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload:
                return payload[key]

        for value in payload.values():
            found = nested_value(
                value,
                keys,
            )

            if found is not None:
                return found

    elif isinstance(payload, list):
        for value in payload:
            found = nested_value(
                value,
                keys,
            )

            if found is not None:
                return found

    return None


# ----------------------------------------------------------
# Run Validation
# ----------------------------------------------------------

REQUIRED_RUN_FILES = (
    "run_manifest.json",
    "run_summary.json",
    "raw_provider_responses.jsonl",
    "transcripts.json",
    "evaluation_results.jsonl",
    "evaluation_scores.csv",
)


def validate_run_directory(
    run_directory: Path,
) -> None:
    """Validate the expected evidence files."""

    if not run_directory.exists():
        raise FileNotFoundError(
            f"Run directory does not exist: "
            f"{run_directory}"
        )

    if not run_directory.is_dir():
        raise ValueError(
            f"Run path is not a directory: "
            f"{run_directory}"
        )

    missing = [
        filename
        for filename in REQUIRED_RUN_FILES
        if not (
            run_directory / filename
        ).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Run directory is missing required files: "
            + ", ".join(missing)
        )


# ----------------------------------------------------------
# Aggregation
# ----------------------------------------------------------

def record_score_percentage(
    record: Mapping[str, Any],
) -> float | None:
    """
    Calculate the mean score percentage for one evaluator
    result from its configured score components.
    """

    scores = record.get(
        "scores",
        [],
    )

    if not isinstance(scores, list):
        return None

    percentages = []

    for score in scores:
        if not isinstance(score, Mapping):
            continue

        value = safe_float(
            score.get("score")
        )
        maximum = safe_float(
            score.get("maximum")
        )

        if (
            value is None
            or maximum is None
            or maximum == 0
        ):
            continue

        percentages.append(
            (value / maximum) * 100.0
        )

    return mean_or_none(percentages)


def evaluator_summaries(
    records: Sequence[Mapping[str, Any]],
) -> list[EvaluatorSummary]:
    """Aggregate pass rates and mean scores by evaluator."""

    grouped: dict[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for record in records:
        evaluator = str(
            record.get(
                "evaluator",
                "<missing>",
            )
        )
        grouped[evaluator].append(record)

    summaries = []

    for evaluator, group in grouped.items():
        passed = sum(
            1
            for record in group
            if parse_bool(
                record.get("passed")
            ) is True
        )
        failed = sum(
            1
            for record in group
            if parse_bool(
                record.get("passed")
            ) is False
        )

        scored = passed + failed

        pass_rate = (
            (passed / scored) * 100.0
            if scored
            else 0.0
        )

        mean_score = mean_or_none(
            score
            for score in (
                record_score_percentage(
                    record
                )
                for record in group
            )
            if score is not None
        )

        summaries.append(
            EvaluatorSummary(
                evaluator=evaluator,
                records=len(group),
                passed=passed,
                failed=failed,
                pass_rate=pass_rate,
                mean_score=mean_score,
            )
        )

    return sorted(
        summaries,
        key=lambda item: (
            item.pass_rate,
            item.evaluator.lower(),
        ),
    )


def session_summaries(
    records: Sequence[Mapping[str, Any]],
) -> list[SessionSummary]:
    """Aggregate evaluator-result pass rates by session."""

    grouped: dict[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for record in records:
        session_id = str(
            record.get(
                "session_id",
                "<missing>",
            )
        )
        grouped[session_id].append(record)

    summaries = []

    for session_id, group in grouped.items():
        passed = sum(
            1
            for record in group
            if parse_bool(
                record.get("passed")
            ) is True
        )
        failed = sum(
            1
            for record in group
            if parse_bool(
                record.get("passed")
            ) is False
        )

        scored = passed + failed

        pass_rate = (
            (passed / scored) * 100.0
            if scored
            else 0.0
        )

        summaries.append(
            SessionSummary(
                session_id=session_id,
                records=len(group),
                passed=passed,
                failed=failed,
                pass_rate=pass_rate,
            )
        )

    return sorted(
        summaries,
        key=lambda item: item.session_id,
    )


def observation_trajectory(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, float]]:
    """
    Return evaluator mean scores at configured observation
    states.
    """

    grouped: dict[
        str,
        dict[int, list[float]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    for record in records:
        state = safe_int(
            record.get(
                "observation_state"
            )
        )

        if state is None:
            continue

        score = record_score_percentage(
            record
        )

        if score is None:
            continue

        evaluator = str(
            record.get(
                "evaluator",
                "<missing>",
            )
        )

        grouped[evaluator][state].append(
            score
        )

    trajectory: dict[
        str,
        dict[int, float],
    ] = {}

    for evaluator, states in grouped.items():
        trajectory[evaluator] = {
            state: statistics.fmean(values)
            for state, values in states.items()
            if values
        }

    return dict(
        sorted(
            trajectory.items(),
            key=lambda item: item[0].lower(),
        )
    )


def unique_response_metrics(
    score_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Deduplicate response-level latency and token metrics
    repeated across evaluator score rows.
    """

    unique: dict[
        tuple[str, int, int],
        dict[str, Any],
    ] = {}

    for row in score_rows:
        session_id = str(
            row.get(
                "session_id",
                "<missing>",
            )
        )
        turn_number = safe_int(
            row.get(
                "turn_number"
            )
        )
        round_number = safe_int(
            row.get(
                "round_number"
            )
        )

        if (
            turn_number is None
            or round_number is None
        ):
            continue

        key = (
            session_id,
            turn_number,
            round_number,
        )

        candidate = {
            "session_id": session_id,
            "turn_number": turn_number,
            "round_number": round_number,
            "latency_seconds": safe_float(
                row.get(
                    "latency_seconds"
                )
            ),
            "input_tokens": safe_int(
                row.get(
                    "input_tokens"
                )
            ),
            "output_tokens": safe_int(
                row.get(
                    "output_tokens"
                )
            ),
        }

        existing = unique.get(key)

        if existing is None:
            unique[key] = candidate
            continue

        for metric in (
            "latency_seconds",
            "input_tokens",
            "output_tokens",
        ):
            if (
                existing.get(metric) is None
                and candidate.get(metric) is not None
            ):
                existing[metric] = candidate[metric]

    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item["round_number"],
            item["session_id"],
            item["turn_number"],
        ),
    )

    for index, item in enumerate(
        ordered,
        start=1,
    ):
        item["execution_index"] = index

    return ordered


# ----------------------------------------------------------
# Chart Helpers
# ----------------------------------------------------------

def prepare_figure(
    title: str,
    subtitle: str,
    *,
    width: float = 12.0,
    height: float = 7.0,
) -> tuple[Any, Any]:
    """Create one independent accessible figure."""

    figure, axis = plt.subplots(
        figsize=(
            width,
            height,
        )
    )

    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    figure.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    axis.set_title(
        subtitle,
        fontsize=10,
        pad=14,
    )

    axis.tick_params(
        labelsize=10,
    )

    return figure, axis


def save_figure(
    figure: Any,
    path: Path,
) -> None:
    """Write and close one chart."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.94,
        )
    )

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)


def annotate_horizontal_bars(
    axis: Any,
    values: Sequence[float],
) -> None:
    """Add percentage labels to horizontal bars."""

    for index, value in enumerate(values):
        axis.text(
            min(
                value + 1.0,
                98.0,
            ),
            index,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )


def chart_evaluator_pass_rate(
    summaries: Sequence[EvaluatorSummary],
    path: Path,
) -> bool:
    """Generate evaluator pass-rate chart."""

    if not summaries:
        return False

    labels = [
        item.evaluator
        for item in summaries
    ]
    values = [
        item.pass_rate
        for item in summaries
    ]

    figure, axis = prepare_figure(
        "Evaluator pass rate",
        (
            "Percentage of evaluator result records marked "
            "passed. This reports stored evaluator outputs; "
            "it does not independently validate them."
        ),
        height=max(
            6.5,
            len(labels) * 0.65,
        ),
    )

    positions = range(len(labels))

    axis.barh(
        positions,
        values,
    )
    axis.set_yticks(
        list(positions),
        labels=labels,
    )
    axis.set_xlim(0, 100)
    axis.set_xlabel("Pass rate (%)")
    axis.grid(
        axis="x",
        alpha=0.25,
    )

    annotate_horizontal_bars(
        axis,
        values,
    )

    save_figure(
        figure,
        path,
    )

    return True


def chart_evaluator_mean_score(
    summaries: Sequence[EvaluatorSummary],
    path: Path,
) -> bool:
    """Generate evaluator mean-score chart."""

    available = [
        item
        for item in summaries
        if item.mean_score is not None
    ]

    if not available:
        return False

    labels = [
        item.evaluator
        for item in available
    ]
    values = [
        float(item.mean_score)
        for item in available
        if item.mean_score is not None
    ]

    figure, axis = prepare_figure(
        "Evaluator mean score",
        (
            "Mean percentage across each evaluator result's "
            "configured score components."
        ),
        height=max(
            6.5,
            len(labels) * 0.65,
        ),
    )

    positions = range(len(labels))

    axis.barh(
        positions,
        values,
    )
    axis.set_yticks(
        list(positions),
        labels=labels,
    )
    axis.set_xlim(0, 100)
    axis.set_xlabel("Mean score (%)")
    axis.grid(
        axis="x",
        alpha=0.25,
    )

    annotate_horizontal_bars(
        axis,
        values,
    )

    save_figure(
        figure,
        path,
    )

    return True


def chart_session_pass_rate(
    summaries: Sequence[SessionSummary],
    path: Path,
) -> bool:
    """Generate session pass-rate chart."""

    if not summaries:
        return False

    labels = [
        item.session_id
        for item in summaries
    ]
    values = [
        item.pass_rate
        for item in summaries
    ]

    figure, axis = prepare_figure(
        "Pass rate by benchmark session",
        (
            "Share of stored evaluator result records marked "
            "passed within each continuing conversation."
        ),
    )

    positions = range(len(labels))

    axis.bar(
        positions,
        values,
    )
    axis.set_xticks(
        list(positions),
        labels=labels,
        rotation=20,
        ha="right",
    )
    axis.set_ylim(0, 100)
    axis.set_ylabel("Pass rate (%)")
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for index, value in enumerate(values):
        axis.text(
            index,
            min(
                value + 1.5,
                98.0,
            ),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(
        figure,
        path,
    )

    return True


def chart_observation_trajectory(
    trajectory: Mapping[
        str,
        Mapping[int, float],
    ],
    path: Path,
) -> bool:
    """Generate evaluator trajectories at observation states."""

    if not trajectory:
        return False

    all_states = sorted(
        {
            state
            for states in trajectory.values()
            for state in states
        }
    )

    if not all_states:
        return False

    figure, axis = prepare_figure(
        "Evaluator score trajectory",
        (
            "Mean evaluator result score at the configured "
            "observation states."
        ),
        width=13.0,
        height=8.0,
    )

    plotted = False

    for evaluator, states in trajectory.items():
        x_values = []
        y_values = []

        for state in all_states:
            if state not in states:
                continue

            x_values.append(state)
            y_values.append(states[state])

        if not x_values:
            continue

        axis.plot(
            x_values,
            y_values,
            marker="o",
            label=evaluator,
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        return False

    axis.set_xticks(all_states)
    axis.set_xlim(
        min(all_states) - 0.25,
        max(all_states) + 0.25,
    )
    axis.set_ylim(0, 100)
    axis.set_xlabel("Observation state")
    axis.set_ylabel("Mean score (%)")
    axis.grid(alpha=0.25)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(
            1.01,
            0.5,
        ),
        fontsize=9,
    )

    save_figure(
        figure,
        path,
    )

    return True


def chart_latency_by_session(
    metrics: Sequence[Mapping[str, Any]],
    path: Path,
) -> bool:
    """Generate response-latency trajectories."""

    grouped: dict[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for item in metrics:
        if item.get(
            "latency_seconds"
        ) is None:
            continue

        grouped[
            str(item["session_id"])
        ].append(item)

    if not grouped:
        return False

    figure, axis = prepare_figure(
        "Response latency by turn",
        (
            "Provider latency for each response in the four "
            "continuing-conversation sessions."
        ),
        width=13.0,
        height=7.5,
    )

    for session_id, items in sorted(
        grouped.items()
    ):
        ordered = sorted(
            items,
            key=lambda item: item[
                "turn_number"
            ],
        )

        axis.plot(
            [
                item["turn_number"]
                for item in ordered
            ],
            [
                item["latency_seconds"]
                for item in ordered
            ],
            marker="o",
            label=session_id,
        )

    axis.set_xticks(
        range(1, 10)
    )
    axis.set_xlabel("Turn number")
    axis.set_ylabel("Latency (seconds)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    save_figure(
        figure,
        path,
    )

    return True


def chart_token_usage(
    metrics: Sequence[Mapping[str, Any]],
    path: Path,
) -> bool:
    """Generate prompt and completion token trajectories."""

    usable = [
        item
        for item in metrics
        if (
            item.get(
                "input_tokens"
            ) is not None
            or item.get(
                "output_tokens"
            ) is not None
        )
    ]

    if not usable:
        return False

    figure, axis = prepare_figure(
        "Token usage across benchmark execution",
        (
            "Prompt and completion token counts for each "
            "stored model response."
        ),
        width=13.0,
        height=7.5,
    )

    x_values = [
        item["execution_index"]
        for item in usable
    ]

    input_values = [
        item.get(
            "input_tokens"
        )
        for item in usable
    ]
    output_values = [
        item.get(
            "output_tokens"
        )
        for item in usable
    ]

    if any(
        value is not None
        for value in input_values
    ):
        axis.plot(
            x_values,
            [
                value
                if value is not None
                else math.nan
                for value in input_values
            ],
            marker="o",
            label="Prompt tokens",
        )

    if any(
        value is not None
        for value in output_values
    ):
        axis.plot(
            x_values,
            [
                value
                if value is not None
                else math.nan
                for value in output_values
            ],
            marker="o",
            label="Completion tokens",
        )

    axis.set_xlabel("Response execution index")
    axis.set_ylabel("Tokens")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    save_figure(
        figure,
        path,
    )

    return True


# ----------------------------------------------------------
# Markdown Report
# ----------------------------------------------------------

def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> str:
    """Build one Markdown table."""

    lines = [
        "| "
        + " | ".join(
            escape_markdown(header)
            for header in headers
        )
        + " |",
        "| "
        + " | ".join(
            "---"
            for _ in headers
        )
        + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                escape_markdown(value)
                for value in row
            )
            + " |"
        )

    return "\n".join(lines)


def build_markdown_report(
    *,
    run_directory: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    raw_response_count: int,
    evaluation_count: int,
    score_row_count: int,
    evaluator_data: Sequence[EvaluatorSummary],
    session_data: Sequence[SessionSummary],
    observation_data: Mapping[
        str,
        Mapping[int, float],
    ],
    chart_paths: Sequence[Path],
    skipped_charts: Sequence[str],
) -> str:
    """Create the readable benchmark report."""

    run_id = summary.get(
        "run_id",
        manifest.get(
            "run_id",
            run_directory.name,
        ),
    )

    evaluator_rows = [
        (
            item.evaluator,
            item.records,
            item.passed,
            item.failed,
            percentage_text(
                item.pass_rate
            ),
            percentage_text(
                item.mean_score
            ),
        )
        for item in evaluator_data
    ]

    session_rows = [
        (
            item.session_id,
            item.records,
            item.passed,
            item.failed,
            percentage_text(
                item.pass_rate
            ),
        )
        for item in session_data
    ]

    states = sorted(
        {
            state
            for state_map in observation_data.values()
            for state in state_map
        }
    )

    observation_rows = []

    for evaluator, state_map in observation_data.items():
        observation_rows.append(
            (
                evaluator,
                *(
                    percentage_text(
                        state_map.get(state)
                    )
                    for state in states
                ),
            )
        )

    lines = [
        "# GenAI Evaluation Matrix Benchmark Report",
        "",
        f"**Run ID:** `{run_id}`  ",
        f"**Generated:** `{utc_timestamp()}`  ",
        f"**Provider:** `{summary.get('provider', manifest.get('provider', 'unknown'))}`  ",
        f"**Model:** `{summary.get('model', manifest.get('model', 'unknown'))}`  ",
        f"**Run status:** `{summary.get('status', 'unknown')}`",
        "",
        "## Evidence integrity",
        "",
        "This report is generated only from the stored run evidence. "
        "It does not call a provider, rerun an evaluator, verify external "
        "facts, or fill missing values. The charts visualise the evaluator "
        "outputs present in this run; evaluator correctness must be audited "
        "separately.",
        "",
        markdown_table(
            (
                "Evidence item",
                "Count",
            ),
            (
                (
                    "Raw provider responses",
                    raw_response_count,
                ),
                (
                    "Evaluator result records",
                    evaluation_count,
                ),
                (
                    "Evaluator score rows",
                    score_row_count,
                ),
                (
                    "Completed model responses",
                    summary.get(
                        "successful_model_responses",
                        summary.get(
                            "completed_turns",
                            "unknown",
                        ),
                    ),
                ),
                (
                    "Failed model responses",
                    summary.get(
                        "failed_turns",
                        "unknown",
                    ),
                ),
                (
                    "Evaluator execution errors",
                    summary.get(
                        "evaluator_errors_written",
                        "unknown",
                    ),
                ),
            ),
        ),
        "",
        "## Evaluator results",
        "",
        markdown_table(
            (
                "Evaluator",
                "Records",
                "Passed",
                "Failed",
                "Pass rate",
                "Mean score",
            ),
            evaluator_rows,
        ),
        "",
        "## Session results",
        "",
        markdown_table(
            (
                "Session",
                "Evaluator records",
                "Passed",
                "Failed",
                "Pass rate",
            ),
            session_rows,
        ),
    ]

    if states and observation_rows:
        lines.extend(
            [
                "",
                "## Observation-state trajectory",
                "",
                markdown_table(
                    (
                        "Evaluator",
                        *(
                            f"State {state}"
                            for state in states
                        ),
                    ),
                    observation_rows,
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Generated graphs",
            "",
        ]
    )

    if chart_paths:
        for chart_path in chart_paths:
            relative_path = chart_path.relative_to(
                run_directory
            )

            title = chart_path.stem.replace(
                "_",
                " ",
            ).title()

            lines.extend(
                [
                    f"### {title}",
                    "",
                    f"![{title}]({relative_path.as_posix()})",
                    "",
                ]
            )
    else:
        lines.append(
            "No graphs could be generated from the available data."
        )

    if skipped_charts:
        lines.extend(
            [
                "",
                "## Skipped graphs",
                "",
            ]
        )

        for reason in skipped_charts:
            lines.append(
                f"- {reason}"
            )

    lines.extend(
        [
            "",
            "## Source evidence files",
            "",
            *(
                f"- `{filename}`"
                for filename in REQUIRED_RUN_FILES
            ),
            "",
        ]
    )

    return "\n".join(lines)


# ----------------------------------------------------------
# Complete Report Generation
# ----------------------------------------------------------

def generate_report(
    run_directory: Path,
) -> ReportOutputs:
    """Generate graphs, report and report manifest."""

    run_directory = (
        run_directory
        .expanduser()
        .resolve()
    )

    validate_run_directory(
        run_directory
    )

    run_manifest = read_json(
        run_directory
        / "run_manifest.json"
    )
    run_summary = read_json(
        run_directory
        / "run_summary.json"
    )
    raw_responses = read_jsonl(
        run_directory
        / "raw_provider_responses.jsonl"
    )
    evaluation_records = read_jsonl(
        run_directory
        / "evaluation_results.jsonl"
    )
    score_rows = read_csv(
        run_directory
        / "evaluation_scores.csv"
    )

    evaluator_data = evaluator_summaries(
        evaluation_records
    )
    session_data = session_summaries(
        evaluation_records
    )
    observation_data = observation_trajectory(
        evaluation_records
    )
    response_metrics = unique_response_metrics(
        score_rows
    )

    charts_directory = (
        run_directory
        / "charts"
    )
    charts_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    chart_jobs = (
        (
            "Evaluator pass-rate chart lacked evaluator records.",
            charts_directory
            / "01_evaluator_pass_rate.png",
            lambda path: chart_evaluator_pass_rate(
                evaluator_data,
                path,
            ),
        ),
        (
            "Evaluator mean-score chart lacked numeric scores.",
            charts_directory
            / "02_evaluator_mean_score.png",
            lambda path: chart_evaluator_mean_score(
                evaluator_data,
                path,
            ),
        ),
        (
            "Session pass-rate chart lacked session results.",
            charts_directory
            / "03_session_pass_rate.png",
            lambda path: chart_session_pass_rate(
                session_data,
                path,
            ),
        ),
        (
            "Observation trajectory lacked scored observation states.",
            charts_directory
            / "04_observation_state_trajectory.png",
            lambda path: chart_observation_trajectory(
                observation_data,
                path,
            ),
        ),
        (
            "Latency chart lacked response latency values.",
            charts_directory
            / "05_response_latency.png",
            lambda path: chart_latency_by_session(
                response_metrics,
                path,
            ),
        ),
        (
            "Token chart lacked prompt or completion token values.",
            charts_directory
            / "06_token_usage.png",
            lambda path: chart_token_usage(
                response_metrics,
                path,
            ),
        ),
    )

    generated_charts: list[Path] = []
    skipped_charts: list[str] = []

    for (
        skip_reason,
        chart_path,
        chart_function,
    ) in chart_jobs:
        generated = chart_function(
            chart_path
        )

        if generated:
            generated_charts.append(
                chart_path
            )
        else:
            skipped_charts.append(
                skip_reason
            )

    report_path = (
        run_directory
        / "benchmark_report.md"
    )

    report_text = build_markdown_report(
        run_directory=run_directory,
        manifest=run_manifest,
        summary=run_summary,
        raw_response_count=len(
            raw_responses
        ),
        evaluation_count=len(
            evaluation_records
        ),
        score_row_count=len(
            score_rows
        ),
        evaluator_data=evaluator_data,
        session_data=session_data,
        observation_data=observation_data,
        chart_paths=generated_charts,
        skipped_charts=skipped_charts,
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    manifest_path = (
        run_directory
        / "report_manifest.json"
    )

    report_manifest = {
        "run_id": run_summary.get(
            "run_id",
            run_manifest.get(
                "run_id",
                run_directory.name,
            ),
        ),
        "generated_at": utc_timestamp(),
        "source_run_directory": str(
            run_directory
        ),
        "source_evidence": {
            filename: {
                "path": str(
                    run_directory
                    / filename
                ),
                "sha256": sha256_file(
                    run_directory
                    / filename
                ),
            }
            for filename in REQUIRED_RUN_FILES
        },
        "report": {
            "path": str(
                report_path
            ),
            "sha256": sha256_file(
                report_path
            ),
        },
        "charts_generated": len(
            generated_charts
        ),
        "charts": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in generated_charts
        ],
        "skipped_charts": skipped_charts,
        "generator": {
            "file": "report_generator.py",
            "scope": (
                "stored evidence visualisation only"
            ),
            "provider_called": False,
            "evaluators_rerun": False,
        },
    }

    manifest_path.write_text(
        json.dumps(
            report_manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return ReportOutputs(
        report_path=report_path,
        manifest_path=manifest_path,
        chart_paths=tuple(
            generated_charts
        ),
        skipped_charts=tuple(
            skipped_charts
        ),
    )


# ----------------------------------------------------------
# Self-Test
# ----------------------------------------------------------

def write_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Write temporary self-test JSONL."""

    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def run_self_test() -> None:
    """Generate a complete temporary synthetic report."""

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="gaiem_report_test_"
        )
    )

    try:
        run_directory = (
            temporary_root
            / "synthetic_run"
        )
        run_directory.mkdir(
            parents=True
        )

        (
            run_directory
            / "run_manifest.json"
        ).write_text(
            json.dumps(
                {
                    "run_id": "synthetic_run",
                    "provider": "test",
                    "model": "test-model",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            run_directory
            / "run_summary.json"
        ).write_text(
            json.dumps(
                {
                    "run_id": "synthetic_run",
                    "status": "completed",
                    "provider": "test",
                    "model": "test-model",
                    "completed_turns": 4,
                    "failed_turns": 0,
                    "successful_model_responses": 4,
                    "evaluator_errors_written": 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        write_jsonl(
            run_directory
            / "raw_provider_responses.jsonl",
            [
                {
                    "session_id": "SESSION-A",
                    "turn_number": turn,
                    "latency_seconds": float(turn),
                }
                for turn in range(
                    1,
                    5,
                )
            ],
        )

        (
            run_directory
            / "transcripts.json"
        ).write_text(
            json.dumps(
                {
                    "run_id": "synthetic_run",
                    "sessions": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        evaluation_records = []

        for turn in range(1, 5):
            for evaluator in (
                "Evaluator A",
                "Evaluator B",
            ):
                passed = (
                    turn % 2 == 0
                    if evaluator == "Evaluator A"
                    else turn != 3
                )

                evaluation_records.append(
                    {
                        "run_id": "synthetic_run",
                        "session_id": "SESSION-A",
                        "turn_number": turn,
                        "round_number": 1,
                        "observation_state": turn,
                        "evaluator": evaluator,
                        "passed": passed,
                        "scores": [
                            {
                                "name": evaluator,
                                "score": (
                                    1.0
                                    if passed
                                    else 0.0
                                ),
                                "maximum": 1.0,
                                "passed": passed,
                            }
                        ],
                    }
                )

        write_jsonl(
            run_directory
            / "evaluation_results.jsonl",
            evaluation_records,
        )

        fieldnames = (
            "run_id",
            "provider",
            "model",
            "session_id",
            "turn_number",
            "case_id",
            "round_number",
            "checkpoint",
            "observation_state",
            "evaluator",
            "scope",
            "score_name",
            "score",
            "maximum",
            "percentage",
            "passed",
            "current_turn_passed",
            "conversation_passed",
            "latency_seconds",
            "input_tokens",
            "output_tokens",
            "detected_claims",
            "missing_facts",
            "lost_protected_terms",
            "newly_introduced_claims",
            "failing_turns",
        )

        with (
            run_directory
            / "evaluation_scores.csv"
        ).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for turn in range(1, 5):
                writer.writerow(
                    {
                        "run_id": "synthetic_run",
                        "provider": "test",
                        "model": "test-model",
                        "session_id": "SESSION-A",
                        "turn_number": turn,
                        "case_id": f"CASE-{turn}",
                        "round_number": 1,
                        "checkpoint": True,
                        "observation_state": turn,
                        "evaluator": "Evaluator A",
                        "scope": "turn",
                        "score_name": "Evaluator A",
                        "score": 1.0,
                        "maximum": 1.0,
                        "percentage": 1.0,
                        "passed": True,
                        "latency_seconds": float(turn),
                        "input_tokens": turn * 10,
                        "output_tokens": turn * 5,
                    }
                )

        outputs = generate_report(
            run_directory
        )

        required_outputs = (
            outputs.report_path,
            outputs.manifest_path,
            *outputs.chart_paths,
        )

        if len(
            outputs.chart_paths
        ) != 6:
            raise RuntimeError(
                "Self-test expected six generated graphs; "
                f"found {len(outputs.chart_paths)}."
            )

        missing = [
            str(path)
            for path in required_outputs
            if not path.is_file()
        ]

        if missing:
            raise RuntimeError(
                "Self-test outputs missing: "
                + ", ".join(missing)
            )

        print(
            "Report generator self-test: passed"
        )
        print(
            "No benchmark model responses were generated."
        )
        print(
            "No self-test output directory was retained."
        )

    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="report_generator.py",
        description=(
            "Generate evidence-backed graphs and a Markdown "
            "report from one completed GAIEM run."
        ),
    )

    parser.add_argument(
        "--run",
        type=Path,
        help=(
            "Completed result directory, for example "
            "results/<run-id>."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run a temporary synthetic self-test without "
            "using benchmark responses."
        ),
    )

    return parser


def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.self_test:
        if arguments.run is not None:
            parser.error(
                "--run cannot be combined with --self-test."
            )

        run_self_test()
        return 0

    if arguments.run is None:
        parser.error(
            "--run is required unless --self-test is used."
        )

    try:
        outputs = generate_report(
            arguments.run
        )

    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"Error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print("=" * 72)
    print("GAIEM Report Generation Complete")
    print("=" * 72)
    print(f"Report   : {outputs.report_path}")
    print(f"Manifest : {outputs.manifest_path}")
    print(f"Graphs   : {len(outputs.chart_paths)}")

    for chart_path in outputs.chart_paths:
        print(f"  - {chart_path}")

    if outputs.skipped_charts:
        print("Skipped:")

        for reason in outputs.skipped_charts:
            print(f"  - {reason}")

    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
