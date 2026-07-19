"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Clinical Candidate Selector

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Rank audited MTS-Dialog records for a chosen clinical
scenario without modifying the source data.

The selector reads:
    data/clinical/processed/mts_dialog/
    accepted_candidates.jsonl

It writes:
    data/clinical/review/mts_dialog_triage/
    ranked_candidates.jsonl
    ranked_candidates.csv
    selection_summary.json

The selector does not:
- diagnose patients;
- generate benchmark prompts;
- call a model;
- rewrite source wording;
- include rejected source records.

Default scenario
----------------
Chest-pain triage with associated red-flag language.

Self-test
---------
python clinical_candidate_selector.py --self-test
==========================================================
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


# ----------------------------------------------------------
# Defaults
# ----------------------------------------------------------

DEFAULT_INPUT = Path(
    "data/clinical/processed/mts_dialog/"
    "accepted_candidates.jsonl"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/clinical/review/mts_dialog_triage"
)

DEFAULT_QUERY_TERMS = (
    "chest pain",
    "chest pressure",
    "chest tightness",
    "shortness of breath",
    "breathless",
    "difficulty breathing",
    "dizziness",
    "dizzy",
    "warfarin",
    "blood thinner",
    "anticoagulant",
    "penicillin",
    "allergy",
    "worse with walking",
    "worse on exertion",
    "exertion",
    "indigestion",
    "heartburn",
)

RED_FLAG_TERMS = {
    "chest pain",
    "chest pressure",
    "chest tightness",
    "shortness of breath",
    "breathless",
    "difficulty breathing",
    "dizziness",
    "dizzy",
    "warfarin",
    "blood thinner",
    "anticoagulant",
    "worse with walking",
    "worse on exertion",
    "exertion",
}

CONVERSATIONAL_MARKERS = {
    "i have",
    "i've",
    "i am",
    "i'm",
    "it started",
    "when did",
    "how long",
    "what happened",
    "tell me",
    "do you",
    "are you",
    "patient:",
    "doctor:",
}


# ----------------------------------------------------------
# Data Models
# ----------------------------------------------------------

@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    score: float
    source_id: str
    source_row_number: int
    section_header: str
    matched_terms: tuple[str, ...]
    red_flag_matches: tuple[str, ...]
    patient_turn_count: int
    clinician_turn_count: int
    dialogue_turn_count: int
    dialogue_preview: str
    section_text_preview: str
    source_file: str
    source_sha256: str


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def normalise(value: Any) -> str:
    """Normalise text for deterministic matching."""

    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def lower_text(value: Any) -> str:
    """Return normalised lowercase text."""

    return normalise(value).lower()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read audited accepted-candidate JSONL."""

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
                    f"Invalid JSON at {path}:"
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number} must "
                    "contain a JSON object."
                )

            if record.get("accepted") is not True:
                raise ValueError(
                    f"{path}:{line_number} contains "
                    "a record not marked accepted."
                )

            records.append(record)

    return records


def token_count(text: str) -> int:
    """Count simple word tokens."""

    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
        )
    )


def matched_terms(
    text: str,
    terms: Sequence[str],
) -> tuple[str, ...]:
    """Return query terms found in text."""

    found = [
        term
        for term in terms
        if term.lower() in text
    ]

    return tuple(
        dict.fromkeys(found)
    )


def count_roles(
    turns: Any,
) -> tuple[int, int, int]:
    """Count patient, clinician and total parsed turns."""

    if not isinstance(turns, list):
        return 0, 0, 0

    patient = 0
    clinician = 0

    for turn in turns:
        if not isinstance(turn, Mapping):
            continue

        role = lower_text(
            turn.get("role")
        )

        if role == "patient":
            patient += 1
        elif role == "clinician":
            clinician += 1

    return patient, clinician, len(turns)


def score_record(
    record: Mapping[str, Any],
    *,
    terms: Sequence[str],
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    """
    Score one record using transparent lexical criteria.

    The weighting rewards:
    - exact clinical query-term matches;
    - multiple red-flag matches;
    - both patient and clinician turns;
    - conversational markers;
    - sufficient but not excessive dialogue length.
    """

    dialogue = lower_text(
        record.get("dialogue_raw")
    )
    section_text = lower_text(
        record.get("section_text")
    )
    combined = f"{dialogue} {section_text}"

    found = matched_terms(
        combined,
        terms,
    )

    red_flags = tuple(
        term
        for term in found
        if term in RED_FLAG_TERMS
    )

    patient_turns, clinician_turns, total_turns = (
        count_roles(
            record.get("turns")
        )
    )

    conversation_hits = sum(
        1
        for marker in CONVERSATIONAL_MARKERS
        if marker in dialogue
    )

    dialogue_words = token_count(
        dialogue
    )

    score = 0.0
    score += len(found) * 4.0
    score += len(red_flags) * 3.0
    score += min(
        patient_turns,
        4,
    ) * 1.5
    score += min(
        clinician_turns,
        4,
    ) * 1.5
    score += min(
        conversation_hits,
        6,
    ) * 0.75

    if 20 <= dialogue_words <= 500:
        score += 2.0

    if total_turns >= 4:
        score += 2.0

    if not found:
        score = 0.0

    return score, found, red_flags


def preview(
    value: Any,
    *,
    maximum: int = 500,
) -> str:
    """Return a bounded one-line preview."""

    text = normalise(value)

    if len(text) <= maximum:
        return text

    return text[: maximum - 1] + "…"


# ----------------------------------------------------------
# Ranking
# ----------------------------------------------------------

def rank_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    terms: Sequence[str],
    minimum_score: float,
    limit: int,
) -> list[RankedCandidate]:
    """Rank clinically relevant records."""

    provisional: list[
        tuple[
            float,
            Mapping[str, Any],
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = []

    for record in records:
        score, found, red_flags = score_record(
            record,
            terms=terms,
        )

        if score < minimum_score:
            continue

        provisional.append(
            (
                score,
                record,
                found,
                red_flags,
            )
        )

    provisional.sort(
        key=lambda item: (
            -item[0],
            str(
                item[1].get(
                    "source_id",
                    "",
                )
            ),
        )
    )

    output: list[RankedCandidate] = []

    for rank, (
        score,
        record,
        found,
        red_flags,
    ) in enumerate(
        provisional[:limit],
        start=1,
    ):
        patient_turns, clinician_turns, total_turns = (
            count_roles(
                record.get("turns")
            )
        )

        output.append(
            RankedCandidate(
                rank=rank,
                score=round(score, 3),
                source_id=str(
                    record.get(
                        "source_id",
                        "",
                    )
                ),
                source_row_number=int(
                    record.get(
                        "source_row_number",
                        0,
                    )
                ),
                section_header=str(
                    record.get(
                        "section_header",
                        "",
                    )
                ),
                matched_terms=found,
                red_flag_matches=red_flags,
                patient_turn_count=patient_turns,
                clinician_turn_count=clinician_turns,
                dialogue_turn_count=total_turns,
                dialogue_preview=preview(
                    record.get(
                        "dialogue_raw"
                    )
                ),
                section_text_preview=preview(
                    record.get(
                        "section_text"
                    )
                ),
                source_file=str(
                    record.get(
                        "source_file",
                        "",
                    )
                ),
                source_sha256=str(
                    record.get(
                        "source_sha256",
                        "",
                    )
                ),
            )
        )

    return output


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

def candidate_to_dict(
    candidate: RankedCandidate,
) -> dict[str, Any]:
    """Convert a candidate to JSON-safe data."""

    return {
        "rank": candidate.rank,
        "score": candidate.score,
        "source_id": candidate.source_id,
        "source_row_number": (
            candidate.source_row_number
        ),
        "section_header": (
            candidate.section_header
        ),
        "matched_terms": list(
            candidate.matched_terms
        ),
        "red_flag_matches": list(
            candidate.red_flag_matches
        ),
        "patient_turn_count": (
            candidate.patient_turn_count
        ),
        "clinician_turn_count": (
            candidate.clinician_turn_count
        ),
        "dialogue_turn_count": (
            candidate.dialogue_turn_count
        ),
        "dialogue_preview": (
            candidate.dialogue_preview
        ),
        "section_text_preview": (
            candidate.section_text_preview
        ),
        "source_file": candidate.source_file,
        "source_sha256": (
            candidate.source_sha256
        ),
    }


def write_jsonl(
    path: Path,
    candidates: Iterable[RankedCandidate],
) -> None:
    """Write ranked JSONL."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for candidate in candidates:
            output_file.write(
                json.dumps(
                    candidate_to_dict(
                        candidate
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_csv(
    path: Path,
    candidates: Sequence[RankedCandidate],
) -> None:
    """Write a review-friendly CSV."""

    fieldnames = [
        "rank",
        "score",
        "source_id",
        "source_row_number",
        "section_header",
        "matched_terms",
        "red_flag_matches",
        "patient_turn_count",
        "clinician_turn_count",
        "dialogue_turn_count",
        "dialogue_preview",
        "section_text_preview",
        "source_file",
        "source_sha256",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for candidate in candidates:
            row = candidate_to_dict(
                candidate
            )
            row["matched_terms"] = "; ".join(
                candidate.matched_terms
            )
            row["red_flag_matches"] = "; ".join(
                candidate.red_flag_matches
            )
            writer.writerow(row)


def select_candidates(
    *,
    input_path: Path,
    output_directory: Path,
    terms: Sequence[str],
    minimum_score: float,
    limit: int,
) -> dict[str, Any]:
    """Run the complete selection pipeline."""

    input_path = (
        input_path
        .expanduser()
        .resolve()
    )
    output_directory = (
        output_directory
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Accepted candidate file not found: "
            f"{input_path}"
        )

    records = read_jsonl(
        input_path
    )

    ranked = rank_candidates(
        records,
        terms=terms,
        minimum_score=minimum_score,
        limit=limit,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonl_path = (
        output_directory
        / "ranked_candidates.jsonl"
    )
    csv_path = (
        output_directory
        / "ranked_candidates.csv"
    )
    summary_path = (
        output_directory
        / "selection_summary.json"
    )

    write_jsonl(
        jsonl_path,
        ranked,
    )
    write_csv(
        csv_path,
        ranked,
    )

    summary = {
        "source": str(input_path),
        "records_available": len(records),
        "records_selected": len(ranked),
        "minimum_score": minimum_score,
        "limit": limit,
        "query_terms": list(terms),
        "selection_method": (
            "deterministic lexical ranking"
        ),
        "model_calls_made": False,
        "source_text_rewritten": False,
        "outputs": {
            "ranked_candidates_jsonl": str(
                jsonl_path
            ),
            "ranked_candidates_csv": str(
                csv_path
            ),
            "selection_summary": str(
                summary_path
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


# ----------------------------------------------------------
# Self-Test
# ----------------------------------------------------------

def run_self_test() -> None:
    """Run an isolated synthetic ranking test."""

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="gaiem_candidate_selector_"
        )
    )

    try:
        input_path = (
            temporary_root
            / "accepted_candidates.jsonl"
        )
        output_directory = (
            temporary_root
            / "review"
        )

        records = [
            {
                "accepted": True,
                "source_id": "high",
                "source_row_number": 2,
                "section_header": "GENHX",
                "section_text": (
                    "Chest pain with shortness "
                    "of breath on exertion."
                ),
                "dialogue_raw": (
                    "Patient: I have chest pain "
                    "and feel breathless when I "
                    "walk. Doctor: When did it "
                    "start?"
                ),
                "turns": [
                    {
                        "role": "patient",
                        "content": "I have chest pain.",
                    },
                    {
                        "role": "clinician",
                        "content": "When did it start?",
                    },
                ],
                "source_file": "sample.csv",
                "source_sha256": "abc",
            },
            {
                "accepted": True,
                "source_id": "low",
                "source_row_number": 3,
                "section_header": "GENHX",
                "section_text": (
                    "Routine skin review."
                ),
                "dialogue_raw": (
                    "Patient: My skin is dry. "
                    "Doctor: How long?"
                ),
                "turns": [
                    {
                        "role": "patient",
                        "content": "My skin is dry.",
                    },
                    {
                        "role": "clinician",
                        "content": "How long?",
                    },
                ],
                "source_file": "sample.csv",
                "source_sha256": "abc",
            },
        ]

        with input_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record)
                    + "\n"
                )

        summary = select_candidates(
            input_path=input_path,
            output_directory=output_directory,
            terms=DEFAULT_QUERY_TERMS,
            minimum_score=1.0,
            limit=10,
        )

        if (
            summary["records_available"] != 2
            or summary["records_selected"] != 1
        ):
            raise RuntimeError(
                "Candidate selector self-test "
                "counts were incorrect."
            )

        required = (
            output_directory
            / "ranked_candidates.jsonl",
            output_directory
            / "ranked_candidates.csv",
            output_directory
            / "selection_summary.json",
        )

        missing = [
            str(path)
            for path in required
            if not path.is_file()
        ]

        if missing:
            raise RuntimeError(
                "Candidate selector self-test "
                "outputs missing: "
                + ", ".join(missing)
            )

        print(
            "Clinical candidate selector "
            "self-test: passed"
        )
        print(
            "No model calls were made."
        )
        print(
            "No self-test output directory "
            "was retained."
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
    """Build command-line options."""

    parser = argparse.ArgumentParser(
        prog="clinical_candidate_selector.py",
        description=(
            "Rank audited MTS-Dialog records for "
            "clinical benchmark review."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Accepted candidate JSONL. Default: "
            "processed MTS-Dialog candidates."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Review output directory."
        ),
    )

    parser.add_argument(
        "--term",
        action="append",
        dest="terms",
        help=(
            "Clinical query term. Repeat to "
            "supply multiple terms. Defaults "
            "to chest-pain triage terms."
        ),
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=1.0,
        help=(
            "Minimum lexical ranking score. "
            "Default: 1.0"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "Maximum selected candidates. "
            "Default: 50"
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run an isolated synthetic self-test."
        ),
    )

    return parser


def main() -> int:
    """Command-line entry point."""

    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.self_test:
        run_self_test()
        return 0

    if arguments.limit < 1:
        parser.error(
            "--limit must be 1 or greater."
        )

    terms = tuple(
        normalise(term).lower()
        for term in (
            arguments.terms
            or DEFAULT_QUERY_TERMS
        )
        if normalise(term)
    )

    if not terms:
        parser.error(
            "At least one non-empty query "
            "term is required."
        )

    try:
        summary = select_candidates(
            input_path=arguments.input,
            output_directory=(
                arguments.output
            ),
            terms=terms,
            minimum_score=(
                arguments.minimum_score
            ),
            limit=arguments.limit,
        )

    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"Error: {type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )
        return 1

    print("=" * 72)
    print(
        "Clinical Candidate Selection Complete"
    )
    print("=" * 72)
    print(
        f"Available : "
        f"{summary['records_available']}"
    )
    print(
        f"Selected  : "
        f"{summary['records_selected']}"
    )
    print(
        f"Threshold : "
        f"{summary['minimum_score']}"
    )
    print(
        f"Output    : "
        f"{arguments.output.expanduser().resolve()}"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
