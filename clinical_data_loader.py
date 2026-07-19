"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

MTS-Dialog Clinical Data Loader

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Read the MTS-Dialog CSV source without modifying it, parse
doctor/patient turns, preserve source provenance, and separate:

- accepted conversational candidates;
- rejected or malformed records;
- an audit summary.

The loader does not:
- diagnose patients;
- generate benchmark prompts;
- alter the source dataset;
- silently repair corrupted dialogue;
- call a model or external API.

Default usage
-------------
python clinical_data_loader.py

Self-test
---------
python clinical_data_loader.py --self-test
==========================================================
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence


# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

DEFAULT_INPUT = Path(
    "data/clinical/MTS-Dialog/Main-Dataset/"
    "MTS-Dialog-TrainingSet.csv"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/clinical/processed/mts_dialog"
)

EXPECTED_COLUMNS = {
    "ID",
    "section_header",
    "section_text",
    "dialogue",
}

SPEAKER_PATTERN = re.compile(
    r"(?P<label>"
    r"Doctor|Patient|Physician|Clinician|Provider|Nurse"
    r")\s*:\s*",
    flags=re.IGNORECASE,
)

SUSPICIOUS_LABEL_PATTERN = re.compile(
    r"\b("
    r"Doctent|Patien|Docto|Physican|Clincian"
    r")\s*:",
    flags=re.IGNORECASE,
)


# ----------------------------------------------------------
# Data Models
# ----------------------------------------------------------

@dataclass(frozen=True)
class DialogueTurn:
    role: str
    content: str


@dataclass(frozen=True)
class SourceRecord:
    source_dataset: str
    source_file: str
    source_sha256: str
    source_row_number: int
    source_id: str
    section_header: str
    section_text: str
    dialogue_raw: str
    turns: tuple[DialogueTurn, ...]
    quality_flags: tuple[str, ...]
    accepted: bool


# ----------------------------------------------------------
# General Helpers
# ----------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def normalise_whitespace(value: str) -> str:
    """Collapse repeated whitespace while preserving wording."""

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def normalise_role(label: str) -> str:
    """Map source speaker labels to benchmark-style roles."""

    normalised = label.strip().lower()

    if normalised in {
        "doctor",
        "physician",
        "clinician",
        "provider",
        "nurse",
    }:
        return "clinician"

    if normalised == "patient":
        return "patient"

    return "unknown"


def json_ready(record: SourceRecord) -> dict[str, Any]:
    """Convert an immutable record to JSON-compatible data."""

    payload = asdict(record)
    payload["turns"] = [
        asdict(turn)
        for turn in record.turns
    ]
    payload["quality_flags"] = list(
        record.quality_flags
    )
    return payload


def write_jsonl(
    path: Path,
    records: Iterable[SourceRecord],
) -> int:
    """Write records to JSONL and return the count."""

    count = 0

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    json_ready(record),
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return count


# ----------------------------------------------------------
# Dialogue Parsing
# ----------------------------------------------------------

def parse_dialogue(
    dialogue: str,
) -> tuple[DialogueTurn, ...]:
    """
    Parse explicitly labelled speaker turns.

    Text before the first recognised speaker label is not
    assigned to a speaker and is handled as a quality flag.
    """

    matches = list(
        SPEAKER_PATTERN.finditer(dialogue)
    )

    if not matches:
        return ()

    turns: list[DialogueTurn] = []

    for index, match in enumerate(matches):
        content_start = match.end()
        content_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(dialogue)
        )

        content = normalise_whitespace(
            dialogue[
                content_start:content_end
            ]
        )

        if not content:
            continue

        turns.append(
            DialogueTurn(
                role=normalise_role(
                    match.group("label")
                ),
                content=content,
            )
        )

    return tuple(turns)


def quality_flags(
    *,
    source_id: str,
    section_header: str,
    section_text: str,
    dialogue: str,
    turns: Sequence[DialogueTurn],
) -> tuple[str, ...]:
    """Return deterministic quality-control flags."""

    flags: list[str] = []

    if not source_id:
        flags.append("missing_source_id")

    if not section_header:
        flags.append("missing_section_header")

    if not section_text:
        flags.append("missing_section_text")

    if not dialogue:
        flags.append("missing_dialogue")
        return tuple(flags)

    if SUSPICIOUS_LABEL_PATTERN.search(
        dialogue
    ):
        flags.append(
            "suspicious_speaker_label"
        )

    first_match = SPEAKER_PATTERN.search(
        dialogue
    )

    if (
        first_match is not None
        and dialogue[
            :first_match.start()
        ].strip()
    ):
        flags.append(
            "unassigned_text_before_first_speaker"
        )

    if not turns:
        flags.append(
            "no_parseable_speaker_turns"
        )
        return tuple(flags)

    roles = {
        turn.role
        for turn in turns
    }

    if "patient" not in roles:
        flags.append(
            "missing_patient_turn"
        )

    if "clinician" not in roles:
        flags.append(
            "missing_clinician_turn"
        )

    if len(turns) < 2:
        flags.append(
            "fewer_than_two_turns"
        )

    if any(
        len(turn.content) < 2
        for turn in turns
    ):
        flags.append(
            "near_empty_turn"
        )

    if any(
        turn.role == "unknown"
        for turn in turns
    ):
        flags.append(
            "unknown_speaker_role"
        )

    if "\ufffd" in dialogue:
        flags.append(
            "unicode_replacement_character"
        )

    # Flag obvious dialogue corruption without claiming to
    # repair it.
    if re.search(
        r"\b(?:ou feel|o\.\s*\?|It ou)\b",
        dialogue,
        flags=re.IGNORECASE,
    ):
        flags.append(
            "possible_text_corruption"
        )

    return tuple(
        dict.fromkeys(flags)
    )


def record_is_accepted(
    flags: Sequence[str],
) -> bool:
    """
    Accept only records with a usable two-sided conversation.

    Non-fatal metadata omissions may remain flagged, but
    malformed dialogue is rejected.
    """

    rejection_flags = {
        "missing_dialogue",
        "suspicious_speaker_label",
        "no_parseable_speaker_turns",
        "missing_patient_turn",
        "missing_clinician_turn",
        "fewer_than_two_turns",
        "near_empty_turn",
        "unknown_speaker_role",
        "unicode_replacement_character",
        "possible_text_corruption",
    }

    return not any(
        flag in rejection_flags
        for flag in flags
    )


# ----------------------------------------------------------
# Dataset Loading
# ----------------------------------------------------------

def load_mts_dialog(
    input_path: Path,
    *,
    limit: int | None = None,
) -> list[SourceRecord]:
    """Read and audit an MTS-Dialog CSV file."""

    input_path = (
        input_path
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"MTS-Dialog CSV not found: "
            f"{input_path}"
        )

    source_hash = sha256_file(
        input_path
    )

    records: list[SourceRecord] = []

    with input_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as source:
        reader = csv.DictReader(source)

        columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            EXPECTED_COLUMNS - columns
        )

        if missing_columns:
            raise ValueError(
                "MTS-Dialog CSV is missing columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            if (
                limit is not None
                and len(records) >= limit
            ):
                break

            source_id = normalise_whitespace(
                row.get("ID", "")
            )
            section_header = (
                normalise_whitespace(
                    row.get(
                        "section_header",
                        "",
                    )
                )
            )
            section_text = (
                normalise_whitespace(
                    row.get(
                        "section_text",
                        "",
                    )
                )
            )
            dialogue = normalise_whitespace(
                row.get(
                    "dialogue",
                    "",
                )
            )

            turns = parse_dialogue(
                dialogue
            )

            flags = quality_flags(
                source_id=source_id,
                section_header=(
                    section_header
                ),
                section_text=section_text,
                dialogue=dialogue,
                turns=turns,
            )

            records.append(
                SourceRecord(
                    source_dataset=(
                        "MTS-Dialog"
                    ),
                    source_file=str(
                        input_path
                    ),
                    source_sha256=source_hash,
                    source_row_number=(
                        row_number
                    ),
                    source_id=source_id,
                    section_header=(
                        section_header
                    ),
                    section_text=section_text,
                    dialogue_raw=dialogue,
                    turns=turns,
                    quality_flags=flags,
                    accepted=(
                        record_is_accepted(
                            flags
                        )
                    ),
                )
            )

    return records


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

def build_summary(
    records: Sequence[SourceRecord],
    *,
    input_path: Path,
) -> dict[str, Any]:
    """Build a deterministic audit summary."""

    accepted = [
        record
        for record in records
        if record.accepted
    ]
    rejected = [
        record
        for record in records
        if not record.accepted
    ]

    flag_counts: dict[str, int] = {}

    for record in records:
        for flag in record.quality_flags:
            flag_counts[flag] = (
                flag_counts.get(flag, 0)
                + 1
            )

    return {
        "dataset": "MTS-Dialog",
        "source_file": str(
            input_path
            .expanduser()
            .resolve()
        ),
        "source_sha256": sha256_file(
            input_path
            .expanduser()
            .resolve()
        ),
        "records_read": len(records),
        "records_accepted": len(
            accepted
        ),
        "records_rejected": len(
            rejected
        ),
        "acceptance_rate": (
            len(accepted) / len(records)
            if records
            else 0.0
        ),
        "quality_flag_counts": dict(
            sorted(
                flag_counts.items()
            )
        ),
        "policy": {
            "source_modified": False,
            "silent_repairs_applied": False,
            "model_calls_made": False,
            "accepted_requires": [
                "at least two parsed turns",
                "patient speaker present",
                "clinician speaker present",
                "no detected corruption flags",
            ],
        },
    }


def process_dataset(
    input_path: Path,
    output_directory: Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Load, classify, and write audited dataset records."""

    records = load_mts_dialog(
        input_path,
        limit=limit,
    )

    output_directory = (
        output_directory
        .expanduser()
        .resolve()
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    accepted = [
        record
        for record in records
        if record.accepted
    ]
    rejected = [
        record
        for record in records
        if not record.accepted
    ]

    accepted_path = (
        output_directory
        / "accepted_candidates.jsonl"
    )
    rejected_path = (
        output_directory
        / "rejected_records.jsonl"
    )
    summary_path = (
        output_directory
        / "audit_summary.json"
    )

    write_jsonl(
        accepted_path,
        accepted,
    )
    write_jsonl(
        rejected_path,
        rejected,
    )

    summary = build_summary(
        records,
        input_path=input_path,
    )

    summary["outputs"] = {
        "accepted_candidates": str(
            accepted_path
        ),
        "rejected_records": str(
            rejected_path
        ),
        "audit_summary": str(
            summary_path
        ),
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
    """Run an isolated synthetic loader test."""

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="gaiem_clinical_loader_"
        )
    )

    try:
        input_path = (
            temporary_root
            / "sample.csv"
        )
        output_directory = (
            temporary_root
            / "processed"
        )

        with input_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "ID",
                    "section_header",
                    "section_text",
                    "dialogue",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "ID": "accepted",
                        "section_header": (
                            "GENHX"
                        ),
                        "section_text": (
                            "Chest pain history."
                        ),
                        "dialogue": (
                            "Patient: I have had "
                            "chest pain for two "
                            "hours. Doctor: Are "
                            "you short of breath?"
                        ),
                    },
                    {
                        "ID": "rejected",
                        "section_header": (
                            "GENHX"
                        ),
                        "section_text": (
                            "Malformed source."
                        ),
                        "dialogue": (
                            "Doctent: broken "
                            "speaker label"
                        ),
                    },
                ]
            )

        summary = process_dataset(
            input_path,
            output_directory,
        )

        if (
            summary["records_read"] != 2
            or summary[
                "records_accepted"
            ] != 1
            or summary[
                "records_rejected"
            ] != 1
        ):
            raise RuntimeError(
                "Clinical loader self-test "
                "counts were incorrect."
            )

        required = (
            output_directory
            / "accepted_candidates.jsonl",
            output_directory
            / "rejected_records.jsonl",
            output_directory
            / "audit_summary.json",
        )

        missing = [
            str(path)
            for path in required
            if not path.is_file()
        ]

        if missing:
            raise RuntimeError(
                "Clinical loader self-test "
                "outputs missing: "
                + ", ".join(missing)
            )

        print(
            "Clinical data loader self-test: passed"
        )
        print(
            "No source dataset was modified."
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
    """Build command-line options."""

    parser = argparse.ArgumentParser(
        prog="clinical_data_loader.py",
        description=(
            "Audit and parse MTS-Dialog conversations "
            "without modifying the source dataset."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "MTS-Dialog CSV source. "
            "Default: training set."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Processed-data directory. "
            "Default: data/clinical/processed/"
            "mts_dialog"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Optional maximum number of source "
            "records to audit."
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
        if arguments.limit is not None:
            parser.error(
                "--limit cannot be used with "
                "--self-test."
            )

        run_self_test()
        return 0

    if (
        arguments.limit is not None
        and arguments.limit < 1
    ):
        parser.error(
            "--limit must be 1 or greater."
        )

    try:
        summary = process_dataset(
            arguments.input,
            arguments.output,
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
    print("MTS-Dialog Clinical Data Audit Complete")
    print("=" * 72)
    print(
        f"Source   : "
        f"{summary['source_file']}"
    )
    print(
        f"Read     : "
        f"{summary['records_read']}"
    )
    print(
        f"Accepted : "
        f"{summary['records_accepted']}"
    )
    print(
        f"Rejected : "
        f"{summary['records_rejected']}"
    )
    print(
        f"Rate     : "
        f"{summary['acceptance_rate']:.1%}"
    )
    print(
        f"Output   : "
        f"{arguments.output.expanduser().resolve()}"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
