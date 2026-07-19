"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Clinical Language Curator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Extract clinically relevant patient-language examples from
audited MTS-Dialog conversations while distinguishing:

- positive symptom statements;
- negative/denied symptoms;
- uncertain statements;
- short answers that inherit meaning from the preceding
  clinician question.

This module is a quality-control stage. It does not:

- diagnose patients;
- generate benchmark cases;
- alter source text;
- silently repair source text;
- call a language model or external service.

Default input
-------------
data/clinical/processed/mts_dialog/
accepted_candidates.jsonl

Default output
--------------
data/clinical/review/mts_dialog_language/
patient_language_candidates.jsonl
patient_language_candidates.csv
excluded_language_records.jsonl
curation_summary.json

Self-test
---------
python clinical_language_curator.py --self-test
==========================================================
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


# ----------------------------------------------------------
# Defaults and Clinical Concept Vocabulary
# ----------------------------------------------------------

DEFAULT_INPUT = Path(
    "data/clinical/processed/mts_dialog/"
    "accepted_candidates.jsonl"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/clinical/review/mts_dialog_language"
)

CONCEPT_TERMS: dict[str, tuple[str, ...]] = {
    "chest_pain": (
        "chest pain",
        "pain in my chest",
        "pain across my chest",
        "chest pressure",
        "pressure in my chest",
        "chest tightness",
        "tightness in my chest",
        "heavy feeling in my chest",
    ),
    "shortness_of_breath": (
        "shortness of breath",
        "short of breath",
        "breathless",
        "difficulty breathing",
        "trouble breathing",
        "can't breathe",
        "cannot breathe",
        "lose my breath",
        "out of breath",
    ),
    "dizziness": (
        "dizziness",
        "dizzy",
        "lightheaded",
        "light-headed",
        "vertigo",
        "faint",
        "fainting",
    ),
    "anticoagulant": (
        "warfarin",
        "coumadin",
        "blood thinner",
        "blood thinners",
        "anticoagulant",
        "anticoagulation",
    ),
    "allergy": (
        "penicillin",
        "allergic",
        "allergy",
        "allergies",
    ),
    "exertion": (
        "when i walk",
        "while walking",
        "walking",
        "stairs",
        "on exertion",
        "with exertion",
        "when i move",
        "moving around",
        "physical activity",
        "exercise",
        "running",
    ),
    "indigestion": (
        "indigestion",
        "heartburn",
        "acid reflux",
        "reflux",
        "upset stomach",
    ),
    "onset": (
        "started",
        "began",
        "came on",
        "since",
        "hours ago",
        "hour ago",
        "minutes ago",
        "today",
        "this morning",
        "this afternoon",
        "last night",
    ),
    "progression": (
        "getting worse",
        "got worse",
        "worse",
        "worsening",
        "not settling",
        "hasn't gone away",
        "has not gone away",
        "still there",
        "more severe",
    ),
}

NEGATION_CUES = (
    "no",
    "not",
    "never",
    "none",
    "without",
    "deny",
    "denies",
    "denied",
    "don't",
    "do not",
    "doesn't",
    "does not",
    "didn't",
    "did not",
    "haven't",
    "have not",
    "hasn't",
    "has not",
    "negative for",
    "free of",
)

UNCERTAINTY_CUES = (
    "maybe",
    "might",
    "possibly",
    "probably",
    "i think",
    "i believe",
    "i'm not sure",
    "i am not sure",
    "not certain",
    "could be",
    "seems like",
    "sort of",
    "kind of",
)

AFFIRMATIVE_SHORT_ANSWERS = (
    "yes",
    "yeah",
    "yep",
    "i do",
    "i am",
    "correct",
    "that's right",
    "that is right",
)

NEGATIVE_SHORT_ANSWERS = (
    "no",
    "nope",
    "i don't",
    "i do not",
    "not really",
    "none",
    "negative",
)

ACUTE_CUES = (
    "now",
    "right now",
    "currently",
    "today",
    "this morning",
    "this afternoon",
    "sudden",
    "suddenly",
    "started",
    "began",
    "hours ago",
    "hour ago",
    "minutes ago",
    "urgent",
    "emergency",
    "getting worse",
    "worsening",
)

HISTORICAL_CUES = (
    "history of",
    "used to",
    "last year",
    "years ago",
    "previously",
    "in the past",
    "had before",
    "was hospitalized",
    "was admitted",
)


# ----------------------------------------------------------
# Data Models
# ----------------------------------------------------------

@dataclass(frozen=True)
class LanguageCandidate:
    source_id: str
    source_row_number: int
    source_file: str
    source_sha256: str
    section_header: str
    patient_turn_index: int
    clinician_context: str
    patient_utterance: str
    concepts: tuple[str, ...]
    matched_terms: tuple[str, ...]
    assertion: str
    inherited_from_context: bool
    acute_cues: tuple[str, ...]
    historical_cues: tuple[str, ...]
    quality_score: float
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class ExcludedLanguageRecord:
    source_id: str
    source_row_number: int
    patient_turn_index: int
    clinician_context: str
    patient_utterance: str
    reason: str


# ----------------------------------------------------------
# General Helpers
# ----------------------------------------------------------

def normalise(value: Any) -> str:
    """Collapse repeated whitespace without changing wording."""

    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def lower_text(value: Any) -> str:
    """Return whitespace-normalised lowercase text."""

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
                    f"Invalid JSON in {path} at "
                    f"line {line_number}: {error}"
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
    """Count simple word-like tokens."""

    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
        )
    )


def matching_items(
    text: str,
    values: Sequence[str],
) -> tuple[str, ...]:
    """Return case-insensitive literal matches."""

    lowered = lower_text(text)

    return tuple(
        value
        for value in values
        if value in lowered
    )


def concepts_in_text(
    text: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    """Return matched concept names and concrete terms."""

    lowered = lower_text(text)
    concepts: list[str] = []
    terms: list[str] = []

    for concept, concept_terms in (
        CONCEPT_TERMS.items()
    ):
        matched = [
            term
            for term in concept_terms
            if term in lowered
        ]

        if matched:
            concepts.append(concept)
            terms.extend(matched)

    return (
        tuple(
            dict.fromkeys(concepts)
        ),
        tuple(
            dict.fromkeys(terms)
        ),
    )


def starts_with_any(
    text: str,
    values: Sequence[str],
) -> bool:
    """Check short-answer response prefixes."""

    lowered = lower_text(text)

    return any(
        lowered == value
        or lowered.startswith(
            value + " "
        )
        or lowered.startswith(
            value + ","
        )
        or lowered.startswith(
            value + "."
        )
        for value in values
    )


# ----------------------------------------------------------
# Assertion Classification
# ----------------------------------------------------------

def term_is_negated(
    text: str,
    term: str,
) -> bool:
    """
    Detect simple local negation around one matched term.

    This is deliberately conservative and auditable. It does
    not claim to be a general clinical NLP negation engine.
    """

    lowered = lower_text(text)
    start = lowered.find(term)

    if start < 0:
        return False

    before = lowered[
        max(
            0,
            start - 55,
        ):start
    ]
    after = lowered[
        start + len(term):
        start + len(term) + 35
    ]

    before_tokens = re.findall(
        r"\b[\w']+\b",
        before,
    )[-8:]

    before_window = " ".join(
        before_tokens
    )
    after_window = after.strip()

    if any(
        cue in before_window
        for cue in NEGATION_CUES
    ):
        return True

    if re.match(
        r"^(?:,?\s*)"
        r"(?:no|not really|none|negative)\b",
        after_window,
    ):
        return True

    return False


def classify_direct_assertion(
    utterance: str,
    matched_terms: Sequence[str],
) -> str:
    """Classify direct concept mentions in patient text."""

    if not matched_terms:
        return "none"

    negated = [
        term
        for term in matched_terms
        if term_is_negated(
            utterance,
            term,
        )
    ]

    uncertain = matching_items(
        utterance,
        UNCERTAINTY_CUES,
    )

    if len(negated) == len(
        matched_terms
    ):
        return "negative"

    if uncertain:
        return "uncertain"

    if negated:
        return "mixed"

    return "positive"


def classify_inherited_assertion(
    patient_utterance: str,
) -> str:
    """
    Classify a short patient response to the preceding
    clinician question.
    """

    if starts_with_any(
        patient_utterance,
        NEGATIVE_SHORT_ANSWERS,
    ):
        return "negative"

    if starts_with_any(
        patient_utterance,
        AFFIRMATIVE_SHORT_ANSWERS,
    ):
        return "positive"

    if matching_items(
        patient_utterance,
        UNCERTAINTY_CUES,
    ):
        return "uncertain"

    return "contextual"


# ----------------------------------------------------------
# Quality Scoring
# ----------------------------------------------------------

def quality_flags(
    *,
    patient_utterance: str,
    clinician_context: str,
    assertion: str,
    inherited_from_context: bool,
    concepts: Sequence[str],
) -> tuple[str, ...]:
    """Return transparent review flags."""

    flags: list[str] = []

    patient_tokens = token_count(
        patient_utterance
    )

    if patient_tokens < 3:
        flags.append(
            "very_short_patient_answer"
        )

    if patient_tokens > 120:
        flags.append(
            "long_patient_utterance"
        )

    if inherited_from_context:
        flags.append(
            "meaning_inherited_from_clinician_question"
        )

    if assertion in {
        "negative",
        "mixed",
        "uncertain",
        "contextual",
    }:
        flags.append(
            f"assertion_{assertion}"
        )

    if not clinician_context:
        flags.append(
            "no_preceding_clinician_context"
        )

    if not concepts:
        flags.append(
            "no_target_clinical_concept"
        )

    return tuple(flags)


def candidate_quality_score(
    *,
    patient_utterance: str,
    concepts: Sequence[str],
    assertion: str,
    inherited_from_context: bool,
    acute_cues: Sequence[str],
    historical_cues: Sequence[str],
) -> float:
    """Score usability as a language donor."""

    words = token_count(
        patient_utterance
    )

    score = 0.0
    score += len(concepts) * 4.0
    score += len(acute_cues) * 2.0

    if assertion == "positive":
        score += 8.0
    elif assertion == "uncertain":
        score += 5.0
    elif assertion == "mixed":
        score += 2.0
    elif assertion == "negative":
        score -= 4.0
    else:
        score -= 1.0

    if 6 <= words <= 80:
        score += 3.0

    if 12 <= words <= 50:
        score += 2.0

    if inherited_from_context:
        score -= 2.0

    score -= len(
        historical_cues
    ) * 2.0

    return round(
        score,
        3,
    )


# ----------------------------------------------------------
# Extraction
# ----------------------------------------------------------

def extract_language_candidates(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    list[LanguageCandidate],
    list[ExcludedLanguageRecord],
]:
    """Extract assertion-aware patient-language examples."""

    candidates: list[
        LanguageCandidate
    ] = []
    excluded: list[
        ExcludedLanguageRecord
    ] = []

    for record in records:
        turns = record.get(
            "turns"
        )

        if not isinstance(
            turns,
            list,
        ):
            continue

        for index, turn in enumerate(
            turns
        ):
            if not isinstance(
                turn,
                Mapping,
            ):
                continue

            role = lower_text(
                turn.get("role")
            )

            if role != "patient":
                continue

            patient_utterance = normalise(
                turn.get("content")
            )

            clinician_context = ""

            if index > 0:
                previous = turns[
                    index - 1
                ]

                if (
                    isinstance(
                        previous,
                        Mapping,
                    )
                    and lower_text(
                        previous.get(
                            "role"
                        )
                    )
                    == "clinician"
                ):
                    clinician_context = (
                        normalise(
                            previous.get(
                                "content"
                            )
                        )
                    )

            direct_concepts, direct_terms = (
                concepts_in_text(
                    patient_utterance
                )
            )

            inherited = False
            concepts = direct_concepts
            terms = direct_terms

            if not concepts:
                (
                    context_concepts,
                    context_terms,
                ) = concepts_in_text(
                    clinician_context
                )

                if context_concepts:
                    inherited = True
                    concepts = (
                        context_concepts
                    )
                    terms = context_terms

            if not concepts:
                excluded.append(
                    ExcludedLanguageRecord(
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
                        patient_turn_index=(
                            index
                        ),
                        clinician_context=(
                            clinician_context
                        ),
                        patient_utterance=(
                            patient_utterance
                        ),
                        reason=(
                            "no_target_concept"
                        ),
                    )
                )
                continue

            if inherited:
                assertion = (
                    classify_inherited_assertion(
                        patient_utterance
                    )
                )
            else:
                assertion = (
                    classify_direct_assertion(
                        patient_utterance,
                        terms,
                    )
                )

            acute_cues = matching_items(
                patient_utterance,
                ACUTE_CUES,
            )
            historical_cues = (
                matching_items(
                    patient_utterance,
                    HISTORICAL_CUES,
                )
            )

            flags = quality_flags(
                patient_utterance=(
                    patient_utterance
                ),
                clinician_context=(
                    clinician_context
                ),
                assertion=assertion,
                inherited_from_context=(
                    inherited
                ),
                concepts=concepts,
            )

            score = candidate_quality_score(
                patient_utterance=(
                    patient_utterance
                ),
                concepts=concepts,
                assertion=assertion,
                inherited_from_context=(
                    inherited
                ),
                acute_cues=acute_cues,
                historical_cues=(
                    historical_cues
                ),
            )

            candidates.append(
                LanguageCandidate(
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
                    section_header=str(
                        record.get(
                            "section_header",
                            "",
                        )
                    ),
                    patient_turn_index=(
                        index
                    ),
                    clinician_context=(
                        clinician_context
                    ),
                    patient_utterance=(
                        patient_utterance
                    ),
                    concepts=concepts,
                    matched_terms=terms,
                    assertion=assertion,
                    inherited_from_context=(
                        inherited
                    ),
                    acute_cues=acute_cues,
                    historical_cues=(
                        historical_cues
                    ),
                    quality_score=score,
                    quality_flags=flags,
                )
            )

    candidates.sort(
        key=lambda item: (
            -item.quality_score,
            item.source_id,
            item.patient_turn_index,
        )
    )

    return candidates, excluded


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

def candidate_to_dict(
    candidate: LanguageCandidate,
) -> dict[str, Any]:
    """Convert one candidate to JSON-safe data."""

    payload = asdict(candidate)

    for field in (
        "concepts",
        "matched_terms",
        "acute_cues",
        "historical_cues",
        "quality_flags",
    ):
        payload[field] = list(
            payload[field]
        )

    return payload


def excluded_to_dict(
    record: ExcludedLanguageRecord,
) -> dict[str, Any]:
    """Convert one excluded record to JSON-safe data."""

    return asdict(record)


def write_jsonl(
    path: Path,
    records: Iterable[
        Mapping[str, Any]
    ],
) -> int:
    """Write JSONL records and return count."""

    count = 0

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return count


def write_candidate_csv(
    path: Path,
    candidates: Sequence[
        LanguageCandidate
    ],
) -> None:
    """Write human-review CSV."""

    fieldnames = [
        "quality_score",
        "assertion",
        "concepts",
        "matched_terms",
        "acute_cues",
        "historical_cues",
        "inherited_from_context",
        "source_id",
        "source_row_number",
        "section_header",
        "patient_turn_index",
        "clinician_context",
        "patient_utterance",
        "quality_flags",
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
            payload = (
                candidate_to_dict(
                    candidate
                )
            )

            for field in (
                "concepts",
                "matched_terms",
                "acute_cues",
                "historical_cues",
                "quality_flags",
            ):
                payload[field] = "; ".join(
                    payload[field]
                )

            writer.writerow(payload)


def curate_language(
    *,
    input_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Run the complete language-curation pipeline."""

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
            f"Accepted candidate JSONL "
            f"not found: {input_path}"
        )

    source_records = read_jsonl(
        input_path
    )

    candidates, excluded = (
        extract_language_candidates(
            source_records
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_jsonl_path = (
        output_directory
        / "patient_language_candidates.jsonl"
    )
    candidate_csv_path = (
        output_directory
        / "patient_language_candidates.csv"
    )
    excluded_path = (
        output_directory
        / "excluded_language_records.jsonl"
    )
    summary_path = (
        output_directory
        / "curation_summary.json"
    )

    write_jsonl(
        candidate_jsonl_path,
        (
            candidate_to_dict(
                candidate
            )
            for candidate in candidates
        ),
    )

    write_candidate_csv(
        candidate_csv_path,
        candidates,
    )

    write_jsonl(
        excluded_path,
        (
            excluded_to_dict(
                record
            )
            for record in excluded
        ),
    )

    assertion_counts: dict[str, int] = {}

    for candidate in candidates:
        assertion_counts[
            candidate.assertion
        ] = (
            assertion_counts.get(
                candidate.assertion,
                0,
            )
            + 1
        )

    concept_counts: dict[str, int] = {}

    for candidate in candidates:
        for concept in candidate.concepts:
            concept_counts[concept] = (
                concept_counts.get(
                    concept,
                    0,
                )
                + 1
            )

    summary = {
        "source": str(
            input_path
        ),
        "accepted_source_records": len(
            source_records
        ),
        "language_candidates": len(
            candidates
        ),
        "excluded_patient_turns": len(
            excluded
        ),
        "assertion_counts": dict(
            sorted(
                assertion_counts.items()
            )
        ),
        "concept_counts": dict(
            sorted(
                concept_counts.items()
            )
        ),
        "curation_method": (
            "deterministic concept and "
            "assertion matching"
        ),
        "limitations": [
            (
                "The negation rules are "
                "conservative lexical rules, "
                "not a clinical diagnosis engine."
            ),
            (
                "Candidates require human review "
                "before benchmark use."
            ),
            (
                "Source wording is preserved and "
                "not silently repaired."
            ),
        ],
        "model_calls_made": False,
        "source_text_rewritten": False,
        "outputs": {
            "candidate_jsonl": str(
                candidate_jsonl_path
            ),
            "candidate_csv": str(
                candidate_csv_path
            ),
            "excluded_jsonl": str(
                excluded_path
            ),
            "summary": str(
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
    """Run an isolated synthetic assertion test."""

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="gaiem_language_curator_"
        )
    )

    try:
        input_path = (
            temporary_root
            / "accepted_candidates.jsonl"
        )
        output_directory = (
            temporary_root
            / "output"
        )

        source_record = {
            "accepted": True,
            "source_id": "test",
            "source_row_number": 2,
            "source_file": "sample.csv",
            "source_sha256": "abc",
            "section_header": "GENHX",
            "turns": [
                {
                    "role": "clinician",
                    "content": (
                        "Are you having chest "
                        "pain right now?"
                    ),
                },
                {
                    "role": "patient",
                    "content": (
                        "No, I don't have any."
                    ),
                },
                {
                    "role": "clinician",
                    "content": (
                        "What happens when you "
                        "walk upstairs?"
                    ),
                },
                {
                    "role": "patient",
                    "content": (
                        "I get short of breath "
                        "and dizzy when I walk."
                    ),
                },
                {
                    "role": "clinician",
                    "content": (
                        "When did that begin?"
                    ),
                },
                {
                    "role": "patient",
                    "content": (
                        "It started two hours ago."
                    ),
                },
            ],
        }

        input_path.write_text(
            json.dumps(
                source_record
            )
            + "\n",
            encoding="utf-8",
        )

        summary = curate_language(
            input_path=input_path,
            output_directory=(
                output_directory
            ),
        )

        candidate_path = (
            output_directory
            / "patient_language_candidates.jsonl"
        )

        generated = read_jsonl_unchecked(
            candidate_path
        )

        assertions = {
            record["assertion"]
            for record in generated
        }

        concepts = {
            concept
            for record in generated
            for concept in record[
                "concepts"
            ]
        }

        if (
            summary[
                "accepted_source_records"
            ]
            != 1
            or "negative" not in assertions
            or "positive" not in assertions
            or "shortness_of_breath"
            not in concepts
            or "dizziness" not in concepts
            or "onset" not in concepts
        ):
            raise RuntimeError(
                "Clinical language curator "
                "self-test assertions failed."
            )

        print(
            "Clinical language curator "
            "self-test: passed"
        )
        print(
            "Positive and negative assertions "
            "were distinguished."
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


def read_jsonl_unchecked(
    path: Path,
) -> list[dict[str, Any]]:
    """Read generated self-test JSONL."""

    output: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            if line.strip():
                output.append(
                    json.loads(line)
                )

    return output


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command-line options."""

    parser = argparse.ArgumentParser(
        prog="clinical_language_curator.py",
        description=(
            "Extract assertion-aware patient "
            "language from audited MTS-Dialog "
            "conversations."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Accepted MTS-Dialog candidate "
            "JSONL."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Clinical language review output "
            "directory."
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

    try:
        summary = curate_language(
            input_path=arguments.input,
            output_directory=(
                arguments.output
            ),
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
        "Clinical Language Curation Complete"
    )
    print("=" * 72)
    print(
        f"Source records : "
        f"{summary['accepted_source_records']}"
    )
    print(
        f"Candidates     : "
        f"{summary['language_candidates']}"
    )
    print(
        f"Excluded turns : "
        f"{summary['excluded_patient_turns']}"
    )
    print("Assertions:")

    for assertion, count in (
        summary[
            "assertion_counts"
        ].items()
    ):
        print(
            f"  {assertion:<10} {count}"
        )

    print(
        f"Output         : "
        f"{arguments.output.expanduser().resolve()}"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
