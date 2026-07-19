"""
==========================================================
GenAI Evaluation Matrix (GAIEM)
Benchmark Loader

Copyright (c) 2026 NashMarkAI
https://nashmarkai.com

Purpose
-------
Loads, validates, and exposes benchmark definitions stored
in JSON files.

GAIEM supports two benchmark structures:

1. Conversation sessions

   sessions
   └── ordered turns

   Each turn is sent inside the same continuing model
   conversation. This is the primary structure used for
   conversational drift, hallucination accumulation,
   consistency, uncertainty, and instruction-retention
   testing.

2. Legacy isolated cases

   cases

   Each legacy case is converted into its own one-turn
   session. This keeps existing benchmark files loadable
   without falsely treating unrelated prompts as one chat.
==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, TypeAlias


# ----------------------------------------------------------
# Type Aliases
# ----------------------------------------------------------

FactDefinition: TypeAlias = str | dict[str, Any]


# ----------------------------------------------------------
# Exceptions
# ----------------------------------------------------------

class BenchmarkError(Exception):
    """
    Base exception for benchmark loading and validation
    errors.
    """


class BenchmarkFormatError(BenchmarkError):
    """
    Raised when the benchmark JSON structure is invalid.
    """


# ----------------------------------------------------------
# Benchmark Turn
# ----------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkCase:
    """
    Represents one user turn and its evaluation rules inside
    a benchmark conversation session.

    Core execution fields:
        case_id
        prompt
        category

    Reference evaluation fields:
        expected_answer
        instructions
        expected_facts
        known_false_claims
        contradiction_pairs
        uncertainty_required
        required_uncertainty_markers
        forbidden_certainty_markers
        baseline_response
        drift_threshold
        protected_terms

    Legacy compatibility fields:
        expected_keywords
        forbidden_keywords

    Conversation location:
        session_id
        turn_number

    Additional information:
        metadata
    """

    case_id: str
    prompt: str
    category: str

    expected_answer: str | None

    instructions: dict[str, Any]

    expected_facts: tuple[FactDefinition, ...]
    known_false_claims: tuple[str, ...]

    contradiction_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    uncertainty_required: bool | None

    required_uncertainty_markers: tuple[str, ...]
    forbidden_certainty_markers: tuple[str, ...]

    baseline_response: str | None
    drift_threshold: float
    protected_terms: tuple[str, ...]

    expected_keywords: tuple[str, ...]
    forbidden_keywords: tuple[str, ...]

    metadata: dict[str, Any]

    session_id: str = ""
    turn_number: int = 1


# ----------------------------------------------------------
# Conversation Session
# ----------------------------------------------------------

@dataclass(frozen=True)
class ConversationSession:
    """
    Represents one continuing GenAI conversation.

    The runner must execute turns in tuple order while
    retaining the complete user and assistant message
    history for the same session.
    """

    session_id: str
    name: str
    system_prompt: str | None
    turns: tuple[BenchmarkCase, ...]
    metadata: dict[str, Any]

    def __len__(self) -> int:
        """
        Return the number of ordered user turns.
        """

        return len(self.turns)

    def __iter__(self) -> Iterator[BenchmarkCase]:
        """
        Iterate through turns in conversation order.
        """

        return iter(self.turns)

    def get_turn(
        self,
        turn_number: int,
    ) -> BenchmarkCase:
        """
        Return one turn by its one-based turn number.
        """

        if (
            not isinstance(turn_number, int)
            or isinstance(turn_number, bool)
            or turn_number < 1
        ):
            raise ValueError(
                "turn_number must be an integer "
                "greater than zero."
            )

        try:
            return self.turns[
                turn_number - 1
            ]

        except IndexError as error:
            raise KeyError(
                f"Session '{self.session_id}' has no "
                f"turn {turn_number}."
            ) from error


# ----------------------------------------------------------
# Benchmark Collection
# ----------------------------------------------------------

@dataclass(frozen=True)
class Benchmark:
    """
    Represents a complete benchmark containing one or more
    conversation sessions.

    For compatibility, direct iteration and len(benchmark)
    operate on flattened turns. New conversation-aware code
    should iterate benchmark.sessions explicitly.
    """

    name: str
    version: str
    description: str
    sessions: tuple[ConversationSession, ...]
    source_format: str

    @property
    def cases(self) -> tuple[BenchmarkCase, ...]:
        """
        Return all turns flattened in session order.
        """

        return tuple(
            turn
            for session in self.sessions
            for turn in session.turns
        )

    @property
    def session_count(self) -> int:
        """
        Return the number of conversation sessions.
        """

        return len(self.sessions)

    @property
    def turn_count(self) -> int:
        """
        Return the total number of benchmark turns.
        """

        return sum(
            len(session)
            for session in self.sessions
        )

    def __len__(self) -> int:
        """
        Return the total number of benchmark turns.
        """

        return self.turn_count

    def __iter__(self) -> Iterator[BenchmarkCase]:
        """
        Iterate over flattened turns for compatibility with
        the existing single-turn runner.
        """

        return iter(self.cases)

    def get_session(
        self,
        session_id: str,
    ) -> ConversationSession:
        """
        Return a session by identifier.
        """

        normalized_id = require_non_empty_string(
            session_id,
            "session_id",
        )

        for session in self.sessions:
            if session.session_id == normalized_id:
                return session

        raise KeyError(
            f"Unknown benchmark session: {normalized_id}"
        )


# ----------------------------------------------------------
# Validation Helpers
# ----------------------------------------------------------

def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """
    Validate and return a required non-empty string.
    """

    if not isinstance(value, str):
        raise BenchmarkFormatError(
            f"'{field_name}' must be a string."
        )

    cleaned_value = value.strip()

    if not cleaned_value:
        raise BenchmarkFormatError(
            f"'{field_name}' cannot be empty."
        )

    return cleaned_value


def optional_string(
    value: Any,
    field_name: str,
) -> str | None:
    """
    Validate an optional string value.
    """

    if value is None:
        return None

    if not isinstance(value, str):
        raise BenchmarkFormatError(
            f"'{field_name}' must be a string or null."
        )

    cleaned_value = value.strip()

    return cleaned_value or None


def optional_boolean(
    value: Any,
    field_name: str,
) -> bool | None:
    """
    Validate an optional boolean value.
    """

    if value is None:
        return None

    if not isinstance(value, bool):
        raise BenchmarkFormatError(
            f"'{field_name}' must be a boolean or null."
        )

    return value


def positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """
    Validate a positive integer.
    """

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise BenchmarkFormatError(
            f"'{field_name}' must be an integer "
            "greater than zero."
        )

    return value


def bounded_float(
    value: Any,
    field_name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Validate a bounded numeric value.
    """

    if value is None:
        return default

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise BenchmarkFormatError(
            f"'{field_name}' must be a number."
        )

    numeric_value = float(value)

    if not minimum <= numeric_value <= maximum:
        raise BenchmarkFormatError(
            f"'{field_name}' must be between "
            f"{minimum} and {maximum}."
        )

    return numeric_value


def string_tuple(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    """
    Validate a JSON array of strings and return a tuple.
    """

    if value is None:
        return ()

    if not isinstance(value, list):
        raise BenchmarkFormatError(
            f"'{field_name}' must be a JSON array."
        )

    cleaned_values: list[str] = []

    for index, item in enumerate(value):
        cleaned_item = require_non_empty_string(
            item,
            f"{field_name}[{index}]",
        )

        if cleaned_item not in cleaned_values:
            cleaned_values.append(cleaned_item)

    return tuple(cleaned_values)


def fact_definition_tuple(
    value: Any,
    field_name: str,
) -> tuple[FactDefinition, ...]:
    """
    Validate expected factual requirements.

    Supported JSON forms:

    [
        "Paris",
        {
            "fact": "gold has the symbol Au",
            "aliases": [
                "Au is the symbol for gold"
            ]
        }
    ]
    """

    if value is None:
        return ()

    if not isinstance(value, list):
        raise BenchmarkFormatError(
            f"'{field_name}' must be a JSON array."
        )

    cleaned_facts: list[FactDefinition] = []

    for index, item in enumerate(value):
        item_path = f"{field_name}[{index}]"

        if isinstance(item, str):
            cleaned_facts.append(
                require_non_empty_string(
                    item,
                    item_path,
                )
            )
            continue

        if not isinstance(item, dict):
            raise BenchmarkFormatError(
                f"'{item_path}' must be a string or "
                "JSON object."
            )

        canonical_value = (
            item.get("fact")
            or item.get("canonical")
            or item.get("value")
        )

        canonical_fact = require_non_empty_string(
            canonical_value,
            f"{item_path}.fact",
        )

        aliases = string_tuple(
            item.get("aliases"),
            f"{item_path}.aliases",
        )

        cleaned_facts.append(
            {
                "fact": canonical_fact,
                "aliases": aliases,
            }
        )

    return tuple(cleaned_facts)


def contradiction_pair_tuple(
    value: Any,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    """
    Validate contradiction pairs.

    Expected JSON structure:

    [
        [
            "First claim",
            "Contradictory claim"
        ]
    ]
    """

    if value is None:
        return ()

    if not isinstance(value, list):
        raise BenchmarkFormatError(
            f"'{field_name}' must be a JSON array."
        )

    cleaned_pairs: list[
        tuple[str, str]
    ] = []

    for index, pair in enumerate(value):
        pair_path = f"{field_name}[{index}]"

        if not isinstance(pair, list):
            raise BenchmarkFormatError(
                f"'{pair_path}' must be a JSON array."
            )

        if len(pair) != 2:
            raise BenchmarkFormatError(
                f"'{pair_path}' must contain exactly "
                "two claims."
            )

        first_claim = require_non_empty_string(
            pair[0],
            f"{pair_path}[0]",
        )

        second_claim = require_non_empty_string(
            pair[1],
            f"{pair_path}[1]",
        )

        if first_claim.lower() == second_claim.lower():
            raise BenchmarkFormatError(
                f"'{pair_path}' must contain two "
                "different claims."
            )

        cleaned_pairs.append(
            (
                first_claim,
                second_claim,
            )
        )

    return tuple(cleaned_pairs)


def object_value(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    """
    Validate and copy a JSON object.
    """

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise BenchmarkFormatError(
            f"'{field_name}' must be a JSON object."
        )

    return dict(value)


# ----------------------------------------------------------
# Instruction Parsing
# ----------------------------------------------------------

def parse_instructions(
    raw_case: dict[str, Any],
    field_prefix: str,
    expected_keywords: tuple[str, ...],
    forbidden_keywords: tuple[str, ...],
) -> dict[str, Any]:
    """
    Parse instruction-following configuration.

    Legacy expected_keywords and forbidden_keywords values
    are mapped into instructions when explicit instruction
    values are absent.
    """

    instructions = object_value(
        raw_case.get("instructions"),
        f"{field_prefix}.instructions",
    )

    if (
        expected_keywords
        and "required_keywords" not in instructions
    ):
        instructions["required_keywords"] = list(
            expected_keywords
        )

    if (
        forbidden_keywords
        and "forbidden_keywords" not in instructions
    ):
        instructions["forbidden_keywords"] = list(
            forbidden_keywords
        )

    return instructions


# ----------------------------------------------------------
# Turn Parsing
# ----------------------------------------------------------

def parse_benchmark_case(
    raw_case: Any,
    index: int,
    *,
    session_id: str = "",
    turn_number: int = 1,
    field_prefix: str = "cases",
    generated_case_id: str | None = None,
) -> BenchmarkCase:
    """
    Validate and convert one raw JSON benchmark turn.

    The default arguments preserve compatibility with code
    that previously called parse_benchmark_case(raw, index).
    """

    if not isinstance(raw_case, dict):
        raise BenchmarkFormatError(
            f"Benchmark turn at index {index} "
            "must be an object."
        )

    case_path = f"{field_prefix}[{index}]"

    raw_case_id = raw_case.get("id")

    if raw_case_id is None and generated_case_id is not None:
        case_id = generated_case_id
    else:
        case_id = require_non_empty_string(
            raw_case_id,
            f"{case_path}.id",
        )

    prompt = require_non_empty_string(
        raw_case.get("prompt"),
        f"{case_path}.prompt",
    )

    category = require_non_empty_string(
        raw_case.get("category", "general"),
        f"{case_path}.category",
    )

    expected_answer = optional_string(
        raw_case.get("expected_answer"),
        f"{case_path}.expected_answer",
    )

    expected_keywords = string_tuple(
        raw_case.get("expected_keywords"),
        f"{case_path}.expected_keywords",
    )

    forbidden_keywords = string_tuple(
        raw_case.get("forbidden_keywords"),
        f"{case_path}.forbidden_keywords",
    )

    instructions = parse_instructions(
        raw_case=raw_case,
        field_prefix=case_path,
        expected_keywords=expected_keywords,
        forbidden_keywords=forbidden_keywords,
    )

    expected_facts = fact_definition_tuple(
        raw_case.get("expected_facts"),
        f"{case_path}.expected_facts",
    )

    known_false_claims = string_tuple(
        raw_case.get("known_false_claims"),
        f"{case_path}.known_false_claims",
    )

    contradiction_pairs = contradiction_pair_tuple(
        raw_case.get("contradiction_pairs"),
        f"{case_path}.contradiction_pairs",
    )

    uncertainty_required = optional_boolean(
        raw_case.get("uncertainty_required"),
        f"{case_path}.uncertainty_required",
    )

    required_uncertainty_markers = string_tuple(
        raw_case.get("required_uncertainty_markers"),
        (
            f"{case_path}."
            "required_uncertainty_markers"
        ),
    )

    forbidden_certainty_markers = string_tuple(
        raw_case.get("forbidden_certainty_markers"),
        (
            f"{case_path}."
            "forbidden_certainty_markers"
        ),
    )

    baseline_response = optional_string(
        raw_case.get("baseline_response"),
        f"{case_path}.baseline_response",
    )

    drift_threshold = bounded_float(
        raw_case.get("drift_threshold"),
        f"{case_path}.drift_threshold",
        default=0.75,
        minimum=0.0,
        maximum=1.0,
    )

    protected_terms = string_tuple(
        raw_case.get("protected_terms"),
        f"{case_path}.protected_terms",
    )

    metadata = object_value(
        raw_case.get("metadata"),
        f"{case_path}.metadata",
    )

    return BenchmarkCase(
        case_id=case_id,
        prompt=prompt,
        category=category,
        expected_answer=expected_answer,
        instructions=instructions,
        expected_facts=expected_facts,
        known_false_claims=known_false_claims,
        contradiction_pairs=contradiction_pairs,
        uncertainty_required=uncertainty_required,
        required_uncertainty_markers=(
            required_uncertainty_markers
        ),
        forbidden_certainty_markers=(
            forbidden_certainty_markers
        ),
        baseline_response=baseline_response,
        drift_threshold=drift_threshold,
        protected_terms=protected_terms,
        expected_keywords=expected_keywords,
        forbidden_keywords=forbidden_keywords,
        metadata=metadata,
        session_id=session_id,
        turn_number=turn_number,
    )


# ----------------------------------------------------------
# Conversation Session Parsing
# ----------------------------------------------------------

def parse_conversation_session(
    raw_session: Any,
    index: int,
) -> ConversationSession:
    """
    Validate and convert one continuing conversation
    session.
    """

    if not isinstance(raw_session, dict):
        raise BenchmarkFormatError(
            f"Session at index {index} must be an object."
        )

    session_path = f"sessions[{index}]"

    session_id = require_non_empty_string(
        raw_session.get("id"),
        f"{session_path}.id",
    )

    session_name = (
        optional_string(
            raw_session.get("name"),
            f"{session_path}.name",
        )
        or session_id
    )

    system_prompt = optional_string(
        raw_session.get("system_prompt"),
        f"{session_path}.system_prompt",
    )

    metadata = object_value(
        raw_session.get("metadata"),
        f"{session_path}.metadata",
    )

    raw_turns = raw_session.get("turns")

    if not isinstance(raw_turns, list):
        raise BenchmarkFormatError(
            f"'{session_path}.turns' must be a JSON array."
        )

    if not raw_turns:
        raise BenchmarkFormatError(
            f"'{session_path}.turns' must contain at "
            "least one turn."
        )

    turns: list[BenchmarkCase] = []

    for turn_index, raw_turn in enumerate(raw_turns):
        turn_number = turn_index + 1

        if isinstance(raw_turn, dict) and "turn" in raw_turn:
            configured_turn = positive_integer(
                raw_turn.get("turn"),
                (
                    f"{session_path}.turns["
                    f"{turn_index}].turn"
                ),
            )

            if configured_turn != turn_number:
                raise BenchmarkFormatError(
                    f"'{session_path}.turns["
                    f"{turn_index}].turn' must be "
                    f"{turn_number} to preserve ordered, "
                    "gap-free conversation turns."
                )

        generated_case_id = (
            f"{session_id}-T{turn_number:03d}"
        )

        turns.append(
            parse_benchmark_case(
                raw_turn,
                turn_index,
                session_id=session_id,
                turn_number=turn_number,
                field_prefix=(
                    f"{session_path}.turns"
                ),
                generated_case_id=generated_case_id,
            )
        )

    return ConversationSession(
        session_id=session_id,
        name=session_name,
        system_prompt=system_prompt,
        turns=tuple(turns),
        metadata=metadata,
    )


# ----------------------------------------------------------
# Legacy Case Conversion
# ----------------------------------------------------------

def parse_legacy_case_sessions(
    raw_cases: list[Any],
) -> tuple[ConversationSession, ...]:
    """
    Convert each legacy isolated case into its own one-turn
    session.

    Unrelated legacy prompts are never grouped into one
    artificial conversation.
    """

    sessions: list[ConversationSession] = []

    for index, raw_case in enumerate(raw_cases):
        case = parse_benchmark_case(
            raw_case,
            index,
            field_prefix="cases",
        )

        session_id = case.case_id

        session_case = BenchmarkCase(
            case_id=case.case_id,
            prompt=case.prompt,
            category=case.category,
            expected_answer=case.expected_answer,
            instructions=case.instructions,
            expected_facts=case.expected_facts,
            known_false_claims=case.known_false_claims,
            contradiction_pairs=case.contradiction_pairs,
            uncertainty_required=case.uncertainty_required,
            required_uncertainty_markers=(
                case.required_uncertainty_markers
            ),
            forbidden_certainty_markers=(
                case.forbidden_certainty_markers
            ),
            baseline_response=case.baseline_response,
            drift_threshold=case.drift_threshold,
            protected_terms=case.protected_terms,
            expected_keywords=case.expected_keywords,
            forbidden_keywords=case.forbidden_keywords,
            metadata=case.metadata,
            session_id=session_id,
            turn_number=1,
        )

        sessions.append(
            ConversationSession(
                session_id=session_id,
                name=case.case_id,
                system_prompt=None,
                turns=(session_case,),
                metadata={
                    "legacy_isolated_case": True,
                },
            )
        )

    return tuple(sessions)


# ----------------------------------------------------------
# Duplicate Detection
# ----------------------------------------------------------

def validate_unique_session_ids(
    sessions: tuple[ConversationSession, ...],
) -> None:
    """
    Confirm that every conversation session has a unique
    identifier.
    """

    seen_ids: set[str] = set()

    for session in sessions:
        if session.session_id in seen_ids:
            raise BenchmarkFormatError(
                "Duplicate benchmark session ID: "
                f"{session.session_id}"
            )

        seen_ids.add(session.session_id)


def validate_unique_case_ids(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    """
    Confirm that every turn has a globally unique case ID.
    """

    seen_ids: set[str] = set()

    for case in cases:
        if case.case_id in seen_ids:
            raise BenchmarkFormatError(
                "Duplicate benchmark turn ID: "
                f"{case.case_id}"
            )

        seen_ids.add(case.case_id)


# ----------------------------------------------------------
# Benchmark Loading
# ----------------------------------------------------------

def load_benchmark(
    file_path: Path | str,
) -> Benchmark:
    """
    Load and validate a benchmark JSON file.
    """

    benchmark_path = (
        Path(file_path)
        .expanduser()
        .resolve()
    )

    if not benchmark_path.exists():
        raise FileNotFoundError(
            "Benchmark file not found: "
            f"{benchmark_path}"
        )

    if not benchmark_path.is_file():
        raise BenchmarkError(
            "Benchmark path is not a file: "
            f"{benchmark_path}"
        )

    try:
        with benchmark_path.open(
            mode="r",
            encoding="utf-8",
        ) as benchmark_file:
            raw_data = json.load(benchmark_file)

    except json.JSONDecodeError as error:
        raise BenchmarkFormatError(
            "Invalid JSON in benchmark file "
            f"'{benchmark_path}'. "
            f"Line {error.lineno}, "
            f"column {error.colno}: "
            f"{error.msg}"
        ) from error

    except OSError as error:
        raise BenchmarkError(
            "Unable to read benchmark file: "
            f"{benchmark_path}"
        ) from error

    if not isinstance(raw_data, dict):
        raise BenchmarkFormatError(
            "Benchmark root must be a JSON object."
        )

    name = require_non_empty_string(
        raw_data.get("name"),
        "name",
    )

    version = require_non_empty_string(
        raw_data.get("version", "1.0"),
        "version",
    )

    description = (
        optional_string(
            raw_data.get("description"),
            "description",
        )
        or ""
    )

    has_sessions = "sessions" in raw_data
    has_cases = "cases" in raw_data

    if has_sessions and has_cases:
        raise BenchmarkFormatError(
            "Benchmark root must contain either "
            "'sessions' or legacy 'cases', not both."
        )

    if not has_sessions and not has_cases:
        raise BenchmarkFormatError(
            "Benchmark root must contain 'sessions' "
            "or legacy 'cases'."
        )

    if has_sessions:
        raw_sessions = raw_data.get("sessions")

        if not isinstance(raw_sessions, list):
            raise BenchmarkFormatError(
                "'sessions' must be a JSON array."
            )

        if not raw_sessions:
            raise BenchmarkFormatError(
                "Benchmark must contain at least one "
                "conversation session."
            )

        sessions = tuple(
            parse_conversation_session(
                raw_session,
                index,
            )
            for index, raw_session
            in enumerate(raw_sessions)
        )

        source_format = "sessions"

    else:
        raw_cases = raw_data.get("cases")

        if not isinstance(raw_cases, list):
            raise BenchmarkFormatError(
                "'cases' must be a JSON array."
            )

        if not raw_cases:
            raise BenchmarkFormatError(
                "Benchmark must contain at least one case."
            )

        sessions = parse_legacy_case_sessions(
            raw_cases
        )

        source_format = "legacy_cases"

    validate_unique_session_ids(sessions)

    flattened_cases = tuple(
        turn
        for session in sessions
        for turn in session.turns
    )

    validate_unique_case_ids(flattened_cases)

    return Benchmark(
        name=name,
        version=version,
        description=description,
        sessions=sessions,
        source_format=source_format,
    )


# ----------------------------------------------------------
# Standalone Test
# ----------------------------------------------------------

def main() -> int:
    """
    Load the default benchmark and display its sessions and
    ordered turns.
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    default_benchmark = (
        project_root
        / "benchmarks"
        / "default.json"
    )

    try:
        benchmark = load_benchmark(
            default_benchmark
        )

    except (
        FileNotFoundError,
        BenchmarkError,
    ) as error:
        print(
            f"Error: {error}"
        )

        return 1

    print("=" * 60)
    print("GenAI Evaluation Matrix")
    print("Benchmark Loader")
    print("=" * 60)
    print(f"Name       : {benchmark.name}")
    print(f"Version    : {benchmark.version}")
    print(f"Description: {benchmark.description}")
    print(f"Format     : {benchmark.source_format}")
    print(f"Sessions   : {benchmark.session_count}")
    print(f"Turns      : {benchmark.turn_count}")
    print("=" * 60)

    for session in benchmark.sessions:
        print(
            f"{session.session_id} | "
            f"turns={len(session)} | "
            f"{session.name}"
        )

        for turn in session:
            print(
                f"  Turn {turn.turn_number:03d} | "
                f"{turn.case_id} | "
                f"{turn.category} | "
                f"{turn.prompt}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
