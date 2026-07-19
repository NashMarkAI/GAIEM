"""
==========================================================
GenAI Evaluation Matrix (GAIEM)
Conversation-Aware Benchmark Runner

Copyright (c) 2026 NashMarkAI
https://nashmarkai.com

Purpose
-------
Runs equal-length, continuing-conversation benchmark
sessions against a provider. Normal execution uses the
unscaffolded chat benchmark automatically:

    python runner.py

For every session:
- the full accumulated transcript is sent on every turn;
- the returned provider response is preserved;
- every eligible evaluator runs after that turn;
- raw responses, transcripts, evaluator results and score
  rows are written as auditable evidence;
- evidence-backed graphs and a readable report are generated
  after the run completes.

Operational entry points are centralised here:
- normal benchmark execution and reporting;
- validation without a model request;
- evaluator-registry inspection;
- report regeneration from an existing evidence directory.

Individual evaluator module commands remain developer
self-tests only.

This runner never generates placeholder graph data.
==========================================================
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import platform
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from benchmarks.benchmark import (
    Benchmark,
    BenchmarkError,
    ConversationSession,
    load_benchmark,
)
from config import (
    PROJECT_NAME,
    VERSION,
    config,
)
from evaluators.base_evaluator import (
    ConversationMessage,
    EvaluationContext,
)
from evaluators.evaluator_factory import (
    create_default_factory,
)
from providers.base_provider import (
    BaseProvider,
    ProviderError,
)
from providers.provider_factory import (
    create_provider,
)
from response_store import (
    ResponseStore,
    ResponseStoreError,
    StoredBenchmarkRun,
    create_completed_result,
    create_failed_result,
    utc_timestamp,
)


# ----------------------------------------------------------
# Command-Line Arguments
# ----------------------------------------------------------

DEFAULT_BENCHMARK_FILE = (
    config.BENCHMARK_DIR
    / "chat_unscaffolded.json"
)

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="runner.py",
        description=(
            "Run the GenAI Evaluation Matrix as unscaffolded "
            "continuing-conversation sessions by default and store "
            "auditable evidence."
        ),
    )

    parser.add_argument(
        "--provider",
        default=config.DEFAULT_PROVIDER,
        help=(
            "Provider adapter. "
            f"Default: {config.DEFAULT_PROVIDER}"
        ),
    )

    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        help=(
            "Provider model identifier. "
            f"Default: {config.DEFAULT_MODEL}"
        ),
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help=(
            "Benchmark JSON file override. When omitted, runner.py "
            "uses benchmarks/chat_unscaffolded.json automatically."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_DIR,
        help="Root directory for run evidence.",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help=(
            "Complete benchmark rounds. "
            "Each round starts new conversations. Default: 1"
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help=(
            "Sampling temperature. "
            f"Default: {config.DEFAULT_TEMPERATURE}"
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.DEFAULT_MAX_TOKENS,
        help=(
            "Maximum output tokens per response. "
            f"Default: {config.DEFAULT_MAX_TOKENS}"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help=(
            "Provider seed where supported. "
            f"Default: {config.DEFAULT_SEED}"
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Continue with later sessions after a provider "
            "or evaluator error. Default: enabled"
        ),
    )

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "reports"
        ),
        help=(
            "Root directory for generated reports. "
            "Default: ./reports"
        ),
    )

    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "graphs"
        ),
        help=(
            "Root directory for generated graphs. "
            "Default: ./graphs"
        ),
    )

    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "docs"
        ),
        help=(
            "Root directory for generated plain-text "
            "documentation. Default: ./docs"
        ),
    )

    parser.add_argument(
        "--generate-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate graphs and a Markdown report after the "
            "benchmark. Default: enabled"
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate configuration, benchmark design, provider "
            "adapter and evaluator registry without calling a model."
        ),
    )

    parser.add_argument(
        "--list-evaluators",
        action="store_true",
        help=(
            "Print the registered evaluator names, versions, scopes "
            "and implementation modules, then exit."
        ),
    )

    parser.add_argument(
        "--report-only",
        type=Path,
        default=None,
        metavar="RUN_DIRECTORY",
        help=(
            "Regenerate graphs and reports from an existing run "
            "evidence directory without calling a provider or "
            "rerunning evaluators."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROJECT_NAME} {VERSION}",
    )

    return parser


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Validate command-line values."""

    if not isinstance(
        arguments.provider,
        str,
    ) or not arguments.provider.strip():
        raise ValueError(
            "Provider cannot be empty."
        )

    if not isinstance(
        arguments.model,
        str,
    ) or not arguments.model.strip():
        raise ValueError(
            "Model cannot be empty."
        )

    if arguments.rounds < 1:
        raise ValueError(
            "Rounds must be at least 1."
        )

    if not 0.0 <= arguments.temperature <= 2.0:
        raise ValueError(
            "Temperature must be between 0.0 and 2.0."
        )

    if arguments.max_tokens < 1:
        raise ValueError(
            "Maximum tokens must be at least 1."
        )

    selected_modes = sum(
        (
            bool(arguments.validate_only),
            bool(arguments.list_evaluators),
            arguments.report_only is not None,
        )
    )

    if selected_modes > 1:
        raise ValueError(
            "--validate-only, --list-evaluators and "
            "--report-only are mutually exclusive."
        )


# ----------------------------------------------------------
# Benchmark Design Validation
# ----------------------------------------------------------

def validate_benchmark_design(
    benchmark: Benchmark,
) -> None:
    """
    Enforce the current GAIEM comparison profile:

    - four continuing-conversation sessions;
    - nine responses per session;
    - observation states 1, 3, 7 and 9.

    This validates benchmark shape only. It does not add a system
    prompt or otherwise scaffold model behaviour.
    """

    expected_session_count = 4
    expected_turn_count = (
        expected_session_count
        * config.RESPONSES_PER_SESSION
    )

    if benchmark.session_count != expected_session_count:
        raise ValueError(
            "GAIEM benchmark profile requires exactly "
            f"{expected_session_count} sessions; found "
            f"{benchmark.session_count}."
        )

    if benchmark.turn_count != expected_turn_count:
        raise ValueError(
            "GAIEM benchmark profile requires exactly "
            f"{expected_turn_count} turns; found "
            f"{benchmark.turn_count}."
        )

    expected_states = set(
        config.OBSERVATION_STATES
    )

    for session in benchmark.sessions:
        if len(session) != config.RESPONSES_PER_SESSION:
            raise ValueError(
                f"Session {session.session_id} contains "
                f"{len(session)} turns; expected "
                f"{config.RESPONSES_PER_SESSION}."
            )

        actual_states = {
            turn.turn_number
            for turn in session.turns
            if bool(
                turn.metadata.get(
                    "checkpoint",
                    False,
                )
            )
        }

        if actual_states != expected_states:
            raise ValueError(
                f"Session {session.session_id} has observation "
                f"states {sorted(actual_states)}; expected "
                f"{sorted(expected_states)}."
            )


def benchmark_mode(
    benchmark: Benchmark,
) -> str:
    """Return the declared or structurally inferred chat mode."""

    declared_modes: set[str] = set()

    for session in benchmark.sessions:
        session_mode = session.metadata.get(
            "chat_mode"
        )

        if isinstance(session_mode, str) and session_mode.strip():
            declared_modes.add(
                session_mode.strip().lower()
            )

        for case in session.turns:
            turn_mode = case.metadata.get(
                "chat_mode"
            )

            if isinstance(turn_mode, str) and turn_mode.strip():
                declared_modes.add(
                    turn_mode.strip().lower()
                )

    if len(declared_modes) == 1:
        return next(iter(declared_modes))

    if len(declared_modes) > 1:
        return "mixed"

    system_prompts = [
        session.system_prompt
        for session in benchmark.sessions
    ]

    if all(
        prompt is None
        or not str(prompt).strip()
        for prompt in system_prompts
    ):
        return "unscaffolded"

    if all(
        prompt is not None
        and bool(str(prompt).strip())
        for prompt in system_prompts
    ):
        return "scaffolded"

    return "mixed"


def system_prompt_session_count(
    benchmark: Benchmark,
) -> int:
    """Count sessions that define a non-empty system prompt."""

    return sum(
        1
        for session in benchmark.sessions
        if session.system_prompt is not None
        and bool(str(session.system_prompt).strip())
    )


def validate_default_operational_profile(
    *,
    benchmark: Benchmark,
    evaluator_registry: Iterable[Mapping[str, Any]],
) -> None:
    """
    Validate the zero-argument operational profile.

    A plain ``python runner.py`` invocation must run the
    unscaffolded conversation benchmark with no benchmark-defined
    system prompts and with Conversation State Retention 1.1
    registered before any provider request is sent.
    """

    mode = benchmark_mode(benchmark)

    if mode != "unscaffolded":
        raise ValueError(
            "The default runner profile must be unscaffolded; "
            f"loaded mode was {mode!r}."
        )

    prompt_count = system_prompt_session_count(
        benchmark
    )

    if prompt_count != 0:
        raise ValueError(
            "The default unscaffolded benchmark must define zero "
            f"system prompts; found {prompt_count}."
        )

    descriptors = tuple(
        evaluator_registry
    )

    retention = next(
        (
            descriptor
            for descriptor in descriptors
            if str(
                descriptor.get(
                    "name",
                    "",
                )
            )
            == "Conversation State Retention"
        ),
        None,
    )

    if retention is None:
        raise ValueError(
            "The default runner profile requires the "
            "Conversation State Retention evaluator."
        )

    retention_version = str(
        retention.get(
            "version",
            "",
        )
    )

    if retention_version != "1.1":
        raise ValueError(
            "The default runner profile requires Conversation "
            "State Retention version 1.1; loaded version was "
            f"{retention_version!r}."
        )


# ----------------------------------------------------------
# Serialisation Helpers
# ----------------------------------------------------------

def serialise_object(
    value: Any,
) -> Any:
    """Convert framework objects into JSON-safe data."""

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return serialise_object(
            asdict(value)
        )

    if isinstance(value, Mapping):
        return {
            str(key): serialise_object(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            serialise_object(item)
            for item in value
        ]

    to_dict = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(to_dict):
        return serialise_object(
            to_dict()
        )

    if hasattr(value, "__dict__"):
        return serialise_object(
            vars(value)
        )

    return str(value)


def nested_value(
    payload: Any,
    key: str,
) -> Any:
    """Find the first matching key inside nested evidence."""

    if isinstance(payload, Mapping):
        if key in payload:
            return payload[key]

        for value in payload.values():
            found = nested_value(
                value,
                key,
            )

            if found is not None:
                return found

    elif isinstance(payload, list):
        for value in payload:
            found = nested_value(
                value,
                key,
            )

            if found is not None:
                return found

    return None


def file_sha256(
    path: Path,
) -> str:
    """Calculate the benchmark file hash."""

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# ----------------------------------------------------------
# Run Identity
# ----------------------------------------------------------

def safe_identifier(
    value: str,
) -> str:
    """Convert a provider/model value into a path-safe label."""

    cleaned = "".join(
        character.lower()
        if character.isalnum()
        else "_"
        for character in value.strip()
    )

    while "__" in cleaned:
        cleaned = cleaned.replace(
            "__",
            "_",
        )

    return cleaned.strip("_") or "unknown"


def create_run_id(
    provider_name: str,
    model_name: str,
) -> str:
    """Create a unique UTC run identifier."""

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    return (
        f"{timestamp}_"
        f"{safe_identifier(provider_name)}_"
        f"{safe_identifier(model_name)}"
    )


# ----------------------------------------------------------
# Evaluator Compatibility Helpers
# ----------------------------------------------------------

def create_conversation_message(
    *,
    role: str,
    content: str,
    turn_number: int,
    metadata: Mapping[str, Any] | None = None,
) -> ConversationMessage:
    """
    Construct the shared evaluator message while remaining
    compatible with optional fields in the installed dataclass.
    """

    parameters = inspect.signature(
        ConversationMessage
    ).parameters

    values: dict[str, Any] = {
        "role": role,
        "content": content,
    }

    optional_values = {
        "turn_number": turn_number,
        "metadata": dict(
            metadata or {}
        ),
    }

    for name, value in optional_values.items():
        if name in parameters:
            values[name] = value

    return ConversationMessage(
        **values
    )


def create_evaluation_context(
    *,
    case: Any,
    response: Any,
    history: Iterable[ConversationMessage],
    session_id: str,
    turn_number: int,
    round_number: int,
    metadata: Mapping[str, Any],
) -> EvaluationContext:
    """Construct the shared conversation-aware context."""

    parameters = inspect.signature(
        EvaluationContext
    ).parameters

    candidate_values: dict[str, Any] = {
        "case": case,
        "response": response,
        "history": tuple(history),
        "session_id": session_id,
        "turn_number": turn_number,
        "round_number": round_number,
        "metadata": dict(metadata),
    }

    accepted_values = {
        name: value
        for name, value in candidate_values.items()
        if name in parameters
    }

    missing_required = [
        name
        for name, parameter in parameters.items()
        if (
            parameter.default
            is inspect.Parameter.empty
            and name not in accepted_values
        )
    ]

    if missing_required:
        raise TypeError(
            "EvaluationContext has unsupported required "
            "fields: "
            + ", ".join(missing_required)
        )

    return EvaluationContext(
        **accepted_values
    )


def factory_evaluators(
    factory: Any,
) -> tuple[Any, ...]:
    """Return every registered evaluator from the factory."""

    all_method = getattr(
        factory,
        "all",
        None,
    )

    if not callable(all_method):
        raise TypeError(
            "Evaluator factory does not expose all()."
        )

    registered = all_method()

    if isinstance(registered, Mapping):
        registered = registered.values()

    evaluators = tuple(registered)

    if not evaluators:
        raise ValueError(
            "Evaluator registry is empty."
        )

    return evaluators


def evaluator_display_name(
    evaluator: Any,
) -> str:
    """Return one stable evaluator name."""

    for attribute_name in (
        "evaluator_name",
        "name",
    ):
        value = getattr(
            evaluator,
            attribute_name,
            None,
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

    return evaluator.__class__.__name__


def registered_evaluator_names(
    factory: Any,
) -> tuple[str, ...]:
    """Return the evaluator registry names."""

    names_method = getattr(
        factory,
        "names",
        None,
    )

    if callable(names_method):
        names = tuple(
            str(name)
            for name in names_method()
        )

        if names:
            return names

    return tuple(
        evaluator_display_name(evaluator)
        for evaluator in factory_evaluators(
            factory
        )
    )


def evaluator_descriptor(
    evaluator: Any,
) -> dict[str, Any]:
    """Return reproducibility metadata for one evaluator."""

    evaluator_class = evaluator.__class__
    module = inspect.getmodule(
        evaluator_class
    )

    module_file = getattr(
        module,
        "__file__",
        None,
    )

    return {
        "name": evaluator_display_name(
            evaluator
        ),
        "version": str(
            getattr(
                evaluator,
                "version",
                "unversioned",
            )
        ),
        "scope": str(
            getattr(
                evaluator,
                "scope",
                "turn",
            )
        ),
        "class": evaluator_class.__name__,
        "module": evaluator_class.__module__,
        "module_file": (
            str(
                Path(module_file).resolve()
            )
            if module_file
            else None
        ),
    }


def validate_evaluator_registry(
    factory: Any,
) -> tuple[dict[str, Any], ...]:
    """
    Validate evaluator contracts and return their descriptors.

    This check runs before any provider request. It catches an
    empty registry, duplicate evaluator names and components
    that do not expose the required supports/evaluate methods.
    """

    descriptors: list[
        dict[str, Any]
    ] = []
    seen_names: set[str] = set()

    for evaluator in factory_evaluators(
        factory
    ):
        descriptor = evaluator_descriptor(
            evaluator
        )
        name = str(descriptor["name"])

        if name in seen_names:
            raise ValueError(
                "Duplicate evaluator name: "
                f"{name}"
            )

        seen_names.add(name)

        for method_name in (
            "supports",
            "evaluate",
        ):
            method = getattr(
                evaluator,
                method_name,
                None,
            )

            if not callable(method):
                raise TypeError(
                    f"Evaluator {name} does not expose "
                    f"{method_name}(context)."
                )

            parameters = inspect.signature(
                method
            ).parameters

            if not parameters:
                raise TypeError(
                    f"Evaluator {name} has an invalid "
                    f"{method_name}() signature."
                )

        descriptors.append(
            descriptor
        )

    if not descriptors:
        raise ValueError(
            "Evaluator registry is empty."
        )

    return tuple(descriptors)


def print_evaluator_registry(
    descriptors: Iterable[
        Mapping[str, Any]
    ],
    *,
    heading: str = "Evaluator Registry",
) -> None:
    """Print registered evaluator implementation details."""

    items = tuple(descriptors)

    print("=" * 72)
    print(heading)
    print("=" * 72)
    print(f"Registered evaluators: {len(items)}")

    for descriptor in items:
        print(
            "  - "
            f"{descriptor.get('name')} "
            f"| version={descriptor.get('version')} "
            f"| scope={descriptor.get('scope')}"
        )
        print(
            "    "
            f"{descriptor.get('module')}."
            f"{descriptor.get('class')}"
        )
        print(
            "    file="
            f"{descriptor.get('module_file')}"
        )

    print("=" * 72)


# ----------------------------------------------------------
# Evaluation Records and CSV Rows
# ----------------------------------------------------------

def evaluation_record(
    *,
    run: StoredBenchmarkRun,
    context: EvaluationContext,
    result: Any,
    evaluator: Any,
    checkpoint: bool,
    observation_state: int | None,
) -> dict[str, Any]:
    """Build one complete evaluator-result evidence record."""

    payload = serialise_object(
        result
    )

    if not isinstance(payload, dict):
        payload = {
            "result": payload,
        }

    payload.setdefault(
        "run_id",
        run.run_id,
    )
    payload.setdefault(
        "provider",
        run.provider,
    )
    payload.setdefault(
        "model",
        run.model,
    )
    payload.setdefault(
        "session_id",
        getattr(
            context,
            "session_id",
            "",
        ),
    )
    payload.setdefault(
        "turn_number",
        getattr(
            context,
            "turn_number",
            0,
        ),
    )
    payload.setdefault(
        "round_number",
        getattr(
            context,
            "round_number",
            0,
        ),
    )
    payload.setdefault(
        "case_id",
        getattr(
            getattr(context, "case", None),
            "case_id",
            "",
        ),
    )
    descriptor = evaluator_descriptor(
        evaluator
    )

    payload.setdefault(
        "evaluator",
        descriptor["name"],
    )
    payload.setdefault(
        "evaluator_version",
        descriptor["version"],
    )
    payload.setdefault(
        "scope",
        descriptor["scope"],
    )
    payload.setdefault(
        "evaluator_class",
        descriptor["class"],
    )
    payload.setdefault(
        "evaluator_module",
        descriptor["module"],
    )

    payload["checkpoint"] = checkpoint
    payload["observation_state"] = (
        observation_state
    )
    payload["recorded_at"] = utc_timestamp()

    return payload


def evaluation_error_record(
    *,
    run: StoredBenchmarkRun,
    session_id: str,
    turn_number: int,
    round_number: int,
    case_id: str,
    evaluator_name: str,
    stage: str,
    error: Exception,
    checkpoint: bool,
    observation_state: int | None,
) -> dict[str, Any]:
    """Build a truthful evaluator-execution error record."""

    return {
        "run_id": run.run_id,
        "provider": run.provider,
        "model": run.model,
        "session_id": session_id,
        "turn_number": turn_number,
        "round_number": round_number,
        "case_id": case_id,
        "evaluator": evaluator_name,
        "status": "error",
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "checkpoint": checkpoint,
        "observation_state": observation_state,
        "recorded_at": utc_timestamp(),
        "scores": [],
    }


def score_rows_from_record(
    *,
    record: Mapping[str, Any],
    response: Any,
) -> list[dict[str, Any]]:
    """Flatten evaluator score components into CSV rows."""

    raw_scores = record.get(
        "scores",
        [],
    )

    if not isinstance(raw_scores, list):
        return []

    evidence = record.get(
        "evidence",
        {},
    )

    metadata = record.get(
        "metadata",
        {},
    )

    output_rows: list[
        dict[str, Any]
    ] = []

    for raw_score in raw_scores:
        score = serialise_object(
            raw_score
        )

        if not isinstance(score, dict):
            continue

        numeric_score = score.get(
            "score"
        )
        maximum = score.get(
            "maximum",
            1.0,
        )

        percentage: float | None = None

        if (
            isinstance(numeric_score, (int, float))
            and isinstance(maximum, (int, float))
            and maximum != 0
        ):
            percentage = (
                float(numeric_score)
                / float(maximum)
            )

        output_rows.append(
            {
                "run_id": record.get(
                    "run_id"
                ),
                "provider": record.get(
                    "provider"
                ),
                "model": record.get(
                    "model"
                ),
                "session_id": record.get(
                    "session_id"
                ),
                "turn_number": record.get(
                    "turn_number"
                ),
                "case_id": record.get(
                    "case_id"
                ),
                "round_number": record.get(
                    "round_number"
                ),
                "checkpoint": record.get(
                    "checkpoint"
                ),
                "observation_state": record.get(
                    "observation_state"
                ),
                "evaluator": record.get(
                    "evaluator"
                ),
                "scope": record.get(
                    "scope",
                    metadata.get(
                        "scope",
                        "turn",
                    )
                    if isinstance(
                        metadata,
                        Mapping,
                    )
                    else "turn",
                ),
                "score_name": score.get(
                    "name",
                    record.get(
                        "evaluator"
                    ),
                ),
                "score": numeric_score,
                "maximum": maximum,
                "percentage": percentage,
                "passed": score.get(
                    "passed",
                    record.get(
                        "passed"
                    ),
                ),
                "current_turn_passed": nested_value(
                    evidence,
                    "current_turn_passed",
                ),
                "conversation_passed": nested_value(
                    evidence,
                    "conversation_passed",
                ),
                "latency_seconds": getattr(
                    response,
                    "latency_seconds",
                    None,
                ),
                "input_tokens": getattr(
                    response,
                    "prompt_tokens",
                    None,
                ),
                "output_tokens": getattr(
                    response,
                    "completion_tokens",
                    None,
                ),
                "detected_claims": nested_value(
                    evidence,
                    "detected_claims",
                ),
                "missing_facts": nested_value(
                    evidence,
                    "missing_facts",
                ),
                "lost_protected_terms": nested_value(
                    evidence,
                    "lost_protected_terms",
                ),
                "newly_introduced_claims": nested_value(
                    evidence,
                    "newly_introduced_claims",
                ),
                "failing_turns": nested_value(
                    evidence,
                    "failing_turns",
                ),
            }
        )

    return output_rows


# ----------------------------------------------------------
# Display
# ----------------------------------------------------------

def print_validation_summary(
    *,
    benchmark: Benchmark,
    benchmark_file: Path,
    benchmark_selection: str,
    provider: BaseProvider,
    evaluator_registry: tuple[
        Mapping[str, Any],
        ...,
    ],
) -> None:
    """Display validation-only results."""

    print("=" * 72)
    print(PROJECT_NAME)
    print("Conversation Runner Validation")
    print("=" * 72)
    print(f"Version            : {VERSION}")
    print(f"Benchmark          : {benchmark.name}")
    print(f"Benchmark file     : {benchmark_file}")
    print(f"Benchmark selection: {benchmark_selection}")
    print(f"Benchmark version  : {benchmark.version}")
    print(f"Chat mode          : {benchmark_mode(benchmark)}")
    print(
        "System prompts     : "
        f"{system_prompt_session_count(benchmark)} "
        "session(s)"
    )
    print(f"Sessions           : {benchmark.session_count}")
    print(f"Turns              : {benchmark.turn_count}")
    print(
        "Turns per session  : "
        + ", ".join(
            str(len(session))
            for session in benchmark.sessions
        )
    )
    print(
        "Observation states : "
        + ", ".join(
            str(state)
            for state in config.OBSERVATION_STATES
        )
    )
    print(f"Provider           : {provider.provider_name}")
    print(f"Model              : {provider.default_model}")
    print(
        f"Provider class     : "
        f"{provider.__class__.__module__}."
        f"{provider.__class__.__name__}"
    )
    print(
        f"Evaluators         : "
        f"{len(evaluator_registry)}"
    )

    for descriptor in evaluator_registry:
        print(
            "  - "
            f"{descriptor.get('name')} "
            f"| version={descriptor.get('version')} "
            f"| scope={descriptor.get('scope')}"
        )
        print(
            "    file="
            f"{descriptor.get('module_file')}"
        )

    print("=" * 72)
    print("Validation passed. No model requests were sent.")


def compact_text(
    text: str,
    maximum_length: int = 160,
) -> str:
    """Return a one-line response preview."""

    compact = " ".join(
        text.split()
    )

    if len(compact) <= maximum_length:
        return compact

    return (
        compact[
            : maximum_length - 3
        ]
        + "..."
    )


# ----------------------------------------------------------
# Session Execution
# ----------------------------------------------------------

def evaluator_history_from_messages(
    provider_messages: Iterable[
        Mapping[str, Any]
    ],
) -> tuple[ConversationMessage, ...]:
    """Convert prior provider messages for evaluator context."""

    history: list[
        ConversationMessage
    ] = []

    for message in provider_messages:
        role = str(
            message.get(
                "role",
                "",
            )
        )
        content = str(
            message.get(
                "content",
                "",
            )
        )
        metadata = message.get(
            "metadata",
            {},
        )

        turn_number = 0

        if isinstance(metadata, Mapping):
            raw_turn_number = metadata.get(
                "turn_number",
                0,
            )

            if isinstance(
                raw_turn_number,
                int,
            ):
                turn_number = raw_turn_number

        history.append(
            create_conversation_message(
                role=role,
                content=content,
                turn_number=turn_number,
                metadata=(
                    metadata
                    if isinstance(
                        metadata,
                        Mapping,
                    )
                    else {}
                ),
            )
        )

    return tuple(history)


def append_provider_evidence(
    *,
    store: ResponseStore,
    run: StoredBenchmarkRun,
    session: ConversationSession,
    case: Any,
    round_number: int,
    request_messages: list[dict[str, Any]],
    response: Any,
    completed_at: str,
) -> None:
    """Append the raw provider response with its request context."""

    raw_response = getattr(
        response,
        "raw_response",
        {},
    )

    store.append_raw_response(
        {
            "run_id": run.run_id,
            "provider": run.provider,
            "model": run.model,
            "round_number": round_number,
            "session_id": session.session_id,
            "turn_number": case.turn_number,
            "case_id": case.case_id,
            "category": case.category,
            "checkpoint": bool(
                case.metadata.get(
                    "checkpoint",
                    False,
                )
            ),
            "observation_state": case.metadata.get(
                "observation_state"
            ),
            "system_prompt": session.system_prompt,
            "messages_sent": request_messages,
            "completed_at": completed_at,
            "latency_seconds": getattr(
                response,
                "latency_seconds",
                None,
            ),
            "raw_provider_response": (
                raw_response
                if isinstance(
                    raw_response,
                    dict,
                )
                else serialise_object(
                    raw_response
                )
            ),
        }
    )


def execute_successful_evaluations(
    *,
    factory: Any,
    context: EvaluationContext,
    response: Any,
    run: StoredBenchmarkRun,
    store: ResponseStore,
    case: Any,
) -> tuple[int, int]:
    """
    Run every eligible evaluator.

    Returns:
        evaluator records written,
        evaluator errors written.
    """

    records_written = 0
    errors_written = 0

    checkpoint = bool(
        case.metadata.get(
            "checkpoint",
            False,
        )
    )
    observation_state = case.metadata.get(
        "observation_state"
    )

    for evaluator in factory_evaluators(
        factory
    ):
        evaluator_name = evaluator_display_name(
            evaluator
        )

        supports = getattr(
            evaluator,
            "supports",
            None,
        )

        if not callable(supports):
            error = TypeError(
                "Evaluator does not expose supports(context)."
            )
            store.append_evaluation(
                evaluation_error_record(
                    run=run,
                    session_id=case.session_id,
                    turn_number=case.turn_number,
                    round_number=getattr(
                        context,
                        "round_number",
                        0,
                    ),
                    case_id=case.case_id,
                    evaluator_name=evaluator_name,
                    stage="supports",
                    error=error,
                    checkpoint=checkpoint,
                    observation_state=(
                        observation_state
                    ),
                )
            )
            errors_written += 1
            continue

        try:
            is_supported = bool(
                supports(context)
            )
        except Exception as error:
            store.append_evaluation(
                evaluation_error_record(
                    run=run,
                    session_id=case.session_id,
                    turn_number=case.turn_number,
                    round_number=getattr(
                        context,
                        "round_number",
                        0,
                    ),
                    case_id=case.case_id,
                    evaluator_name=evaluator_name,
                    stage="supports",
                    error=error,
                    checkpoint=checkpoint,
                    observation_state=(
                        observation_state
                    ),
                )
            )
            errors_written += 1
            continue

        if not is_supported:
            continue

        evaluate = getattr(
            evaluator,
            "evaluate",
            None,
        )

        if not callable(evaluate):
            error = TypeError(
                "Evaluator does not expose evaluate(context)."
            )
            store.append_evaluation(
                evaluation_error_record(
                    run=run,
                    session_id=case.session_id,
                    turn_number=case.turn_number,
                    round_number=getattr(
                        context,
                        "round_number",
                        0,
                    ),
                    case_id=case.case_id,
                    evaluator_name=evaluator_name,
                    stage="evaluate",
                    error=error,
                    checkpoint=checkpoint,
                    observation_state=(
                        observation_state
                    ),
                )
            )
            errors_written += 1
            continue

        try:
            result = evaluate(
                context
            )

            record = evaluation_record(
                run=run,
                context=context,
                result=result,
                evaluator=evaluator,
                checkpoint=checkpoint,
                observation_state=(
                    observation_state
                ),
            )

            store.append_evaluation(
                record
            )

            store.append_score_rows(
                score_rows_from_record(
                    record=record,
                    response=response,
                )
            )

            records_written += 1

        except Exception as error:
            store.append_evaluation(
                evaluation_error_record(
                    run=run,
                    session_id=case.session_id,
                    turn_number=case.turn_number,
                    round_number=getattr(
                        context,
                        "round_number",
                        0,
                    ),
                    case_id=case.case_id,
                    evaluator_name=evaluator_name,
                    stage="evaluate",
                    error=error,
                    checkpoint=checkpoint,
                    observation_state=(
                        observation_state
                    ),
                )
            )
            errors_written += 1

    return (
        records_written,
        errors_written,
    )


def execute_session(
    *,
    session: ConversationSession,
    round_number: int,
    provider: BaseProvider,
    factory: Any,
    run: StoredBenchmarkRun,
    store: ResponseStore,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    continue_on_error: bool,
    total_turns: int,
    progress_counter: list[int],
) -> tuple[
    dict[str, Any],
    int,
    int,
    int,
]:
    """
    Execute one continuing conversation session.

    Returns:
        transcript record,
        successful model responses,
        evaluator records,
        evaluator errors.
    """

    provider_history: list[
        dict[str, Any]
    ] = []

    transcript_messages: list[
        dict[str, Any]
    ] = []

    successful_responses = 0
    evaluator_records = 0
    evaluator_errors = 0

    session_failed = False

    for case in session.turns:
        progress_counter[0] += 1

        print(
            f"[{progress_counter[0]}/{total_turns}] "
            f"Round {round_number} | "
            f"{session.session_id} | "
            f"Turn {case.turn_number}/"
            f"{len(session)} | "
            f"{case.case_id}"
        )

        current_user_message = {
            "role": "user",
            "content": case.prompt,
            "metadata": {
                "session_id": session.session_id,
                "turn_number": case.turn_number,
                "case_id": case.case_id,
                "category": case.category,
            },
        }

        request_messages = [
            dict(message)
            for message in provider_history
        ]
        request_messages.append(
            current_user_message
        )

        started_at = utc_timestamp()

        try:
            response = provider.generate(
                messages=request_messages,
                system_prompt=session.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                metadata={
                    "run_id": run.run_id,
                    "round_number": round_number,
                    "session_id": session.session_id,
                    "turn_number": case.turn_number,
                    "case_id": case.case_id,
                    "category": case.category,
                    "checkpoint": bool(
                        case.metadata.get(
                            "checkpoint",
                            False,
                        )
                    ),
                    "observation_state": (
                        case.metadata.get(
                            "observation_state"
                        )
                    ),
                },
            )

        except Exception as error:
            failed_result = create_failed_result(
                case=case,
                round_number=round_number,
                error=error,
                started_at=started_at,
                transcript=request_messages,
                system_prompt=session.system_prompt,
            )

            run.add_result(
                failed_result
            )

            store.append_raw_response(
                {
                    "run_id": run.run_id,
                    "provider": run.provider,
                    "model": run.model,
                    "round_number": round_number,
                    "session_id": session.session_id,
                    "turn_number": case.turn_number,
                    "case_id": case.case_id,
                    "category": case.category,
                    "status": "failed",
                    "system_prompt": session.system_prompt,
                    "messages_sent": request_messages,
                    "started_at": started_at,
                    "completed_at": utc_timestamp(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

            transcript_messages.append(
                current_user_message
            )
            transcript_messages.append(
                {
                    "role": "provider_error",
                    "content": str(error),
                    "metadata": {
                        "turn_number": case.turn_number,
                        "error_type": (
                            type(error).__name__
                        ),
                    },
                }
            )

            print(
                "    Provider failed: "
                f"{type(error).__name__}: {error}"
            )

            session_failed = True

            if not continue_on_error:
                raise

            break

        completed_at = utc_timestamp()

        append_provider_evidence(
            store=store,
            run=run,
            session=session,
            case=case,
            round_number=round_number,
            request_messages=request_messages,
            response=response,
            completed_at=completed_at,
        )

        assistant_message = {
            "role": "assistant",
            "content": response.text,
            "metadata": {
                "session_id": session.session_id,
                "turn_number": case.turn_number,
                "case_id": case.case_id,
                "provider": response.provider,
                "model": response.model,
            },
        }

        evaluation_history = (
            evaluator_history_from_messages(
                provider_history
            )
        )

        context = create_evaluation_context(
            case=case,
            response=response,
            history=evaluation_history,
            session_id=session.session_id,
            turn_number=case.turn_number,
            round_number=round_number,
            metadata={
                "run_id": run.run_id,
                "system_prompt": session.system_prompt,
                "checkpoint": bool(
                    case.metadata.get(
                        "checkpoint",
                        False,
                    )
                ),
                "observation_state": (
                    case.metadata.get(
                        "observation_state"
                    )
                ),
            },
        )

        (
            written,
            errors,
        ) = execute_successful_evaluations(
            factory=factory,
            context=context,
            response=response,
            run=run,
            store=store,
            case=case,
        )

        evaluator_records += written
        evaluator_errors += errors

        provider_history.append(
            current_user_message
        )
        provider_history.append(
            assistant_message
        )

        transcript_messages.append(
            current_user_message
        )
        transcript_messages.append(
            assistant_message
        )

        completed_result = create_completed_result(
            case=case,
            round_number=round_number,
            response=response,
            started_at=started_at,
            transcript=provider_history,
            system_prompt=session.system_prompt,
        )

        run.add_result(
            completed_result
        )

        successful_responses += 1

        print(
            f"    {response.latency_seconds:.3f}s | "
            f"evaluators={written} | "
            f"evaluation_errors={errors}"
        )
        print(
            "    Response: "
            + compact_text(
                response.text
            )
        )

    return (
        {
            "round_number": round_number,
            "session_id": session.session_id,
            "name": session.name,
            "system_prompt": session.system_prompt,
            "planned_turns": len(session),
            "completed_model_responses": (
                successful_responses
            ),
            "session_failed": session_failed,
            "messages": transcript_messages,
        },
        successful_responses,
        evaluator_records,
        evaluator_errors,
    )


# ----------------------------------------------------------
# Complete Run
# ----------------------------------------------------------

def execute_benchmark(
    *,
    benchmark: Benchmark,
    benchmark_file: Path,
    benchmark_selection: str,
    provider: BaseProvider,
    factory: Any,
    rounds: int,
    output_directory: Path,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    continue_on_error: bool,
) -> tuple[
    StoredBenchmarkRun,
    Path,
]:
    """Execute the full provider-neutral conversation benchmark."""

    run_id = create_run_id(
        provider.provider_name,
        provider.default_model,
    )

    evaluator_registry = (
        validate_evaluator_registry(
            factory
        )
    )
    evaluator_names = tuple(
        str(descriptor["name"])
        for descriptor in evaluator_registry
    )

    run = StoredBenchmarkRun(
        run_id=run_id,
        project_name=PROJECT_NAME,
        project_version=VERSION,
        benchmark_name=benchmark.name,
        benchmark_version=benchmark.version,
        benchmark_file=str(
            benchmark_file
        ),
        provider=provider.provider_name,
        model=provider.default_model,
        requested_rounds=rounds,
        started_at=utc_timestamp(),
        metadata={
            "benchmark_sha256": file_sha256(
                benchmark_file
            ),
            "benchmark_selection": benchmark_selection,
            "source_format": (
                benchmark.source_format
            ),
            "benchmark_mode": benchmark_mode(
                benchmark
            ),
            "system_prompt_sessions": (
                system_prompt_session_count(
                    benchmark
                )
            ),
            "runner_adds_system_prompt": False,
            "full_history_resent_each_turn": True,
            "session_count": (
                benchmark.session_count
            ),
            "turns_per_round": (
                benchmark.turn_count
            ),
            "responses_per_session": (
                config.RESPONSES_PER_SESSION
            ),
            "observation_states": list(
                config.OBSERVATION_STATES
            ),
            "evaluate_every_turn": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "continue_on_error": (
                continue_on_error
            ),
            "evaluators": list(
                evaluator_names
            ),
            "evaluator_registry": [
                dict(descriptor)
                for descriptor in evaluator_registry
            ],
            "runner_file": str(
                Path(__file__).resolve()
            ),
            "runner_sha256": file_sha256(
                Path(__file__).resolve()
            ),
            "provider_class": (
                provider.__class__.__name__
            ),
            "provider_module": (
                provider.__class__.__module__
            ),
            "python_version": (
                platform.python_version()
            ),
            "platform": platform.platform(),
        },
    )

    store = ResponseStore(
        output_directory
    )
    run_directory = store.initialise(
        run
    )

    transcripts: list[
        dict[str, Any]
    ] = []

    total_planned_turns = (
        benchmark.turn_count
        * rounds
    )

    progress_counter = [0]

    successful_responses = 0
    evaluator_records = 0
    evaluator_errors = 0

    print("=" * 72)
    print(PROJECT_NAME)
    print("Conversation Benchmark Run")
    print("=" * 72)
    print(f"Run ID       : {run.run_id}")
    print(f"Benchmark    : {benchmark.name}")
    print(f"Benchmark file: {benchmark_file}")
    print(f"Selection    : {benchmark_selection}")
    print(f"Chat mode    : {benchmark_mode(benchmark)}")
    print(
        "System prompts: "
        f"{system_prompt_session_count(benchmark)} "
        "session(s)"
    )
    print(f"Provider     : {provider.provider_name}")
    print(f"Model        : {provider.default_model}")
    print(f"Rounds       : {rounds}")
    print(f"Sessions     : {benchmark.session_count * rounds}")
    print(f"Planned turns: {total_planned_turns}")
    print(f"Evidence     : {run_directory}")
    print(f"Evaluators   : {len(evaluator_registry)}")

    for descriptor in evaluator_registry:
        print(
            "  - "
            f"{descriptor['name']} "
            f"| version={descriptor['version']} "
            f"| scope={descriptor['scope']}"
        )

    print("=" * 72)

    fatal_error: Exception | None = None

    try:
        for round_number in range(
            1,
            rounds + 1,
        ):
            for session in benchmark.sessions:
                (
                    transcript,
                    response_count,
                    record_count,
                    error_count,
                ) = execute_session(
                    session=session,
                    round_number=round_number,
                    provider=provider,
                    factory=factory,
                    run=run,
                    store=store,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    continue_on_error=(
                        continue_on_error
                    ),
                    total_turns=total_planned_turns,
                    progress_counter=(
                        progress_counter
                    ),
                )

                transcripts.append(
                    transcript
                )
                successful_responses += (
                    response_count
                )
                evaluator_records += (
                    record_count
                )
                evaluator_errors += (
                    error_count
                )

    except Exception as error:
        fatal_error = error

    finally:
        run.finish()

        transcript_payload = {
            "run_id": run.run_id,
            "provider": run.provider,
            "model": run.model,
            "benchmark": run.benchmark_name,
            "rounds": rounds,
            "sessions": transcripts,
        }

        store.write_transcripts(
            transcript_payload
        )

        summary = run.to_summary()
        summary.update(
            {
                "planned_model_responses": (
                    total_planned_turns
                ),
                "successful_model_responses": (
                    successful_responses
                ),
                "missing_model_responses": (
                    total_planned_turns
                    - successful_responses
                ),
                "evaluator_results_written": (
                    evaluator_records
                ),
                "evaluator_errors_written": (
                    evaluator_errors
                ),
                "raw_provider_responses_file": str(
                    store.raw_responses_path
                ),
                "transcripts_file": str(
                    store.transcripts_path
                ),
                "evaluation_results_file": str(
                    store.evaluation_results_path
                ),
                "evaluation_scores_file": str(
                    store.score_csv_path
                ),
                "graphs_generated": False,
                "graph_count": 0,
                "report_generated": False,
                "report_file": None,
                "report_manifest_file": None,
                "fatal_error": (
                    {
                        "type": type(
                            fatal_error
                        ).__name__,
                        "message": str(
                            fatal_error
                        ),
                    }
                    if fatal_error is not None
                    else None
                ),
            }
        )

        store.write_summary(
            summary
        )

    if fatal_error is not None:
        raise fatal_error

    return (
        run,
        run_directory,
    )


# ----------------------------------------------------------
# Reporting
# ----------------------------------------------------------

def write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write a complete JSON object using atomic replacement."""

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)


def generate_run_report(
    *,
    run_directory: Path,
    reports_directory: Path,
    graphs_directory: Path,
    docs_directory: Path,
) -> Any:
    """
    Generate evidence-backed graphs and reports.

    The call is signature-aware so runner.py remains compatible
    with earlier report_generator.py versions. Only keyword
    arguments accepted by the installed generate_report()
    function are passed.

    Reporting is deliberately separate from model execution.
    """

    import report_generator

    generate_report = (
        report_generator.generate_report
    )
    signature = inspect.signature(
        generate_report
    )
    parameters = signature.parameters

    keyword_arguments: dict[str, Path] = {}

    if "reports_root" in parameters:
        keyword_arguments[
            "reports_root"
        ] = reports_directory

    if "graphs_root" in parameters:
        keyword_arguments[
            "graphs_root"
        ] = graphs_directory

    if "docs_root" in parameters:
        keyword_arguments[
            "docs_root"
        ] = docs_directory

    print(
        "    Report module : "
        f"{Path(report_generator.__file__).resolve()}"
    )
    print(
        "    Report API    : "
        f"{signature}"
    )

    outputs = generate_report(
        run_directory,
        **keyword_arguments,
    )

    summary_path = (
        run_directory
        / "run_summary.json"
    )
    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    chart_paths = tuple(
        getattr(
            outputs,
            "chart_paths",
            (),
        )
    )
    skipped_charts = tuple(
        getattr(
            outputs,
            "skipped_charts",
            (),
        )
    )

    report_path = getattr(
        outputs,
        "report_path",
        None,
    )
    manifest_path = getattr(
        outputs,
        "manifest_path",
        None,
    )
    document_path = getattr(
        outputs,
        "document_path",
        None,
    )

    summary_update: dict[str, Any] = {
        "graphs_generated": bool(
            chart_paths
        ),
        "graph_count": len(
            chart_paths
        ),
        "graph_files": [
            str(path)
            for path in chart_paths
        ],
        "report_generated": (
            report_path is not None
        ),
        "skipped_graphs": list(
            skipped_charts
        ),
        "report_generator_module": str(
            Path(
                report_generator.__file__
            ).resolve()
        ),
        "report_generator_signature": str(
            signature
        ),
    }

    if report_path is not None:
        summary_update[
            "report_file"
        ] = str(report_path)

    if manifest_path is not None:
        summary_update[
            "report_manifest_file"
        ] = str(manifest_path)

    if document_path is not None:
        summary_update[
            "documentation_file"
        ] = str(document_path)

    summary.update(
        summary_update
    )

    write_json_atomic(
        summary_path,
        summary,
    )

    return outputs


def validate_run_evidence_directory(
    run_directory: Path,
) -> None:
    """Validate the stored evidence needed for report generation."""

    required_files = (
        "run_manifest.json",
        "run_summary.json",
        "raw_provider_responses.jsonl",
        "transcripts.json",
        "evaluation_results.jsonl",
        "evaluation_scores.csv",
    )

    missing = [
        name
        for name in required_files
        if not (
            run_directory
            / name
        ).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Run evidence directory is incomplete. "
            "Missing: "
            + ", ".join(missing)
        )


def print_report_outputs(
    *,
    run_directory: Path,
    outputs: Any,
    heading: str,
) -> None:
    """Print generated report and graph locations."""

    chart_paths = tuple(
        getattr(
            outputs,
            "chart_paths",
            (),
        )
    )

    print("=" * 72)
    print(heading)
    print("=" * 72)
    print(f"Evidence   : {run_directory}")
    print(f"Graphs     : {len(chart_paths)}")
    print(
        "Report     : "
        f"{getattr(outputs, 'report_path', 'not generated')}"
    )

    document_path = getattr(
        outputs,
        "document_path",
        None,
    )

    if document_path is not None:
        print(
            f"Document   : {document_path}"
        )

    print("=" * 72)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main() -> int:
    """
    Run one of the centralised operational modes:

    - list evaluator implementations;
    - regenerate a report from stored evidence;
    - validate a benchmark configuration;
    - execute the benchmark, evaluators and report.
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    try:
        validate_arguments(
            arguments
        )

        reports_directory = (
            arguments.reports_dir
            .expanduser()
            .resolve()
        )
        graphs_directory = (
            arguments.graphs_dir
            .expanduser()
            .resolve()
        )
        docs_directory = (
            arguments.docs_dir
            .expanduser()
            .resolve()
        )

        if arguments.report_only is not None:
            run_directory = (
                arguments.report_only
                .expanduser()
                .resolve()
            )

            validate_run_evidence_directory(
                run_directory
            )

            print(
                "Regenerating graphs and report "
                "from stored evidence..."
            )

            report_outputs = generate_run_report(
                run_directory=run_directory,
                reports_directory=reports_directory,
                graphs_directory=graphs_directory,
                docs_directory=docs_directory,
            )

            print_report_outputs(
                run_directory=run_directory,
                outputs=report_outputs,
                heading="Report Regeneration Finished",
            )
            return 0

        factory = create_default_factory()
        evaluator_registry = (
            validate_evaluator_registry(
                factory
            )
        )

        if arguments.list_evaluators:
            print_evaluator_registry(
                evaluator_registry
            )
            return 0

        using_default_benchmark = (
            arguments.benchmark is None
        )

        benchmark_argument = (
            DEFAULT_BENCHMARK_FILE
            if using_default_benchmark
            else arguments.benchmark
        )

        if benchmark_argument is None:
            raise ValueError(
                "Benchmark path could not be resolved."
            )

        benchmark_file = (
            benchmark_argument
            .expanduser()
            .resolve()
        )
        output_directory = (
            arguments.output
            .expanduser()
            .resolve()
        )

        benchmark = load_benchmark(
            benchmark_file
        )
        validate_benchmark_design(
            benchmark
        )

        if using_default_benchmark:
            validate_default_operational_profile(
                benchmark=benchmark,
                evaluator_registry=evaluator_registry,
            )

        provider = create_provider(
            provider_name=(
                arguments.provider
            ),
            model=arguments.model,
        )
        provider.validate_configuration()

        if arguments.validate_only:
            print_validation_summary(
                benchmark=benchmark,
                benchmark_file=benchmark_file,
                benchmark_selection=(
                    "default unscaffolded"
                    if using_default_benchmark
                    else "explicit override"
                ),
                provider=provider,
                evaluator_registry=(
                    evaluator_registry
                ),
            )
            return 0

        run, run_directory = (
            execute_benchmark(
                benchmark=benchmark,
                benchmark_file=(
                    benchmark_file
                ),
                benchmark_selection=(
                    "default unscaffolded"
                    if using_default_benchmark
                    else "explicit override"
                ),
                provider=provider,
                factory=factory,
                rounds=arguments.rounds,
                output_directory=(
                    output_directory
                ),
                temperature=(
                    arguments.temperature
                ),
                max_tokens=(
                    arguments.max_tokens
                ),
                seed=arguments.seed,
                continue_on_error=(
                    arguments.continue_on_error
                ),
            )
        )

        report_outputs = None
        report_error: Exception | None = None

        if arguments.generate_report:
            print()
            print(
                "Generating graphs and report "
                "from stored evidence..."
            )

            try:
                report_outputs = generate_run_report(
                    run_directory=run_directory,
                    reports_directory=(
                        reports_directory
                    ),
                    graphs_directory=(
                        graphs_directory
                    ),
                    docs_directory=(
                        docs_directory
                    ),
                )

            except Exception as error:
                report_error = error

                summary_path = (
                    run_directory
                    / "run_summary.json"
                )

                try:
                    summary = json.loads(
                        summary_path.read_text(
                            encoding="utf-8"
                        )
                    )
                    summary.update(
                        {
                            "graphs_generated": False,
                            "report_generated": False,
                            "reporting_error_type": (
                                type(error).__name__
                            ),
                            "reporting_error_message": str(
                                error
                            ),
                        }
                    )
                    write_json_atomic(
                        summary_path,
                        summary,
                    )
                except Exception:
                    pass

                print(
                    "Reporting warning: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                print(
                    "The benchmark evidence is complete. "
                    "Rerun reporting through runner.py with:"
                )
                print(
                    "    python runner.py --report-only "
                    f"{run_directory}"
                )

    except (
        BenchmarkError,
        ProviderError,
        ResponseStoreError,
        FileNotFoundError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"Error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Interrupted by user.",
            file=sys.stderr,
        )
        return 130

    print("=" * 72)
    print("Benchmark Run Finished")
    print("=" * 72)
    print(f"Run ID     : {run.run_id}")
    print(f"Status     : {run.status}")
    print(f"Completed  : {run.completed_count}")
    print(f"Failed     : {run.failed_count}")
    print(f"Evidence   : {run_directory}")

    if report_outputs is not None:
        chart_paths = tuple(
            getattr(
                report_outputs,
                "chart_paths",
                (),
            )
        )

        print(
            f"Graphs     : "
            f"{len(chart_paths)}"
        )
        print(
            f"Graphs dir : "
            f"{chart_paths[0].parent}"
            if chart_paths
            else "Graphs dir : no graphs generated"
        )
        print(
            f"Report     : "
            f"{getattr(report_outputs, 'report_path', 'not generated')}"
        )

        document_path = getattr(
            report_outputs,
            "document_path",
            None,
        )

        if document_path is not None:
            print(
                f"Document   : "
                f"{document_path}"
            )

    elif report_error is not None:
        print(
            "Reporting  : failed after evidence completion"
        )
        print(
            f"Reason     : "
            f"{type(report_error).__name__}: "
            f"{report_error}"
        )

    print("=" * 72)

    if run.status == "failed":
        return 1

    if run.status == "completed_with_errors":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
