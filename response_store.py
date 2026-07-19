"""
==========================================================
GenAI Evaluation Matrix (GAIEM)
Append-Only Evidence Store

Copyright (c) 2026 NashMarkAI
https://nashmarkai.com

Purpose
-------
Stores one benchmark run as auditable evidence.

The application writes:
- run_manifest.json
- raw_provider_responses.jsonl
- transcripts.json
- evaluation_results.jsonl
- evaluation_scores.csv
- run_summary.json

JSONL and CSV evidence is appended. Existing run directories
are not reused, preventing accidental replacement of an
earlier run.
==========================================================
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


class ResponseStoreError(Exception):
    """Raised when run evidence cannot be stored safely."""


def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """
    Convert common Python objects into JSON-serialisable data
    without altering strings, numbers, booleans or dictionaries.
    """

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item)
            for item in value
        ]

    to_dict = getattr(value, "to_dict", None)

    if callable(to_dict):
        return _json_safe(to_dict())

    return str(value)


def _require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """Validate one required text value."""

    if not isinstance(value, str):
        raise ResponseStoreError(
            f"{field_name} must be a string."
        )

    cleaned = value.strip()

    if not cleaned:
        raise ResponseStoreError(
            f"{field_name} cannot be empty."
        )

    return cleaned


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, Any] | list[Any],
    *,
    allow_replace: bool = False,
) -> None:
    """Write one complete JSON document atomically."""

    if path.exists() and not allow_replace:
        raise ResponseStoreError(
            f"Evidence file already exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialisable = _json_safe(payload)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(
                serialisable,
                temporary_file,
                ensure_ascii=False,
                indent=2,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

            temporary_path = Path(
                temporary_file.name
            )

        if path.exists() and not allow_replace:
            raise ResponseStoreError(
                f"Evidence file already exists: {path}"
            )

        temporary_path.replace(path)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def _append_json_line(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Append one complete JSON object as one physical line."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    try:
        with path.open(
            "a",
            encoding="utf-8",
        ) as output_file:
            output_file.write(encoded)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

    except OSError as error:
        raise ResponseStoreError(
            f"Could not append evidence to {path}: {error}"
        ) from error


@dataclass
class StoredTurnResult:
    """One provider execution inside a conversation session."""

    session_id: str
    turn_number: int
    case_id: str
    category: str
    round_number: int

    status: str
    prompt: str

    started_at: str
    completed_at: str

    system_prompt: str | None = None
    transcript: list[dict[str, Any]] = field(
        default_factory=list
    )

    response_text: str | None = None
    latency_seconds: float | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    finish_reason: str | None = None
    request_id: str | None = None

    raw_provider_response: dict[str, Any] = field(
        default_factory=dict
    )
    provider_metadata: dict[str, Any] = field(
        default_factory=dict
    )

    error_type: str | None = None
    error_message: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.session_id = _require_non_empty_string(
            self.session_id,
            "session_id",
        )
        self.case_id = _require_non_empty_string(
            self.case_id,
            "case_id",
        )
        self.category = _require_non_empty_string(
            self.category,
            "category",
        )
        self.prompt = _require_non_empty_string(
            self.prompt,
            "prompt",
        )

        if self.turn_number < 1:
            raise ResponseStoreError(
                "turn_number must be at least 1."
            )

        if self.round_number < 1:
            raise ResponseStoreError(
                "round_number must be at least 1."
            )

        if self.status not in {
            "completed",
            "failed",
        }:
            raise ResponseStoreError(
                "Turn status must be completed or failed."
            )

        if self.status == "completed":
            if self.response_text is None:
                raise ResponseStoreError(
                    "A completed turn requires response_text."
                )

        if self.status == "failed":
            if not self.error_message:
                raise ResponseStoreError(
                    "A failed turn requires error_message."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete stored turn record."""

        return _json_safe(asdict(self))


@dataclass
class StoredBenchmarkRun:
    """Mutable in-memory state for one benchmark execution."""

    run_id: str

    project_name: str
    project_version: str

    benchmark_name: str
    benchmark_version: str
    benchmark_file: str

    provider: str
    model: str

    requested_rounds: int
    started_at: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    status: str = "running"
    completed_at: str | None = None

    results: list[StoredTurnResult] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.run_id = _require_non_empty_string(
            self.run_id,
            "run_id",
        )

        if self.requested_rounds < 1:
            raise ResponseStoreError(
                "requested_rounds must be at least 1."
            )

    @property
    def completed_count(self) -> int:
        return sum(
            result.status == "completed"
            for result in self.results
        )

    @property
    def failed_count(self) -> int:
        return sum(
            result.status == "failed"
            for result in self.results
        )

    @property
    def total_case_executions(self) -> int:
        return len(self.results)

    def add_result(
        self,
        result: StoredTurnResult,
    ) -> None:
        """Add one turn result while the run is active."""

        if self.status != "running":
            raise ResponseStoreError(
                "Cannot add a result after the run finishes."
            )

        if not isinstance(result, StoredTurnResult):
            raise ResponseStoreError(
                "result must be a StoredTurnResult."
            )

        identity = (
            result.round_number,
            result.session_id,
            result.turn_number,
            result.case_id,
        )

        existing_identities = {
            (
                existing.round_number,
                existing.session_id,
                existing.turn_number,
                existing.case_id,
            )
            for existing in self.results
        }

        if identity in existing_identities:
            raise ResponseStoreError(
                "Duplicate turn result: "
                f"round={result.round_number}, "
                f"session={result.session_id}, "
                f"turn={result.turn_number}, "
                f"case={result.case_id}"
            )

        self.results.append(result)

    def finish(self) -> None:
        """Close the run and derive its final status."""

        if self.status != "running":
            return

        self.completed_at = utc_timestamp()

        if not self.results:
            self.status = "failed"
        elif self.failed_count == len(self.results):
            self.status = "failed"
        elif self.failed_count:
            self.status = "completed_with_errors"
        else:
            self.status = "completed"

    def to_manifest(self) -> dict[str, Any]:
        """Return immutable run-start information."""

        return {
            "run_id": self.run_id,
            "project_name": self.project_name,
            "project_version": self.project_version,
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "benchmark_file": self.benchmark_file,
            "provider": self.provider,
            "model": self.model,
            "requested_rounds": self.requested_rounds,
            "started_at": self.started_at,
            "metadata": self.metadata,
        }

    def to_summary(self) -> dict[str, Any]:
        """Return final run counts and status."""

        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "provider": self.provider,
            "model": self.model,
            "completed_turns": self.completed_count,
            "failed_turns": self.failed_count,
            "total_turn_executions": (
                self.total_case_executions
            ),
        }


SCORE_COLUMNS = (
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


class ResponseStore:
    """Append-only filesystem store for one unique run."""

    def __init__(
        self,
        results_root: str | Path,
        run_id: str | None = None,
    ) -> None:
        self.results_root = Path(
            results_root
        ).expanduser()

        self.run_id = (
            _require_non_empty_string(
                run_id,
                "run_id",
            )
            if run_id is not None
            else None
        )

        self.run_directory: Path | None = None

        self._saved_result_count: dict[str, int] = {}

        if self.run_id is not None:
            self._bind_new_run_directory(
                self.run_id
            )

    def _bind_new_run_directory(
        self,
        run_id: str,
    ) -> Path:
        """Create a unique run directory exactly once."""

        run_directory = (
            self.results_root
            / run_id
        )

        try:
            run_directory.mkdir(
                parents=True,
                exist_ok=False,
            )
        except FileExistsError as error:
            raise ResponseStoreError(
                "Run directory already exists; refusing "
                f"to overwrite evidence: {run_directory}"
            ) from error
        except OSError as error:
            raise ResponseStoreError(
                f"Could not create run directory: {error}"
            ) from error

        self.run_id = run_id
        self.run_directory = run_directory

        return run_directory

    def initialise(
        self,
        run: StoredBenchmarkRun,
    ) -> Path:
        """Create the run directory and write its manifest."""

        if self.run_directory is None:
            self._bind_new_run_directory(
                run.run_id
            )
        elif self.run_id != run.run_id:
            raise ResponseStoreError(
                "ResponseStore is bound to a different run."
            )

        self.write_manifest(
            run.to_manifest()
        )

        return self._required_run_directory()

    def _required_run_directory(self) -> Path:
        if self.run_directory is None:
            raise ResponseStoreError(
                "The store has not been initialised."
            )

        return self.run_directory

    @property
    def manifest_path(self) -> Path:
        return (
            self._required_run_directory()
            / "run_manifest.json"
        )

    @property
    def raw_responses_path(self) -> Path:
        return (
            self._required_run_directory()
            / "raw_provider_responses.jsonl"
        )

    @property
    def transcripts_path(self) -> Path:
        return (
            self._required_run_directory()
            / "transcripts.json"
        )

    @property
    def evaluation_results_path(self) -> Path:
        return (
            self._required_run_directory()
            / "evaluation_results.jsonl"
        )

    @property
    def score_csv_path(self) -> Path:
        return (
            self._required_run_directory()
            / "evaluation_scores.csv"
        )

    @property
    def summary_path(self) -> Path:
        return (
            self._required_run_directory()
            / "run_summary.json"
        )

    @property
    def charts_directory(self) -> Path:
        path = (
            self._required_run_directory()
            / "charts"
        )
        path.mkdir(
            parents=True,
            exist_ok=True,
        )
        return path

    def write_manifest(
        self,
        manifest: Mapping[str, Any],
    ) -> Path:
        """Write the run manifest once."""

        _atomic_write_json(
            self.manifest_path,
            dict(manifest),
        )
        return self.manifest_path

    def append_raw_response(
        self,
        record: Mapping[str, Any],
    ) -> Path:
        """
        Append one provider turn.

        The record should include the complete provider response
        dictionary under raw_provider_response. That dictionary
        is written without deleting provider-specific fields.
        """

        payload = dict(record)
        payload.setdefault(
            "run_id",
            self.run_id,
        )

        _append_json_line(
            self.raw_responses_path,
            payload,
        )

        return self.raw_responses_path

    def append_evaluation(
        self,
        record: Mapping[str, Any],
    ) -> Path:
        """Append one complete evaluator result."""

        payload = dict(record)
        payload.setdefault(
            "run_id",
            self.run_id,
        )

        _append_json_line(
            self.evaluation_results_path,
            payload,
        )

        return self.evaluation_results_path

    def append_score_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> Path:
        """Append one CSV row per evaluator score component."""

        rows_list = [
            dict(row)
            for row in rows
        ]

        if not rows_list:
            return self.score_csv_path

        path = self.score_csv_path
        write_header = not path.exists()

        try:
            with path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=SCORE_COLUMNS,
                    extrasaction="ignore",
                )

                if write_header:
                    writer.writeheader()

                for row in rows_list:
                    normalised = {
                        column: _json_safe(
                            row.get(column)
                        )
                        for column in SCORE_COLUMNS
                    }

                    for column, value in normalised.items():
                        if isinstance(
                            value,
                            (dict, list),
                        ):
                            normalised[column] = json.dumps(
                                value,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )

                    writer.writerow(normalised)

                output_file.flush()
                os.fsync(output_file.fileno())

        except OSError as error:
            raise ResponseStoreError(
                f"Could not append score rows: {error}"
            ) from error

        return path

    def write_transcripts(
        self,
        transcripts: Mapping[str, Any] | list[Any],
    ) -> Path:
        """Write the completed transcripts once."""

        _atomic_write_json(
            self.transcripts_path,
            transcripts,
        )
        return self.transcripts_path

    def write_summary(
        self,
        summary: Mapping[str, Any],
    ) -> Path:
        """Write the final run summary once."""

        _atomic_write_json(
            self.summary_path,
            dict(summary),
        )
        return self.summary_path

    def save(
        self,
        run: StoredBenchmarkRun,
    ) -> Path:
        """
        Compatibility method for the existing runner.

        It writes the manifest once and appends only turn results
        that have not already been stored by this process.
        """

        if self.run_directory is None:
            self.initialise(run)

        elif not self.manifest_path.exists():
            self.write_manifest(
                run.to_manifest()
            )

        saved_count = self._saved_result_count.get(
            run.run_id,
            0,
        )

        for result in run.results[saved_count:]:
            self.append_raw_response(
                {
                    "run_id": run.run_id,
                    "provider": run.provider,
                    "model": run.model,
                    **result.to_dict(),
                }
            )

        self._saved_result_count[
            run.run_id
        ] = len(run.results)

        if (
            run.status != "running"
            and not self.summary_path.exists()
        ):
            self.write_summary(
                run.to_summary()
            )

        return self._required_run_directory()


def create_completed_result(
    *,
    case: Any,
    round_number: int,
    response: Any,
    started_at: str,
    transcript: Iterable[Mapping[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> StoredTurnResult:
    """Create one successful turn record."""

    response_dictionary = (
        response.to_dict()
        if callable(
            getattr(response, "to_dict", None)
        )
        else _json_safe(response)
    )

    if not isinstance(
        response_dictionary,
        dict,
    ):
        raise ResponseStoreError(
            "Provider response must serialise to a dictionary."
        )

    return StoredTurnResult(
        session_id=(
            getattr(case, "session_id", "")
            or "legacy"
        ),
        turn_number=int(
            getattr(case, "turn_number", 1)
        ),
        case_id=str(
            getattr(case, "case_id", "")
        ),
        category=str(
            getattr(case, "category", "")
        ),
        round_number=round_number,
        status="completed",
        prompt=str(
            getattr(case, "prompt", "")
        ),
        system_prompt=system_prompt,
        transcript=[
            dict(message)
            for message in (transcript or [])
        ],
        started_at=started_at,
        completed_at=utc_timestamp(),
        response_text=response_dictionary.get(
            "text",
            "",
        ),
        latency_seconds=response_dictionary.get(
            "latency_seconds"
        ),
        prompt_tokens=response_dictionary.get(
            "prompt_tokens"
        ),
        completion_tokens=response_dictionary.get(
            "completion_tokens"
        ),
        total_tokens=response_dictionary.get(
            "total_tokens"
        ),
        finish_reason=response_dictionary.get(
            "finish_reason"
        ),
        request_id=response_dictionary.get(
            "request_id"
        ),
        raw_provider_response=dict(
            response_dictionary.get(
                "raw_response",
                {},
            )
        ),
        provider_metadata=dict(
            response_dictionary.get(
                "metadata",
                {},
            )
        ),
        metadata={
            "checkpoint": getattr(
                case,
                "metadata",
                {},
            ).get(
                "checkpoint",
                False,
            ),
            "observation_state": getattr(
                case,
                "metadata",
                {},
            ).get(
                "observation_state"
            ),
        },
    )


def create_failed_result(
    *,
    case: Any,
    round_number: int,
    error: Exception,
    started_at: str,
    transcript: Iterable[Mapping[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> StoredTurnResult:
    """Create one failed turn record."""

    return StoredTurnResult(
        session_id=(
            getattr(case, "session_id", "")
            or "legacy"
        ),
        turn_number=int(
            getattr(case, "turn_number", 1)
        ),
        case_id=str(
            getattr(case, "case_id", "")
        ),
        category=str(
            getattr(case, "category", "")
        ),
        round_number=round_number,
        status="failed",
        prompt=str(
            getattr(case, "prompt", "")
        ),
        system_prompt=system_prompt,
        transcript=[
            dict(message)
            for message in (transcript or [])
        ],
        started_at=started_at,
        completed_at=utc_timestamp(),
        error_type=type(error).__name__,
        error_message=str(error),
        metadata={
            "checkpoint": getattr(
                case,
                "metadata",
                {},
            ).get(
                "checkpoint",
                False,
            ),
            "observation_state": getattr(
                case,
                "metadata",
                {},
            ).get(
                "observation_state"
            ),
        },
    )


def main() -> int:
    """
    Validate the evidence-store implementation in a temporary
    directory. No benchmark result is created.
    """

    with tempfile.TemporaryDirectory() as directory:
        run = StoredBenchmarkRun(
            run_id="storage-self-test",
            project_name="GenAI Evaluation Matrix",
            project_version="0.2.0",
            benchmark_name="Storage self-test",
            benchmark_version="0",
            benchmark_file="not-a-benchmark-run",
            provider="none",
            model="none",
            requested_rounds=1,
            started_at=utc_timestamp(),
            metadata={
                "self_test": True,
            },
        )

        store = ResponseStore(
            directory
        )
        store.initialise(run)

        run.finish()
        store.save(run)

        required_files = {
            "run_manifest.json",
            "run_summary.json",
        }

        created_files = {
            path.name
            for path in store.run_directory.iterdir()
        }

        missing = (
            required_files
            - created_files
        )

        if missing:
            print(
                "Storage self-test failed. Missing: "
                + ", ".join(sorted(missing))
            )
            return 1

    print("Response store self-test: passed")
    print("No benchmark responses were generated.")
    print("No result directory was retained.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
