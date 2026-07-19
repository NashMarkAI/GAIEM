"""
==========================================================
GenAI Evaluation Matrix (GAIEM)
Application Configuration

Copyright (c) 2026 NashMarkAI
https://nashmarkai.com

Purpose
-------
Defines project paths and default execution settings.

Ollama is the operational baseline provider. Environment
variables or command-line arguments may override defaults.
==========================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_NAME = "GenAI Evaluation Matrix"
VERSION = os.getenv("GAIEM_VERSION", "0.1.0")

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_local_environment() -> None:
    """Load the project's local .env file when available."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(PROJECT_ROOT / ".env")


def _path_from_environment(
    variable_name: str,
    default_relative_path: str,
) -> Path:
    """
    Resolve a configurable project path.

    Relative values are resolved from the project root.
    Absolute values are preserved.
    """

    raw_value = os.getenv(
        variable_name,
        default_relative_path,
    ).strip()

    candidate = Path(raw_value).expanduser()

    if candidate.is_absolute():
        return candidate

    return PROJECT_ROOT / candidate


def _float_from_environment(
    variable_name: str,
    default: float,
) -> float:
    """Read and validate one floating-point setting."""

    raw_value = os.getenv(variable_name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{variable_name} must be numeric."
        ) from error


def _integer_from_environment(
    variable_name: str,
    default: int,
) -> int:
    """Read and validate one integer setting."""

    raw_value = os.getenv(variable_name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{variable_name} must be an integer."
        ) from error


def _optional_integer_from_environment(
    variable_name: str,
    default: int | None,
) -> int | None:
    """Read an optional integer setting."""

    raw_value = os.getenv(variable_name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{variable_name} must be an integer or empty."
        ) from error


_load_local_environment()


@dataclass(frozen=True)
class AppConfig:
    """
    Immutable application configuration.

    The runner already expects these public attributes:
    DEFAULT_PROVIDER
    DEFAULT_MODEL
    BENCHMARK_DIR
    RESULTS_DIR
    """

    DEFAULT_PROVIDER: str
    DEFAULT_MODEL: str

    BENCHMARK_DIR: Path
    RESULTS_DIR: Path

    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str
    OLLAMA_TIMEOUT_SECONDS: float

    DEFAULT_TEMPERATURE: float
    DEFAULT_MAX_TOKENS: int
    DEFAULT_SEED: int | None

    RESPONSES_PER_SESSION: int
    OBSERVATION_STATES: tuple[int, ...]

    def validate(self) -> None:
        """Validate configuration without contacting a provider."""

        if not self.DEFAULT_PROVIDER.strip():
            raise ValueError(
                "DEFAULT_PROVIDER cannot be empty."
            )

        if not self.DEFAULT_MODEL.strip():
            raise ValueError(
                "DEFAULT_MODEL cannot be empty."
            )

        if not self.OLLAMA_BASE_URL.strip():
            raise ValueError(
                "OLLAMA_BASE_URL cannot be empty."
            )

        if self.OLLAMA_TIMEOUT_SECONDS <= 0:
            raise ValueError(
                "OLLAMA_TIMEOUT_SECONDS must be greater than zero."
            )

        if not 0.0 <= self.DEFAULT_TEMPERATURE <= 2.0:
            raise ValueError(
                "DEFAULT_TEMPERATURE must be between 0.0 and 2.0."
            )

        if self.DEFAULT_MAX_TOKENS < 1:
            raise ValueError(
                "DEFAULT_MAX_TOKENS must be at least 1."
            )

        if self.RESPONSES_PER_SESSION != 9:
            raise ValueError(
                "The controlled benchmark requires exactly "
                "9 responses per session."
            )

        if self.OBSERVATION_STATES != (1, 3, 7, 9):
            raise ValueError(
                "Observation states must be exactly 1, 3, 7 and 9."
            )

        if self.OBSERVATION_STATES[-1] != (
            self.RESPONSES_PER_SESSION
        ):
            raise ValueError(
                "The final observation state must equal the "
                "session response count."
            )

    def ensure_output_directory(self) -> Path:
        """Create and return the root results directory."""

        self.RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self.RESULTS_DIR


config = AppConfig(
    DEFAULT_PROVIDER=os.getenv(
        "DEFAULT_PROVIDER",
        "ollama",
    ).strip(),
    DEFAULT_MODEL=os.getenv(
        "DEFAULT_MODEL",
        os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:latest",
        ),
    ).strip(),
    BENCHMARK_DIR=_path_from_environment(
        "BENCHMARK_DIR",
        "benchmarks",
    ),
    RESULTS_DIR=_path_from_environment(
        "RESULTS_DIR",
        "results",
    ),
    OLLAMA_BASE_URL=os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ).rstrip("/"),
    OLLAMA_MODEL=os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:latest",
    ).strip(),
    OLLAMA_TIMEOUT_SECONDS=_float_from_environment(
        "OLLAMA_TIMEOUT_SECONDS",
        180.0,
    ),
    DEFAULT_TEMPERATURE=_float_from_environment(
        "DEFAULT_TEMPERATURE",
        0.0,
    ),
    DEFAULT_MAX_TOKENS=_integer_from_environment(
        "DEFAULT_MAX_TOKENS",
        1024,
    ),
    DEFAULT_SEED=_optional_integer_from_environment(
        "DEFAULT_SEED",
        42,
    ),
    RESPONSES_PER_SESSION=9,
    OBSERVATION_STATES=(1, 3, 7, 9),
)


def main() -> int:
    """Run the standalone configuration validation."""

    try:
        config.validate()
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    default_benchmark = (
        config.BENCHMARK_DIR
        / "default.json"
    )

    print(f"Project              : {PROJECT_NAME}")
    print(f"Version              : {VERSION}")
    print(f"Default provider     : {config.DEFAULT_PROVIDER}")
    print(f"Default model        : {config.DEFAULT_MODEL}")
    print(f"Ollama endpoint      : {config.OLLAMA_BASE_URL}")
    print(f"Benchmark            : {default_benchmark}")
    print(f"Benchmark exists     : {default_benchmark.is_file()}")
    print(f"Results directory    : {config.RESULTS_DIR}")
    print(f"Responses per session: {config.RESPONSES_PER_SESSION}")
    print(
        "Observation states   : "
        + ", ".join(
            str(value)
            for value in config.OBSERVATION_STATES
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
