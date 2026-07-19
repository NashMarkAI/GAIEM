"""
==========================================================
GenAI Evaluation Matrix (GAIEM)
Provider Factory

Copyright (c) 2026 NashMarkAI
https://nashmarkai.com

Purpose
-------
Creates the requested provider adapter while keeping the
runner provider-neutral.

Operational baseline:
- Ollama

Optional external adapters:
- OpenAI
- DeepSeek

Declared extension placeholders:
- Anthropic
- Gemini
==========================================================
"""

from __future__ import annotations

import os

from .base_provider import (
    BaseProvider,
    ProviderConfigurationError,
)


def _load_local_environment() -> None:
    """Load a local .env file when python-dotenv is installed."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def _environment_float(name: str, default: float) -> float:
    """Read one positive floating-point environment value."""

    raw_value = os.environ.get(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value)
    except ValueError as error:
        raise ProviderConfigurationError(
            f"{name} must be numeric."
        ) from error

    if value <= 0:
        raise ProviderConfigurationError(
            f"{name} must be greater than zero."
        )

    return value


def _environment_integer(name: str, default: int) -> int:
    """Read one non-negative integer environment value."""

    raw_value = os.environ.get(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ProviderConfigurationError(
            f"{name} must be an integer."
        ) from error

    if value < 0:
        raise ProviderConfigurationError(
            f"{name} cannot be negative."
        )

    return value


def _environment_boolean(name: str, default: bool) -> bool:
    """Read a boolean environment value."""

    raw_value = os.environ.get(name)

    if raw_value is None or not raw_value.strip():
        return default

    normalized = raw_value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ProviderConfigurationError(
        f"{name} must be true or false."
    )


def _selected_model(
    requested_model: str | None,
    environment_name: str,
    default_model: str | None = None,
) -> str:
    """Resolve CLI override, environment value, or provider default."""

    if isinstance(requested_model, str) and requested_model.strip():
        return requested_model.strip()

    environment_model = os.environ.get(
        environment_name,
        "",
    ).strip()

    if environment_model:
        return environment_model

    if default_model:
        return default_model

    raise ProviderConfigurationError(
        f"No model was supplied. Set {environment_name} or pass --model."
    )


def available_providers() -> tuple[str, ...]:
    """Return provider names recognised by the factory."""

    return (
        "ollama",
        "openai",
        "deepseek",
        "anthropic",
        "gemini",
    )


def create_provider(
    provider_name: str,
    model: str | None = None,
) -> BaseProvider:
    """Create one provider adapter."""

    _load_local_environment()

    if not isinstance(provider_name, str):
        raise ProviderConfigurationError(
            "Provider name must be a string."
        )

    normalized_name = provider_name.strip().lower()

    if not normalized_name:
        raise ProviderConfigurationError(
            "Provider name cannot be empty."
        )

    if normalized_name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            default_model=_selected_model(
                model,
                "OLLAMA_MODEL",
                "llama3.2:latest",
            ),
            base_url=os.environ.get(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ),
            timeout_seconds=_environment_float(
                "OLLAMA_TIMEOUT_SECONDS",
                180.0,
            ),
        )

    if normalized_name == "openai":
        from .openai_provider import OpenAIProvider

        base_url = os.environ.get(
            "OPENAI_BASE_URL",
            "",
        ).strip() or None

        return OpenAIProvider(
            api_key=os.environ.get(
                "OPENAI_API_KEY",
                "",
            ),
            default_model=_selected_model(
                model,
                "OPENAI_MODEL",
            ),
            base_url=base_url,
            timeout_seconds=_environment_float(
                "OPENAI_TIMEOUT_SECONDS",
                120.0,
            ),
            max_retries=_environment_integer(
                "OPENAI_MAX_RETRIES",
                2,
            ),
            store_responses=_environment_boolean(
                "OPENAI_STORE_RESPONSES",
                False,
            ),
        )

    if normalized_name == "deepseek":
        from .deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(
            api_key=os.environ.get(
                "DEEPSEEK_API_KEY",
                "",
            ),
            default_model=_selected_model(
                model,
                "DEEPSEEK_MODEL",
                "deepseek-v4-flash",
            ),
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL",
                "https://api.deepseek.com",
            ),
            timeout_seconds=_environment_float(
                "DEEPSEEK_TIMEOUT_SECONDS",
                120.0,
            ),
            max_retries=_environment_integer(
                "DEEPSEEK_MAX_RETRIES",
                2,
            ),
        )

    if normalized_name == "anthropic":
        raise ProviderConfigurationError(
            "Anthropic is a declared extension placeholder. "
            "No Anthropic provider adapter is implemented."
        )

    if normalized_name == "gemini":
        raise ProviderConfigurationError(
            "Gemini is a declared extension placeholder. "
            "No Gemini provider adapter is implemented."
        )

    supported = ", ".join(available_providers())

    raise ProviderConfigurationError(
        f"Unsupported provider '{provider_name}'. "
        f"Recognised providers: {supported}."
    )


def main() -> int:
    """Verify that the factory creates the local Ollama adapter."""

    try:
        provider = create_provider(
            provider_name="ollama",
            model="llama3.2:latest",
        )
    except ProviderConfigurationError as error:
        print(f"Error: {error}")
        return 1

    print(f"Provider class : {provider.__class__.__name__}")
    print(f"Provider name  : {provider.provider_name}")
    print(f"Model          : {provider.default_model}")
    print(f"Health check   : {provider.health_check()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
