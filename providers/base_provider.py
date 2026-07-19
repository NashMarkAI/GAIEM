"""
==========================================================
GenAI Evaluation Matrix (GAIEM)
Base Provider Interface

Copyright (c) 2026 NashMarkAI
https://nashmarkai.com

Purpose
-------
Defines the common interface that every model provider must
implement.

Provider adapters accept either:

- one independent prompt; or
- an ordered conversation transcript containing user and
  assistant messages from the same continuing session.

Provider adapters return a standard ProviderResponse so the
benchmark and evaluation layers remain provider-neutral.
==========================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


# ----------------------------------------------------------
# Provider Exceptions
# ----------------------------------------------------------

class ProviderError(Exception):
    """
    Base exception for provider-related failures.
    """


class ProviderConfigurationError(ProviderError):
    """
    Raised when a provider is missing required configuration.
    """


class ProviderConnectionError(ProviderError):
    """
    Raised when a provider cannot be reached.
    """


class ProviderAuthenticationError(ProviderError):
    """
    Raised when provider authentication fails.
    """


class ProviderResponseError(ProviderError):
    """
    Raised when a provider returns an invalid response.
    """


# ----------------------------------------------------------
# Conversation Message
# ----------------------------------------------------------

@dataclass(frozen=True)
class GenerationMessage:
    """
    Provider-neutral conversation message.

    Attributes:
        role:
            Message role. GAIEM conversation execution uses
            user and assistant roles. System and developer
            roles are also accepted for provider adapters
            that support them.

        content:
            Message text sent to the model.

        metadata:
            Optional local benchmark information. Metadata is
            not automatically sent to the model.
    """

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate the conversation message.
        """

        if not isinstance(self.role, str):
            raise ValueError(
                "Conversation message role must be a string."
            )

        normalized_role = self.role.strip().lower()

        if normalized_role not in {
            "system",
            "developer",
            "user",
            "assistant",
        }:
            raise ValueError(
                "Conversation message role must be one of: "
                "system, developer, user, assistant."
            )

        if not isinstance(self.content, str):
            raise ValueError(
                "Conversation message content must be a string."
            )

        if not self.content.strip():
            raise ValueError(
                "Conversation message content cannot be empty."
            )

        if not isinstance(self.metadata, dict):
            raise ValueError(
                "Conversation message metadata must be a dictionary."
            )

        object.__setattr__(
            self,
            "role",
            normalized_role,
        )

    def to_dict(self) -> dict[str, str]:
        """
        Convert the message into provider input form.
        """

        return {
            "role": self.role,
            "content": self.content,
        }


# ----------------------------------------------------------
# Generation Request
# ----------------------------------------------------------

@dataclass(frozen=True)
class GenerationRequest:
    """
    Standard request sent to a model provider.

    Exactly one input mode must be supplied:

    - prompt:
        One independent user prompt.

    - messages:
        The complete ordered transcript for one continuing
        conversation, including the current user turn.

    Attributes:
        model:
            Provider-specific model identifier.

        prompt:
            Independent user prompt.

        messages:
            Ordered conversation transcript.

        system_prompt:
            Optional stable system-level instruction.

        temperature:
            Sampling temperature.

        max_tokens:
            Maximum number of output tokens.

        seed:
            Optional deterministic seed where supported.

        metadata:
            Additional request information.
    """

    model: str
    prompt: str | None = None
    messages: tuple[GenerationMessage, ...] = ()
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate request values after construction.
        """

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model cannot be empty.")

        has_prompt = (
            isinstance(self.prompt, str)
            and bool(self.prompt.strip())
        )

        if self.prompt is not None and not isinstance(
            self.prompt,
            str,
        ):
            raise ValueError(
                "Prompt must be a string or None."
            )

        if not isinstance(self.messages, tuple):
            raise ValueError(
                "Messages must be stored as a tuple."
            )

        for message in self.messages:
            if not isinstance(message, GenerationMessage):
                raise ValueError(
                    "Messages must contain GenerationMessage "
                    "objects."
                )

        has_messages = bool(self.messages)

        if has_prompt == has_messages:
            raise ValueError(
                "Supply exactly one request input: either "
                "'prompt' or 'messages'."
            )

        if self.system_prompt is not None:
            if not isinstance(self.system_prompt, str):
                raise ValueError(
                    "System prompt must be a string or None."
                )

            if not self.system_prompt.strip():
                raise ValueError(
                    "System prompt cannot be empty."
                )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "Temperature must be between 0.0 and 2.0."
            )

        if self.max_tokens < 1:
            raise ValueError(
                "max_tokens must be 1 or greater."
            )

        if not isinstance(self.metadata, dict):
            raise ValueError(
                "Request metadata must be a dictionary."
            )

    @property
    def is_conversation(self) -> bool:
        """
        Return True when the request contains a transcript.
        """

        return bool(self.messages)

    @property
    def message_count(self) -> int:
        """
        Return the number of supplied conversation messages.
        """

        return len(self.messages)


# ----------------------------------------------------------
# Provider Response
# ----------------------------------------------------------

@dataclass(frozen=True)
class ProviderResponse:
    """
    Standard response returned by every provider adapter.

    Attributes:
        provider:
            Provider name.

        model:
            Model used to generate the response.

        text:
            Generated response text.

        latency_seconds:
            Total request execution time.

        prompt_tokens:
            Number of input tokens where reported.

        completion_tokens:
            Number of generated tokens where reported.

        total_tokens:
            Total token usage where reported.

        finish_reason:
            Provider-specific completion reason.

        request_id:
            Provider request identifier where available.

        raw_response:
            Original provider response converted into a
            serialisable dictionary.

        metadata:
            Additional provider-specific data.
    """

    provider: str
    model: str
    text: str
    latency_seconds: float

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    finish_reason: str | None = None
    request_id: str | None = None

    raw_response: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate standard provider response fields.
        """

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("Provider name cannot be empty.")

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model name cannot be empty.")

        if not isinstance(self.text, str):
            raise ValueError("Response text must be a string.")

        if self.latency_seconds < 0:
            raise ValueError(
                "Latency cannot be negative."
            )

        if not isinstance(self.raw_response, dict):
            raise ValueError(
                "raw_response must be a dictionary."
            )

        if not isinstance(self.metadata, dict):
            raise ValueError(
                "Response metadata must be a dictionary."
            )

    @property
    def has_text(self) -> bool:
        """
        Return True when the response contains non-whitespace text.
        """

        return bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the response into a serialisable dictionary.
        """

        return {
            "provider": self.provider,
            "model": self.model,
            "text": self.text,
            "latency_seconds": self.latency_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "raw_response": self.raw_response,
            "metadata": self.metadata,
        }


# ----------------------------------------------------------
# Base Provider
# ----------------------------------------------------------

class BaseProvider(ABC):
    """
    Abstract base class for all model provider adapters.
    """

    provider_name: str = "base"

    def __init__(self, default_model: str) -> None:
        """
        Initialise the provider.

        Args:
            default_model:
                Model used when no model override is supplied.
        """

        if not isinstance(default_model, str):
            raise ProviderConfigurationError(
                "Default model must be a string."
            )

        cleaned_model = default_model.strip()

        if not cleaned_model:
            raise ProviderConfigurationError(
                "Default model cannot be empty."
            )

        self.default_model = cleaned_model

    @abstractmethod
    def validate_configuration(self) -> None:
        """
        Validate credentials, endpoints and provider settings.
        """

    @abstractmethod
    def _generate(
        self,
        request: GenerationRequest,
    ) -> ProviderResponse:
        """
        Execute one provider-specific generation request.

        Provider adapters implement this method.
        """

    @staticmethod
    def _normalise_messages(
        messages: Iterable[
            GenerationMessage | Mapping[str, Any]
        ],
    ) -> tuple[GenerationMessage, ...]:
        """
        Convert message objects or dictionaries into one
        validated immutable transcript.
        """

        if isinstance(messages, (str, bytes)):
            raise ValueError(
                "Messages must be an iterable of message "
                "objects, not a string."
            )

        normalised_messages: list[GenerationMessage] = []

        for index, message in enumerate(messages):
            if isinstance(message, GenerationMessage):
                normalised_messages.append(message)
                continue

            if not isinstance(message, Mapping):
                raise ValueError(
                    f"Message at index {index} must be a "
                    "GenerationMessage or mapping."
                )

            role = message.get("role")
            content = message.get("content")
            raw_metadata = message.get("metadata", {})

            if raw_metadata is None:
                raw_metadata = {}

            if not isinstance(raw_metadata, dict):
                raise ValueError(
                    f"Message metadata at index {index} "
                    "must be a dictionary."
                )

            normalised_messages.append(
                GenerationMessage(
                    role=role,
                    content=content,
                    metadata=dict(raw_metadata),
                )
            )

        if not normalised_messages:
            raise ValueError(
                "Conversation messages cannot be empty."
            )

        if normalised_messages[-1].role != "user":
            raise ValueError(
                "The final conversation message must have "
                "the role 'user' before generation."
            )

        return tuple(normalised_messages)

    def generate(
        self,
        prompt: str | None = None,
        *,
        messages: Iterable[
            GenerationMessage | Mapping[str, Any]
        ] | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        """
        Validate and execute one model generation request.

        Use prompt for an independent request.

        Use messages for a continuing conversation. The
        messages must contain the complete ordered transcript
        and end with the current user turn.
        """

        selected_model = (
            model.strip()
            if isinstance(model, str) and model.strip()
            else self.default_model
        )

        has_prompt = (
            isinstance(prompt, str)
            and bool(prompt.strip())
        )

        has_messages = messages is not None

        if has_prompt == has_messages:
            raise ValueError(
                "Supply exactly one input: either 'prompt' "
                "or 'messages'."
            )

        normalised_messages: tuple[
            GenerationMessage,
            ...,
        ] = ()

        if messages is not None:
            normalised_messages = self._normalise_messages(
                messages
            )

        request = GenerationRequest(
            model=selected_model,
            prompt=prompt if has_prompt else None,
            messages=normalised_messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            metadata=metadata or {},
        )

        self.validate_configuration()

        started_at = perf_counter()

        response = self._generate(request)

        measured_latency = perf_counter() - started_at

        if not isinstance(response, ProviderResponse):
            raise ProviderResponseError(
                f"{self.provider_name} returned an invalid "
                "response object."
            )

        if response.provider != self.provider_name:
            raise ProviderResponseError(
                "Provider response name does not match the "
                f"adapter name: expected '{self.provider_name}', "
                f"received '{response.provider}'."
            )

        if response.latency_seconds <= 0:
            response = ProviderResponse(
                provider=response.provider,
                model=response.model,
                text=response.text,
                latency_seconds=measured_latency,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                finish_reason=response.finish_reason,
                request_id=response.request_id,
                raw_response=response.raw_response,
                metadata=response.metadata,
            )

        return response

    def health_check(self) -> bool:
        """
        Validate whether the provider is configured.

        Provider adapters may override this method to perform a
        real connection test.
        """

        try:
            self.validate_configuration()
        except ProviderError:
            return False

        return True

    def __repr__(self) -> str:
        """
        Return a readable provider representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"provider_name='{self.provider_name}', "
            f"default_model='{self.default_model}'"
            f")"
        )


# ----------------------------------------------------------
# Stand-alone Test
# ----------------------------------------------------------

if __name__ == "__main__":

    class DummyProvider(BaseProvider):
        """
        Local provider used to verify request construction.
        """

        provider_name = "dummy"

        def validate_configuration(self) -> None:
            return None

        def _generate(
            self,
            request: GenerationRequest,
        ) -> ProviderResponse:
            if request.is_conversation:
                response_text = (
                    f"conversation_messages="
                    f"{request.message_count}"
                )
            else:
                response_text = (
                    f"prompt={request.prompt}"
                )

            return ProviderResponse(
                provider=self.provider_name,
                model=request.model,
                text=response_text,
                latency_seconds=0.0,
                metadata={
                    "is_conversation": (
                        request.is_conversation
                    ),
                    "message_count": (
                        request.message_count
                    ),
                },
            )

    provider = DummyProvider(
        default_model="dummy-model"
    )

    single_response = provider.generate(
        "Independent prompt"
    )

    conversation_response = provider.generate(
        messages=[
            {
                "role": "user",
                "content": "Turn one.",
            },
            {
                "role": "assistant",
                "content": "Response one.",
            },
            {
                "role": "user",
                "content": "Turn two.",
            },
        ]
    )

    print(
        f"Single prompt: "
        f"{single_response.text}"
    )

    print(
        f"Conversation: "
        f"{conversation_response.text}"
    )
