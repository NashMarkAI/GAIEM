"""
==========================================================
GenAI Evaluation Matrix (GAIEM)
Ollama Provider Adapter

Copyright (c) 2026 NashMarkAI
https://nashmarkai.com

Purpose
-------
Connects GAIEM to a local Ollama installation through the
/api/chat endpoint.

The adapter accepts either:

- one independent prompt; or
- the complete ordered transcript for one continuing
  conversation.

Every provider response is returned through the shared
ProviderResponse class and includes the original Ollama JSON
for later evidence storage.
==========================================================
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .base_provider import (
    BaseProvider,
    GenerationRequest,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderResponse,
    ProviderResponseError,
)


class OllamaProvider(BaseProvider):
    """
    Local Ollama provider using the native /api/chat route.

    The runner remains responsible for maintaining session
    history. For a continuing conversation, the runner passes
    every previous user and assistant message plus the current
    user message to BaseProvider.generate(messages=...).
    """

    provider_name = "ollama"

    def __init__(
        self,
        default_model: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 180.0,
    ) -> None:
        """
        Initialise the Ollama adapter.

        Args:
            default_model:
                Ollama model identifier used when the runner
                does not provide an override.

            base_url:
                Root URL of the local Ollama service.

            timeout_seconds:
                Maximum duration of one HTTP request.
        """

        super().__init__(default_model=default_model)

        if not isinstance(base_url, str):
            raise ProviderConfigurationError(
                "Ollama base URL must be a string."
            )

        cleaned_url = base_url.strip().rstrip("/")

        if not cleaned_url:
            raise ProviderConfigurationError(
                "Ollama base URL cannot be empty."
            )

        if not isinstance(timeout_seconds, (int, float)):
            raise ProviderConfigurationError(
                "Ollama timeout must be numeric."
            )

        if timeout_seconds <= 0:
            raise ProviderConfigurationError(
                "Ollama timeout must be greater than zero."
            )

        self.base_url = cleaned_url
        self.timeout_seconds = float(timeout_seconds)

    @property
    def chat_url(self) -> str:
        """Return the Ollama chat endpoint URL."""

        return f"{self.base_url}/api/chat"

    @property
    def tags_url(self) -> str:
        """Return the Ollama model-list endpoint URL."""

        return f"{self.base_url}/api/tags"

    def validate_configuration(self) -> None:
        """
        Validate local configuration without making a request.

        BaseProvider invokes this before every generation, so
        it deliberately performs no network call.
        """

        parsed_url = urlparse(self.base_url)

        if parsed_url.scheme not in {"http", "https"}:
            raise ProviderConfigurationError(
                "Ollama base URL must use http or https."
            )

        if not parsed_url.netloc:
            raise ProviderConfigurationError(
                "Ollama base URL must include a host and port."
            )

    @staticmethod
    def _provider_messages(
        request: GenerationRequest,
    ) -> list[dict[str, str]]:
        """
        Convert a provider-neutral request into Ollama messages.

        A system prompt is inserted once at the beginning. A
        developer message is mapped to a system message because
        Ollama's portable chat roles are system, user and
        assistant.
        """

        messages: list[dict[str, str]] = []

        if request.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )

        if request.is_conversation:
            for message in request.messages:
                role = (
                    "system"
                    if message.role == "developer"
                    else message.role
                )

                messages.append(
                    {
                        "role": role,
                        "content": message.content,
                    }
                )
        else:
            if request.prompt is None:
                raise ProviderResponseError(
                    "Ollama request contains no prompt or messages."
                )

            messages.append(
                {
                    "role": "user",
                    "content": request.prompt,
                }
            )

        if not messages:
            raise ProviderResponseError(
                "Ollama request contains no messages."
            )

        return messages

    @staticmethod
    def _decode_json_response(
        response_body: bytes,
        *,
        endpoint: str,
    ) -> dict[str, Any]:
        """Decode and validate one Ollama JSON response."""

        try:
            decoded = response_body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProviderResponseError(
                f"Ollama returned non-UTF-8 data from {endpoint}."
            ) from error

        try:
            data = json.loads(decoded)
        except json.JSONDecodeError as error:
            raise ProviderResponseError(
                f"Ollama returned invalid JSON from {endpoint}: "
                f"{decoded[:300]}"
            ) from error

        if not isinstance(data, dict):
            raise ProviderResponseError(
                "Ollama response must be a JSON object."
            )

        return data

    def _request_json(
        self,
        *,
        url: str,
        method: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one JSON request against the Ollama service."""

        encoded_payload: bytes | None = None
        headers = {"Accept": "application/json"}

        if payload is not None:
            encoded_payload = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            url=url,
            data=encoded_payload,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read()

        except HTTPError as error:
            error_body = error.read().decode(
                "utf-8",
                errors="replace",
            )

            raise ProviderResponseError(
                "Ollama returned HTTP "
                f"{error.code} from {url}: "
                f"{error_body[:500]}"
            ) from error

        except (URLError, TimeoutError, OSError) as error:
            raise ProviderConnectionError(
                f"Could not connect to Ollama at {url}: {error}"
            ) from error

        return self._decode_json_response(
            response_body,
            endpoint=url,
        )

    def _generate(
        self,
        request: GenerationRequest,
    ) -> ProviderResponse:
        """
        Send one prompt or full transcript to /api/chat.
        """

        messages = self._provider_messages(request)

        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        }

        if request.seed is not None:
            options["seed"] = request.seed

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        started_at = perf_counter()

        data = self._request_json(
            url=self.chat_url,
            method="POST",
            payload=payload,
        )

        latency_seconds = perf_counter() - started_at

        message = data.get("message")

        if not isinstance(message, dict):
            raise ProviderResponseError(
                "Ollama response is missing the 'message' object."
            )

        text = message.get("content")

        if not isinstance(text, str):
            raise ProviderResponseError(
                "Ollama response message is missing string content."
            )

        returned_model = data.get("model", request.model)

        if not isinstance(returned_model, str) or not returned_model.strip():
            returned_model = request.model

        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")

        if not isinstance(prompt_tokens, int):
            prompt_tokens = None

        if not isinstance(completion_tokens, int):
            completion_tokens = None

        total_tokens: int | None = None

        if (
            prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens

        finish_reason = data.get("done_reason")

        if finish_reason is not None and not isinstance(
            finish_reason,
            str,
        ):
            finish_reason = str(finish_reason)

        created_at = data.get("created_at")
        request_id = (
            created_at
            if isinstance(created_at, str)
            else None
        )

        return ProviderResponse(
            provider=self.provider_name,
            model=returned_model,
            text=text,
            latency_seconds=latency_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            request_id=request_id,
            raw_response=data,
            metadata={
                "base_url": self.base_url,
                "endpoint": self.chat_url,
                "request_mode": (
                    "conversation"
                    if request.is_conversation
                    else "prompt"
                ),
                "message_count": len(messages),
                "system_prompt_supplied": (
                    request.system_prompt is not None
                ),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "seed": request.seed,
                "ollama_total_duration_ns": data.get(
                    "total_duration"
                ),
                "ollama_load_duration_ns": data.get(
                    "load_duration"
                ),
                "ollama_prompt_eval_duration_ns": data.get(
                    "prompt_eval_duration"
                ),
                "ollama_eval_duration_ns": data.get(
                    "eval_duration"
                ),
            },
        )

    def health_check(self) -> bool:
        """
        Return True only when Ollama responds and the configured
        default model appears in /api/tags.
        """

        try:
            self.validate_configuration()

            data = self._request_json(
                url=self.tags_url,
                method="GET",
            )
        except (
            ProviderConfigurationError,
            ProviderConnectionError,
            ProviderResponseError,
        ):
            return False

        models = data.get("models")

        if not isinstance(models, list):
            return False

        available_names = {
            model.get("name")
            for model in models
            if isinstance(model, dict)
            and isinstance(model.get("name"), str)
        }

        available_models = {
            model.get("model")
            for model in models
            if isinstance(model, dict)
            and isinstance(model.get("model"), str)
        }

        return self.default_model in (
            available_names | available_models
        )


# ----------------------------------------------------------
# Stand-alone Local Test
# ----------------------------------------------------------

if __name__ == "__main__":
    provider = OllamaProvider(
        default_model="llama3.2:latest",
        base_url="http://localhost:11434",
    )

    print(f"Health check: {provider.health_check()}")

    response = provider.generate(
        messages=[
            {
                "role": "user",
                "content": (
                    "Record that Project Cedar is owned "
                    "by Priya."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Project Cedar is owned by Priya."
                ),
            },
            {
                "role": "user",
                "content": "Who owns Project Cedar?",
            },
        ],
        temperature=0.0,
        max_tokens=256,
        seed=42,
    )

    print(f"Provider       : {response.provider}")
    print(f"Model          : {response.model}")
    print(f"Prompt tokens  : {response.prompt_tokens}")
    print(f"Output tokens  : {response.completion_tokens}")
    print(f"Finish reason  : {response.finish_reason}")
    print(f"Latency        : {response.latency_seconds:.3f}s")
    print(f"Response       : {response.text}")
