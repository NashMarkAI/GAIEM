"""
==========================================================
GenAI Evaluation Matrix (GAIEM)
OpenAI Provider Adapter

Copyright (c) 2026 NashMarkAI
https://nashmarkai.com

Purpose
-------
Connects the provider-neutral evaluation framework to the
OpenAI Responses API.

The adapter supports:

- one independent prompt; and
- a complete ordered conversation transcript containing
  prior user and assistant messages from the same session.
==========================================================
"""

from __future__ import annotations

from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from providers.base_provider import (
    BaseProvider,
    GenerationRequest,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderResponse,
    ProviderResponseError,
)


class OpenAIProvider(BaseProvider):
    """
    OpenAI implementation of the common provider interface.
    """

    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        default_model: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        store_responses: bool = False,
    ) -> None:
        """
        Initialise the OpenAI provider.

        Args:
            api_key:
                OpenAI API key.

            default_model:
                Default OpenAI model identifier.

            base_url:
                Optional alternative API endpoint.

            timeout_seconds:
                Maximum time allowed for one API request.

            max_retries:
                Number of automatic retries performed by the
                OpenAI SDK.

            store_responses:
                Whether OpenAI should persist response objects.
                GAIEM defaults this to False.
        """

        super().__init__(default_model=default_model)

        self.api_key = api_key.strip() if api_key else ""
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.store_responses = store_responses

        self._client: OpenAI | None = None

    def validate_configuration(self) -> None:
        """
        Validate the OpenAI provider configuration.
        """

        if not self.api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is missing. "
                "Add it to the local .env file."
            )

        if self.timeout_seconds <= 0:
            raise ProviderConfigurationError(
                "OpenAI timeout must be greater than zero."
            )

        if self.max_retries < 0:
            raise ProviderConfigurationError(
                "OpenAI max_retries cannot be negative."
            )

        if not isinstance(self.store_responses, bool):
            raise ProviderConfigurationError(
                "OpenAI store_responses must be a boolean."
            )

    def _get_client(self) -> OpenAI:
        """
        Create the OpenAI client only when first required.
        """

        if self._client is None:
            client_arguments: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout_seconds,
                "max_retries": self.max_retries,
            }

            if self.base_url:
                client_arguments["base_url"] = self.base_url

            self._client = OpenAI(**client_arguments)

        return self._client

    @staticmethod
    def _build_input(
        request: GenerationRequest,
    ) -> str | list[dict[str, str]]:
        """
        Build the Responses API input value.

        Independent requests use one string.

        Conversation requests use the complete ordered list
        of user and assistant messages supplied by GAIEM.
        """

        if request.is_conversation:
            return [
                message.to_dict()
                for message in request.messages
            ]

        if request.prompt is None:
            raise ProviderConfigurationError(
                "OpenAI request contains no prompt input."
            )

        return request.prompt

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """
        Extract generated text from an OpenAI response.

        The SDK's output_text convenience property is used
        first. A structured-output fallback is retained.
        """

        output_text = getattr(
            response,
            "output_text",
            None,
        )

        if isinstance(output_text, str):
            return output_text

        collected_text: list[str] = []

        for output_item in (
            getattr(response, "output", [])
            or []
        ):
            for content_item in (
                getattr(output_item, "content", [])
                or []
            ):
                text = getattr(
                    content_item,
                    "text",
                    None,
                )

                if isinstance(text, str):
                    collected_text.append(text)

        return "\n".join(collected_text)

    @staticmethod
    def _extract_usage(
        response: Any,
    ) -> tuple[int | None, int | None, int | None]:
        """
        Extract token usage values where available.
        """

        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None:
            return None, None, None

        prompt_tokens = getattr(
            usage,
            "input_tokens",
            None,
        )

        completion_tokens = getattr(
            usage,
            "output_tokens",
            None,
        )

        total_tokens = getattr(
            usage,
            "total_tokens",
            None,
        )

        return (
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

    @staticmethod
    def _serialise_response(
        response: Any,
    ) -> dict[str, Any]:
        """
        Convert the SDK response into a JSON-compatible
        dictionary.
        """

        if hasattr(response, "model_dump"):
            return response.model_dump(
                mode="json"
            )

        if hasattr(response, "to_dict"):
            return response.to_dict()

        return {
            "id": getattr(
                response,
                "id",
                None,
            ),
            "model": getattr(
                response,
                "model",
                None,
            ),
            "status": getattr(
                response,
                "status",
                None,
            ),
        }

    def _generate(
        self,
        request: GenerationRequest,
    ) -> ProviderResponse:
        """
        Execute one request through the OpenAI Responses API.
        """

        client = self._get_client()

        request_arguments: dict[str, Any] = {
            "model": request.model,
            "input": self._build_input(request),
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "store": self.store_responses,
        }

        if request.system_prompt:
            request_arguments[
                "instructions"
            ] = request.system_prompt

        if request.seed is not None:
            request_arguments[
                "seed"
            ] = request.seed

        try:
            response = client.responses.create(
                **request_arguments
            )

        except AuthenticationError as error:
            raise ProviderAuthenticationError(
                "OpenAI authentication failed. "
                "Check OPENAI_API_KEY."
            ) from error

        except RateLimitError as error:
            raise ProviderConnectionError(
                "OpenAI rate limit or usage limit reached."
            ) from error

        except APIConnectionError as error:
            raise ProviderConnectionError(
                "Unable to connect to the OpenAI API."
            ) from error

        except APIStatusError as error:
            raise ProviderResponseError(
                "OpenAI API request failed with status "
                f"{error.status_code}: {error.message}"
            ) from error

        except OpenAIError as error:
            raise ProviderError(
                f"OpenAI request failed: {error}"
            ) from error

        response_text = self._extract_output_text(
            response
        )

        if not response_text.strip():
            raise ProviderResponseError(
                "OpenAI returned no generated text."
            )

        (
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ) = self._extract_usage(
            response
        )

        response_model = getattr(
            response,
            "model",
            request.model,
        )

        response_status = getattr(
            response,
            "status",
            None,
        )

        request_id = getattr(
            response,
            "id",
            None,
        )

        return ProviderResponse(
            provider=self.provider_name,
            model=response_model,
            text=response_text,
            latency_seconds=0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=response_status,
            request_id=request_id,
            raw_response=self._serialise_response(
                response
            ),
            metadata={
                **request.metadata,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "seed": request.seed,
                "request_mode": (
                    "conversation"
                    if request.is_conversation
                    else "single_prompt"
                ),
                "conversation_message_count": (
                    request.message_count
                ),
                "store_response": (
                    self.store_responses
                ),
            },
        )


def main() -> int:
    """
    Perform direct provider tests using config.py.

    The first call is independent. The second call supplies
    a three-message transcript and therefore tests the
    conversation input path.
    """

    from config import config

    try:
        provider = OpenAIProvider(
            api_key=config.OPENAI_API_KEY,
            default_model=config.DEFAULT_MODEL,
        )

        independent_response = provider.generate(
            "Respond with exactly one word: equilibrium",
            temperature=0.0,
            max_tokens=20,
        )

        conversation_response = provider.generate(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Remember that Project Cedar is "
                        "owned by Priya."
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
                    "content": (
                        "Who owns Project Cedar?"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=40,
        )

    except ProviderError as error:
        print(f"Error: {error}")
        return 1

    print("=" * 60)
    print("OpenAI Provider Test")
    print("=" * 60)
    print(
        f"Independent response : "
        f"{independent_response.text}"
    )
    print(
        f"Conversation response: "
        f"{conversation_response.text}"
    )
    print(
        f"Conversation messages: "
        f"{conversation_response.metadata.get(
            'conversation_message_count'
        )}"
    )
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
