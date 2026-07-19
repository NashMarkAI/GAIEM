"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Base Evaluator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Defines the common conversation context, score, result,
and evaluator interfaces used throughout the GenAI
Evaluation Matrix.

Evaluators receive:

- the benchmark case;
- the current model response;
- the complete conversation history;
- the session identifier;
- the current turn number;
- the benchmark round number.

This allows evaluators to assess both individual responses
and behaviour across a continuing conversation.
==========================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------
# Conversation Message
# ---------------------------------------------------------

@dataclass(
    slots=True,
    frozen=True,
)
class ConversationMessage:
    """
    Represents one message in a continuing model session.

    Attributes:
        role:
            Message role, normally system, user, or assistant.

        content:
            Message text.

        turn_number:
            Conversation turn associated with the message.

        metadata:
            Optional provider or benchmark information.
    """

    role: str
    content: str
    turn_number: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate the immutable conversation message.
        """

        if not isinstance(self.role, str):
            raise TypeError(
                "Conversation message role "
                "must be a string."
            )

        if not self.role.strip():
            raise ValueError(
                "Conversation message role "
                "cannot be empty."
            )

        if not isinstance(self.content, str):
            raise TypeError(
                "Conversation message content "
                "must be a string."
            )

        if self.turn_number < 1:
            raise ValueError(
                "Conversation message turn_number "
                "must be 1 or greater."
            )


# ---------------------------------------------------------
# Evaluation Context
# ---------------------------------------------------------

@dataclass(slots=True)
class EvaluationContext:
    """
    Complete input supplied to every evaluator.

    The history contains all messages already present in
    the conversation before the current model response.

    This normally includes the current user message but
    does not include the current assistant response, which
    is supplied separately through response.
    """

    case: Any
    response: Any

    history: tuple[
        ConversationMessage,
        ...,
    ] = ()

    session_id: str = ""
    turn_number: int = 1
    round_number: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate the evaluation context.
        """

        if self.turn_number < 1:
            raise ValueError(
                "Evaluation turn_number must be "
                "1 or greater."
            )

        if self.round_number < 1:
            raise ValueError(
                "Evaluation round_number must be "
                "1 or greater."
            )

        if not isinstance(self.history, tuple):
            self.history = tuple(
                self.history
            )

        for message in self.history:

            if not isinstance(
                message,
                ConversationMessage,
            ):
                raise TypeError(
                    "Evaluation history must contain "
                    "ConversationMessage objects."
                )

    # -----------------------------------------------------

    @property
    def case_id(
        self,
    ) -> str:
        """
        Return the benchmark case identifier.
        """

        return str(
            getattr(
                self.case,
                "case_id",
                "unknown_case",
            )
        )

    # -----------------------------------------------------

    @property
    def category(
        self,
    ) -> str:
        """
        Return the benchmark category.
        """

        return str(
            getattr(
                self.case,
                "category",
                "general",
            )
        )

    # -----------------------------------------------------

    @property
    def current_text(
        self,
    ) -> str:
        """
        Return the current model response text.
        """

        value = getattr(
            self.response,
            "text",
            "",
        )

        if value is None:
            return ""

        return str(
            value
        )

    # -----------------------------------------------------

    @property
    def prior_assistant_messages(
        self,
    ) -> tuple[
        ConversationMessage,
        ...,
    ]:
        """
        Return assistant messages produced before the
        current response.
        """

        return tuple(
            message
            for message in self.history
            if message.role.lower() == "assistant"
        )

    # -----------------------------------------------------

    @property
    def prior_user_messages(
        self,
    ) -> tuple[
        ConversationMessage,
        ...,
    ]:
        """
        Return all user messages in the supplied history.
        """

        return tuple(
            message
            for message in self.history
            if message.role.lower() == "user"
        )

    # -----------------------------------------------------

    @property
    def prior_assistant_texts(
        self,
    ) -> tuple[str, ...]:
        """
        Return prior assistant response texts.
        """

        return tuple(
            message.content
            for message
            in self.prior_assistant_messages
        )

    # -----------------------------------------------------

    @property
    def prior_user_texts(
        self,
    ) -> tuple[str, ...]:
        """
        Return prior user message texts.
        """

        return tuple(
            message.content
            for message
            in self.prior_user_messages
        )

    # -----------------------------------------------------

    @property
    def full_conversation(
        self,
    ) -> tuple[
        ConversationMessage,
        ...,
    ]:
        """
        Return the supplied history plus the current model
        response as the final assistant message.
        """

        current_message = ConversationMessage(
            role="assistant",
            content=self.current_text,
            turn_number=self.turn_number,
            metadata={
                "current_response": True,
            },
        )

        return (
            *self.history,
            current_message,
        )

    # -----------------------------------------------------

    @property
    def assistant_response_count(
        self,
    ) -> int:
        """
        Return the number of assistant responses including
        the current response.
        """

        return (
            len(
                self.prior_assistant_messages
            )
            + 1
        )


# ---------------------------------------------------------
# Score
# ---------------------------------------------------------

@dataclass(slots=True)
class EvaluationScore:
    """
    Standardised evaluator score.
    """

    name: str
    score: float

    maximum: float = 1.0
    passed: bool = False
    confidence: float = 1.0


# ---------------------------------------------------------
# Result
# ---------------------------------------------------------

@dataclass(slots=True)
class EvaluationResult:
    """
    Result returned by every evaluator.
    """

    evaluator: str
    case_id: str
    passed: bool

    session_id: str = ""
    turn_number: int = 1
    round_number: int = 1
    scope: str = "turn"

    scores: list[EvaluationScore] = field(
        default_factory=list
    )

    summary: str = ""

    evidence: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # -----------------------------------------------------

    @property
    def total_score(
        self,
    ) -> float:
        """
        Return the combined evaluator score.
        """

        return sum(
            score.score
            for score in self.scores
        )

    # -----------------------------------------------------

    @property
    def maximum_score(
        self,
    ) -> float:
        """
        Return the combined maximum score.
        """

        return sum(
            score.maximum
            for score in self.scores
        )

    # -----------------------------------------------------

    @property
    def percentage(
        self,
    ) -> float:
        """
        Return the result percentage.
        """

        maximum = self.maximum_score

        if maximum == 0:
            return 0.0

        return (
            self.total_score
            / maximum
        ) * 100.0


# ---------------------------------------------------------
# Base Evaluator
# ---------------------------------------------------------

class BaseEvaluator(ABC):
    """
    Parent class for every evaluator.

    Evaluators may operate at either:

    - turn scope:
        the current response is assessed independently;

    - conversation scope:
        the current response is compared with earlier
        messages and responses from the same session.
    """

    evaluator_name = "Base Evaluator"
    version = "1.0"
    scope = "turn"

    def __init__(
        self,
    ) -> None:
        """
        Initialise the evaluator.
        """

        self.enabled = True

    # -----------------------------------------------------

    @abstractmethod
    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """
        Evaluate the current response within its complete
        conversation context.
        """

    # -----------------------------------------------------

    def supports(
        self,
        context: EvaluationContext,
    ) -> bool:
        """
        Return whether the evaluator supports the supplied
        benchmark and conversation state.
        """

        return True

    # -----------------------------------------------------

    @staticmethod
    def validate_score(
        score: float,
        maximum: float,
    ) -> float:
        """
        Clamp a score between zero and its maximum.
        """

        if maximum <= 0:
            raise ValueError(
                "Score maximum must be greater than zero."
            )

        if score < 0:
            return 0.0

        if score > maximum:
            return float(
                maximum
            )

        return float(
            score
        )

    # -----------------------------------------------------

    def create_score(
        self,
        *,
        name: str,
        score: float,
        maximum: float = 1.0,
        passed: bool | None = None,
        confidence: float = 1.0,
    ) -> EvaluationScore:
        """
        Create a validated evaluator score.
        """

        validated_score = self.validate_score(
            score,
            maximum,
        )

        validated_confidence = (
            self.validate_score(
                confidence,
                1.0,
            )
        )

        if passed is None:

            passed = (
                validated_score
                >= maximum
            )

        return EvaluationScore(
            name=name,
            score=validated_score,
            maximum=maximum,
            passed=passed,
            confidence=validated_confidence,
        )

    # -----------------------------------------------------

    def create_result(
        self,
        *,
        context: EvaluationContext,
        passed: bool,
    ) -> EvaluationResult:
        """
        Create a result containing the standard session and
        turn identifiers.
        """

        return EvaluationResult(
            evaluator=self.evaluator_name,
            case_id=context.case_id,
            passed=passed,
            session_id=context.session_id,
            turn_number=context.turn_number,
            round_number=context.round_number,
            scope=self.scope,
        )


# ==========================================================
# Stand-alone Test
# ==========================================================

if __name__ == "__main__":

    class DummyCase:

        case_id = "conversation_demo"
        category = "conversation_drift"


    class DummyResponse:

        text = (
            "The current model response."
        )


    history = (
        ConversationMessage(
            role="user",
            content="Initial user message.",
            turn_number=1,
        ),
        ConversationMessage(
            role="assistant",
            content="Initial assistant response.",
            turn_number=1,
        ),
        ConversationMessage(
            role="user",
            content="Follow-up user message.",
            turn_number=2,
        ),
    )

    context = EvaluationContext(
        case=DummyCase(),
        response=DummyResponse(),
        history=history,
        session_id="SESSION-001",
        turn_number=2,
        round_number=1,
    )

    print(
        f"Case ID: "
        f"{context.case_id}"
    )

    print(
        f"Session ID: "
        f"{context.session_id}"
    )

    print(
        f"Turn: "
        f"{context.turn_number}"
    )

    print(
        f"Prior assistant responses: "
        f"{len(context.prior_assistant_messages)}"
    )

    print(
        f"Assistant responses including current: "
        f"{context.assistant_response_count}"
    )

    print(
        f"Current response: "
        f"{context.current_text}"
    )