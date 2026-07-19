"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Exact Match Evaluator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Checks whether the current model response exactly matches
the benchmark-defined expected answer.

This evaluator operates on one turn inside a continuing
conversation.

It does NOT:
- measure conversational drift;
- compare earlier assistant responses;
- perform semantic matching;
- verify external facts.
==========================================================
"""

from __future__ import annotations

from evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
)


class ExactMatchEvaluator(BaseEvaluator):
    """
    Deterministic exact-response evaluator.
    """

    evaluator_name = "Exact Match"
    version = "1.0"
    scope = "turn"

    # -----------------------------------------------------

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        """
        Remove surrounding whitespace without changing
        spelling, punctuation, or letter case.
        """

        return value.strip()

    # -----------------------------------------------------

    def supports(
        self,
        context: EvaluationContext,
    ) -> bool:
        """
        Return True when the benchmark defines a non-empty
        expected answer.
        """

        expected_answer = getattr(
            context.case,
            "expected_answer",
            None,
        )

        return bool(
            isinstance(
                expected_answer,
                str,
            )
            and expected_answer.strip()
        )

    # -----------------------------------------------------

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """
        Compare the current response with the benchmark's
        expected answer.
        """

        expected_answer = getattr(
            context.case,
            "expected_answer",
            None,
        )

        if not isinstance(
            expected_answer,
            str,
        ):
            raise TypeError(
                "'expected_answer' must be a string."
            )

        if not expected_answer.strip():
            raise ValueError(
                "'expected_answer' must not be empty."
            )

        actual_text = context.current_text

        if not isinstance(
            actual_text,
            str,
        ):
            raise TypeError(
                "The response text must be a string."
            )

        normalized_expected = self._normalize_text(
            expected_answer
        )

        normalized_actual = self._normalize_text(
            actual_text
        )

        passed = (
            normalized_actual
            == normalized_expected
        )

        score = (
            1.0
            if passed
            else 0.0
        )

        result = self.create_result(
            context=context,
            passed=passed,
        )

        result.scores.append(
            self.create_score(
                name="Exact Match",
                score=score,
                maximum=1.0,
                passed=passed,
                confidence=1.0,
            )
        )

        if passed:

            result.summary = (
                "The current response exactly matches "
                "the expected answer."
            )

        else:

            result.summary = (
                "The current response does not exactly "
                "match the expected answer."
            )

        result.evidence = {
            "expected": normalized_expected,
            "actual": normalized_actual,
            "comparison": (
                "case-sensitive exact string match "
                "after removing surrounding whitespace"
            ),
        }

        result.metadata = {
            "session_id": context.session_id,
            "turn_number": context.turn_number,
            "round_number": context.round_number,
            "assistant_response_count": (
                context.assistant_response_count
            ),
        }

        return result


# ==========================================================
# Stand-alone Test
# ==========================================================

if __name__ == "__main__":

    from evaluators.base_evaluator import (
        ConversationMessage,
    )

    class DummyCase:

        case_id = "exact_match_demo"
        category = "goal_retention"
        expected_answer = "42"


    class DummyResponse:

        text = "42"


    history = (
        ConversationMessage(
            role="user",
            content=(
                "Remember that the required final "
                "answer is 42."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="assistant",
            content=(
                "Understood. The required answer is 42."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="user",
            content=(
                "Now return only the required answer."
            ),
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

    evaluator = ExactMatchEvaluator()

    print(
        f"Supported: "
        f"{evaluator.supports(context)}"
    )

    result = evaluator.evaluate(
        context
    )

    print(result)

    print(
        f"Percentage: "
        f"{result.percentage:.1f}%"
    )

    for score in result.scores:

        print(
            f"{score.name}: "
            f"{score.score:.2f} / "
            f"{score.maximum:.2f} | "
            f"passed={score.passed}"
        )