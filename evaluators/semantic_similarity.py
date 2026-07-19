"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Semantic Similarity Evaluator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Measures deterministic token similarity between the
current model response and the benchmark-defined expected
answer.

This evaluator operates on one response at a specified
turn within a continuing conversation.

It does NOT:
- measure drift across conversation turns;
- verify factual accuracy;
- infer semantic meaning using an external model;
- compare separate conversations.
==========================================================
"""

from __future__ import annotations

import re

from evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
)


class SemanticSimilarityEvaluator(BaseEvaluator):
    """
    Deterministic normalized-token similarity evaluator.

    The default comparison method is Jaccard token overlap:

        intersection size / union size
    """

    evaluator_name = "Semantic Similarity"
    version = "1.0"
    scope = "turn"

    similarity_threshold = 0.80

    # -----------------------------------------------------

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        """
        Normalize text for deterministic comparison.
        """

        text = value.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # -----------------------------------------------------

    @staticmethod
    def _tokenize(
        value: str,
    ) -> set[str]:
        """
        Extract lowercase alphanumeric word tokens.
        """

        return set(
            re.findall(
                r"\b[\w']+\b",
                value.lower(),
            )
        )

    # -----------------------------------------------------

    @staticmethod
    def _jaccard_similarity(
        expected_tokens: set[str],
        actual_tokens: set[str],
    ) -> float:
        """
        Calculate Jaccard token similarity.
        """

        if (
            not expected_tokens
            and not actual_tokens
        ):
            return 1.0

        union = (
            expected_tokens
            | actual_tokens
        )

        if not union:
            return 0.0

        intersection = (
            expected_tokens
            & actual_tokens
        )

        return (
            len(intersection)
            / len(union)
        )

    # -----------------------------------------------------

    def supports(
        self,
        context: EvaluationContext,
    ) -> bool:
        """
        Return True when the case defines a non-empty
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
        Compare the current response with the expected
        answer using deterministic token overlap.
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

        normalized_expected = self._normalize_text(
            expected_answer
        )

        normalized_actual = self._normalize_text(
            actual_text
        )

        expected_tokens = self._tokenize(
            normalized_expected
        )

        actual_tokens = self._tokenize(
            normalized_actual
        )

        score = self._jaccard_similarity(
            expected_tokens,
            actual_tokens,
        )

        score = self.validate_score(
            score,
            maximum=1.0,
        )

        passed = (
            score
            >= self.similarity_threshold
        )

        result = self.create_result(
            context=context,
            passed=passed,
        )

        result.scores.append(
            self.create_score(
                name="Semantic Similarity",
                score=score,
                maximum=1.0,
                passed=passed,
                confidence=1.0,
            )
        )

        result.summary = (
            f"Semantic similarity: {score:.3f}"
        )

        result.evidence = {
            "expected": normalized_expected,
            "actual": normalized_actual,
            "expected_tokens": sorted(
                expected_tokens
            ),
            "actual_tokens": sorted(
                actual_tokens
            ),
            "shared_tokens": sorted(
                expected_tokens
                & actual_tokens
            ),
            "missing_tokens": sorted(
                expected_tokens
                - actual_tokens
            ),
            "additional_tokens": sorted(
                actual_tokens
                - expected_tokens
            ),
        }

        result.metadata = {
            "threshold": (
                self.similarity_threshold
            ),
            "comparison_method": (
                "normalized Jaccard token overlap"
            ),
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

        case_id = "semantic_similarity_demo"
        category = "semantic_similarity"

        expected_answer = (
            "Paris is the capital of France."
        )


    class DummyResponse:

        text = (
            "Paris is the capital city of France."
        )


    history = (
        ConversationMessage(
            role="user",
            content=(
                "We are discussing European capitals."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="assistant",
            content=(
                "France is a country in Europe."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="user",
            content=(
                "What is the capital of France?"
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

    evaluator = SemanticSimilarityEvaluator()

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