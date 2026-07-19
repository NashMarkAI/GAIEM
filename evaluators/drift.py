"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Conversation Drift Evaluator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Measures configured response drift across one continuing
conversation.

The evaluator compares the current assistant response with:

- the configured benchmark baseline;
- the first assistant response in the session;
- the immediately preceding assistant response;
- benchmark-defined protected terms.

It detects:

- deviation from the intended baseline;
- movement away from the initial response;
- sudden change from the previous turn;
- loss of protected facts, terms, instructions, medicines,
  symptoms, constraints, or other conversation anchors.

This deterministic evaluator uses normalized lexical token
overlap and configured protected-term matching.

It does NOT:
- determine whether ordinary conversation development is
  drift;
- perform external factual verification;
- infer semantic equivalence with another AI model;
- operate without earlier responses from the same session.

The benchmark must explicitly configure baseline_response
or protected_terms before this evaluator will run.
==========================================================
"""

from __future__ import annotations

import re
from typing import Any

from evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
)


class DriftEvaluator(BaseEvaluator):
    """
    Deterministic conversation-level drift evaluator.

    Supported benchmark fields:

    - baseline_response
    - drift_threshold
    - protected_terms
    """

    evaluator_name = "Conversation Drift"
    version = "1.0"
    scope = "conversation"

    default_drift_threshold = 0.80

    # -----------------------------------------------------

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:
        """
        Normalize text for deterministic comparison.
        """

        if value is None:
            return ""

        text = str(value).lower()

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
        first_tokens: set[str],
        second_tokens: set[str],
    ) -> float:
        """
        Calculate normalized Jaccard token overlap.
        """

        if (
            not first_tokens
            and not second_tokens
        ):
            return 1.0

        union = (
            first_tokens
            | second_tokens
        )

        if not union:
            return 0.0

        intersection = (
            first_tokens
            & second_tokens
        )

        return (
            len(intersection)
            / len(union)
        )

    # -----------------------------------------------------

    @staticmethod
    def _contains_term(
        text: str,
        term: str,
    ) -> bool:
        """
        Match a complete configured word or phrase without
        accidental partial substring matching.
        """

        term_parts = term.strip().split()

        if not term_parts:
            return False

        pattern = (
            r"(?<!\w)"
            + r"\s+".join(
                re.escape(part)
                for part in term_parts
            )
            + r"(?!\w)"
        )

        return bool(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        )

    # -----------------------------------------------------

    @classmethod
    def _extract_protected_terms(
        cls,
        raw_terms: Any,
    ) -> list[str]:
        """
        Convert protected_terms into a validated list.
        """

        if raw_terms is None:
            return []

        if isinstance(
            raw_terms,
            str,
        ):

            term_items = [
                raw_terms,
            ]

        elif isinstance(
            raw_terms,
            (list, tuple, set),
        ):

            term_items = list(
                raw_terms
            )

        else:

            raise TypeError(
                "'protected_terms' must be a string, "
                "list, tuple, or set."
            )

        protected_terms = []

        for term in term_items:

            if not isinstance(
                term,
                str,
            ):
                raise TypeError(
                    "Each protected term must be "
                    "a string."
                )

            normalized_term = cls._normalize_text(
                term
            )

            if not normalized_term:
                raise ValueError(
                    "Protected terms must not be empty."
                )

            if normalized_term not in protected_terms:

                protected_terms.append(
                    normalized_term
                )

        return protected_terms

    # -----------------------------------------------------

    @classmethod
    def _extract_baseline(
        cls,
        context: EvaluationContext,
    ) -> str:
        """
        Return the configured normalized baseline response.
        """

        baseline = getattr(
            context.case,
            "baseline_response",
            None,
        )

        if baseline is None:
            return ""

        if not isinstance(
            baseline,
            str,
        ):
            raise TypeError(
                "'baseline_response' must be a string "
                "or None."
            )

        return cls._normalize_text(
            baseline
        )

    # -----------------------------------------------------

    def _extract_threshold(
        self,
        context: EvaluationContext,
    ) -> float:
        """
        Return and validate the configured drift threshold.
        """

        threshold = getattr(
            context.case,
            "drift_threshold",
            None,
        )

        if threshold is None:

            threshold = (
                self.default_drift_threshold
            )

        if (
            not isinstance(
                threshold,
                (int, float),
            )
            or isinstance(
                threshold,
                bool,
            )
        ):
            raise TypeError(
                "'drift_threshold' must be a number."
            )

        threshold = float(
            threshold
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "'drift_threshold' must be between "
                "0.0 and 1.0."
            )

        return threshold

    # -----------------------------------------------------

    def _compare_texts(
        self,
        *,
        comparison_name: str,
        reference_text: str,
        current_text: str,
        threshold: float,
        reference_turn: int | None,
    ) -> dict[str, Any]:
        """
        Compare current response text with one reference.
        """

        reference_tokens = self._tokenize(
            reference_text
        )

        current_tokens = self._tokenize(
            current_text
        )

        shared_tokens = (
            reference_tokens
            & current_tokens
        )

        lost_tokens = (
            reference_tokens
            - current_tokens
        )

        introduced_tokens = (
            current_tokens
            - reference_tokens
        )

        similarity = self._jaccard_similarity(
            reference_tokens,
            current_tokens,
        )

        passed = (
            similarity
            >= threshold
        )

        return {
            "name": comparison_name,
            "reference_turn": reference_turn,
            "reference_text": reference_text,
            "current_text": current_text,
            "similarity": similarity,
            "threshold": threshold,
            "passed": passed,
            "shared_tokens": sorted(
                shared_tokens
            ),
            "lost_tokens": sorted(
                lost_tokens
            ),
            "introduced_tokens": sorted(
                introduced_tokens
            ),
        }

    # -----------------------------------------------------

    def supports(
        self,
        context: EvaluationContext,
    ) -> bool:
        """
        Return True only when:

        - the session contains an earlier assistant response;
        - and the benchmark configures a baseline response
          or protected terms.
        """

        if not context.prior_assistant_messages:
            return False

        baseline = self._extract_baseline(
            context
        )

        protected_terms = (
            self._extract_protected_terms(
                getattr(
                    context.case,
                    "protected_terms",
                    None,
                )
            )
        )

        return bool(
            baseline
            or protected_terms
        )

    # -----------------------------------------------------

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """
        Measure drift in the current response against
        earlier responses from the same conversation.
        """

        prior_messages = (
            context.prior_assistant_messages
        )

        if not prior_messages:
            raise ValueError(
                "Conversation drift requires at least "
                "one earlier assistant response from "
                "the same session."
            )

        current_text = self._normalize_text(
            context.current_text
        )

        baseline = self._extract_baseline(
            context
        )

        protected_terms = (
            self._extract_protected_terms(
                getattr(
                    context.case,
                    "protected_terms",
                    None,
                )
            )
        )

        if not baseline and not protected_terms:
            raise ValueError(
                "The case must configure either "
                "'baseline_response' or "
                "'protected_terms'."
            )

        threshold = self._extract_threshold(
            context
        )

        first_message = prior_messages[0]
        previous_message = prior_messages[-1]

        first_text = self._normalize_text(
            first_message.content
        )

        previous_text = self._normalize_text(
            previous_message.content
        )

        comparisons = []

        # ==================================================
        # Configured Baseline
        # ==================================================

        if baseline:

            comparisons.append(
                self._compare_texts(
                    comparison_name=(
                        "Configured Baseline Retention"
                    ),
                    reference_text=baseline,
                    current_text=current_text,
                    threshold=threshold,
                    reference_turn=None,
                )
            )

        # ==================================================
        # First Conversation Response
        # ==================================================

        comparisons.append(
            self._compare_texts(
                comparison_name=(
                    "Initial Turn Retention"
                ),
                reference_text=first_text,
                current_text=current_text,
                threshold=threshold,
                reference_turn=(
                    first_message.turn_number
                ),
            )
        )

        # ==================================================
        # Immediately Previous Response
        # ==================================================

        if (
            previous_message.turn_number
            != first_message.turn_number
        ):

            comparisons.append(
                self._compare_texts(
                    comparison_name=(
                        "Previous Turn Stability"
                    ),
                    reference_text=previous_text,
                    current_text=current_text,
                    threshold=threshold,
                    reference_turn=(
                        previous_message.turn_number
                    ),
                )
            )

        # ==================================================
        # Protected Terms
        # ==================================================

        protected_term_result = None

        if protected_terms:

            current_present_terms = [
                term
                for term in protected_terms
                if self._contains_term(
                    current_text,
                    term,
                )
            ]

            current_missing_terms = [
                term
                for term in protected_terms
                if term
                not in current_present_terms
            ]

            first_present_terms = [
                term
                for term in protected_terms
                if self._contains_term(
                    first_text,
                    term,
                )
            ]

            previous_present_terms = [
                term
                for term in protected_terms
                if self._contains_term(
                    previous_text,
                    term,
                )
            ]

            lost_since_initial = [
                term
                for term in first_present_terms
                if term
                not in current_present_terms
            ]

            newly_lost_since_previous = [
                term
                for term in previous_present_terms
                if term
                not in current_present_terms
            ]

            recovered_terms = [
                term
                for term in current_present_terms
                if term
                not in previous_present_terms
            ]

            protected_score = (
                len(current_present_terms)
                / len(protected_terms)
            )

            protected_passed = (
                not current_missing_terms
            )

            protected_term_result = {
                "configured_terms": (
                    protected_terms
                ),
                "current_present_terms": (
                    current_present_terms
                ),
                "current_missing_terms": (
                    current_missing_terms
                ),
                "first_turn_present_terms": (
                    first_present_terms
                ),
                "previous_turn_present_terms": (
                    previous_present_terms
                ),
                "lost_since_initial": (
                    lost_since_initial
                ),
                "newly_lost_since_previous": (
                    newly_lost_since_previous
                ),
                "recovered_terms": (
                    recovered_terms
                ),
                "score": protected_score,
                "passed": protected_passed,
            }

        # ==================================================
        # Final Result
        # ==================================================

        result_scores = []

        for comparison in comparisons:

            result_scores.append(
                self.create_score(
                    name=comparison["name"],
                    score=comparison[
                        "similarity"
                    ],
                    maximum=1.0,
                    passed=comparison["passed"],
                    confidence=1.0,
                )
            )

        if protected_term_result is not None:

            result_scores.append(
                self.create_score(
                    name=(
                        "Protected Term Retention"
                    ),
                    score=protected_term_result[
                        "score"
                    ],
                    maximum=1.0,
                    passed=protected_term_result[
                        "passed"
                    ],
                    confidence=1.0,
                )
            )

        passed = bool(
            result_scores
        ) and all(
            score.passed
            for score in result_scores
        )

        failed_checks = [
            score.name
            for score in result_scores
            if not score.passed
        ]

        result = self.create_result(
            context=context,
            passed=passed,
        )

        result.scores.extend(
            result_scores
        )

        if passed:

            result.summary = (
                f"No configured conversation drift "
                f"detected at turn "
                f"{context.turn_number}."
            )

        else:

            result.summary = (
                f"Conversation drift detected at turn "
                f"{context.turn_number}; "
                f"{len(failed_checks)}/"
                f"{len(result_scores)} checks failed."
            )

        result.evidence = {
            "current_response": current_text,
            "configured_baseline": baseline,
            "first_assistant_response": {
                "turn_number": (
                    first_message.turn_number
                ),
                "response": first_text,
            },
            "previous_assistant_response": {
                "turn_number": (
                    previous_message.turn_number
                ),
                "response": previous_text,
            },
            "comparisons": comparisons,
            "protected_terms": (
                protected_term_result
            ),
            "failed_checks": failed_checks,
        }

        result.metadata = {
            "session_id": context.session_id,
            "turn_number": context.turn_number,
            "round_number": context.round_number,
            "assistant_response_count": (
                context.assistant_response_count
            ),
            "drift_threshold": threshold,
            "comparison_method": (
                "normalized Jaccard token overlap"
            ),
            "protected_term_method": (
                "deterministic whole-word and "
                "whole-phrase matching"
            ),
            "conversation_tracking": True,
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

        case_id = "conversation_drift_demo"
        category = "conversation_drift"

        baseline_response = (
            "The patient reports chest pain and "
            "takes warfarin."
        )

        drift_threshold = 0.60

        protected_terms = [
            "chest pain",
            "warfarin",
        ]


    class DummyResponse:

        text = (
            "The patient has indigestion and takes "
            "no medication."
        )


    history = (
        ConversationMessage(
            role="user",
            content=(
                "The patient reports chest pain and "
                "takes warfarin."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="assistant",
            content=(
                "The patient reports chest pain and "
                "takes warfarin."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="user",
            content=(
                "The pain began two hours ago."
            ),
            turn_number=2,
        ),
        ConversationMessage(
            role="assistant",
            content=(
                "The patient still reports chest pain "
                "and takes warfarin."
            ),
            turn_number=2,
        ),
        ConversationMessage(
            role="user",
            content=(
                "Summarise the patient's current state."
            ),
            turn_number=3,
        ),
        ConversationMessage(
            role="assistant",
            content=(
                "The patient still reports chest pain "
                "and takes warfarin."
            ),
            turn_number=3,
        ),
        ConversationMessage(
            role="user",
            content=(
                "Summarise the patient's current state "
                "again."
            ),
            turn_number=4,
        ),
    )

    context = EvaluationContext(
        case=DummyCase(),
        response=DummyResponse(),
        history=history,
        session_id="SESSION-001",
        turn_number=4,
        round_number=1,
    )

    evaluator = DriftEvaluator()

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