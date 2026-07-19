"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Consistency Evaluator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Detects benchmark-defined contradictions across one
continuing conversation.

The evaluator checks:

- contradictions inside the current response;
- contradictions between the current response and earlier
  assistant responses;
- contradictions introduced on earlier turns;
- cumulative contradiction across the whole session.

It does NOT:
- independently decide whether two unconfigured claims
  contradict each other;
- perform external factual verification;
- use another language model as a judge.
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


class ConsistencyEvaluator(BaseEvaluator):
    """
    Deterministic conversation-level contradiction
    evaluator.

    Supported benchmark field:

        contradiction_pairs

    Example:

        contradiction_pairs = [
            [
                "same input produces the same output",
                "same input produces a different output"
            ]
        ]
    """

    evaluator_name = "Consistency"
    version = "1.0"
    scope = "conversation"

    # -----------------------------------------------------

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:
        """
        Normalize text for deterministic phrase matching.
        """

        if value is None:
            return ""

        text = str(value)

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # -----------------------------------------------------

    @classmethod
    def _extract_contradiction_pairs(
        cls,
        raw_pairs: Any,
    ) -> list[tuple[str, str]]:
        """
        Convert contradiction_pairs into validated pairs.

        Supported pair forms:

        [
            "claim A",
            "claim B"
        ]

        {
            "claim_a": "claim A",
            "claim_b": "claim B"
        }
        """

        if raw_pairs is None:
            return []

        if not isinstance(
            raw_pairs,
            (list, tuple, set),
        ):
            raise TypeError(
                "'contradiction_pairs' must be a "
                "list, tuple, or set."
            )

        normalized_pairs: list[
            tuple[str, str]
        ] = []

        for item in raw_pairs:

            if isinstance(
                item,
                dict,
            ):

                claim_a = (
                    item.get("claim_a")
                    or item.get("first")
                    or item.get("a")
                )

                claim_b = (
                    item.get("claim_b")
                    or item.get("second")
                    or item.get("b")
                )

            elif isinstance(
                item,
                (list, tuple),
            ):

                if len(item) != 2:
                    raise ValueError(
                        "Each contradiction pair must "
                        "contain exactly two claims."
                    )

                claim_a = item[0]
                claim_b = item[1]

            else:

                raise TypeError(
                    "Each contradiction pair must be "
                    "a two-item sequence or dictionary."
                )

            if not isinstance(
                claim_a,
                str,
            ) or not isinstance(
                claim_b,
                str,
            ):
                raise TypeError(
                    "Both values in a contradiction "
                    "pair must be strings."
                )

            normalized_a = cls._normalize_text(
                claim_a
            )

            normalized_b = cls._normalize_text(
                claim_b
            )

            if not normalized_a or not normalized_b:
                raise ValueError(
                    "Contradiction claims must not "
                    "be empty."
                )

            if normalized_a.lower() == normalized_b.lower():
                raise ValueError(
                    "The two contradiction claims must "
                    "be different."
                )

            pair = (
                normalized_a,
                normalized_b,
            )

            if pair not in normalized_pairs:
                normalized_pairs.append(
                    pair
                )

        return normalized_pairs

    # -----------------------------------------------------

    @staticmethod
    def _contains_claim(
        text: str,
        claim: str,
    ) -> bool:
        """
        Match a complete configured word or phrase without
        accidental partial substring matching.
        """

        claim_parts = claim.strip().split()

        if not claim_parts:
            return False

        pattern = (
            r"(?<!\w)"
            + r"\s+".join(
                re.escape(part)
                for part in claim_parts
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

    def supports(
        self,
        context: EvaluationContext,
    ) -> bool:
        """
        Return True when the case defines at least one
        contradiction pair.
        """

        contradiction_pairs = (
            self._extract_contradiction_pairs(
                getattr(
                    context.case,
                    "contradiction_pairs",
                    None,
                )
            )
        )

        return bool(
            contradiction_pairs
        )

    # -----------------------------------------------------

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """
        Detect configured contradictions in the current
        response and across the complete conversation.
        """

        contradiction_pairs = (
            self._extract_contradiction_pairs(
                getattr(
                    context.case,
                    "contradiction_pairs",
                    None,
                )
            )
        )

        if not contradiction_pairs:
            raise ValueError(
                "The case does not define any "
                "'contradiction_pairs'."
            )

        current_text = self._normalize_text(
            context.current_text
        )

        pair_results = []

        current_conflict_count = 0
        conversation_conflict_count = 0

        for pair_index, (
            claim_a,
            claim_b,
        ) in enumerate(
            contradiction_pairs,
            start=1,
        ):

            current_contains_a = (
                self._contains_claim(
                    current_text,
                    claim_a,
                )
            )

            current_contains_b = (
                self._contains_claim(
                    current_text,
                    claim_b,
                )
            )

            prior_a_turns = []
            prior_b_turns = []

            prior_occurrences = []

            for message in (
                context.prior_assistant_messages
            ):

                message_text = self._normalize_text(
                    message.content
                )

                contains_a = self._contains_claim(
                    message_text,
                    claim_a,
                )

                contains_b = self._contains_claim(
                    message_text,
                    claim_b,
                )

                if contains_a:
                    prior_a_turns.append(
                        message.turn_number
                    )

                if contains_b:
                    prior_b_turns.append(
                        message.turn_number
                    )

                if contains_a or contains_b:

                    prior_occurrences.append(
                        {
                            "turn_number": (
                                message.turn_number
                            ),
                            "contains_claim_a": (
                                contains_a
                            ),
                            "contains_claim_b": (
                                contains_b
                            ),
                            "response": message_text,
                        }
                    )

            prior_contains_a = bool(
                prior_a_turns
            )

            prior_contains_b = bool(
                prior_b_turns
            )

            current_internal_conflict = (
                current_contains_a
                and current_contains_b
            )

            current_vs_prior_conflict = (
                (
                    current_contains_a
                    and prior_contains_b
                )
                or (
                    current_contains_b
                    and prior_contains_a
                )
            )

            current_conflict = (
                current_internal_conflict
                or current_vs_prior_conflict
            )

            conversation_contains_a = (
                prior_contains_a
                or current_contains_a
            )

            conversation_contains_b = (
                prior_contains_b
                or current_contains_b
            )

            conversation_conflict = (
                conversation_contains_a
                and conversation_contains_b
            )

            if current_conflict:
                current_conflict_count += 1

            if conversation_conflict:
                conversation_conflict_count += 1

            conflict_types = []

            if current_internal_conflict:
                conflict_types.append(
                    "current_response_internal"
                )

            if current_vs_prior_conflict:
                conflict_types.append(
                    "current_response_vs_prior_turn"
                )

            if (
                prior_contains_a
                and prior_contains_b
            ):
                conflict_types.append(
                    "prior_conversation_conflict"
                )

            pair_results.append(
                {
                    "pair_index": pair_index,
                    "claim_a": claim_a,
                    "claim_b": claim_b,
                    "current_contains_claim_a": (
                        current_contains_a
                    ),
                    "current_contains_claim_b": (
                        current_contains_b
                    ),
                    "prior_claim_a_turns": (
                        prior_a_turns
                    ),
                    "prior_claim_b_turns": (
                        prior_b_turns
                    ),
                    "current_conflict": (
                        current_conflict
                    ),
                    "conversation_conflict": (
                        conversation_conflict
                    ),
                    "conflict_types": conflict_types,
                    "prior_occurrences": (
                        prior_occurrences
                    ),
                }
            )

        total_pairs = len(
            contradiction_pairs
        )

        current_score = (
            total_pairs
            - current_conflict_count
        ) / total_pairs

        conversation_score = (
            total_pairs
            - conversation_conflict_count
        ) / total_pairs

        current_passed = (
            current_conflict_count == 0
        )

        conversation_passed = (
            conversation_conflict_count == 0
        )

        passed = (
            current_passed
            and conversation_passed
        )

        result = self.create_result(
            context=context,
            passed=passed,
        )

        result.scores.append(
            self.create_score(
                name="Current Turn Consistency",
                score=current_score,
                maximum=1.0,
                passed=current_passed,
                confidence=1.0,
            )
        )

        result.scores.append(
            self.create_score(
                name="Conversation Consistency",
                score=conversation_score,
                maximum=1.0,
                passed=conversation_passed,
                confidence=1.0,
            )
        )

        result.summary = (
            f"Current turn conflicts "
            f"{current_conflict_count}/"
            f"{total_pairs}; conversation conflicts "
            f"{conversation_conflict_count}/"
            f"{total_pairs}."
        )

        result.evidence = {
            "current_response": current_text,
            "pair_results": pair_results,
            "counts": {
                "configured_pairs": total_pairs,
                "current_conflicts": (
                    current_conflict_count
                ),
                "conversation_conflicts": (
                    conversation_conflict_count
                ),
            },
        }

        result.metadata = {
            "session_id": context.session_id,
            "turn_number": context.turn_number,
            "round_number": context.round_number,
            "assistant_response_count": (
                context.assistant_response_count
            ),
            "matching_method": (
                "deterministic configured "
                "contradiction-pair matching"
            ),
            "conversation_tracking": True,
            "verification_scope": (
                "benchmark-defined contradiction "
                "pairs only"
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

        case_id = "consistency_demo"
        category = "consistency"

        contradiction_pairs = [
            [
                "same input produces the same output",
                "same input produces a different output",
            ],
            [
                "deterministic output is predictable",
                "deterministic output is unpredictable",
            ],
        ]


    class DummyResponse:

        text = (
            "The same input produces a different output."
        )


    history = (
        ConversationMessage(
            role="user",
            content=(
                "Define deterministic output."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="assistant",
            content=(
                "The same input produces the same output, "
                "so deterministic output is predictable."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="user",
            content=(
                "Explain that definition again."
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

    evaluator = ConsistencyEvaluator()

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