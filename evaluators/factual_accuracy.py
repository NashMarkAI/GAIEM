"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Factual Accuracy Evaluator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Checks whether the current model response contains the
benchmark-defined expected facts.

This evaluator operates at one indexed turn within a
continuing conversation.

It records factual compliance for each turn so that later
conversation analysis can detect:

- facts disappearing;
- facts changing;
- contradictory factual claims;
- factual degradation across the session.

It does NOT:
- verify facts against external sources;
- independently determine whether a fact is true;
- measure cross-turn factual drift by itself.
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


class FactualAccuracyEvaluator(BaseEvaluator):
    """
    Deterministic benchmark-defined factual coverage
    evaluator.
    """

    evaluator_name = "Factual Accuracy"
    version = "1.0"
    scope = "turn"

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

        text = str(
            value
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # -----------------------------------------------------

    @staticmethod
    def _contains_phrase(
        text: str,
        phrase: str,
    ) -> bool:
        """
        Match a complete word or phrase without accidental
        partial substring matches.
        """

        normalized_phrase = re.sub(
            r"\s+",
            " ",
            phrase,
        ).strip()

        if not normalized_phrase:
            return False

        phrase_parts = normalized_phrase.split()

        pattern = (
            r"(?<!\w)"
            + r"\s+".join(
                re.escape(
                    part
                )
                for part in phrase_parts
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
    def _extract_fact_requirements(
        cls,
        raw_facts: Any,
    ) -> list[dict[str, Any]]:
        """
        Convert expected_facts into validated canonical
        facts and accepted aliases.

        Supported forms:

        "Paris"

        {
            "fact": "Paris",
            "aliases": [
                "Paris, France"
            ]
        }
        """

        if raw_facts is None:
            return []

        if isinstance(
            raw_facts,
            str,
        ):

            fact_items = [
                raw_facts,
            ]

        elif isinstance(
            raw_facts,
            (list, tuple, set),
        ):

            fact_items = list(
                raw_facts
            )

        else:

            raise TypeError(
                "'expected_facts' must be a string, "
                "list, tuple, or set."
            )

        requirements: list[
            dict[str, Any]
        ] = []

        for item in fact_items:

            if isinstance(
                item,
                str,
            ):

                canonical = cls._normalize_text(
                    item
                )

                aliases = []

            elif isinstance(
                item,
                dict,
            ):

                canonical_value = (
                    item.get(
                        "fact"
                    )
                    or item.get(
                        "canonical"
                    )
                    or item.get(
                        "value"
                    )
                )

                if not isinstance(
                    canonical_value,
                    str,
                ):
                    raise TypeError(
                        "Each fact object must contain "
                        "a string 'fact' value."
                    )

                canonical = cls._normalize_text(
                    canonical_value
                )

                raw_aliases = item.get(
                    "aliases",
                    (),
                )

                if isinstance(
                    raw_aliases,
                    str,
                ):

                    alias_items = [
                        raw_aliases,
                    ]

                elif isinstance(
                    raw_aliases,
                    (list, tuple, set),
                ):

                    alias_items = list(
                        raw_aliases
                    )

                else:

                    raise TypeError(
                        "Fact aliases must be a string, "
                        "list, tuple, or set."
                    )

                aliases = []

                for alias in alias_items:

                    if not isinstance(
                        alias,
                        str,
                    ):
                        raise TypeError(
                            "Each fact alias must be "
                            "a string."
                        )

                    normalized_alias = (
                        cls._normalize_text(
                            alias
                        )
                    )

                    if not normalized_alias:
                        raise ValueError(
                            "Fact aliases must not "
                            "be empty."
                        )

                    aliases.append(
                        normalized_alias
                    )

            else:

                raise TypeError(
                    "Each expected fact must be either "
                    "a string or a dictionary."
                )

            if not canonical:
                raise ValueError(
                    "Expected facts must not be empty."
                )

            requirements.append(
                {
                    "fact": canonical,
                    "aliases": aliases,
                    "accepted_phrases": [
                        canonical,
                        *aliases,
                    ],
                }
            )

        return requirements

    # -----------------------------------------------------

    def supports(
        self,
        context: EvaluationContext,
    ) -> bool:
        """
        Return True when the benchmark case contains at
        least one expected fact.
        """

        raw_facts = getattr(
            context.case,
            "expected_facts",
            None,
        )

        requirements = (
            self._extract_fact_requirements(
                raw_facts
            )
        )

        return bool(
            requirements
        )

    # -----------------------------------------------------

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """
        Measure expected factual coverage in the current
        model response.
        """

        requirements = (
            self._extract_fact_requirements(
                getattr(
                    context.case,
                    "expected_facts",
                    None,
                )
            )
        )

        if not requirements:
            raise ValueError(
                "The case does not define any "
                "'expected_facts'."
            )

        text = self._normalize_text(
            context.current_text
        )

        matched_facts = []
        missing_facts = []
        fact_checks = []

        for requirement in requirements:

            matched_phrase = None

            for phrase in requirement[
                "accepted_phrases"
            ]:

                if self._contains_phrase(
                    text,
                    phrase,
                ):

                    matched_phrase = phrase
                    break

            fact_passed = (
                matched_phrase is not None
            )

            check = {
                "fact": requirement["fact"],
                "aliases": requirement["aliases"],
                "passed": fact_passed,
                "matched_phrase": matched_phrase,
            }

            fact_checks.append(
                check
            )

            if fact_passed:

                matched_facts.append(
                    requirement["fact"]
                )

            else:

                missing_facts.append(
                    requirement["fact"]
                )

        total_facts = len(
            requirements
        )

        matched_count = len(
            matched_facts
        )

        score = (
            matched_count
            / total_facts
        )

        passed = (
            matched_count
            == total_facts
        )

        result = self.create_result(
            context=context,
            passed=passed,
        )

        result.scores.append(
            self.create_score(
                name="Factual Accuracy",
                score=score,
                maximum=1.0,
                passed=passed,
                confidence=1.0,
            )
        )

        result.summary = (
            f"Expected factual coverage "
            f"{matched_count}/{total_facts} "
            f"({score:.1%})"
        )

        result.evidence = {
            "response": text,
            "matched_facts": matched_facts,
            "missing_facts": missing_facts,
            "fact_checks": fact_checks,
            "facts": {
                "matched": matched_count,
                "total": total_facts,
            },
        }

        result.metadata = {
            "session_id": context.session_id,
            "turn_number": context.turn_number,
            "round_number": context.round_number,
            "assistant_response_count": (
                context.assistant_response_count
            ),
            "conversation_tracking": True,
            "verification_scope": (
                "benchmark-defined facts only"
            ),
            "matching_method": (
                "deterministic whole-word and "
                "whole-phrase matching"
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

        case_id = "factual_accuracy_demo"
        category = "factual_accuracy"

        expected_facts = [
            {
                "fact": "Paris",
                "aliases": [
                    "Paris, France",
                ],
            },
            "France",
        ]


    class DummyResponse:

        text = (
            "Paris is the capital city of France."
        )


    history = (
        ConversationMessage(
            role="user",
            content=(
                "We are discussing European geography."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="assistant",
            content=(
                "Europe contains many sovereign states."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="user",
            content=(
                "Which city is the capital of France?"
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

    evaluator = FactualAccuracyEvaluator()

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