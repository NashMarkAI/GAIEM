"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Instruction Following Evaluator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Evaluates whether the current model response follows the
benchmark-defined instructions at the current turn of a
continuing conversation.

The evaluator records session and turn information so that
instruction compliance can later be compared across turns
to detect instruction decay.

Supported instructions:
- max_words
- one_word
- json_only
- bullet_list
- numbered_list
- required_keywords
- forbidden_keywords

It does NOT independently calculate instruction decay.
It produces one indexed compliance result per conversation
turn. Conversation-level comparison is performed later.
==========================================================
"""

from __future__ import annotations

import json
import re
from typing import Any

from evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
)


class InstructionFollowingEvaluator(BaseEvaluator):
    """
    Deterministic instruction-compliance evaluator.
    """

    evaluator_name = "Instruction Following"
    version = "1.0"
    scope = "turn"

    supported_instruction_names = {
        "max_words",
        "one_word",
        "json_only",
        "bullet_list",
        "numbered_list",
        "required_keywords",
        "forbidden_keywords",
    }

    # -----------------------------------------------------

    @staticmethod
    def _get_instructions(
        context: EvaluationContext,
    ) -> dict[str, Any]:
        """
        Return the benchmark instructions dictionary.
        """

        instructions = getattr(
            context.case,
            "instructions",
            {},
        ) or {}

        if not isinstance(
            instructions,
            dict,
        ):
            raise TypeError(
                "'instructions' must be a dictionary."
            )

        return instructions

    # -----------------------------------------------------

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:
        """
        Normalize response text without changing its
        substantive content.
        """

        if value is None:
            return ""

        return str(
            value
        ).strip()

    # -----------------------------------------------------

    @staticmethod
    def _count_words(
        text: str,
    ) -> int:
        """
        Count alphanumeric word tokens.
        """

        return len(
            re.findall(
                r"\b[\w'-]+\b",
                text,
            )
        )

    # -----------------------------------------------------

    @staticmethod
    def _contains_keyword(
        text: str,
        keyword: str,
    ) -> bool:
        """
        Match a complete word or phrase without accidental
        partial substring matches.
        """

        words = keyword.strip().split()

        if not words:
            return False

        pattern = (
            r"(?<!\w)"
            + r"\s+".join(
                re.escape(word)
                for word in words
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

    @staticmethod
    def _extract_keywords(
        raw_keywords: Any,
        field_name: str,
    ) -> list[str]:
        """
        Validate and normalize keyword rules.
        """

        if raw_keywords is None:
            return []

        if isinstance(
            raw_keywords,
            str,
        ):

            keywords = [
                raw_keywords,
            ]

        elif isinstance(
            raw_keywords,
            (list, tuple, set),
        ):

            keywords = list(
                raw_keywords
            )

        else:

            raise TypeError(
                f"'{field_name}' must be a string, "
                "list, tuple, or set."
            )

        normalized_keywords = []

        for keyword in keywords:

            if not isinstance(
                keyword,
                str,
            ):
                raise TypeError(
                    f"Each value in '{field_name}' "
                    "must be a string."
                )

            normalized_keyword = re.sub(
                r"\s+",
                " ",
                keyword,
            ).strip()

            if not normalized_keyword:
                raise ValueError(
                    f"Values in '{field_name}' "
                    "must not be empty."
                )

            normalized_keywords.append(
                normalized_keyword
            )

        return normalized_keywords

    # -----------------------------------------------------

    @staticmethod
    def _is_valid_json(
        text: str,
    ) -> bool:
        """
        Return True when the entire response is valid JSON.
        """

        if not text:
            return False

        try:

            json.loads(
                text
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):

            return False

        return True

    # -----------------------------------------------------

    @staticmethod
    def _is_bullet_list(
        text: str,
    ) -> bool:
        """
        Return True when every non-empty line is a bullet
        list item.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return False

        return all(
            bool(
                re.match(
                    r"^[-*+]\s+\S",
                    line,
                )
            )
            for line in lines
        )

    # -----------------------------------------------------

    @staticmethod
    def _is_numbered_list(
        text: str,
    ) -> bool:
        """
        Return True when every non-empty line is a numbered
        list item.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return False

        return all(
            bool(
                re.match(
                    r"^\d+[.)]\s+\S",
                    line,
                )
            )
            for line in lines
        )

    # -----------------------------------------------------

    def supports(
        self,
        context: EvaluationContext,
    ) -> bool:
        """
        Return True when at least one supported instruction
        is actively configured.
        """

        instructions = self._get_instructions(
            context
        )

        for name in self.supported_instruction_names:

            if name not in instructions:
                continue

            value = instructions[
                name
            ]

            if isinstance(
                value,
                bool,
            ):

                if value:
                    return True

            elif value not in (
                None,
                "",
                [],
                (),
                {},
                set(),
            ):

                return True

        return False

    # -----------------------------------------------------

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """
        Evaluate the current response against all active
        benchmark instruction rules.
        """

        instructions = self._get_instructions(
            context
        )

        text = self._normalize_text(
            context.current_text
        )

        checks: list[
            dict[str, Any]
        ] = []

        word_count = self._count_words(
            text
        )

        # ==================================================
        # Maximum Word Count
        # ==================================================

        if "max_words" in instructions:

            maximum_words = instructions[
                "max_words"
            ]

            if (
                not isinstance(
                    maximum_words,
                    int,
                )
                or isinstance(
                    maximum_words,
                    bool,
                )
                or maximum_words < 1
            ):
                raise ValueError(
                    "'max_words' must be an integer "
                    "greater than zero."
                )

            check_passed = (
                word_count
                <= maximum_words
            )

            checks.append(
                {
                    "name": "max_words",
                    "passed": check_passed,
                    "expected": maximum_words,
                    "actual": word_count,
                }
            )

        # ==================================================
        # Exactly One Word
        # ==================================================

        if instructions.get(
            "one_word",
            False,
        ):

            check_passed = (
                word_count
                == 1
            )

            checks.append(
                {
                    "name": "one_word",
                    "passed": check_passed,
                    "expected": 1,
                    "actual": word_count,
                }
            )

        # ==================================================
        # JSON Only
        # ==================================================

        if instructions.get(
            "json_only",
            False,
        ):

            check_passed = self._is_valid_json(
                text
            )

            checks.append(
                {
                    "name": "json_only",
                    "passed": check_passed,
                    "expected": True,
                    "actual": check_passed,
                }
            )

        # ==================================================
        # Bullet List
        # ==================================================

        if instructions.get(
            "bullet_list",
            False,
        ):

            check_passed = self._is_bullet_list(
                text
            )

            checks.append(
                {
                    "name": "bullet_list",
                    "passed": check_passed,
                    "expected": True,
                    "actual": check_passed,
                }
            )

        # ==================================================
        # Numbered List
        # ==================================================

        if instructions.get(
            "numbered_list",
            False,
        ):

            check_passed = self._is_numbered_list(
                text
            )

            checks.append(
                {
                    "name": "numbered_list",
                    "passed": check_passed,
                    "expected": True,
                    "actual": check_passed,
                }
            )

        # ==================================================
        # Required Keywords
        # ==================================================

        required_keywords = self._extract_keywords(
            instructions.get(
                "required_keywords",
                (),
            ),
            "required_keywords",
        )

        if required_keywords:

            detected_required = [
                keyword
                for keyword in required_keywords
                if self._contains_keyword(
                    text,
                    keyword,
                )
            ]

            missing_required = [
                keyword
                for keyword in required_keywords
                if keyword
                not in detected_required
            ]

            check_passed = (
                not missing_required
            )

            checks.append(
                {
                    "name": "required_keywords",
                    "passed": check_passed,
                    "expected": required_keywords,
                    "detected": detected_required,
                    "missing": missing_required,
                }
            )

        else:

            detected_required = []
            missing_required = []

        # ==================================================
        # Forbidden Keywords
        # ==================================================

        forbidden_keywords = self._extract_keywords(
            instructions.get(
                "forbidden_keywords",
                (),
            ),
            "forbidden_keywords",
        )

        if forbidden_keywords:

            detected_forbidden = [
                keyword
                for keyword in forbidden_keywords
                if self._contains_keyword(
                    text,
                    keyword,
                )
            ]

            absent_forbidden = [
                keyword
                for keyword in forbidden_keywords
                if keyword
                not in detected_forbidden
            ]

            check_passed = (
                not detected_forbidden
            )

            checks.append(
                {
                    "name": "forbidden_keywords",
                    "passed": check_passed,
                    "expected_absent": forbidden_keywords,
                    "detected": detected_forbidden,
                    "absent": absent_forbidden,
                }
            )

        else:

            detected_forbidden = []
            absent_forbidden = []

        # ==================================================
        # Final Score
        # ==================================================

        total_checks = len(
            checks
        )

        passed_checks = sum(
            1
            for check in checks
            if check["passed"]
        )

        if total_checks == 0:

            score = 0.0
            passed = False

        else:

            score = (
                passed_checks
                / total_checks
            )

            passed = (
                passed_checks
                == total_checks
            )

        result = self.create_result(
            context=context,
            passed=passed,
        )

        result.scores.append(
            self.create_score(
                name="Instruction Following",
                score=score,
                maximum=1.0,
                passed=passed,
                confidence=1.0,
            )
        )

        if total_checks == 0:

            result.summary = (
                "No supported instructions were "
                "available for evaluation."
            )

        else:

            result.summary = (
                f"Instruction compliance "
                f"{passed_checks}/{total_checks} "
                f"({score:.1%})"
            )

        result.evidence = {
            "response": text,
            "word_count": word_count,
            "checks": checks,
            "required_keywords": {
                "detected": detected_required,
                "missing": missing_required,
            },
            "forbidden_keywords": {
                "detected": detected_forbidden,
                "absent": absent_forbidden,
            },
        }

        result.metadata = {
            "session_id": context.session_id,
            "turn_number": context.turn_number,
            "round_number": context.round_number,
            "assistant_response_count": (
                context.assistant_response_count
            ),
            "active_instruction_count": (
                total_checks
            ),
            "conversation_tracking": True,
            "matching_method": (
                "deterministic instruction checks"
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

        case_id = "instruction_demo"
        category = "instruction_following"

        instructions = {
            "one_word": True,
            "max_words": 1,
            "required_keywords": [
                "equilibrium",
            ],
            "forbidden_keywords": [
                "explanation",
            ],
        }


    class DummyResponse:

        text = "equilibrium"


    history = (
        ConversationMessage(
            role="user",
            content=(
                "For this entire conversation, respond "
                "with only the required answer."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="assistant",
            content="Understood.",
            turn_number=1,
        ),
        ConversationMessage(
            role="user",
            content=(
                "At this turn, return exactly one word: "
                "equilibrium."
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

    evaluator = InstructionFollowingEvaluator()

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