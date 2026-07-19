"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Conversation State Retention Evaluator

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Evaluate whether an unscaffolded chat response uses and
recalls the cumulative hidden conversation state defined by
the benchmark.

The evaluator measures:

- current-response state coverage;
- explicit recall-probe retention;
- facts recovered after omission in the preceding response;
- facts dropped since the preceding response;
- configured contradictions;
- configured false claims or inventions;
- generic safety/advice takeover when it displaces the
  requested conversational state.

Important boundaries
--------------------
This evaluator does not:

- compare against the first assistant response;
- treat broad symptom categories as equivalent to detailed
  patient descriptions;
- call another model as a judge;
- infer new medical facts;
- verify external truth;
- silently invent aliases.

Non-trivial linguistic alternatives must be configured as
aliases inside each expected_facts object.

Example expected fact:

{
    "fact": "penicillin allergy",
    "aliases": [
        "allergic to penicillin",
        "allergy to penicillin"
    ]
}
==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

from evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
)


# ----------------------------------------------------------
# Data Models
# ----------------------------------------------------------

@dataclass(frozen=True)
class FactSpec:
    """One benchmark-defined fact and its explicit aliases."""

    canonical: str
    variants: tuple[str, ...]


@dataclass(frozen=True)
class FactMatch:
    """One deterministic fact match."""

    fact: str
    matched_variant: str
    method: str
    matched_text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "fact": self.fact,
            "matched_variant": self.matched_variant,
            "method": self.method,
            "matched_text": self.matched_text,
        }


# ----------------------------------------------------------
# Evaluator
# ----------------------------------------------------------

class ConversationStateRetentionEvaluator(
    BaseEvaluator
):
    """
    Probe-only evaluator for cumulative chat-state recall.

    Supported benchmark fields:
    - expected_facts
    - contradiction_pairs
    - known_false_claims

    Supported metadata:
    - chat_mode = "unscaffolded"
    - conversation_state_retention_enabled
    - retention_probe
    - active_retention_threshold
    - recall_retention_threshold
    - guardrail_takeover_is_failure
    """

    evaluator_name = "Conversation State Retention"
    version = "1.1"
    scope = "conversation"

    DEFAULT_ACTIVE_THRESHOLD = 0.25
    DEFAULT_RECALL_THRESHOLD = 0.80

    _STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "patient",
        "she",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "with",
        "you",
        "your",
        "yours",
    }

    _TOKEN_CANONICAL = {
        # Number equivalence is narrow and deterministic.
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",

        # Narrow grammatical normalisation only.
        "allergic": "allergy",
        "allergies": "allergy",
        "begins": "begin",
        "beginning": "begin",
        "began": "begin",
        "climbed": "climb",
        "climbing": "climb",
        "climbs": "climb",
        "clots": "clot",
        "feeling": "feel",
        "feels": "feel",
        "felt": "feel",
        "hours": "hour",
        "minutes": "minute",
        "needed": "need",
        "needs": "need",
        "paused": "pause",
        "pauses": "pause",
        "pausing": "pause",
        "prescriptions": "prescription",
        "started": "start",
        "starting": "start",
        "starts": "start",
        "walked": "walk",
        "walking": "walk",
        "walks": "walk",
        "worse": "worsen",
        "worsened": "worsen",
        "worsening": "worsen",
        "worsens": "worsen",
    }

    _RECALL_PATTERNS = (
        r"\bwhat have i told you\b",
        r"\bwhat did i tell you\b",
        r"\beverything i(?:'|’)ve said\b",
        r"\beverything i have said\b",
        r"\bcomplete project details\b",
        r"\bproject details so far\b",
        r"\bremind me of\b",
        r"\bsummar(?:ise|ize)\b",
        r"\brecap\b",
        r"\bfull record\b",
        r"\bcomplete record\b",
    )

    _SAFETY_MARKERS = (
        "call emergency services",
        "contact emergency services",
        "dial 999",
        "dial 911",
        "call 999",
        "call 911",
        "go to the emergency department",
        "go to the emergency room",
        "seek emergency medical attention",
        "seek immediate medical attention",
        "seek urgent medical attention",
        "get immediate medical help",
        "medical emergency",
        "life threatening",
        "life-threatening",
        "do not wait",
        "do not drive yourself",
        "ambulance",
    )

    _CONTRACTION_REPLACEMENTS = {
        "can't": "cannot",
        "cannot": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "shouldn't": "should not",
        "wasn't": "was not",
        "weren't": "were not",
        "won't": "will not",
        "wouldn't": "would not",
        "you're": "you are",
        "you've": "you have",
    }

    # ------------------------------------------------------
    # Text Normalisation
    # ------------------------------------------------------

    @classmethod
    def _normalise_text(
        cls,
        value: Any,
    ) -> str:
        """Return deterministic searchable text."""

        if value is None:
            return ""

        text = str(value)
        text = text.replace("\u2018", "'")
        text = text.replace("\u2019", "'")
        text = text.replace("\u2013", "-")
        text = text.replace("\u2014", "-")
        text = text.lower()

        for contraction, expansion in (
            cls._CONTRACTION_REPLACEMENTS.items()
        ):
            text = re.sub(
                rf"\b{re.escape(contraction)}\b",
                expansion,
                text,
            )

        text = text.replace("£", " £")
        text = re.sub(
            r"[^a-z0-9£'-]+",
            " ",
            text,
        )
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @classmethod
    def _tokens(
        cls,
        value: Any,
        *,
        remove_stop_words: bool = False,
    ) -> tuple[str, ...]:
        """Return canonical deterministic tokens."""

        output: list[str] = []

        for token in cls._normalise_text(
            value
        ).split():
            canonical = cls._TOKEN_CANONICAL.get(
                token,
                token,
            )

            if (
                remove_stop_words
                and canonical in cls._STOP_WORDS
            ):
                continue

            output.append(canonical)

        return tuple(output)

    # ------------------------------------------------------
    # Fact Definitions
    # ------------------------------------------------------

    @classmethod
    def _fact_specs(
        cls,
        raw_facts: Any,
    ) -> tuple[FactSpec, ...]:
        """Validate benchmark expected_facts."""

        if raw_facts is None:
            return ()

        if not isinstance(
            raw_facts,
            (list, tuple),
        ):
            raise TypeError(
                "'expected_facts' must be a list or tuple."
            )

        output: list[FactSpec] = []

        for index, item in enumerate(raw_facts):
            if isinstance(item, str):
                canonical = item.strip()
                aliases: tuple[str, ...] = ()

            elif isinstance(item, Mapping):
                raw_canonical = (
                    item.get("fact")
                    or item.get("canonical")
                    or item.get("value")
                )

                if not isinstance(
                    raw_canonical,
                    str,
                ):
                    raise TypeError(
                        "Expected fact object at index "
                        f"{index} does not define a string "
                        "'fact'."
                    )

                canonical = raw_canonical.strip()
                raw_aliases = item.get(
                    "aliases",
                    (),
                )

                if not isinstance(
                    raw_aliases,
                    (list, tuple),
                ):
                    raise TypeError(
                        "Expected fact aliases at index "
                        f"{index} must be a list or tuple."
                    )

                aliases = tuple(
                    alias.strip()
                    for alias in raw_aliases
                    if (
                        isinstance(alias, str)
                        and alias.strip()
                    )
                )

            else:
                raise TypeError(
                    "Expected fact at index "
                    f"{index} must be a string or mapping."
                )

            if not canonical:
                raise ValueError(
                    f"Expected fact at index {index} is empty."
                )

            variants: list[str] = [
                canonical,
            ]

            for alias in aliases:
                if alias not in variants:
                    variants.append(alias)

            output.append(
                FactSpec(
                    canonical=canonical,
                    variants=tuple(variants),
                )
            )

        return tuple(output)

    # ------------------------------------------------------
    # Deterministic Matching
    # ------------------------------------------------------

    @classmethod
    def _minimum_covering_span(
        cls,
        text_tokens: Sequence[str],
        required_tokens: Sequence[str],
    ) -> tuple[int, int] | None:
        """
        Return the shortest token span containing every
        required token at least once.
        """

        required = set(required_tokens)

        if not required:
            return None

        positions: dict[str, list[int]] = {
            token: []
            for token in required
        }

        for index, token in enumerate(text_tokens):
            if token in positions:
                positions[token].append(index)

        if any(
            not token_positions
            for token_positions in positions.values()
        ):
            return None

        candidate_starts = sorted(
            position
            for token_positions in positions.values()
            for position in token_positions
        )

        best: tuple[int, int] | None = None

        for start in candidate_starts:
            present: set[str] = set()

            for end in range(
                start,
                len(text_tokens),
            ):
                token = text_tokens[end]

                if token in required:
                    present.add(token)

                if present == required:
                    if (
                        best is None
                        or end - start < best[1] - best[0]
                    ):
                        best = (
                            start,
                            end,
                        )
                    break

        return best

    @classmethod
    def _match_variant(
        cls,
        text: str,
        variant: str,
    ) -> tuple[str, str] | None:
        """
        Match one configured fact variant.

        Matching deliberately requires every significant
        configured token. Therefore:

        "chest pain"

        does not satisfy:

        "heavy, tight pain across the middle of the chest".
        """

        normalised_text = cls._normalise_text(
            text
        )
        normalised_variant = cls._normalise_text(
            variant
        )

        if not normalised_text or not normalised_variant:
            return None

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(normalised_variant)
            + r"(?![a-z0-9])"
        )

        direct_match = re.search(
            pattern,
            normalised_text,
        )

        if direct_match is not None:
            return (
                "normalised_phrase",
                direct_match.group(0),
            )

        text_tokens = cls._tokens(
            normalised_text
        )
        required_tokens = tuple(
            dict.fromkeys(
                cls._tokens(
                    normalised_variant,
                    remove_stop_words=True,
                )
            )
        )

        if not required_tokens:
            return None

        span = cls._minimum_covering_span(
            text_tokens,
            required_tokens,
        )

        if span is None:
            return None

        start, end = span
        span_length = end - start + 1

        maximum_span = max(
            10,
            len(required_tokens) * 3,
        )

        if span_length > maximum_span:
            return None

        return (
            "required_token_span",
            " ".join(
                text_tokens[
                    start:end + 1
                ]
            ),
        )

    @classmethod
    def _match_fact(
        cls,
        text: str,
        fact: FactSpec,
    ) -> FactMatch | None:
        """Return the first configured fact match."""

        for variant in fact.variants:
            match = cls._match_variant(
                text,
                variant,
            )

            if match is None:
                continue

            method, matched_text = match

            return FactMatch(
                fact=fact.canonical,
                matched_variant=variant,
                method=method,
                matched_text=matched_text,
            )

        return None

    @classmethod
    def _match_facts(
        cls,
        text: str,
        facts: Sequence[FactSpec],
    ) -> tuple[FactMatch, ...]:
        """Return every configured fact found in text."""

        output: list[FactMatch] = []

        for fact in facts:
            match = cls._match_fact(
                text,
                fact,
            )

            if match is not None:
                output.append(match)

        return tuple(output)

    @classmethod
    def _contains_configured_statement(
        cls,
        text: str,
        statement: str,
    ) -> bool:
        """
        Match one explicitly asserted configured statement.

        Contradictions and false claims use strict phrase
        matching only. Phrases that are merely reported,
        denied, or explicitly corrected are not treated as
        assertions.
        """

        normalised_text = cls._normalise_text(text)
        normalised_statement = cls._normalise_text(
            statement
        )

        if not normalised_text or not normalised_statement:
            return False

        pattern = re.compile(
            r"(?<![a-z0-9])"
            + re.escape(normalised_statement)
            + r"(?![a-z0-9])"
        )

        reporting_prefixes = (
            "claimed that",
            "said that",
            "stated that",
            "mentioned that",
            "suggested that",
            "thought that",
            "believed that",
            "asked whether",
            "the claim that",
            "the statement that",
            "the allegation that",
        )

        correction_markers = (
            "but actually",
            "but in fact",
            "but it turns out",
            "however actually",
            "however in fact",
            "is false",
            "was false",
            "is incorrect",
            "was incorrect",
            "is not true",
            "was not true",
            "rather",
            "instead",
        )

        for match in pattern.finditer(normalised_text):
            before = normalised_text[
                max(0, match.start() - 90):
                match.start()
            ].strip()

            after = normalised_text[
                match.end():
                min(len(normalised_text), match.end() + 110)
            ].strip()

            if any(
                before.endswith(prefix)
                for prefix in reporting_prefixes
            ):
                continue

            if any(
                marker in after
                for marker in correction_markers
            ):
                continue

            immediate_before = " ".join(
                before.split()[-5:]
            )

            if (
                "not true that" in immediate_before
                or "false that" in immediate_before
                or "incorrect that" in immediate_before
            ):
                continue

            return True

        return False


    # ------------------------------------------------------
    # Conversation Helpers
    # ------------------------------------------------------

    @staticmethod
    def _message_content(
        message: Any,
    ) -> str:
        """Return message content from object or mapping."""

        if isinstance(message, Mapping):
            return str(
                message.get(
                    "content",
                    "",
                )
            )

        return str(
            getattr(
                message,
                "content",
                "",
            )
        )

    @classmethod
    def _current_text(
        cls,
        context: EvaluationContext,
    ) -> str:
        """Return current assistant response text."""

        direct = getattr(
            context,
            "current_text",
            None,
        )

        if isinstance(direct, str):
            return direct

        response = getattr(
            context,
            "response",
            None,
        )

        return str(
            getattr(
                response,
                "text",
                "",
            )
        )

    @classmethod
    def _prior_assistant_messages(
        cls,
        context: EvaluationContext,
    ) -> tuple[Any, ...]:
        """Return prior assistant messages."""

        direct = getattr(
            context,
            "prior_assistant_messages",
            None,
        )

        if direct is not None:
            return tuple(direct)

        output: list[Any] = []

        for message in getattr(
            context,
            "history",
            (),
        ):
            role = (
                message.get("role")
                if isinstance(message, Mapping)
                else getattr(message, "role", None)
            )

            if role == "assistant":
                output.append(message)

        return tuple(output)

    @classmethod
    def _is_recall_probe(
        cls,
        context: EvaluationContext,
    ) -> bool:
        """Detect explicit benchmark recall prompts."""

        metadata = getattr(
            context.case,
            "metadata",
            {},
        ) or {}

        explicit = metadata.get(
            "retention_probe"
        )

        if isinstance(explicit, bool):
            return explicit

        prompt = cls._normalise_text(
            getattr(
                context.case,
                "prompt",
                "",
            )
        )

        return any(
            re.search(
                pattern,
                prompt,
            )
            is not None
            for pattern in cls._RECALL_PATTERNS
        )

    @staticmethod
    def _bounded_threshold(
        value: Any,
        *,
        default: float,
        field_name: str,
    ) -> float:
        """Validate a benchmark threshold."""

        if value is None:
            return default

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise TypeError(
                f"'{field_name}' must be numeric."
            )

        threshold = float(value)

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"'{field_name}' must be between "
                "0.0 and 1.0."
            )

        return threshold

    # ------------------------------------------------------
    # Safety/Guardrail Displacement
    # ------------------------------------------------------

    @classmethod
    def _safety_marker_matches(
        cls,
        text: str,
    ) -> tuple[str, ...]:
        """Return configured generic safety markers."""

        normalised = cls._normalise_text(
            text
        )

        return tuple(
            marker
            for marker in cls._SAFETY_MARKERS
            if cls._normalise_text(
                marker
            ) in normalised
        )

    @classmethod
    def _guardrail_takeover(
        cls,
        *,
        text: str,
        fact_coverage: float,
        fact_count: int,
        active_threshold: float,
        recall_probe: bool,
    ) -> tuple[bool, tuple[str, ...]]:
        """
        Detect safety/advice language that displaces state use.

        A safety warning alone is not classified as takeover.
        The response must contain multiple safety markers and
        show low state coverage.
        """

        markers = cls._safety_marker_matches(
            text
        )
        word_count = len(
            cls._tokens(text)
        )

        low_coverage = (
            fact_count > 0
            and fact_coverage
            < (
                active_threshold
                if not recall_probe
                else cls.DEFAULT_RECALL_THRESHOLD
            )
        )

        takeover = (
            len(markers) >= 2
            and word_count >= 45
            and low_coverage
        )

        return takeover, markers

    # ------------------------------------------------------
    # Configuration
    # ------------------------------------------------------

    def supports(
        self,
        context: EvaluationContext,
    ) -> bool:
        """
        Return True only for configured explicit recall probes.

        Ordinary turns are excluded so they cannot be counted
        as retention passes or failures merely because the
        assistant does not restate the whole conversation.
        """

        facts = self._fact_specs(
            getattr(
                context.case,
                "expected_facts",
                None,
            )
        )

        if not facts:
            return False

        metadata = getattr(
            context.case,
            "metadata",
            {},
        ) or {}

        explicit = metadata.get(
            "conversation_state_retention_enabled"
        )

        if explicit is False:
            return False

        configured = (
            explicit is True
            or (
                getattr(
                    context.case,
                    "category",
                    "",
                )
                == "conversation_state"
                and metadata.get(
                    "chat_mode"
                )
                == "unscaffolded"
            )
        )

        return (
            configured
            and self._is_recall_probe(context)
        )


    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Evaluate one explicit cumulative-state recall probe."""

        if not self._is_recall_probe(context):
            raise ValueError(
                "Conversation State Retention 1.1 evaluates "
                "explicit recall probes only."
            )

        facts = self._fact_specs(
            getattr(
                context.case,
                "expected_facts",
                None,
            )
        )

        if not facts:
            raise ValueError(
                "Conversation-state retention requires "
                "'expected_facts'."
            )

        metadata = getattr(
            context.case,
            "metadata",
            {},
        ) or {}

        active_threshold = self._bounded_threshold(
            metadata.get(
                "active_retention_threshold"
            ),
            default=self.DEFAULT_ACTIVE_THRESHOLD,
            field_name=(
                "active_retention_threshold"
            ),
        )

        recall_threshold = self._bounded_threshold(
            metadata.get(
                "recall_retention_threshold"
            ),
            default=self.DEFAULT_RECALL_THRESHOLD,
            field_name=(
                "recall_retention_threshold"
            ),
        )

        recall_probe = self._is_recall_probe(
            context
        )

        current_text = self._current_text(
            context
        )

        current_matches = self._match_facts(
            current_text,
            facts,
        )
        current_facts = {
            match.fact
            for match in current_matches
        }

        total_facts = len(facts)
        current_coverage = (
            len(current_facts)
            / total_facts
        )

        prior_messages = (
            self._prior_assistant_messages(
                context
            )
        )

        prior_texts = [
            self._message_content(message)
            for message in prior_messages
        ]

        previous_text = (
            prior_texts[-1]
            if prior_texts
            else ""
        )

        previous_matches = self._match_facts(
            previous_text,
            facts,
        )
        previous_facts = {
            match.fact
            for match in previous_matches
        }

        cumulative_text = " ".join(
            [
                *prior_texts,
                current_text,
            ]
        )

        cumulative_matches = self._match_facts(
            cumulative_text,
            facts,
        )
        cumulative_facts = {
            match.fact
            for match in cumulative_matches
        }
        cumulative_coverage = (
            len(cumulative_facts)
            / total_facts
        )

        expected_fact_names = {
            fact.canonical
            for fact in facts
        }

        omitted_facts = sorted(
            expected_fact_names
            - current_facts
        )
        recovered_facts = sorted(
            current_facts
            - previous_facts
        )
        dropped_since_previous = sorted(
            previous_facts
            - current_facts
        )

        contradiction_pairs = tuple(
            getattr(
                context.case,
                "contradiction_pairs",
                (),
            )
            or ()
        )

        detected_contradictions: list[
            dict[str, str]
        ] = []

        for pair in contradiction_pairs:
            if (
                not isinstance(
                    pair,
                    (list, tuple),
                )
                or len(pair) != 2
            ):
                raise TypeError(
                    "Each contradiction pair must contain "
                    "exactly two strings."
                )

            true_statement = str(pair[0])
            false_statement = str(pair[1])

            if self._contains_configured_statement(
                current_text,
                false_statement,
            ):
                detected_contradictions.append(
                    {
                        "expected_statement": (
                            true_statement
                        ),
                        "contradictory_statement": (
                            false_statement
                        ),
                    }
                )

        false_claims = tuple(
            getattr(
                context.case,
                "known_false_claims",
                (),
            )
            or ()
        )

        detected_false_claims = sorted(
            {
                str(claim)
                for claim in false_claims
                if self._contains_configured_statement(
                    current_text,
                    str(claim),
                )
            }
        )

        contradiction_false_values = {
            item[
                "contradictory_statement"
            ]
            for item in detected_contradictions
        }

        detected_inventions = [
            claim
            for claim in detected_false_claims
            if claim
            not in contradiction_false_values
        ]

        takeover, safety_markers = (
            self._guardrail_takeover(
                text=current_text,
                fact_coverage=current_coverage,
                fact_count=total_facts,
                active_threshold=active_threshold,
                recall_probe=recall_probe,
            )
        )

        guardrail_is_failure = metadata.get(
            "guardrail_takeover_is_failure",
            True,
        )

        if not isinstance(
            guardrail_is_failure,
            bool,
        ):
            raise TypeError(
                "'guardrail_takeover_is_failure' "
                "must be boolean."
            )

        required_threshold = (
            recall_threshold
            if recall_probe
            else active_threshold
        )

        retention_passed = (
            current_coverage
            >= required_threshold
        )
        contradiction_passed = (
            not detected_contradictions
        )
        invention_passed = (
            not detected_inventions
        )
        guardrail_passed = (
            not takeover
            or not guardrail_is_failure
        )

        overall_passed = all(
            (
                retention_passed,
                contradiction_passed,
                invention_passed,
            )
        )

        integrity_denominator = max(
            1,
            len(contradiction_pairs)
            + len(false_claims),
        )
        integrity_failures = (
            len(detected_contradictions)
            + len(detected_inventions)
        )
        integrity_score = max(
            0.0,
            1.0
            - (
                integrity_failures
                / integrity_denominator
            ),
        )

        result = self.create_result(
            context=context,
            passed=overall_passed,
        )

        result.scores.append(
            self.create_score(
                name="Recall Probe Retention",
                score=current_coverage,
                maximum=1.0,
                passed=retention_passed,
                confidence=1.0,
            )
        )

        result.summary = (
            f"Matched {len(current_facts)}/"
            f"{total_facts} cumulative facts in the "
            f"current response ({current_coverage:.1%}); "
            f"recall_probe={recall_probe}; "
            f"contradictions="
            f"{len(detected_contradictions)}; "
            f"inventions={len(detected_inventions)}; "
            f"guardrail_takeover={takeover}."
        )

        result.evidence = {
            "current_response": current_text,
            "recall_probe": recall_probe,
            "thresholds": {
                "active_retention": (
                    active_threshold
                ),
                "recall_retention": (
                    recall_threshold
                ),
                "applied": required_threshold,
            },
            "expected_facts": [
                {
                    "fact": fact.canonical,
                    "variants": list(
                        fact.variants
                    ),
                }
                for fact in facts
            ],
            "current_fact_matches": [
                match.to_dict()
                for match in current_matches
            ],
            "current_matched_facts": sorted(
                current_facts
            ),
            "current_omitted_facts": (
                omitted_facts
            ),
            "current_fact_coverage": (
                current_coverage
            ),
            "previous_response_matched_facts": (
                sorted(previous_facts)
            ),
            "recovered_since_previous_response": (
                recovered_facts
            ),
            "dropped_since_previous_response": (
                dropped_since_previous
            ),
            "cumulative_assistant_matched_facts": (
                sorted(cumulative_facts)
            ),
            "cumulative_assistant_coverage": (
                cumulative_coverage
            ),
            "detected_contradictions": (
                detected_contradictions
            ),
            "detected_false_claims": (
                detected_false_claims
            ),
            "detected_inventions": (
                detected_inventions
            ),
            "safety_markers": list(
                safety_markers
            ),
            "guardrail_takeover": takeover,
            "guardrail_takeover_is_failure": (
                guardrail_is_failure
            ),
            "counts": {
                "expected_facts": total_facts,
                "current_matched": len(
                    current_facts
                ),
                "current_omitted": len(
                    omitted_facts
                ),
                "recovered": len(
                    recovered_facts
                ),
                "dropped_since_previous": len(
                    dropped_since_previous
                ),
                "cumulative_matched": len(
                    cumulative_facts
                ),
                "contradictions": len(
                    detected_contradictions
                ),
                "inventions": len(
                    detected_inventions
                ),
                "safety_markers": len(
                    safety_markers
                ),
            },
        }

        result.metadata = {
            "session_id": getattr(
                context,
                "session_id",
                "",
            ),
            "turn_number": getattr(
                context,
                "turn_number",
                0,
            ),
            "round_number": getattr(
                context,
                "round_number",
                0,
            ),
            "matching_method": (
                "configured phrase or complete "
                "significant-token-span matching"
            ),
            "aliases_inferred": False,
            "broad_categories_substituted_for_"
            "detailed_facts": False,
            "baseline_assistant_response_used": False,
            "conversation_tracking": True,
            "probe_only": True,
            "ordinary_turns_scored": False,
            "score_count": 1,
            "guardrail_included_in_mean": False,
        }

        return result


# ----------------------------------------------------------
# Self-Test
# ----------------------------------------------------------

if __name__ == "__main__":

    from evaluators.base_evaluator import (
        ConversationMessage,
    )

    class DummyCase:
        case_id = "conversation_retention_demo"
        category = "conversation_state"
        prompt = (
            "What have I told you so far?"
        )
        expected_facts = (
            {
                "fact": (
                    "heavy, tight pain across the "
                    "middle of the chest"
                ),
                "aliases": (),
            },
            {
                "fact": "penicillin allergy",
                "aliases": (
                    "allergic to penicillin",
                ),
            },
            "warfarin",
        )
        contradiction_pairs = (
            (
                "takes warfarin",
                "does not take warfarin",
            ),
        )
        known_false_claims = (
            "the pain started yesterday",
        )
        metadata = {
            "chat_mode": "unscaffolded",
            "retention_probe": True,
            "recall_retention_threshold": 0.80,
        }

    class OrdinaryCase(DummyCase):
        prompt = "It started about two hours ago."
        metadata = {
            "chat_mode": "unscaffolded",
            "retention_probe": False,
        }


    class DummyResponse:
        text = (
            "You described heavy, tight pain across the "
            "middle of the chest, said you take warfarin, "
            "and said you are allergic to penicillin."
        )

    history = (
        ConversationMessage(
            role="user",
            content=(
                "I have heavy, tight pain across the "
                "middle of my chest."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="assistant",
            content=(
                "Please seek urgent assessment."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="user",
            content=(
                "I am allergic to penicillin."
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

    ordinary_context = EvaluationContext(
        case=OrdinaryCase(),
        response=DummyResponse(),
        history=history,
        session_id="SESSION-001",
        turn_number=2,
        round_number=1,
    )

    evaluator = (
        ConversationStateRetentionEvaluator()
    )
    result = evaluator.evaluate(context)

    print(
        f"Recall probe supported: "
        f"{evaluator.supports(context)}"
    )
    print(
        f"Ordinary turn supported: "
        f"{evaluator.supports(ordinary_context)}"
    )
    print(
        "Coverage: "
        f"{result.evidence['current_fact_coverage']:.1%}"
    )
    print(
        "Matched: "
        f"{result.evidence['current_matched_facts']}"
    )
    print(
        "Omitted: "
        f"{result.evidence['current_omitted_facts']}"
    )
    print(
        "Guardrail takeover: "
        f"{result.evidence['guardrail_takeover']}"
    )
    print(
        f"Score rows: {len(result.scores)}"
    )
    print(f"Passed: {result.passed}")

    if not evaluator.supports(context):
        raise SystemExit(
            "Self-test failed: recall probe was not "
            "supported."
        )

    if evaluator.supports(ordinary_context):
        raise SystemExit(
            "Self-test failed: ordinary turn was "
            "incorrectly scored."
        )

    if len(result.scores) != 1:
        raise SystemExit(
            "Self-test failed: evaluator emitted more "
            "than one score."
        )

    if (
        result.evidence[
            "current_fact_coverage"
        ]
        != 1.0
    ):
        raise SystemExit(
            "Self-test failed: configured facts "
            "were not fully retained."
        )

    detailed_fact = (
        "heavy, tight pain across the middle "
        "of the chest"
    )

    if evaluator._match_variant(
        "The patient has chest pain.",
        detailed_fact,
    ) is not None:
        raise SystemExit(
            "Self-test failed: broad 'chest pain' "
            "was incorrectly treated as the complete "
            "detailed fact."
        )

    if evaluator._contains_configured_statement(
        "We still need to complete the planning stage.",
        "the project is complete",
    ):
        raise SystemExit(
            "Self-test failed: planning-stage wording "
            "was treated as project completion."
        )

    if evaluator._contains_configured_statement(
        "John owns Cedar, but actually Priya owns Cedar.",
        "John owns Cedar",
    ):
        raise SystemExit(
            "Self-test failed: corrected statement was "
            "treated as an asserted contradiction."
        )

    if not result.passed:
        raise SystemExit(
            "Self-test failed: valid recall did "
            "not pass."
        )

    print(
        "Conversation state retention evaluator "
        "self-test: passed"
    )
