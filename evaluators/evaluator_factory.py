"""
==========================================================
GenAI Evaluation Matrix (GAIEM)

Evaluator Factory

Copyright (c) 2026 NashMarkAI
==========================================================

Purpose
-------
Registers, retrieves, and selects evaluators for benchmark
turns inside continuing conversation sessions.

Every evaluator receives an EvaluationContext containing:

- benchmark case;
- current response;
- complete prior conversation history;
- session identifier;
- turn number;
- round number.
==========================================================
"""

from __future__ import annotations

from collections.abc import Iterable

from evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationContext,
)
from evaluators.consistency import (
    ConsistencyEvaluator,
)
from evaluators.conversation_state_retention import (
    ConversationStateRetentionEvaluator,
)
from evaluators.drift import (
    DriftEvaluator,
)
from evaluators.exact_match import (
    ExactMatchEvaluator,
)
from evaluators.factual_accuracy import (
    FactualAccuracyEvaluator,
)
from evaluators.hallucination import (
    HallucinationEvaluator,
)
from evaluators.instruction_following import (
    InstructionFollowingEvaluator,
)
from evaluators.semantic_similarity import (
    SemanticSimilarityEvaluator,
)
from evaluators.uncertainty import (
    UncertaintyEvaluator,
)


class EvaluatorFactory:
    """
    Registry for GAIEM evaluators.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialise an empty evaluator registry.
        """

        self._evaluators: dict[
            str,
            BaseEvaluator,
        ] = {}

    # -----------------------------------------------------

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        """
        Normalize evaluator names for registry lookup.
        """

        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "Evaluator name must be a string."
            )

        normalized = (
            name.strip().lower()
        )

        if not normalized:
            raise ValueError(
                "Evaluator name must not be empty."
            )

        return normalized

    # -----------------------------------------------------

    def register(
        self,
        evaluator: BaseEvaluator,
        *,
        replace: bool = False,
    ) -> None:
        """
        Register one evaluator instance.
        """

        if not isinstance(
            evaluator,
            BaseEvaluator,
        ):
            raise TypeError(
                "Registered evaluators must inherit "
                "from BaseEvaluator."
            )

        normalized_name = self._normalize_name(
            evaluator.evaluator_name
        )

        if (
            normalized_name in self._evaluators
            and not replace
        ):
            raise ValueError(
                f"Evaluator already registered: "
                f"{evaluator.evaluator_name}"
            )

        self._evaluators[
            normalized_name
        ] = evaluator

    # -----------------------------------------------------

    def register_many(
        self,
        evaluators: Iterable[
            BaseEvaluator
        ],
        *,
        replace: bool = False,
    ) -> None:
        """
        Register multiple evaluator instances.
        """

        for evaluator in evaluators:

            self.register(
                evaluator,
                replace=replace,
            )

    # -----------------------------------------------------

    def unregister(
        self,
        name: str,
    ) -> BaseEvaluator:
        """
        Remove and return a registered evaluator.
        """

        normalized_name = self._normalize_name(
            name
        )

        try:

            return self._evaluators.pop(
                normalized_name
            )

        except KeyError as error:

            raise KeyError(
                f"Unknown evaluator: {name}"
            ) from error

    # -----------------------------------------------------

    def get(
        self,
        name: str,
    ) -> BaseEvaluator:
        """
        Return one registered evaluator.
        """

        normalized_name = self._normalize_name(
            name
        )

        try:

            return self._evaluators[
                normalized_name
            ]

        except KeyError as error:

            raise KeyError(
                f"Unknown evaluator: {name}"
            ) from error

    # -----------------------------------------------------

    def all(
        self,
        *,
        enabled_only: bool = True,
    ) -> tuple[
        BaseEvaluator,
        ...,
    ]:
        """
        Return all registered evaluators.
        """

        evaluators = tuple(
            self._evaluators.values()
        )

        if not enabled_only:
            return evaluators

        return tuple(
            evaluator
            for evaluator in evaluators
            if evaluator.enabled
        )

    # -----------------------------------------------------

    def supported(
        self,
        context: EvaluationContext,
    ) -> tuple[
        BaseEvaluator,
        ...,
    ]:
        """
        Return enabled evaluators that support the supplied
        conversation context.
        """

        if not isinstance(
            context,
            EvaluationContext,
        ):
            raise TypeError(
                "'context' must be an "
                "EvaluationContext."
            )

        supported_evaluators = []

        for evaluator in self.all():

            if evaluator.supports(
                context
            ):

                supported_evaluators.append(
                    evaluator
                )

        return tuple(
            supported_evaluators
        )

    # -----------------------------------------------------

    def names(
        self,
        *,
        enabled_only: bool = True,
    ) -> tuple[str, ...]:
        """
        Return registered evaluator display names.
        """

        return tuple(
            evaluator.evaluator_name
            for evaluator in self.all(
                enabled_only=enabled_only
            )
        )

    # -----------------------------------------------------

    @property
    def count(
        self,
    ) -> int:
        """
        Return the total number of registered evaluators.
        """

        return len(
            self._evaluators
        )


# ---------------------------------------------------------
# Default Factory
# ---------------------------------------------------------

def create_default_factory() -> EvaluatorFactory:
    """
    Create the standard GAIEM evaluator registry.
    """

    evaluator_factory = EvaluatorFactory()

    evaluator_factory.register_many(
        (
            ExactMatchEvaluator(),
            SemanticSimilarityEvaluator(),
            InstructionFollowingEvaluator(),
            FactualAccuracyEvaluator(),
            HallucinationEvaluator(),
            ConsistencyEvaluator(),
            UncertaintyEvaluator(),
            DriftEvaluator(),
            ConversationStateRetentionEvaluator(),
        )
    )

    return evaluator_factory


factory = create_default_factory()


# ==========================================================
# Stand-alone Test
# ==========================================================

if __name__ == "__main__":

    from evaluators.base_evaluator import (
        ConversationMessage,
    )

    class DummyCase:

        case_id = "factory_demo"
        category = "conversation_evaluation"

        expected_answer = "42"

        instructions = {
            "one_word": True,
            "max_words": 1,
            "required_keywords": [
                "42",
            ],
        }

        expected_facts = [
            "42",
        ]

        known_false_claims = [
            "43",
        ]

        contradiction_pairs = [
            [
                "the answer is 42",
                "the answer is 43",
            ],
        ]

        uncertainty_required = None
        required_uncertainty_markers = ()
        forbidden_certainty_markers = ()

        baseline_response = "42"
        drift_threshold = 1.0

        protected_terms = [
            "42",
        ]


    class DummyResponse:

        text = "42"


    history = (
        ConversationMessage(
            role="user",
            content=(
                "Remember that the required answer "
                "is 42."
            ),
            turn_number=1,
        ),
        ConversationMessage(
            role="assistant",
            content="42",
            turn_number=1,
        ),
        ConversationMessage(
            role="user",
            content=(
                "Return the required answer again."
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

    print(
        f"Registered evaluators: "
        f"{factory.count}"
    )

    for evaluator in factory.all():

        print(
            f"- {evaluator.evaluator_name} "
            f"| scope={evaluator.scope}"
        )

    print(
        "\nSupported for current context:"
    )

    supported_evaluators = factory.supported(
        context
    )

    for evaluator in supported_evaluators:

        print(
            f"- {evaluator.evaluator_name} "
            f"| scope={evaluator.scope}"
        )