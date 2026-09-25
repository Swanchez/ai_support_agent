"""Deterministic checks for important properties of a generated support answer."""

from dataclasses import dataclass

from ai_support_agent.schemas import AnswerStatus, SupportResponse


@dataclass(frozen=True)
class AnswerEvaluationCase:
    """Expected observable properties of one answer, not one exact wording."""

    question: str
    expected_status: AnswerStatus
    expected_source_ids: frozenset[str]
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerEvaluationResult:
    """The failures found when one response is compared with its expected behavior."""

    case: AnswerEvaluationCase
    response: SupportResponse
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every deterministic expectation was met."""

        return not self.failures


DEFAULT_ANSWER_EVALUATION_CASES: tuple[AnswerEvaluationCase, ...] = (
    AnswerEvaluationCase(
        question="Когда придут деньги за возврат?",
        expected_status=AnswerStatus.ANSWERED,
        expected_source_ids=frozenset({"refund-policy-v1"}),
        required_phrases=("3-7 рабочих дней",),
        forbidden_phrases=("14 дней", "сразу после отправки"),
    ),
    AnswerEvaluationCase(
        question="Как поменять адрес доставки?",
        expected_status=AnswerStatus.INSUFFICIENT_CONTEXT,
        expected_source_ids=frozenset(),
    ),
    AnswerEvaluationCase(
        question="В течение какого срока можно обменять непродовольственный товар?",
        expected_status=AnswerStatus.INSUFFICIENT_CONTEXT,
        expected_source_ids=frozenset({"consumer-exchange-return-v1"}),
        required_phrases=("справочно:", "14 дней"),
    ),
)


def evaluate_answer(
    case: AnswerEvaluationCase,
    response: SupportResponse,
) -> AnswerEvaluationResult:
    """Check status, real sources, and selected high-risk phrases of one response."""

    failures: list[str] = []
    if response.status is not case.expected_status:
        failures.append(
            f"Expected status {case.expected_status.value}, got {response.status.value}."
        )

    if frozenset(response.sources) != case.expected_source_ids:
        failures.append(
            "Expected sources "
            f"{sorted(case.expected_source_ids)}, got {sorted(response.sources)}."
        )

    response_text = _normalise_response_text(response)
    for phrase in case.required_phrases:
        if _normalise(phrase) not in response_text:
            failures.append(f"Missing required phrase: {phrase!r}.")
    for phrase in case.forbidden_phrases:
        if _normalise(phrase) in response_text:
            failures.append(f"Found forbidden phrase: {phrase!r}.")

    return AnswerEvaluationResult(case, response, tuple(failures))


def _normalise_response_text(response: SupportResponse) -> str:
    """Join user-visible content before checking text expectations."""

    return _normalise(
        " ".join(
            part
            for part in (
                response.answer,
                response.alternative or "",
                *response.recommendations,
            )
            if part
        )
    )


def _normalise(text: str) -> str:
    """Make harmless whitespace and dash variants equivalent in simple phrase checks."""

    return " ".join(text.casefold().replace("–", "-").replace("—", "-").split())
