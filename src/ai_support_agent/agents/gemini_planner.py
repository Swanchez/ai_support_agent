"""Gemini adapter that plans one bounded agent step at a time."""

import json
from dataclasses import dataclass
from typing import Any

from google.genai import errors as gemini_errors
from google.genai._gaos.lib.compat_errors import GeminiNextGenAPIClientError
from pydantic import ValidationError

from ai_support_agent.agents.core import (
    AgentAction,
    AgentActionProposal,
    AgentDecision,
    AgentPlannerResult,
    AgentState,
)
from ai_support_agent.config import GeminiConfig
from ai_support_agent.exceptions import InvalidModelResponseError, LlmRequestError
from ai_support_agent.llm_client import LlmResult, gemini_response_format
from ai_support_agent.schemas import SupportResponse
from ai_support_agent.tools.gemini_adapter import parse_gemini_tool_turn


AGENT_SYSTEM_PROMPT = """Ты — AI Support Agent. Отвечай только на русском языке, кратко и по делу.
Используй только факты из observations и результатов доступных read-инструментов.
За один шаг вызови не более одного инструмента. Не вызывай write-инструменты и не выдумывай данные.
Если observations достаточно, верни финальный ответ строго в согласованном JSON-формате.
Внешний справочный источник не подтверждает внутренние правила магазина."""
AGENT_SYSTEM_PROMPT += """
Если observation содержит result.ok=false, считай это окончательным результатом инструмента для текущего шага.
Не повторяй тот же инструмент с теми же аргументами. При code=order_not_found честно сообщи, что заказ не найден, и верни status=insufficient_context.
При временной технической ошибке честно сообщи о недоступности и предложи повторить попытку позже; сетевые повторы выполняет код клиента, а не агент.
Если search_knowledge_base успешно вернул пустой список chunks, сообщи, что точной информации в доступных источниках нет, и верни status=insufficient_context.
"""
AGENT_SYSTEM_PROMPT += """
Инструменты, обозначенные приложением как proposal-only, не выполняют действие: их вызов лишь создаёт предложение для подтверждения пользователем.
Используй proposal-only отмену заказа только при явной просьбе отменить заказ с указанным номером ORD-<цифры>.
"""


@dataclass
class GeminiAgentPlanner:
    """Stateless planner that chooses exactly one agent action or a final response."""

    config: GeminiConfig
    tool_definitions: list[dict[str, object]]
    proposal_tool_names: frozenset[str] = frozenset()
    sdk_client: Any | None = None

    def __post_init__(self) -> None:
        if self.sdk_client is None:
            from google import genai

            self.sdk_client = genai.Client(api_key=self.config.api_key)

    def decide(self, state: AgentState) -> AgentPlannerResult:
        """Ask Gemini for one next read action or one final structured response."""

        response = self._request(state, tools=self.tool_definitions, limit_reached=False)
        turn = parse_gemini_tool_turn(response)
        if len(turn.calls) > 1:
            raise InvalidModelResponseError("Agent planner proposed more than one action per step.")
        if turn.calls:
            call = turn.calls[0]
            if call.name in self.proposal_tool_names:
                return AgentPlannerResult(
                    AgentDecision(
                        proposal=AgentActionProposal(call.name, call.arguments)
                    ),
                    _llm_result(self.config.model, response),
                )
            return AgentPlannerResult(
                AgentDecision(action=AgentAction(call.name, call.arguments)),
                _llm_result(self.config.model, response),
            )
        return AgentPlannerResult(
            AgentDecision(response=_support_response(response)),
            _llm_result(self.config.model, response),
        )

    def finalize_after_limit(self, state: AgentState) -> AgentPlannerResult:
        """Force a grounded final JSON response after the action budget is exhausted."""

        response = self._request(state, tools=[], limit_reached=True)
        return AgentPlannerResult(
            AgentDecision(response=_support_response(response)),
            _llm_result(self.config.model, response),
        )

    def _request(
        self,
        state: AgentState,
        *,
        tools: list[dict[str, object]],
        limit_reached: bool,
    ) -> object:
        instruction = AGENT_SYSTEM_PROMPT
        if limit_reached:
            instruction += "\nЛимит действий исчерпан: не вызывай инструменты, ответь по текущим observations."
        try:
            return self.sdk_client.interactions.create(
                model=self.config.model,
                system_instruction=instruction,
                input=_render_state(state),
                tools=tools,
                response_format=gemini_response_format(),
                store=False,
            )
        except (gemini_errors.APIError, GeminiNextGenAPIClientError) as error:
            raise LlmRequestError("Gemini agent planning request failed.") from error


def _render_state(state: AgentState) -> str:
    """Send only user question and sanitized observations to the stateless planner."""

    observations = [
        {"tool": item.action.name, "arguments": item.action.arguments, "result": item.result.data}
        for item in state.observations
    ]
    return json.dumps(
        {"question": state.original_question, "observations": observations},
        ensure_ascii=False,
    )


def _support_response(response: object) -> SupportResponse:
    """Validate the final planner output against the existing response contract."""

    try:
        return SupportResponse.model_validate_json(getattr(response, "output_text", "") or "")
    except ValidationError as error:
        raise InvalidModelResponseError("Agent final response does not match SupportResponse.") from error


def _llm_result(model: str, response: object) -> LlmResult:
    """Normalize Gemini usage for one planner request."""

    usage = getattr(response, "usage", None)
    if usage is None:
        raise LlmRequestError("Gemini agent response did not include usage metadata.")
    return LlmResult(
        text=getattr(response, "output_text", "") or "",
        model=model,
        input_tokens=getattr(usage, "total_input_tokens", 0) or 0,
        output_tokens=getattr(usage, "total_output_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )
