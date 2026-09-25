"""Orchestration of RAG evidence and application-controlled tools."""

from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from ai_support_agent.exceptions import InvalidModelResponseError
from ai_support_agent.llm_client import LlmCall
from ai_support_agent.rag.retriever import Retriever
from ai_support_agent.service import (
    AnswerResult,
    RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT,
    RETRIEVAL_THRESHOLD,
    RETRIEVAL_TOP_K,
    SOURCE_TYPE_RULE,
    SYSTEM_PROMPT,
    format_retrieved_context,
    source_ids_from_chunks,
)
from ai_support_agent.schemas import AnswerStatus, SupportResponse
from ai_support_agent.tools.executor import ToolExecutor
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.confirmation import InMemoryPendingActionStore
from ai_support_agent.tools.gemini_tool_client import ToolCallingResult
from ai_support_agent.tools.order_status_prompt import ORDER_TOOL_RULES


class ToolCallingClient(Protocol):
    """Provider boundary for one controlled tool-calling cycle."""

    def complete_with_tools(
        self,
        call: LlmCall,
        executor: ToolExecutor,
        context: ToolExecutionContext,
    ) -> ToolCallingResult:
        """Return the final model text and tools actually executed by application code."""


def build_assistant_tool_call(user_question: str, retrieved_context: str) -> LlmCall:
    """Build one prompt containing grounded evidence and tightly scoped tool rules."""

    return LlmCall(
        messages=[
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n{SOURCE_TYPE_RULE}\n{ORDER_TOOL_RULES}",
            },
            {"role": "user", "content": f"Контекст:\n{retrieved_context}"},
            {"role": "user", "content": user_question},
        ]
    )


@dataclass
class AssistantService:
    """One application service that joins RAG, tool execution, and provenance."""

    retriever: Retriever
    tool_client: ToolCallingClient
    tool_executor: ToolExecutor
    tool_context: ToolExecutionContext
    pending_action_store: InMemoryPendingActionStore

    def answer(self, user_question: str) -> AnswerResult:
        """Retrieve evidence, run an optional tool round, and validate one response contract."""

        chunks = self.retriever.retrieve(
            user_question,
            top_k=RETRIEVAL_TOP_K,
            threshold=RETRIEVAL_THRESHOLD,
            max_chunks_per_document=RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT,
        )
        call = build_assistant_tool_call(user_question, format_retrieved_context(chunks))
        tool_result = self.tool_client.complete_with_tools(
            call,
            self.tool_executor,
            self.tool_context,
        )
        for tool_call in tool_result.pending_tool_calls:
            self.pending_action_store.create(
                user_id=self.tool_context.current_user_id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
            )
        if tool_result.pending_tool_calls:
            result = tool_result.llm_result
            return AnswerResult(
                response=SupportResponse(
                    status=AnswerStatus.CLARIFICATION_NEEDED,
                    answer="Запрошенное действие ожидает вашего подтверждения.",
                    alternative=None,
                    recommendations=["Подтвердите действие или отмените его."],
                    sources=[],
                ),
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
            )
        try:
            response = SupportResponse.model_validate_json(tool_result.llm_result.text)
        except ValidationError as error:
            raise InvalidModelResponseError(
                "LLM response does not match the SupportResponse contract."
            ) from error

        trusted_sources = source_ids_from_chunks(chunks) + tool_result.trusted_source_ids()
        response = response.model_copy(update={"sources": list(dict.fromkeys(trusted_sources))})
        result = tool_result.llm_result
        return AnswerResult(
            response=response,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
        )
