"""Interactive console entry point for the bounded agent with HITL confirmation."""

import sys

from ai_support_agent.agent_assistant_service import AgentAssistantService
from ai_support_agent.agents.runtime import build_gemini_agent_runner
from ai_support_agent.agents.trace import format_agent_trace
from ai_support_agent.config import load_gemini_config
from ai_support_agent.conversation_service import ConversationService
from ai_support_agent.exceptions import (
    ConfigurationError,
    EmbeddingRequestError,
    InvalidModelResponseError,
    LlmRequestError,
)
from ai_support_agent.factory import create_gemini_confirmation_resolver
from ai_support_agent.rag.fallback_retriever import FallbackRetriever
from ai_support_agent.rag.runtime import (
    create_external_reference_gemini_vector_retriever,
    create_gemini_vector_retriever,
)
from ai_support_agent.service import AnswerResult, EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD
from ai_support_agent.tools.confirmation import InMemoryPendingActionStore
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT
from ai_support_agent.tools.executor import DEFAULT_TOOL_EXECUTOR


def main(debug: bool = False) -> None:
    """Keep agent and pending confirmation state alive for one console session."""

    try:
        retriever = FallbackRetriever(
            primary=create_gemini_vector_retriever(),
            fallback_factory=create_external_reference_gemini_vector_retriever,
            fallback_threshold=EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD,
        )
        assistant = AgentAssistantService(
            agent_runner=build_gemini_agent_runner(
                retriever=retriever,
                config=load_gemini_config(),
                executor=DEFAULT_TOOL_EXECUTOR,
                context=DEMO_TOOL_CONTEXT,
            ),
            tool_executor=DEFAULT_TOOL_EXECUTOR,
            tool_context=DEMO_TOOL_CONTEXT,
            pending_action_store=InMemoryPendingActionStore(),
        )
        conversation = ConversationService(
            assistant_service=assistant,
            confirmation_resolver=create_gemini_confirmation_resolver(),
        )
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
        return

    print("Введите вопрос. Для выхода напишите: выход")
    while True:
        pending_before_message = conversation.has_pending_action()
        prompt = (
            "Подтвердите действие (или отмените его): "
            if pending_before_message
            else "Вопрос агенту: "
        )
        user_message = input(prompt).strip()
        if user_message.lower() in {"выход", "exit", "quit"}:
            return
        if not user_message:
            print("Введите вопрос или напишите «выход».")
            continue

        try:
            result = conversation.answer(user_message)
        except (LlmRequestError, EmbeddingRequestError):
            print("Не удалось обратиться к LLM-сервису. Попробуйте позже.")
            continue
        except InvalidModelResponseError:
            print("Не удалось обработать ответ агента. Попробуйте ещё раз.")
            continue

        _print_result(result)
        if debug and not pending_before_message and assistant.last_run is not None:
            print(f"\n{format_agent_trace(assistant.last_run.state)}")


def _print_result(result: AnswerResult) -> None:
    """Render a user answer without leaking pending-action internals."""

    response = result.response
    print(f"\n{response.answer}")
    if response.alternative:
        print(f"\n{response.alternative}")
    if response.recommendations:
        print("\nРекомендации:")
        for recommendation in response.recommendations:
            print(f"- {recommendation}")
    if response.sources:
        print("\nИсточники:")
        for source in response.sources:
            print(f"- {source}")
    print(
        "\n[Agent] "
        f"model={result.model}; input={result.input_tokens}; "
        f"output={result.output_tokens}; total={result.total_tokens}"
    )


if __name__ == "__main__":
    main(debug="--debug" in sys.argv)
