from ai_support_agent.exceptions import (
    ConfigurationError,
    EmbeddingRequestError,
    InvalidModelResponseError,
    LlmRequestError,
)
from ai_support_agent.assistant_service import AssistantService
from ai_support_agent.conversation_service import ConversationService
from ai_support_agent.config import LlmProvider, load_llm_provider
from ai_support_agent.factory import (
    create_gemini_confirmation_resolver,
    create_gemini_tool_calling_client,
    create_llm_client,
)
from ai_support_agent.rag.fallback_retriever import FallbackRetriever
from ai_support_agent.rag.runtime import (
    create_external_reference_gemini_vector_retriever,
    create_gemini_vector_retriever,
)
from ai_support_agent.service import AnswerResult, EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD, answer_question
from ai_support_agent.tools.executor import DEFAULT_TOOL_EXECUTOR
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT
from ai_support_agent.tools.confirmation import InMemoryPendingActionStore


def main() -> None:
    try:
        retriever = FallbackRetriever(
            primary=create_gemini_vector_retriever(),
            fallback_factory=create_external_reference_gemini_vector_retriever,
            fallback_threshold=EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD,
        )
        provider = load_llm_provider()
        if provider is LlmProvider.GEMINI:
            assistant_service = AssistantService(
                retriever=retriever,
                tool_client=create_gemini_tool_calling_client(),
                tool_executor=DEFAULT_TOOL_EXECUTOR,
                tool_context=DEMO_TOOL_CONTEXT,
                pending_action_store=InMemoryPendingActionStore(),
            )
            conversation = ConversationService(
                assistant_service=assistant_service,
                confirmation_resolver=create_gemini_confirmation_resolver(),
            )
            _run_gemini_dialogue(conversation)
            return
        else:
            question = input("Ваш вопрос: ").strip()
            if not question:
                print("Введите вопрос и запустите программу снова.")
                return
            result = answer_question(question, create_llm_client(), retriever)
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
        return
    except (LlmRequestError, EmbeddingRequestError):
        print("Не удалось обратиться к LLM-сервису. Попробуйте позже.")
        return
    except InvalidModelResponseError:
        print("Не удалось обработать ответ ассистента. Попробуйте ещё раз.")
        return

    _print_result(result)


def _run_gemini_dialogue(conversation: ConversationService) -> None:
    """Keep application-owned pending state alive for one local console session."""

    print("Введите вопрос. Для выхода напишите: выход")
    while True:
        prompt = "Подтвердите действие (или отмените его): " if conversation.has_pending_action() else "Ваш вопрос: "
        question = input(prompt).strip()
        if question.lower() in {"выход", "exit", "quit"}:
            return
        if not question:
            print("Введите вопрос или напишите «выход».")
            continue
        try:
            result = conversation.answer(question)
        except (LlmRequestError, EmbeddingRequestError):
            print("Не удалось обратиться к LLM-сервису. Попробуйте позже.")
            continue
        except InvalidModelResponseError:
            print("Не удалось обработать ответ ассистента. Попробуйте ещё раз.")
            continue
        _print_result(result)


def _print_result(result: AnswerResult) -> None:
    """Render an AnswerResult without exposing tool internals or session state."""

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
        "\n[Usage] "
        f"model={result.model}; input={result.input_tokens}; "
        f"output={result.output_tokens}; total={result.total_tokens}"
    )


if __name__ == "__main__":
    main()
