"""Manual console entry point for the Gemini order-status tool-calling scenario."""

from pydantic import ValidationError

from ai_support_agent.config import load_gemini_config
from ai_support_agent.exceptions import ConfigurationError, LlmRequestError
from ai_support_agent.schemas import SupportResponse
from ai_support_agent.tools.executor import DEFAULT_TOOL_EXECUTOR
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT
from ai_support_agent.tools.gemini_tool_client import GeminiToolCallingClient
from ai_support_agent.tools.order_status_prompt import build_order_status_tool_call


def format_support_response(response: SupportResponse) -> str:
    """Render the user-facing portion of a validated tool-assisted support response."""

    lines = [response.answer]
    if response.alternative:
        lines.extend(("", response.alternative))
    if response.recommendations:
        lines.extend(("", "Рекомендации:"))
        lines.extend(f"- {recommendation}" for recommendation in response.recommendations)
    if response.sources:
        lines.extend(("", "Источники:"))
        lines.extend(f"- {source}" for source in response.sources)
    return "\n".join(lines)


def attach_trusted_tool_sources(
    response: SupportResponse,
    source_ids: list[str],
) -> SupportResponse:
    """Replace model-produced provenance with the tools actually run by application code."""

    return response.model_copy(update={"sources": source_ids})


def main() -> None:
    """Ask one status question and run the isolated manual Gemini tool-calling loop."""

    question = input("Ваш вопрос о заказе: ").strip()
    if not question:
        print("Введите вопрос со статусом и номером заказа, например: Где заказ ORD-1001?")
        return

    try:
        client = GeminiToolCallingClient(load_gemini_config())
        tool_result = client.complete_with_tools(
            build_order_status_tool_call(question),
            DEFAULT_TOOL_EXECUTOR,
            DEMO_TOOL_CONTEXT,
        )
        response = SupportResponse.model_validate_json(tool_result.llm_result.text)
        response = attach_trusted_tool_sources(response, tool_result.trusted_source_ids())
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
        return
    except LlmRequestError:
        print("Не удалось обратиться к LLM-сервису. Попробуйте позже.")
        return
    except ValidationError:
        print("Модель не вернула ожидаемый структурированный ответ.")
        return

    print(f"\n{format_support_response(response)}")
    print(
        "\n[Usage] "
        f"model={tool_result.llm_result.model}; input={tool_result.llm_result.input_tokens}; "
        f"output={tool_result.llm_result.output_tokens}; "
        f"total={tool_result.llm_result.total_tokens}"
    )


if __name__ == "__main__":
    main()
