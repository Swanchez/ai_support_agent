from ai_support_agent import cli
from ai_support_agent.config import LlmProvider
from ai_support_agent.exceptions import LlmRequestError


def test_cli_shows_friendly_message_for_llm_request_error(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "Когда придут деньги?")
    monkeypatch.setattr(cli, "create_llm_client", lambda: object())
    monkeypatch.setattr(cli, "create_gemini_vector_retriever", lambda: object())
    monkeypatch.setattr(cli, "load_llm_provider", lambda: LlmProvider.OPENAI)

    def raise_request_error(question: str, client: object, retriever: object) -> object:
        _ = question, client, retriever
        raise LlmRequestError("OpenAI request failed.")

    monkeypatch.setattr(cli, "answer_question", raise_request_error)

    cli.main()

    assert "Не удалось обратиться к LLM-сервису" in capsys.readouterr().out
