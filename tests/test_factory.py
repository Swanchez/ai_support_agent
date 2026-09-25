import pytest

from ai_support_agent.exceptions import ConfigurationError
from ai_support_agent.factory import (
    create_gemini_confirmation_resolver,
    create_gemini_tool_calling_client,
    create_llm_client,
)
from ai_support_agent.llm_client import GeminiLlmClient, OpenAiLlmClient
from ai_support_agent.tools.gemini_tool_client import GeminiToolCallingClient
from ai_support_agent.tools.confirmation_resolver import GeminiConfirmationResolver


def test_create_llm_client_selects_openai() -> None:
    client = create_llm_client(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "test-model",
        }
    )

    assert isinstance(client, OpenAiLlmClient)
    assert client.config.model == "test-model"


def test_create_llm_client_selects_gemini() -> None:
    client = create_llm_client(
        {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
        }
    )

    assert isinstance(client, GeminiLlmClient)
    assert client.config.model == "test-model"


def test_create_gemini_tool_calling_client_uses_selected_gemini_config() -> None:
    client = create_gemini_tool_calling_client(
        {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
        }
    )

    assert isinstance(client, GeminiToolCallingClient)
    assert client.config.model == "test-model"


def test_create_gemini_tool_calling_client_rejects_other_provider() -> None:
    with pytest.raises(ConfigurationError, match="only for Gemini"):
        create_gemini_tool_calling_client(
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "test-model",
            }
        )


def test_create_gemini_confirmation_resolver_uses_selected_gemini_config() -> None:
    resolver = create_gemini_confirmation_resolver(
        {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
        }
    )

    assert isinstance(resolver, GeminiConfirmationResolver)
    assert resolver.config.model == "test-model"
