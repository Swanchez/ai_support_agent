import pytest

from ai_support_agent.config import (
    GeminiEmbeddingConfig,
    GeminiConfig,
    LlmProvider,
    load_gemini_embedding_config,
    OpenAiConfig,
    load_gemini_config,
    load_llm_provider,
    load_openai_config,
)
from ai_support_agent.exceptions import ConfigurationError


def test_load_openai_config_returns_required_settings() -> None:
    config = load_openai_config(
        {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}
    )

    assert config == OpenAiConfig(api_key="test-key", model="test-model")


@pytest.mark.parametrize("missing_name", ["OPENAI_API_KEY", "OPENAI_MODEL"])
def test_load_openai_config_rejects_missing_required_setting(
    missing_name: str,
) -> None:
    env = {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}
    del env[missing_name]

    with pytest.raises(ConfigurationError, match=missing_name):
        load_openai_config(env)


def test_load_llm_provider_normalizes_the_selected_provider() -> None:
    provider = load_llm_provider({"LLM_PROVIDER": " Gemini "})

    assert provider is LlmProvider.GEMINI


@pytest.mark.parametrize("value", ["", "unknown-provider"])
def test_load_llm_provider_rejects_unsupported_value(value: str) -> None:
    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        load_llm_provider({"LLM_PROVIDER": value})


def test_load_gemini_config_returns_required_settings() -> None:
    config = load_gemini_config(
        {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"}
    )

    assert config == GeminiConfig(api_key="test-key", model="test-model")


def test_load_gemini_embedding_config_returns_required_settings() -> None:
    config = load_gemini_embedding_config(
        {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_EMBEDDING_MODEL": "test-embedding-model",
        }
    )

    assert config == GeminiEmbeddingConfig(
        api_key="test-key", model="test-embedding-model"
    )


@pytest.mark.parametrize("missing_name", ["GEMINI_API_KEY", "GEMINI_MODEL"])
def test_load_gemini_config_rejects_missing_required_setting(
    missing_name: str,
) -> None:
    env = {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"}
    del env[missing_name]

    with pytest.raises(ConfigurationError, match=missing_name):
        load_gemini_config(env)


@pytest.mark.parametrize("missing_name", ["GEMINI_API_KEY", "GEMINI_EMBEDDING_MODEL"])
def test_load_gemini_embedding_config_rejects_missing_required_setting(
    missing_name: str,
) -> None:
    env = {
        "GEMINI_API_KEY": "test-key",
        "GEMINI_EMBEDDING_MODEL": "test-embedding-model",
    }
    del env[missing_name]

    with pytest.raises(ConfigurationError, match=missing_name):
        load_gemini_embedding_config(env)
    load_gemini_embedding_config,
