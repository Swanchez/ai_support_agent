import pytest

from ai_support_agent.config import (
    DatabaseConfig,
    GeminiEmbeddingConfig,
    GeminiConfig,
    LlmProvider,
    load_gemini_embedding_config,
    load_database_config,
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


def test_load_database_config_returns_connection_settings() -> None:
    config = load_database_config(
        {
            "POSTGRES_DB": "support",
            "POSTGRES_USER": "app_user",
            "POSTGRES_PASSWORD": "test-password",
            "DATABASE_HOST": "db.internal",
            "DATABASE_PORT": "5433",
        }
    )

    assert config == DatabaseConfig(
        host="db.internal",
        port=5433,
        database="support",
        user="app_user",
        password="test-password",
    )


@pytest.mark.parametrize("value", ["", "not-a-number", "0", "65536"])
def test_load_database_config_rejects_an_invalid_port(value: str) -> None:
    env = {
        "POSTGRES_DB": "support",
        "POSTGRES_USER": "app_user",
        "POSTGRES_PASSWORD": "test-password",
        "DATABASE_PORT": value,
    }

    with pytest.raises(ConfigurationError, match="DATABASE_PORT"):
        load_database_config(env)


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
