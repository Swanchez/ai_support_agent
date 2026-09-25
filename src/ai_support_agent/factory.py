"""Construction of the LLM client selected by local configuration."""

from typing import Mapping

from ai_support_agent.config import (
    LlmProvider,
    load_environment,
    load_gemini_config,
    load_llm_provider,
    load_openai_config,
)
from ai_support_agent.exceptions import ConfigurationError
from ai_support_agent.llm_client import GeminiLlmClient, LlmClient, OpenAiLlmClient
from ai_support_agent.tools.gemini_tool_client import GeminiToolCallingClient
from ai_support_agent.tools.confirmation_resolver import GeminiConfirmationResolver


def create_llm_client(env: Mapping[str, str] | None = None) -> LlmClient:
    """Create the configured provider client without making an LLM request."""

    environment = load_environment(env)
    provider = load_llm_provider(environment)

    match provider:
        case LlmProvider.OPENAI:
            return OpenAiLlmClient(load_openai_config(environment))
        case LlmProvider.GEMINI:
            return GeminiLlmClient(load_gemini_config(environment))


def create_gemini_tool_calling_client(
    env: Mapping[str, str] | None = None,
) -> GeminiToolCallingClient:
    """Create the Gemini-specific client used for application-controlled tools."""

    environment = load_environment(env)
    provider = load_llm_provider(environment)
    if provider is not LlmProvider.GEMINI:
        raise ConfigurationError("Tool calling is currently implemented only for Gemini.")
    return GeminiToolCallingClient(load_gemini_config(environment))


def create_gemini_confirmation_resolver(
    env: Mapping[str, str] | None = None,
) -> GeminiConfirmationResolver:
    """Create the Gemini adapter that resolves a pending action's confirmation."""

    environment = load_environment(env)
    if load_llm_provider(environment) is not LlmProvider.GEMINI:
        raise ConfigurationError("Confirmation resolver is currently implemented only for Gemini.")
    return GeminiConfirmationResolver(load_gemini_config(environment))
