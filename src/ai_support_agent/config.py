"""Loading and validation of local application configuration."""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from ai_support_agent.exceptions import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


class LlmProvider(StrEnum):
    """LLM providers supported by this application."""

    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass(frozen=True)
class OpenAiConfig:
    """Settings necessary to create an OpenAI client."""

    api_key: str
    model: str


@dataclass(frozen=True)
class GeminiConfig:
    """Settings necessary to create a Gemini client."""

    api_key: str
    model: str


@dataclass(frozen=True)
class GeminiEmbeddingConfig:
    """Settings necessary to create a Gemini embedding client."""

    api_key: str
    model: str


def load_environment(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """Load local .env settings unless an explicit environment was supplied."""

    if env is None:
        load_dotenv(DOTENV_PATH, override=False)
        return os.environ

    return env


def load_llm_provider(env: Mapping[str, str] | None = None) -> LlmProvider:
    """Load the selected provider before constructing any network client."""

    environment = load_environment(env)
    value = environment.get("LLM_PROVIDER", "").strip().lower()

    try:
        return LlmProvider(value)
    except ValueError as error:
        supported = ", ".join(provider.value for provider in LlmProvider)
        raise ConfigurationError(
            f"LLM_PROVIDER must be one of: {supported}."
        ) from error


def load_openai_config(env: Mapping[str, str] | None = None) -> OpenAiConfig:
    """Load OpenAI settings without ever exposing the secret in an error."""

    environment = load_environment(env)

    api_key = environment.get("OPENAI_API_KEY", "").strip()
    model = environment.get("OPENAI_MODEL", "").strip()

    if not api_key:
        raise ConfigurationError("OPENAI_API_KEY is not configured.")
    if not model:
        raise ConfigurationError("OPENAI_MODEL is not configured.")

    return OpenAiConfig(api_key=api_key, model=model)


def load_gemini_config(env: Mapping[str, str] | None = None) -> GeminiConfig:
    """Load Gemini settings without ever exposing the secret in an error."""

    environment = load_environment(env)

    api_key = environment.get("GEMINI_API_KEY", "").strip()
    model = environment.get("GEMINI_MODEL", "").strip()

    if not api_key:
        raise ConfigurationError("GEMINI_API_KEY is not configured.")
    if not model:
        raise ConfigurationError("GEMINI_MODEL is not configured.")

    return GeminiConfig(api_key=api_key, model=model)


def load_gemini_embedding_config(
    env: Mapping[str, str] | None = None,
) -> GeminiEmbeddingConfig:
    """Load Gemini embedding settings without exposing the secret in an error."""

    environment = load_environment(env)

    api_key = environment.get("GEMINI_API_KEY", "").strip()
    model = environment.get("GEMINI_EMBEDDING_MODEL", "").strip()

    if not api_key:
        raise ConfigurationError("GEMINI_API_KEY is not configured.")
    if not model:
        raise ConfigurationError("GEMINI_EMBEDDING_MODEL is not configured.")

    return GeminiEmbeddingConfig(api_key=api_key, model=model)
