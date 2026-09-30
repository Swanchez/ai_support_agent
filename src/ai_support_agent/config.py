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


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for the application's PostgreSQL database."""

    host: str
    port: int
    database: str
    user: str
    password: str


@dataclass(frozen=True)
class AuthConfig:
    """Settings for signing short-lived user access tokens."""

    jwt_secret: str
    issuer: str
    access_token_ttl_minutes: int
    cookie_secure: bool


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


def load_database_config(env: Mapping[str, str] | None = None) -> DatabaseConfig:
    """Load PostgreSQL settings without placing the password in an error message."""

    environment = load_environment(env)
    database = environment.get("POSTGRES_DB", "").strip()
    user = environment.get("POSTGRES_USER", "").strip()
    password = environment.get("POSTGRES_PASSWORD", "").strip()
    host = environment.get("DATABASE_HOST", "127.0.0.1").strip()
    raw_port = environment.get("DATABASE_PORT", "5432").strip()

    for name, value in (
        ("POSTGRES_DB", database),
        ("POSTGRES_USER", user),
        ("POSTGRES_PASSWORD", password),
        ("DATABASE_HOST", host),
    ):
        if not value:
            raise ConfigurationError(f"{name} is not configured.")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise ConfigurationError("DATABASE_PORT must be an integer.") from error
    if not 1 <= port <= 65_535:
        raise ConfigurationError("DATABASE_PORT must be between 1 and 65535.")

    return DatabaseConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )


def load_auth_config(env: Mapping[str, str] | None = None) -> AuthConfig:
    """Load JWT settings without ever returning the secret in error text."""

    environment = load_environment(env)
    jwt_secret = environment.get("AUTH_JWT_SECRET", "").strip()
    issuer = environment.get("AUTH_JWT_ISSUER", "ai-support-agent").strip()
    raw_ttl = environment.get("AUTH_ACCESS_TOKEN_TTL_MINUTES", "30").strip()
    raw_cookie_secure = environment.get("AUTH_COOKIE_SECURE", "false").strip().lower()

    if len(jwt_secret) < 32:
        raise ConfigurationError("AUTH_JWT_SECRET must contain at least 32 characters.")
    if not issuer:
        raise ConfigurationError("AUTH_JWT_ISSUER is not configured.")
    try:
        access_token_ttl_minutes = int(raw_ttl)
    except ValueError as error:
        raise ConfigurationError("AUTH_ACCESS_TOKEN_TTL_MINUTES must be an integer.") from error
    if not 1 <= access_token_ttl_minutes <= 1_440:
        raise ConfigurationError(
            "AUTH_ACCESS_TOKEN_TTL_MINUTES must be between 1 and 1440."
        )
    if raw_cookie_secure not in {"true", "false"}:
        raise ConfigurationError("AUTH_COOKIE_SECURE must be true or false.")
    return AuthConfig(
        jwt_secret=jwt_secret,
        issuer=issuer,
        access_token_ttl_minutes=access_token_ttl_minutes,
        cookie_secure=raw_cookie_secure == "true",
    )
