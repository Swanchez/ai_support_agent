"""Contracts and adapters for turning text into semantic vectors."""

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, runtime_checkable

from google.genai import errors as gemini_errors
from google.genai._gaos.lib.compat_errors import GeminiNextGenAPIClientError

from ai_support_agent.config import GeminiEmbeddingConfig
from ai_support_agent.exceptions import EmbeddingRequestError


Embedding: TypeAlias = tuple[float, ...]


class EmbeddingClient(Protocol):
    """A provider-independent service that represents text as a vector."""

    def embed(self, text: str) -> Embedding:
        """Return one embedding vector for one non-empty text."""


@runtime_checkable
class BatchEmbeddingClient(EmbeddingClient, Protocol):
    """An embedding client that can represent several texts in one API request."""

    def embed_many(self, texts: list[str]) -> list[Embedding]:
        """Return vectors in the same order as the input texts."""


GEMINI_EMBEDDING_DIMENSIONS = 768


@dataclass
class GeminiEmbeddingClient:
    """Adapter from Gemini's embedding API to the application's contract."""

    config: GeminiEmbeddingConfig
    sdk_client: Any | None = None

    def __post_init__(self) -> None:
        if self.sdk_client is None:
            from google import genai

            self.sdk_client = genai.Client(api_key=self.config.api_key)

    def embed(self, text: str) -> Embedding:
        """Create one fixed-size embedding vector for non-empty text."""

        if not text.strip():
            raise ValueError("Text for embedding must not be blank.")

        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[Embedding]:
        """Create one vector per text in a single Gemini embedding request."""

        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Text for embedding must not be blank.")

        from google.genai import types

        try:
            response = self.sdk_client.models.embed_content(
                model=self.config.model,
                contents=[
                    types.Content(parts=[types.Part.from_text(text=text)])
                    for text in texts
                ],
                config=types.EmbedContentConfig(
                    output_dimensionality=GEMINI_EMBEDDING_DIMENSIONS
                ),
            )
        except (gemini_errors.APIError, GeminiNextGenAPIClientError) as error:
            status_code = getattr(error, "code", getattr(error, "status_code", None))
            detail = f"HTTP {status_code}" if isinstance(status_code, int) else "unknown status"
            raise EmbeddingRequestError(
                f"Gemini embedding request failed ({detail})."
            ) from error

        embeddings = response.embeddings or []
        if len(embeddings) != len(texts) or any(not embedding.values for embedding in embeddings):
            raise EmbeddingRequestError(
                "Gemini embedding response did not include one vector for every text."
            )
        return [tuple(embedding.values) for embedding in embeddings]
