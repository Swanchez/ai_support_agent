from types import SimpleNamespace

import pytest
from google.genai import errors as gemini_errors

from ai_support_agent.config import GeminiEmbeddingConfig
from ai_support_agent.exceptions import EmbeddingRequestError
from ai_support_agent.rag.embeddings import (
    GEMINI_EMBEDDING_DIMENSIONS,
    GeminiEmbeddingClient,
)


class FakeModelsApi:
    def __init__(self, response: object) -> None:
        self.response = response
        self.received_kwargs: dict[str, object] | None = None

    def embed_content(self, **kwargs: object) -> object:
        self.received_kwargs = kwargs
        return self.response


def test_gemini_embedding_client_translates_our_contract_to_sdk() -> None:
    fake_models_api = FakeModelsApi(
        SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, -0.2, 0.3])])
    )
    client = GeminiEmbeddingClient(
        config=GeminiEmbeddingConfig(api_key="test-key", model="test-embedding-model"),
        sdk_client=SimpleNamespace(models=fake_models_api),
    )

    embedding = client.embed("Текст для поиска")

    assert embedding == (0.1, -0.2, 0.3)
    assert fake_models_api.received_kwargs is not None
    assert fake_models_api.received_kwargs["model"] == "test-embedding-model"
    assert isinstance(fake_models_api.received_kwargs["contents"], list)
    assert len(fake_models_api.received_kwargs["contents"]) == 1
    assert (
        fake_models_api.received_kwargs["config"].output_dimensionality
        == GEMINI_EMBEDDING_DIMENSIONS
    )


def test_gemini_embedding_client_rejects_blank_text_without_sdk_call() -> None:
    fake_models_api = FakeModelsApi(SimpleNamespace(embeddings=[]))
    client = GeminiEmbeddingClient(
        config=GeminiEmbeddingConfig(api_key="test-key", model="test-embedding-model"),
        sdk_client=SimpleNamespace(models=fake_models_api),
    )

    with pytest.raises(ValueError, match="must not be blank"):
        client.embed("   ")

    assert fake_models_api.received_kwargs is None


def test_gemini_embedding_client_embeds_many_texts_in_one_sdk_call() -> None:
    fake_models_api = FakeModelsApi(
        SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.1, 0.2]),
                SimpleNamespace(values=[0.3, 0.4]),
            ]
        )
    )
    client = GeminiEmbeddingClient(
        config=GeminiEmbeddingConfig(api_key="test-key", model="test-embedding-model"),
        sdk_client=SimpleNamespace(models=fake_models_api),
    )

    embeddings = client.embed_many(["First text", "Second text"])

    assert embeddings == [(0.1, 0.2), (0.3, 0.4)]
    assert fake_models_api.received_kwargs is not None
    contents = fake_models_api.received_kwargs["contents"]
    assert isinstance(contents, list)
    assert len(contents) == 2


def test_gemini_embedding_client_wraps_sdk_errors() -> None:
    sdk_error = gemini_errors.ServerError(503, {"message": "unavailable"}, None)

    class FailingModelsApi:
        def embed_content(self, **kwargs: object) -> object:
            _ = kwargs
            raise sdk_error

    client = GeminiEmbeddingClient(
        config=GeminiEmbeddingConfig(api_key="test-key", model="test-embedding-model"),
        sdk_client=SimpleNamespace(models=FailingModelsApi()),
    )

    with pytest.raises(EmbeddingRequestError, match="HTTP 503") as error_info:
        client.embed("Текст для поиска")

    assert error_info.value.__cause__ is sdk_error
