from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as gemini_errors
from google.genai._gaos.lib.compat_errors import NotFoundError
from openai import APIConnectionError

from ai_support_agent.config import GeminiConfig, OpenAiConfig
from ai_support_agent.exceptions import LlmRequestError
from ai_support_agent.llm_client import (
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
    GeminiLlmClient,
    OpenAiLlmClient,
    gemini_generation_config,
    gemini_response_format,
    openai_sdk_options,
)
from ai_support_agent.schemas import support_response_openai_text_format
from ai_support_agent.service import build_llm_call


class FakeResponsesApi:
    def __init__(self, response: object) -> None:
        self.response = response
        self.received_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.received_kwargs = kwargs
        return self.response


class FakeGeminiInteractionsApi:
    def __init__(self, response: object) -> None:
        self.response = response
        self.received_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.received_kwargs = kwargs
        return self.response


def test_openai_sdk_options_are_conservative() -> None:
    assert openai_sdk_options("test-key") == {
        "api_key": "test-key",
        "timeout": OPENAI_TIMEOUT_SECONDS,
        "max_retries": OPENAI_MAX_RETRIES,
    }
    assert OPENAI_TIMEOUT_SECONDS == 20.0
    assert OPENAI_MAX_RETRIES == 0


def test_openai_client_translates_our_contract_to_responses_api() -> None:
    fake_response = SimpleNamespace(
        output_text='{"status": "answered"}',
        model="test-model",
        usage=SimpleNamespace(input_tokens=80, output_tokens=30, total_tokens=110),
    )
    fake_responses_api = FakeResponsesApi(fake_response)
    fake_sdk_client = SimpleNamespace(responses=fake_responses_api)
    config = OpenAiConfig(api_key="test-key", model="test-model")
    client = OpenAiLlmClient(config=config, sdk_client=fake_sdk_client)

    call = build_llm_call("Когда придут деньги за возврат?")
    result = client.complete(call)

    assert result.text == '{"status": "answered"}'
    assert result.model == "test-model"
    assert result.input_tokens == 80
    assert result.output_tokens == 30
    assert result.total_tokens == 110
    assert fake_responses_api.received_kwargs == {
        "model": "test-model",
        "instructions": call.messages[0]["content"],
        "input": call.messages[1:],
        "temperature": call.temperature,
        "max_output_tokens": call.max_output_tokens,
        "store": False,
        "text": support_response_openai_text_format(),
    }


def test_openai_client_wraps_expected_sdk_errors() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    sdk_error = APIConnectionError(request=request)

    class FailingResponsesApi:
        def create(self, **kwargs: object) -> object:
            _ = kwargs
            raise sdk_error

    fake_sdk_client = SimpleNamespace(responses=FailingResponsesApi())
    config = OpenAiConfig(api_key="test-key", model="test-model")
    client = OpenAiLlmClient(config=config, sdk_client=fake_sdk_client)

    with pytest.raises(LlmRequestError) as error_info:
        client.complete(build_llm_call("Когда придут деньги за возврат?"))

    assert error_info.value.__cause__ is sdk_error


def test_gemini_client_translates_our_contract_to_interactions_api() -> None:
    fake_response = SimpleNamespace(
        output_text='{"status": "answered"}',
        usage=SimpleNamespace(
            total_input_tokens=80,
            total_output_tokens=30,
            total_tokens=110,
        ),
    )
    fake_interactions_api = FakeGeminiInteractionsApi(fake_response)
    fake_sdk_client = SimpleNamespace(interactions=fake_interactions_api)
    config = GeminiConfig(api_key="test-key", model="test-model")
    client = GeminiLlmClient(config=config, sdk_client=fake_sdk_client)
    call = build_llm_call("Когда придут деньги за возврат?")

    result = client.complete(call)

    assert result.text == '{"status": "answered"}'
    assert result.model == "test-model"
    assert result.input_tokens == 80
    assert result.output_tokens == 30
    assert result.total_tokens == 110
    assert fake_interactions_api.received_kwargs == {
        "model": "test-model",
        "system_instruction": call.messages[0]["content"],
        "input": "\n\n".join(message["content"] for message in call.messages[1:]),
        "generation_config": gemini_generation_config(call),
        "response_format": gemini_response_format(),
        "store": False,
    }


def test_gemini_client_wraps_expected_sdk_errors() -> None:
    sdk_error = gemini_errors.ServerError(503, {"message": "unavailable"}, None)

    class FailingInteractionsApi:
        def create(self, **kwargs: object) -> object:
            _ = kwargs
            raise sdk_error

    fake_sdk_client = SimpleNamespace(interactions=FailingInteractionsApi())
    config = GeminiConfig(api_key="test-key", model="test-model")
    client = GeminiLlmClient(config=config, sdk_client=fake_sdk_client)

    with pytest.raises(LlmRequestError) as error_info:
        client.complete(build_llm_call("Когда придут деньги за возврат?"))

    assert error_info.value.__cause__ is sdk_error


def test_gemini_client_wraps_interactions_sdk_errors() -> None:
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/interactions")
    response = httpx.Response(404, request=request)
    sdk_error = NotFoundError("model not found", response=response, body={})

    class FailingInteractionsApi:
        def create(self, **kwargs: object) -> object:
            _ = kwargs
            raise sdk_error

    fake_sdk_client = SimpleNamespace(interactions=FailingInteractionsApi())
    config = GeminiConfig(api_key="test-key", model="test-model")
    client = GeminiLlmClient(config=config, sdk_client=fake_sdk_client)

    with pytest.raises(LlmRequestError, match="HTTP 404") as error_info:
        client.complete(build_llm_call("Когда придут деньги за возврат?"))

    assert error_info.value.__cause__ is sdk_error
