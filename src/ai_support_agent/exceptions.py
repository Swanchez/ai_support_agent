class ConfigurationError(RuntimeError):
    """The application cannot start because required configuration is missing."""


class LlmRequestError(RuntimeError):
    """The configured LLM provider could not complete a request."""


class EmbeddingRequestError(RuntimeError):
    """The configured embedding provider could not create a vector."""


class InvalidModelResponseError(RuntimeError):
    """LLM returned data that does not match the application's contract."""
