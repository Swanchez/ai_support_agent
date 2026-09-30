class ConfigurationError(RuntimeError):
    """The application cannot start because required configuration is missing."""


class LlmRequestError(RuntimeError):
    """The configured LLM provider could not complete a request."""


class EmbeddingRequestError(RuntimeError):
    """The configured embedding provider could not create a vector."""


class InvalidModelResponseError(RuntimeError):
    """LLM returned data that does not match the application's contract."""


class OrderNotFoundError(RuntimeError):
    """An order is absent or does not belong to the authenticated user."""


class OrderServiceUnavailableError(RuntimeError):
    """The order backend could not complete a request safely."""


class OrderCancellationConflictError(RuntimeError):
    """An order cannot be cancelled in its current state or with this retry key."""


class InvalidAccessTokenError(RuntimeError):
    """A bearer token cannot establish a trusted application identity."""


class InvalidCredentialsError(RuntimeError):
    """A login attempt did not establish a trusted user identity."""


class AuthenticationServiceUnavailableError(RuntimeError):
    """The authentication backend could not safely complete a login attempt."""
