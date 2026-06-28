"""LLM Provider Exceptions"""


class LLMException(Exception):
    """Base exception for all LLM provider errors"""


class AuthenticationException(LLMException):
    """Raised when API key is invalid or authentication fails"""


class RateLimitException(LLMException):
    """Raised when rate limit is exceeded"""


class ConfigurationException(LLMException):
    """Raised when provider configuration is invalid"""


class APIException(LLMException):
    """Raised when API returns an error"""


class ValidationException(LLMException):
    """Raised when response validation fails"""
