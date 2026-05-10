"""LLM Provider Exceptions"""


class LLMException(Exception):
    """Base exception for all LLM provider errors"""
    pass


class AuthenticationException(LLMException):
    """Raised when API key is invalid or authentication fails"""
    pass


class RateLimitException(LLMException):
    """Raised when rate limit is exceeded"""
    pass


class ConfigurationException(LLMException):
    """Raised when provider configuration is invalid"""
    pass


class APIException(LLMException):
    """Raised when API returns an error"""
    pass


class ValidationException(LLMException):
    """Raised when response validation fails"""
    pass
