"""LLM providers with unified interface."""

"""
Provider factory for LLM APIs.
"""

from enum import Enum
from typing import Optional, Union

from .anthropic import AnthropicProvider
from .base import BaseLLMProvider, Message, CompletionResponse
from .openai import OpenAIProvider


class ProviderType(str, Enum):
    """
    Supported LLM providers.

    Inherits from str for JSON/YAML serialization.
    """
    ANTHROPIC = "anthropic"
    OPENAI = "openai"

    def __str__(self) -> str:
        return self.value


class ProviderFactory:
    """
    Factory for instantiating LLM providers.

    Usage:
        # With Enum (recommended)
        provider = ProviderFactory.create(
            ProviderType.ANTHROPIC,
            api_key="...",
            model="claude-3-5-sonnet-20241022"
        )

        # With string (from config file)
        provider = ProviderFactory.create(
            "anthropic",
            api_key="...",
            model="claude-3-5-sonnet-20241022"
        )
    """

    _PROVIDERS = {
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.OPENAI: OpenAIProvider,
    }

    @classmethod
    def create(
            cls,
            provider_name: Union[str, ProviderType],
            api_key: str,
            model: Optional[str] = None,
            **kwargs
    ) -> BaseLLMProvider:
        """
        Create LLM provider.

        Args:
            provider_name: ProviderType enum or string ("anthropic", "openai")
            api_key: Provider API key
            model: Model name (optional, uses provider default)
            **kwargs: Additional provider-specific parameters

        Returns:
            Instantiated provider

        Raises:
            ValueError: If provider is not supported
        """
        # Normalize string -> Enum
        if isinstance(provider_name, str):
            try:
                provider_name = ProviderType(provider_name.lower())
            except ValueError:
                supported = ", ".join([p.value for p in ProviderType])
                raise ValueError(
                    f"Provider '{provider_name}' not supported. "
                    f"Supported: {supported}"
                )

        # Get provider class
        if provider_name not in cls._PROVIDERS:
            supported = ", ".join([p.value for p in cls._PROVIDERS.keys()])
            raise ValueError(
                f"Provider '{provider_name}' not supported. "
                f"Supported: {supported}"
            )

        provider_class = cls._PROVIDERS[provider_name]

        # Instantiate
        if model:
            return provider_class(api_key=api_key, model=model, **kwargs)
        else:
            return provider_class(api_key=api_key, **kwargs)

    @classmethod
    def register(cls, provider_type: ProviderType, provider_class: type):
        """
        Register new custom provider.

        Args:
            provider_type: ProviderType enum
            provider_class: Class that implements BaseLLMProvider
        """
        cls._PROVIDERS[provider_type] = provider_class

    @classmethod
    def list_providers(cls) -> list[str]:
        """
        List supported providers.

        Returns:
            List of provider names (strings)
        """
        return [p.value for p in cls._PROVIDERS.keys()]


# Explicit export - public API only
__all__ = [
    # Base classes
    "BaseLLMProvider",
    "Message",
    "CompletionResponse",
    # Enums
    "ProviderType",
    # Concrete providers (per uso avanzato)
    "AnthropicProvider",
    "OpenAIProvider",
    # Factory (main entry point)
    "ProviderFactory",
]
