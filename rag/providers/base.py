"""
Abstract LLM provider interface.
Unified interface for all LLM APIs (Anthropic, OpenAI, etc).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Generator


@dataclass
class Message:
    """Message in conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class CompletionResponse:
    """Response from LLM API."""
    content: str
    model: str
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    cost_usd: float
    finish_reason: str  # "stop", "length", "content_filter"

    @property
    def tokens(self) -> Dict[str, int]:
        """Token breakdown for tracking."""
        return {
            "prompt": self.tokens_prompt,
            "completion": self.tokens_completion,
            "total": self.tokens_total
        }


class BaseLLMProvider(ABC):
    """
    Abstract provider for LLM APIs.
    All providers (Anthropic, OpenAI, etc) implement this interface.
    """

    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Args:
            api_key: Provider API key
            model: Model name to use
            **kwargs: Provider-specific parameters
        """
        self.api_key = api_key
        self.model = model
        self.default_params = kwargs

    @abstractmethod
    def complete(
            self,
            messages: List[Message],
            temperature: float = 0.7,
            max_tokens: int = 1000,
            **kwargs
    ) -> CompletionResponse:
        """
        Generate synchronous completion.

        Args:
            messages: List of conversation messages
            temperature: Creativity (0.0-1.0)
            max_tokens: Max response length
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with content and metadata
        """
        pass

    @abstractmethod
    def stream(
            self,
            messages: List[Message],
            temperature: float = 0.7,
            max_tokens: int = 1000,
            **kwargs
    ) -> Generator[str, None, CompletionResponse]:
        """
        Generate streaming completion.

        Yields:
            Token chunks as they arrive

        Returns:
            Final CompletionResponse after streaming complete
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Approximate number of tokens
        """
        pass

    @abstractmethod
    def calculate_cost(self, tokens_prompt: int, tokens_completion: int) -> float:
        """
        Calculate cost in USD for API call.

        Args:
            tokens_prompt: Input tokens
            tokens_completion: Output tokens

        Returns:
            Cost in USD
        """
        pass
