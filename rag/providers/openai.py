"""
OpenAI GPT provider.
"""

from typing import List, Generator

import tiktoken
from openai import OpenAI

from .base import BaseLLMProvider, Message, CompletionResponse


class OpenAIProvider(BaseLLMProvider):
    """
    Provider for OpenAI GPT API.
    """

    # Pricing per 1M tokens (update if pricing changes)
    PRICING = {
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
        "gpt-4": {"prompt": 30.00, "completion": 60.00},
        "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    }

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = OpenAI(api_key=api_key)

        # Tokenizer for accurate counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for new models
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def complete(
            self,
            messages: List[Message],
            temperature: float = 0.7,
            max_tokens: int = 1000,
            **kwargs
    ) -> CompletionResponse:
        """Generate completion using GPT."""

        # Convert Message objects to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **self.default_params,
            **kwargs
        )

        # Extract tokens and calculate cost
        tokens_prompt = response.usage.prompt_tokens
        tokens_completion = response.usage.completion_tokens
        tokens_total = response.usage.total_tokens
        cost = self.calculate_cost(tokens_prompt, tokens_completion)

        return CompletionResponse(
            content=response.choices[0].message.content,
            model=self.model,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_total=tokens_total,
            cost_usd=cost,
            finish_reason=response.choices[0].finish_reason
        )

    def stream(
            self,
            messages: List[Message],
            temperature: float = 0.7,
            max_tokens: int = 1000,
            **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate streaming completion.

        Note: This generator yields text chunks and stores the final CompletionResponse
        in the generator's completion_response attribute after exhausting the generator.
        """

        # Convert Message objects
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # API call with streaming
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **self.default_params,
            **kwargs
        )

        # Accumulate response
        full_content = ""
        finish_reason = None

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                yield content

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        # Calculate tokens (OpenAI doesn't provide in streaming)
        tokens_prompt = self.count_tokens("".join([msg.content for msg in messages]))
        tokens_completion = self.count_tokens(full_content)
        tokens_total = tokens_prompt + tokens_completion
        cost = self.calculate_cost(tokens_prompt, tokens_completion)

        # Store the completion response as an attribute of the provider
        self._last_completion_response = CompletionResponse(
            content=full_content,
            model=self.model,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_total=tokens_total,
            cost_usd=cost,
            finish_reason=finish_reason
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))

    def calculate_cost(self, tokens_prompt: int, tokens_completion: int) -> float:
        """Calculate cost in USD."""
        pricing = self.PRICING.get(self.model)
        if not pricing:
            # Fallback to gpt-4o-mini pricing if unknown model
            pricing = self.PRICING["gpt-4o-mini"]

        cost_prompt = (tokens_prompt / 1_000_000) * pricing["prompt"]
        cost_completion = (tokens_completion / 1_000_000) * pricing["completion"]

        return cost_prompt + cost_completion
