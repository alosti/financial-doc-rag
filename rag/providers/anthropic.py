"""
Anthropic Claude provider.
"""

from typing import List, Generator

import anthropic
from .base import BaseLLMProvider, Message, CompletionResponse


class AnthropicProvider(BaseLLMProvider):
    """
    Provider for Anthropic Claude API.
    """

    # Pricing per 1M tokens (update if pricing changes)
    PRICING = {
        "claude-opus-4-20250514": {"prompt": 15.00, "completion": 75.00},
        "claude-sonnet-4-20250514": {"prompt": 3.00, "completion": 15.00},
        "claude-sonnet-3-5-20241022": {"prompt": 3.00, "completion": 15.00},
        "claude-haiku-3-5-20241022": {"prompt": 0.80, "completion": 4.00},
    }

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(
            self,
            messages: List[Message],
            temperature: float = 0.7,
            max_tokens: int = 1000,
            **kwargs
    ) -> CompletionResponse:
        """Generate completion using Claude."""

        # Separate system message from conversation
        system_msg = None
        conv_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                conv_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # API call
        call_kwargs = {
            "model": self.model,
            "messages": conv_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self.default_params,
            **kwargs
        }

        if system_msg:
            call_kwargs["system"] = system_msg

        response = self.client.messages.create(**call_kwargs)

        # Extract tokens and calculate cost
        tokens_prompt = response.usage.input_tokens
        tokens_completion = response.usage.output_tokens
        tokens_total = tokens_prompt + tokens_completion
        cost = self.calculate_cost(tokens_prompt, tokens_completion)

        return CompletionResponse(
            content=response.content[0].text,
            model=self.model,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_total=tokens_total,
            cost_usd=cost,
            finish_reason=response.stop_reason
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

        # Separate system message
        system_msg = None
        conv_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                conv_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # API call with streaming
        call_kwargs = {
            "model": self.model,
            "messages": conv_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self.default_params,
            **kwargs
        }

        if system_msg:
            call_kwargs["system"] = system_msg

        # Accumulate response for final CompletionResponse
        full_content = ""
        tokens_prompt = 0
        tokens_completion = 0
        finish_reason = None

        with self.client.messages.stream(**call_kwargs) as stream:
            for text in stream.text_stream:
                full_content += text
                yield text

            # Get final message for token counts
            final_message = stream.get_final_message()
            tokens_prompt = final_message.usage.input_tokens
            tokens_completion = final_message.usage.output_tokens
            finish_reason = final_message.stop_reason

        tokens_total = tokens_prompt + tokens_completion
        cost = self.calculate_cost(tokens_prompt, tokens_completion)

        # Store the completion response as an attribute of the generator function
        # This will be accessible after the generator is exhausted
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
        """
        Count tokens using Claude's token counting.
        Approximation: ~4 chars per token for English, ~2-3 for Italian.
        """
        # Claude doesn't provide official tokenizer
        # Use approximation: 4 chars per token (conservative for multilingual)
        return len(text) // 4

    def calculate_cost(self, tokens_prompt: int, tokens_completion: int) -> float:
        """Calculate cost in USD."""
        pricing = self.PRICING.get(self.model)
        if not pricing:
            # Fallback to Sonnet pricing if unknown model
            pricing = self.PRICING["claude-sonnet-4-20250514"]

        cost_prompt = (tokens_prompt / 1_000_000) * pricing["prompt"]
        cost_completion = (tokens_completion / 1_000_000) * pricing["completion"]

        return cost_prompt + cost_completion
