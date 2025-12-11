"""
Request models.
"""

from typing import Literal

from pydantic import BaseModel, Field


class InitializeRequest(BaseModel):
    """Request to initialize RAG system."""
    provider: Literal["anthropic", "openai"] = Field(
        ...,
        description="LLM provider"
    )
    api_key: str = Field(..., description="API key for the provider")
    model: str = Field(..., description="Model name")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "anthropic",
                "api_key": "sk-ant-...",
                "model": "claude-sonnet-4-20250514"
            }
        }


class QueryRequest(BaseModel):
    """Request to query RAG system."""
    question: str = Field(..., description="User question", min_length=1)
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    min_score: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Qual è stata la crescita dei ricavi nel Q4 2023?",
                "k": 5,
                "min_score": 0.5
            }
        }
