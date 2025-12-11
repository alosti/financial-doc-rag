"""
Response models.
"""

from typing import List, Optional

from pydantic import BaseModel


class SourceInfo(BaseModel):
    """Information about a source chunk."""
    text: str
    source: str
    page: int
    score: float
    rank: int


class QueryResponse(BaseModel):
    """Response from query endpoint."""
    answer: str
    sources: List[SourceInfo]
    tokens_used: int
    cost_usd: float
    model: str


class SystemStatusResponse(BaseModel):
    """System status response."""
    initialized: bool
    total_chunks: int
    provider: Optional[str] = None
    model: Optional[str] = None


class SuccessResponse(BaseModel):
    """Generic success response."""
    status: str = "success"
    message: str
    data: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    system_initialized: bool


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    filename: str
    chunks: int
    pages: List[int]


class DocumentListResponse(BaseModel):
    """List of indexed documents."""
    total_chunks: int
    documents: List[DocumentInfo]
