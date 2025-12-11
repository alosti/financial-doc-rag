"""
Custom exceptions for the API.
"""

from typing import Optional

from fastapi import HTTPException


class RAGException(Exception):
    """Base exception for RAG system."""

    def __init__(self, message: str, detail: Optional[str] = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class SystemNotInitializedException(RAGException):
    """Raised when RAG system is not initialized."""

    def __init__(self):
        super().__init__(
            message="RAG system not initialized",
            detail="Call /api/system/initialize first"
        )


class NoDocumentsException(RAGException):
    """Raised when no documents are indexed."""

    def __init__(self):
        super().__init__(
            message="No documents indexed",
            detail="Upload documents via /api/documents/upload first"
        )


class InvalidProviderException(RAGException):
    """Raised when invalid LLM provider specified."""

    def __init__(self, provider: str):
        super().__init__(
            message=f"Invalid provider: {provider}",
            detail="Supported providers: 'anthropic', 'openai'"
        )


class DocumentProcessingException(RAGException):
    """Raised when document processing fails."""
    pass


def rag_exception_handler(exc: RAGException) -> HTTPException:
    """Convert RAGException to HTTPException."""
    return HTTPException(
        status_code=400,
        detail={"error": exc.message, "detail": exc.detail}
    )
