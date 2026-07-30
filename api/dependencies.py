"""
Dependency injection for FastAPI.
"""

from api.services.document_service import DocumentService
from api.services.rag_service import RAGService

# Global service instances (singleton pattern)
_rag_service: RAGService = None
_document_service: DocumentService = None


def get_rag_service() -> RAGService:
    """
    Get RAG service instance.
    Dependency injection for controllers.
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_document_service() -> DocumentService:
    """
    Get document service instance.
    Dependency injection for controllers.
    """
    global _document_service
    if _document_service is None:
        _document_service = DocumentService(get_rag_service())
    return _document_service
