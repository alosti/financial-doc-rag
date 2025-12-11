"""
System management endpoints.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_rag_service
from api.models.requests import InitializeRequest
from api.models.responses import SuccessResponse, SystemStatusResponse
from api.services.rag_service import RAGService

router = APIRouter(prefix="/system", tags=["System"])


@router.post("/initialize", response_model=SuccessResponse)
async def initialize_system(
        request: InitializeRequest,
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Initialize RAG system with LLM provider configuration.
    Must be called before indexing documents or querying.
    """
    data = rag_service.initialize(
        provider=request.provider,
        api_key=request.api_key,
        model=request.model
    )

    return SuccessResponse(
        message="RAG system initialized successfully",
        data=data
    )


@router.get("/status", response_model=SystemStatusResponse)
async def get_status(
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get current system status.
    Returns initialization state and document count.
    """
    return rag_service.get_status()
