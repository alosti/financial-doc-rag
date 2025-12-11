"""
Health check endpoints.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_rag_service
from api.models.responses import HealthResponse
from api.services.rag_service import RAGService

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check(
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Health check endpoint.
    Returns system health status.
    """
    return HealthResponse(
        status="healthy",
        system_initialized=rag_service.is_initialized
    )
