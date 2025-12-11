"""
Document management endpoints.
"""

from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import get_document_service, get_rag_service
from api.models.responses import (
    SuccessResponse,
    DocumentListResponse
)
from api.services.document_service import DocumentService
from api.services.rag_service import RAGService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=SuccessResponse)
async def upload_documents(
        files: List[UploadFile] = File(...),
        document_service: DocumentService = Depends(get_document_service)
):
    """
    Upload and index PDF documents.
    Accepts multiple PDF files and indexes them into the vector store.
    """
    data = await document_service.upload_and_index(files)

    return SuccessResponse(
        message=f"Successfully indexed {data['files_processed']} documents",
        data=data
    )


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    List all indexed documents with metadata.
    Returns document count and chunk information.
    """
    return rag_service.list_documents()


@router.delete("/clear", response_model=SuccessResponse)
async def clear_documents(
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Clear all indexed documents from the vector store.
    This operation cannot be undone.
    """
    data = rag_service.clear_documents()

    return SuccessResponse(
        message="All documents cleared from vector store",
        data=data
    )
