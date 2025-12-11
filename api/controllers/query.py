"""
Query endpoints.
"""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_rag_service
from api.models.requests import QueryRequest
from api.models.responses import QueryResponse
from api.services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_documents(
        request: QueryRequest,
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Query the RAG system with a question.
    Returns answer with source citations and cost metadata.
    """
    return rag_service.query(
        question=request.question,
        k=request.k,
        min_score=request.min_score
    )


@router.post("/stream")
async def query_documents_stream(
        request: QueryRequest,
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Query the RAG system with streaming response.
    Returns Server-Sent Events (SSE) stream with:
    - {"type": "sources", "documents": [...]} - Retrieved documents
    - {"type": "token", "text": "..."} - Text chunks as they arrive
    - {"type": "usage", ...} - Token usage and cost metadata
    - {"type": "done"} - Stream completed
    - {"type": "error", "message": "..."} - Error occurred
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """
        Generate Server-Sent Events from RAG stream.
        Format: data: <json>\n\n
        """
        try:
            # Stream from RAG service
            for event in rag_service.query_stream(
                    question=request.question,
                    k=request.k,
                    min_score=request.min_score
            ):
                # Convert event to SSE format
                sse_data = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                yield sse_data

        except Exception as e:
            # Handle errors that occur before streaming starts
            error_event = {
                "type": "error",
                "message": f"Errore del server: {str(e)}"
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
