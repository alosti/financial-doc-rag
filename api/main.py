"""
FastAPI application entry point.
Clean architecture with controllers and services.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.controllers import health, system, documents, query
from api.exceptions import RAGException

# Logging setup
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api")
app.include_router(system.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "status": "operational"
    }


# Exception handlers
@app.exception_handler(RAGException)
async def handle_rag_exception(request: Request, exc: RAGException):
    """Handle custom RAG exceptions."""
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "detail": exc.detail
        }
    )


@app.exception_handler(Exception)
async def handle_general_exception(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Actions on startup."""
    logger.info(f"{settings.api_title} v{settings.api_version} starting up...")
    logger.info(f"Documentation available at /docs")
    yield
    """Actions on shutdown."""
    logger.info(f"{settings.api_title} shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
