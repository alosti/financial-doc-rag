"""
RAG service - business logic for RAG operations.
"""

import logging
from typing import Optional, Generator, Dict

from api.exceptions import (
    SystemNotInitializedException,
    NoDocumentsException,
    InvalidProviderException
)
from api.models.responses import (
    QueryResponse,
    SourceInfo,
    SystemStatusResponse,
    DocumentInfo,
    DocumentListResponse
)
from rag.complete_rag_system import CompleteRAGSystem
from rag.providers import AnthropicProvider, OpenAIProvider, BaseLLMProvider

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service for RAG operations.
    Encapsulates all business logic related to RAG system.
    """

    def __init__(self):
        """Initialize RAG service."""
        self._rag_system: Optional[CompleteRAGSystem] = None
        self._provider_name: Optional[str] = None
        self._model_name: Optional[str] = None

    @property
    def is_initialized(self) -> bool:
        """Check if RAG system is initialized."""
        return self._rag_system is not None

    @property
    def total_chunks(self) -> int:
        """Get total number of indexed chunks."""
        if not self.is_initialized:
            return 0
        return self._rag_system.vector_store.index.ntotal

    def initialize(self, provider: str, api_key: str, model: str) -> dict:
        """
        Initialize RAG system with LLM provider.

        Args:
            provider: Provider name ('anthropic' or 'openai')
            api_key: API key
            model: Model name

        Returns:
            Initialization result

        Raises:
            InvalidProviderException: If provider not supported
        """
        logger.info(f"Initializing RAG system with {provider} - {model}")

        # Create provider
        llm_provider = self._create_provider(provider, api_key, model)

        # Initialize RAG system
        self._rag_system = CompleteRAGSystem(llm_provider=llm_provider)
        self._provider_name = provider
        self._model_name = model

        logger.info("RAG system initialized successfully")

        return {
            "provider": provider,
            "model": model
        }

    def _create_provider(
            self,
            provider: str,
            api_key: str,
            model: str
    ) -> BaseLLMProvider:
        """
        Create LLM provider instance.

        Args:
            provider: Provider name
            api_key: API key
            model: Model name

        Returns:
            Provider instance

        Raises:
            InvalidProviderException: If provider not supported
        """
        provider_lower = provider.lower()

        if provider_lower == "anthropic":
            return AnthropicProvider(api_key=api_key, model=model)
        elif provider_lower == "openai":
            return OpenAIProvider(api_key=api_key, model=model)
        else:
            raise InvalidProviderException(provider)

    def query(
            self,
            question: str,
            k: int = 5,
            min_score: float = 0.5
    ) -> QueryResponse:
        """
        Query the RAG system.

        Args:
            question: User question
            k: Number of chunks to retrieve
            min_score: Minimum similarity score

        Returns:
            Query response with answer and sources

        Raises:
            SystemNotInitializedException: If system not initialized
            NoDocumentsException: If no documents indexed
        """
        self._ensure_initialized()
        self._ensure_documents_exist()

        logger.info(f"Processing query: {question}")

        # Query RAG system
        response = self._rag_system.query(
            question=question,
            k=k,
            min_score=min_score,
            stream=False
        )

        # Convert to API response format
        sources = [
            SourceInfo(
                text=source.chunk_text,
                source=source.metadata.get("source", "unknown"),
                page=source.metadata.get("page", 0),
                score=source.score,
                rank=source.rank
            )
            for source in response.sources
        ]

        logger.info(
            f"Query completed: {response.tokens_used} tokens, "
            f"${response.cost_usd:.6f}"
        )

        return QueryResponse(
            answer=response.answer,
            sources=sources,
            tokens_used=response.tokens_used,
            cost_usd=response.cost_usd,
            model=response.model
        )

    def query_stream(
            self,
            question: str,
            k: int = 5,
            min_score: float = 0.5
    ) -> Generator[Dict, None, None]:
        """
        Query the RAG system with streaming response.

        Args:
            question: User question
            k: Number of chunks to retrieve
            min_score: Minimum similarity score

        Yields:
            Dictionary events from RAG system stream

        Raises:
            SystemNotInitializedException: If system not initialized
            NoDocumentsException: If no documents indexed
        """
        try:
            self._ensure_initialized()
            self._ensure_documents_exist()

            logger.info(f"Processing streaming query: {question}")

            # Stream from RAG system
            for event in self._rag_system.query_stream(
                question=question,
                k=k,
                min_score=min_score
            ):
                yield event

        except SystemNotInitializedException as e:
            logger.error("System not initialized")
            yield {
                "type": "error",
                "message": "Sistema non inizializzato. Usa /api/system/initialize prima di fare query."
            }
            yield {"type": "done"}
        except NoDocumentsException as e:
            logger.error("No documents indexed")
            yield {
                "type": "error",
                "message": "Nessun documento indicizzato. Carica dei documenti prima di fare query."
            }
            yield {"type": "done"}
        except Exception as e:
            logger.error(f"Error during streaming query: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "message": f"Errore durante la query: {str(e)}"
            }
            yield {"type": "done"}

    def get_status(self) -> SystemStatusResponse:
        """
        Get system status.

        Returns:
            System status information
        """
        if not self.is_initialized:
            return SystemStatusResponse(
                initialized=False,
                total_chunks=0,
                provider=None,
                model=None
            )

        return SystemStatusResponse(
            initialized=True,
            total_chunks=self.total_chunks,
            provider=self._provider_name,
            model=self._model_name
        )

    def list_documents(self) -> DocumentListResponse:
        """
        List all indexed documents.

        Returns:
            List of indexed documents with metadata

        Raises:
            SystemNotInitializedException: If system not initialized
        """
        self._ensure_initialized()

        if not self._rag_system.vector_store.metadata:
            return DocumentListResponse(
                total_chunks=0,
                documents=[]
            )

        # Extract unique documents
        documents_dict = {}
        for metadata in self._rag_system.vector_store.metadata:
            source = metadata.get("source", "unknown")
            if source not in documents_dict:
                documents_dict[source] = {
                    "filename": source,
                    "chunks": 0,
                    "pages": set()
                }
            documents_dict[source]["chunks"] += 1
            documents_dict[source]["pages"].add(metadata.get("page", 0))

        # Convert to response format
        documents = [
            DocumentInfo(
                filename=info["filename"],
                chunks=info["chunks"],
                pages=sorted(list(info["pages"]))
            )
            for info in documents_dict.values()
        ]

        return DocumentListResponse(
            total_chunks=self.total_chunks,
            documents=documents
        )

    def clear_documents(self) -> dict:
        """
        Clear all indexed documents.

        Returns:
            Clear operation result

        Raises:
            SystemNotInitializedException: If system not initialized
        """
        self._ensure_initialized()

        logger.info("Clearing all documents from vector store")

        # Reinitialize vector store
        provider = self._rag_system.rag_generator.llm_provider
        self._rag_system = CompleteRAGSystem(llm_provider=provider)

        logger.info("Documents cleared successfully")

        return {"chunks_cleared": self.total_chunks}

    def _ensure_initialized(self):
        """Ensure RAG system is initialized."""
        if not self.is_initialized:
            raise SystemNotInitializedException()

    def _ensure_documents_exist(self):
        """Ensure documents are indexed."""
        if self.total_chunks == 0:
            raise NoDocumentsException()
