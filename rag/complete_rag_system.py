"""
Complete RAG system: document processing + embedding + retrieval + generation.
End-to-end pipeline.
"""

import logging
from typing import List, Generator, Dict

from . import DocumentProcessor
from .embedding_and_vectorstore import EmbeddingGenerator, VectorStore
from .providers import BaseLLMProvider
from .rag_generator import RAGGenerator, RAGResponse


class CompleteRAGSystem:
    """
    Complete end-to-end RAG system.
    Handles everything from PDF to final answer.
    """

    def __init__(
            self,
            llm_provider: BaseLLMProvider,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ):
        """
        Args:
            llm_provider: LLM provider for answer generation
            embedding_model: Model for embeddings
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.logger = logging.getLogger(__name__)

        # Document processing
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Embeddings
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)

        # Vector store
        self.vector_store = VectorStore(
            dimension=self.embedding_generator.dimension
        )

        # RAG generator
        self.rag_generator = RAGGenerator(llm_provider=llm_provider)

        self.logger.info("Complete RAG System initialized")

    def index_documents(self, pdf_paths: List[str]):
        self.index_financial_document(pdf_paths)

    def index_text_documents(self, pdf_paths: List[str]):
        """
        Index multiple PDF documents.

        Args:
            pdf_paths: List of paths to PDF files
        """
        self.logger.info(f"Indexing {len(pdf_paths)} documents...")

        for pdf_path in pdf_paths:
            self.logger.info(f"Processing: {pdf_path}")

            # Step 1: Extract chunks
            chunks = self.doc_processor.process_pdf(pdf_path)

            # Step 2: Generate embeddings
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_generator.generate(texts, show_progress=False)

            # Step 3: Extract metadata
            metadata = [chunk['metadata'] for chunk in chunks]

            # Step 4: Add to vector store
            self.vector_store.add_documents(texts, embeddings, metadata)

            self.logger.info(f"Indexed {len(chunks)} chunks from {pdf_path}")

        self.logger.info(f"Total documents in vector store: {self.vector_store.index.ntotal}")

    def index_financial_document(self, pdf_paths: List[str]):
        """
        Index a financial document (balance sheet, income statement).
        Uses specialized processor for better table handling.
        """

        self.logger.info(f"Indexing {len(pdf_paths)} documents...")
        for pdf_path in pdf_paths:
            self.logger.info(f"Processing: {pdf_path}")

            # Step 1: Process with financial processor
            chunks = self.doc_processor.process_financial_pdf(pdf_path)

            # Step 2: Generate embeddings
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_generator.generate(texts, show_progress=False)

            # Step 3: Extract metadata
            metadata = [chunk['metadata'] for chunk in chunks]

            # Step 4: Add to vector store
            self.vector_store.add_documents(texts, embeddings, metadata)

            self.logger.info(f"Financial document indexed: {len(chunks)} chunks added")

        self.logger.info(f"Total documents in vector store: {self.vector_store.index.ntotal}")

    def query(
            self,
            question: str,
            k: int = 5,
            min_score: float = 0.5,
            stream: bool = False
    ) -> RAGResponse:
        """
        Query the system and get answer with sources.

        Args:
            question: User question
            k: Number of chunks to retrieve
            min_score: Minimum similarity score
            stream: If True, stream response

        Returns:
            RAGResponse with answer and sources
        """
        self.logger.info(f"Processing query: '{question}'")

        # Step 1: Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(question)

        # Step 2: Search vector store
        search_results = self.vector_store.search(query_embedding, k=k)

        # Step 3: Generate answer with LLM
        response = self.rag_generator.generate_with_threshold(
            query=question,
            search_results=search_results,
            min_score=min_score,
            stream=stream
        )

        return response

    def query_stream(
            self,
            question: str,
            k: int = 5,
            min_score: float = 0.5
    ) -> Generator[Dict, None, None]:
        """
        Query the system with streaming response.
        Yields streaming events as dictionaries.

        Args:
            question: User question
            k: Number of chunks to retrieve
            min_score: Minimum similarity score

        Yields:
            Dictionaries with streaming events:
            - {"type": "sources", "documents": [...]}  # Retrieved documents
            - {"type": "token", "text": "..."}  # Text chunks
            - {"type": "usage", "tokens": ..., "cost": ...}  # Usage metadata
            - {"type": "done"}  # Stream completed
        """
        try:
            self.logger.info(f"Processing streaming query: '{question}'")

            # Step 1: Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(question)

            # Step 2: Search vector store
            search_results = self.vector_store.search(query_embedding, k=k)

            # Filter by score
            filtered_results = [
                r for r in search_results
                if r.score >= min_score
            ]

            # Yield sources first
            sources = [
                {
                    "text": result.chunk_text,
                    "source": result.metadata.get("source", "unknown"),
                    "page": result.metadata.get("page", 0),
                    "score": float(result.score),
                    "rank": result.rank
                }
                for result in filtered_results
            ]

            yield {
                "type": "sources",
                "documents": sources
            }

            if not filtered_results:
                self.logger.warning(
                    f"No results above threshold {min_score}. "
                    f"Best score was {search_results[0].score:.4f}"
                )
                yield {
                    "type": "token",
                    "text": f"Non ho trovato informazioni sufficientemente rilevanti per rispondere a questa domanda (soglia di similarità: {min_score})."
                }
                yield {"type": "done"}
                return

            self.logger.info(
                f"Using {len(filtered_results)}/{len(search_results)} results "
                f"above threshold {min_score}"
            )

            # Step 3: Stream answer from LLM
            stream_generator = self.rag_generator.generate_stream(
                query=question,
                search_results=filtered_results
            )

            # Propagate tokens
            for event in stream_generator:
                yield event

            # Get final usage metadata
            final_response = self.rag_generator.llm_provider._last_completion_response

            # Yield usage metadata
            yield {
                "type": "usage",
                "tokens": final_response.tokens_total,
                "tokens_prompt": final_response.tokens_prompt,
                "tokens_completion": final_response.tokens_completion,
                "cost_usd": final_response.cost_usd,
                "model": final_response.model
            }

            self.logger.info(
                f"Stream completed: {final_response.tokens_total} tokens, "
                f"${final_response.cost_usd:.6f}"
            )

            # Signal completion
            yield {"type": "done"}

        except Exception as e:
            self.logger.error(f"Error during streaming query: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "message": f"Errore durante la query: {str(e)}"
            }
            yield {"type": "done"}

    def save(self, directory: str):
        """Save vector store to disk."""
        self.vector_store.save(directory)
        self.logger.info(f"RAG System saved to {directory}")

    @classmethod
    def load(
            cls,
            directory: str,
            llm_provider: BaseLLMProvider,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> 'CompleteRAGSystem':
        """
        Load existing RAG system.

        Args:
            directory: Directory with saved vector store
            llm_provider: LLM provider for generation
            embedding_model: Embedding model (must match saved)

        Returns:
            CompleteRAGSystem instance
        """
        system = cls(llm_provider=llm_provider, embedding_model=embedding_model)
        system.vector_store = VectorStore.load(directory)
        logger = logging.getLogger(__name__)
        logger.info(f"RAG System loaded from {directory}")
        return system
