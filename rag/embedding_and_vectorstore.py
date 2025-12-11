# embedding_and_vectorstore.py
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag import DocumentProcessor


@dataclass
class SearchResult:
    """Search result from vector store"""
    chunk_text: str
    metadata: Dict
    score: float  # Cosine similarity (0-1, where 1 = identical)
    rank: int


class EmbeddingGenerator:
    """
    Generates embeddings using SentenceTransformers.
    Supports batch processing for efficiency.
    """

    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            device: str = "cpu",
            normalize: bool = True
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda' (if GPU available)
            normalize: If True, normalize vectors for cosine similarity
        """
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def generate(
            self,
            texts: List[str],
            batch_size: int = 32,
            show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed
            batch_size: Number of texts per batch (32 is a good default)
            show_progress: Show progress bar

        Returns:
            Numpy array of shape (len(texts), dimension)
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")

        # Model handles batching internally
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize  # Normalization for cosine similarity
        )

        self.logger.info(f"Embeddings generated: shape {embeddings.shape}")
        return embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        Faster than batch for single text.
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return embedding[0]  # Return 1D array


class VectorStore:
    """
    Manages FAISS vector store for semantic search.
    Uses IndexFlatIP (Inner Product) with normalized vectors = cosine similarity
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Embedding dimension (e.g. 384 for all-MiniLM-L6-v2)
        """
        self.logger = logging.getLogger(__name__)

        self.dimension = dimension

        # IndexFlatIP = Inner Product (with normalized vectors becomes cosine similarity)
        # Pros: 100% accurate, no quality loss
        # Cons: O(n) per query (but OK for <100k documents)
        self.index = faiss.IndexFlatIP(dimension)

        # FAISS doesn't store metadata, we manage it separately
        self.chunks = []  # List of original texts
        self.metadata = []  # List of metadata dicts

        self.logger.info(f"Vector store initialized (dimension={dimension})")

    def add_documents(
            self,
            chunks: List[str],
            embeddings: np.ndarray,
            metadata: List[Dict]
    ):
        """
        Add documents to vector store.

        Args:
            chunks: List of texts
            embeddings: Numpy array (n_chunks, dimension)
            metadata: List of metadata dicts for each chunk
        """
        if len(chunks) != len(embeddings) != len(metadata):
            raise ValueError("chunks, embeddings and metadata must have same length")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != {self.dimension}")

        # Ensure embeddings are float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store chunks and metadata
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

        self.logger.info(f"Added {len(chunks)} documents. Total in vector store: {self.index.ntotal}")

    def search(
            self,
            query_embedding: np.ndarray,
            k: int = 5
    ) -> List[SearchResult]:
        """
        Search for k most similar documents to query.

        Args:
            query_embedding: Query vector (1D array)
            k: Number of results to return

        Returns:
            List of SearchResult ordered by relevance
        """
        if self.index.ntotal == 0:
            self.logger.warning("Vector store is empty, no results")
            return []

        # FAISS requires 2D array (even for single query)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Limit k to available documents
        k = min(k, self.index.ntotal)

        # Search: distances are inner products (= cosine similarity with normalized vectors)
        distances, indices = self.index.search(query_embedding, k)

        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            results.append(SearchResult(
                chunk_text=self.chunks[idx],
                metadata=self.metadata[idx],
                score=float(score),  # Convert from numpy float to Python float
                rank=rank
            ))

        self.logger.info(f"Search completed: {len(results)} results (top score: {results[0].score:.4f})")
        return results

    def save(self, directory: str):
        """
        Save vector store to disk.

        Args:
            directory: Directory where to save files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = directory / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # Save chunks and metadata with pickle
        data_path = directory / "data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)

        self.logger.info(f"Vector store saved to {directory}")

    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        """
        Load vector store from disk.

        Args:
            directory: Directory to load files from

        Returns:
            VectorStore instance
        """
        directory = Path(directory)

        # Load data
        data_path = directory / "data.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Create instance
        store = cls(dimension=data['dimension'])

        # Load FAISS index
        index_path = directory / "faiss.index"
        store.index = faiss.read_index(str(index_path))

        # Restore chunks and metadata
        store.chunks = data['chunks']
        store.metadata = data['metadata']

        logger = logging.getLogger(__name__)
        logger.info(f"Vector store loaded from {directory} ({store.index.ntotal} documents)")
        return store


class RAGPipeline:
    """
    Complete pipeline: from PDF to semantic search.
    Integrates DocumentProcessor, EmbeddingGenerator and VectorStore.
    """

    def __init__(
            self,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ):
        """
        Args:
            embedding_model: Model name for embeddings
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.logger = logging.getLogger(__name__)

        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)

        self.vector_store = VectorStore(
            dimension=self.embedding_generator.dimension
        )

        self.logger.info("RAG Pipeline initialized")

    def index_document(self, pdf_path: str):
        """
        Process a PDF and add it to vector store.

        Args:
            pdf_path: Path to PDF file
        """
        self.logger.info(f"Indexing document: {pdf_path}")

        # Step 1: Extract chunks from PDF
        chunks = self.doc_processor.process_pdf(pdf_path)

        # Step 2: Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_generator.generate(texts)

        # Step 3: Extract metadata
        metadata = [chunk.metadata for chunk in chunks]

        # Step 4: Add to vector store
        self.vector_store.add_documents(texts, embeddings, metadata)

        self.logger.info(f"Document indexed: {len(chunks)} chunks added")

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Search for relevant documents for query.

        Args:
            query: User query
            k: Number of results

        Returns:
            List of SearchResult
        """
        self.logger.info(f"Searching for query: '{query}'")

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)

        # Search in vector store
        results = self.vector_store.search(query_embedding, k=k)

        return results

    def save(self, directory: str):
        """Save vector store"""
        self.vector_store.save(directory)

    @classmethod
    def load(cls, directory: str, embedding_model: str = None) -> 'RAGPipeline':
        """
        Load existing pipeline.

        Args:
            directory: Vector store directory
            embedding_model: Model name (must match the one used to create store)
        """
        # Load vector store
        vector_store = VectorStore.load(directory)

        # Create pipeline
        if embedding_model is None:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

        pipeline = cls(embedding_model=embedding_model)
        pipeline.vector_store = vector_store

        return pipeline
