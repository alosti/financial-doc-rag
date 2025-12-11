"""
Financial Document RAG System

Main components:
- DocumentLoader: PDF loading with fallback strategies
- DocumentChunker: Intelligent text splitting with overlap
- (More components coming in next steps)
"""

__version__ = "0.1.0"

import logging

from rag.chunker import DocumentChunker
from rag.document_loader import DocumentLoader
from rag.processor import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    "DocumentLoader",
    "DocumentChunker",
    "DocumentProcessor",
]
