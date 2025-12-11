import logging
from pathlib import Path
from typing import List, Dict

from rag import DocumentLoader, DocumentChunker


class DocumentProcessor:
    """
    High-level orchestrator: combines loading + chunking.
    Use this as main entry point.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_text_threshold: int = 100):
        """
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            min_text_threshold: Min chars/page for valid extraction
        """
        self.logger = logging.getLogger(__name__)

        self.loader = DocumentLoader(min_text_threshold=min_text_threshold, use_pdfplumber_for_tables=True)
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_pdf(self, file_path: str) -> List[Dict]:
        """
        Complete pipeline: load PDF + chunk.

        Args:
            file_path: Path to PDF

        Returns:
            List of chunks with metadata
        """
        # Load document
        document = self.loader.load_pdf(file_path)

        self.logger.info(
            f"Document loaded: {document['metadata']['num_pages']} pages, "
            f"{len(document['text'])} chars, "
            f"method: {document['method']}"
        )

        # Chunk document
        chunks = self.chunker.chunk_document(document, add_page_numbers=True)

        self.logger.info(f"Document chunked into {len(chunks)} chunks")

        return chunks

    def process_financial_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Process financial PDF with proper table handling.
        Use this instead of process_pdf() for balance sheets.
        """
        from rag.financial_pdf_processor import FinancialPDFProcessor

        # Process PDF to readable text
        processor = FinancialPDFProcessor()
        processed_text = processor.process(pdf_path)

        # Create document structure for chunker
        document = {
            'text': processed_text,
            'metadata': {
                'source': Path(pdf_path).name,
                'type': 'financial_document'
            },
            'method': 'financial_processor'
        }

        # Now chunk the processed text using chunk_document
        chunks = self.chunker.chunk_document(document, add_page_numbers=False)

        self.logger.info(f"Processed financial PDF: {len(chunks)} chunks created")

        return chunks
