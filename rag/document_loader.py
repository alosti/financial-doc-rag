"""
Document loading module with automatic fallback strategies.

Handles PDF documents using:
1. pypdf (fast, simple)
2. pdfplumber fallback (better table extraction)
"""
import logging
from pathlib import Path
from typing import Dict, List

import pdfplumber
import pypdf


class DocumentLoader:
    """
    Loads PDF documents with automatic fallback strategy.

    Strategy:
    1. Try pypdf (fast, simple)
    2. If extraction quality is poor, fallback to pdfplumber
    3. Extract metadata (title, author, pages)
    4. Handle tables (convert to markdown when using pdfplumber)
    """

    def __init__(self, min_text_threshold: int = 100, use_pdfplumber_for_tables: bool = True):
        """
        Args:
            min_text_threshold: Minimum characters per page to consider
                              extraction valid. If below threshold, use fallback.
        """
        self.logger = logging.getLogger(__name__)
        self.min_text_threshold = min_text_threshold
        self.use_pdfplumber_for_tables = use_pdfplumber_for_tables

    def load_pdf(self, file_path: str) -> Dict[str, any]:
        """
        Load PDF and return dict with text and metadata.

        Args:
            file_path: Path to PDF file

        Returns:
            {
                'text': str,              # Full text content
                'pages': List[str],       # Text per page
                'metadata': dict,         # Title, author, num_pages, etc.
                'method': str             # 'pypdf' or 'pdfplumber'
            }

        Raises:
            FileNotFoundError: If PDF doesn't exist
            Exception: If both extraction methods fail
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        self.logger.info(f"Loading PDF: {file_path.name}")

        # If we are required to use pdfplumber
        if self.use_pdfplumber_for_tables:
            try:
                result = self._load_with_pdfplumber(str(file_path))
                self.logger.info("✓ pdfplumber extraction successful")
                return result
            except Exception as e:
                self.logger.error(f"All extraction methods failed: {e}")
                raise

        # Attempt 1: pypdf (fast)
        try:
            result = self._load_with_pypdf(str(file_path))

            # Validate extraction quality
            avg_chars_per_page = len(result['text']) / max(len(result['pages']), 1)

            if avg_chars_per_page >= self.min_text_threshold:
                self.logger.info(
                    f"✓ pypdf extraction successful "
                    f"({avg_chars_per_page:.0f} chars/page)"
                )
                return result
            else:
                self.logger.warning(
                    f"pypdf extraction poor quality "
                    f"({avg_chars_per_page:.0f} chars/page), "
                    f"trying pdfplumber..."
                )
        except Exception as e:
            self.logger.warning(f"pypdf failed: {e}, trying pdfplumber...")

        # Attempt 2: pdfplumber fallback
        try:
            result = self._load_with_pdfplumber(str(file_path))
            self.logger.info("✓ pdfplumber extraction successful")
            return result
        except Exception as e:
            self.logger.error(f"All extraction methods failed: {e}")
            raise

    def _load_with_pypdf(self, file_path: str) -> Dict:
        """Extract text using pypdf (fast, simple)"""
        reader = pypdf.PdfReader(file_path)

        # Extract metadata
        metadata = {
            'author': reader.metadata.get('/Author', 'Unknown') if reader.metadata else 'Unknown',
            'title': reader.metadata.get('/Title', Path(file_path).stem) if reader.metadata else Path(file_path).stem,
            'creation_date': reader.metadata.get('/CreationDate', 'Unknown') if reader.metadata else 'Unknown',
            'num_pages': len(reader.pages),
            'source': Path(file_path).name
        }

        # Extract text page by page
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                pages_text.append(text if text else "")
            except Exception as e:
                self.logger.warning(f"Error extracting page {i + 1}: {e}")
                pages_text.append("")  # Empty page on error

        return {
            'text': '\n\n'.join(pages_text),
            'pages': pages_text,
            'metadata': metadata,
            'method': 'pypdf'
        }

    def _load_with_pdfplumber(self, file_path: str) -> Dict:
        """Extract text using pdfplumber (handles tables better)"""
        pages_text = []
        metadata = {
            'title': Path(file_path).stem,
            'source': Path(file_path).name,
            'num_pages': 0
        }

        with pdfplumber.open(file_path) as pdf:
            metadata['num_pages'] = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()

                    # BONUS: Extract tables and convert to markdown
                    # This is crucial for financial documents!
                    tables = page.extract_tables()
                    if tables:
                        text += "\n\n[TABLES FOUND]\n"
                        for table in tables:
                            text += self._table_to_markdown(table)

                    pages_text.append(text if text else "")

                except Exception as e:
                    self.logger.warning(f"Error extracting page {i + 1} with pdfplumber: {e}")
                    pages_text.append("")

        return {
            'text': '\n\n'.join(pages_text),
            'pages': pages_text,
            'metadata': metadata,
            'method': 'pdfplumber'
        }

    @staticmethod
    def _table_to_markdown(table: List[List[str]]) -> str:
        """
        Convert extracted table to markdown format.
        This helps LLM understand table structure better.

        Args:
            table: List of rows, each row is list of cells

        Returns:
            Markdown formatted table string
        """
        if not table or len(table) == 0:
            return ""

        # Clean None values
        table = [[str(cell) if cell is not None else "" for cell in row] for row in table]

        # Header row
        header = " | ".join(table[0])
        separator = " | ".join(["---"] * len(table[0]))

        # Data rows
        rows = []
        for row in table[1:]:
            rows.append(" | ".join(row))

        return f"\n{header}\n{separator}\n" + "\n".join(rows) + "\n"
