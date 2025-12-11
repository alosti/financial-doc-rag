"""
Document service - handles document upload and indexing.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import UploadFile

from api.exceptions import DocumentProcessingException
from api.services.rag_service import RAGService

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service for document operations.
    Handles file upload, validation, and indexing.
    """

    def __init__(self, rag_service: RAGService):
        """
        Initialize document service.

        Args:
            rag_service: RAG service instance
        """
        self.rag_service = rag_service

    async def upload_and_index(
            self,
            files: List[UploadFile]
    ) -> dict:
        """
        Upload and index PDF documents.

        Args:
            files: List of uploaded files

        Returns:
            Upload result with statistics

        Raises:
            DocumentProcessingException: If processing fails
        """
        if not files:
            raise DocumentProcessingException(
                "No files provided",
                "At least one PDF file is required"
            )

        # Validate files
        self._validate_files(files)

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Save files
            pdf_paths = await self._save_files(files, temp_dir)

            # Index documents
            logger.info(f"Indexing {len(pdf_paths)} documents...")
            initial_chunks = self.rag_service.total_chunks

            self.rag_service._rag_system.index_documents(pdf_paths)

            final_chunks = self.rag_service.total_chunks
            chunks_added = final_chunks - initial_chunks

            logger.info(f"Successfully indexed {len(files)} documents")

            return {
                "files_processed": len(files),
                "chunks_added": chunks_added,
                "total_chunks": final_chunks,
                "filenames": [f.filename for f in files]
            }

        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            raise DocumentProcessingException(
                "Failed to process documents",
                str(e)
            )

        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _validate_files(self, files: List[UploadFile]):
        """
        Validate uploaded files.

        Args:
            files: List of uploaded files

        Raises:
            DocumentProcessingException: If validation fails
        """
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise DocumentProcessingException(
                    f"Invalid file type: {file.filename}",
                    "Only PDF files are allowed"
                )

    async def _save_files(
            self,
            files: List[UploadFile],
            directory: str
    ) -> List[str]:
        """
        Save uploaded files to temporary directory.

        Args:
            files: List of uploaded files
            directory: Target directory

        Returns:
            List of file paths
        """
        pdf_paths = []

        for file in files:
            file_path = Path(directory) / file.filename
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)
            pdf_paths.append(str(file_path))
            logger.info(f"Saved temporary file: {file.filename}")

        return pdf_paths
