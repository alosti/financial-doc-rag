"""
Text chunking module using LangChain RecursiveCharacterTextSplitter.

Maintains context with configurable overlap between chunks.
Tracks page numbers for citation purposes.
"""
import logging
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Intelligent text chunking with overlap for RAG systems.

    Uses LangChain's RecursiveCharacterTextSplitter which:
    - Tries to split on paragraphs first (\n\n)
    - Falls back to sentences (. )
    - Then words ( )
    - Finally characters if needed

    Maintains overlap between chunks to preserve context.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Target chunk size in characters (not strict)
            chunk_overlap: Overlap between consecutive chunks (typically 15-25% of chunk_size)
        """
        self.logger = logging.getLogger(__name__)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize LangChain splitter
        # Separators in order of preference
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraphs (highest preference)
                "\n",  # Lines
                ". ",  # Sentences
                " ",  # Words
                ""  # Characters (fallback)
            ]
        )

    def chunk_document(self,
                       document: Dict,
                       add_page_numbers: bool = True) -> List[Dict]:
        """
        Chunk document with metadata enrichment.

        Args:
            document: Output from DocumentLoader.load_pdf()
            add_page_numbers: If True, track which page each chunk came from

        Returns:
            List of chunks, each is:
            {
                'content': str,          # Chunk text
                'metadata': {
                    'source': str,       # Document filename
                    'page': int,         # Page number (if add_page_numbers=True)
                    'chunk_id': int,     # Sequential chunk ID
                    'total_chunks': int  # Total chunks in document
                }
            }
        """
        chunks = []

        if add_page_numbers and 'pages' in document:
            # Page-aware chunking: maintain page number info
            # This is critical for citations!
            for page_num, page_text in enumerate(document['pages'], start=1):
                if not page_text.strip():
                    continue  # Skip empty pages

                page_chunks = self.splitter.split_text(page_text)

                for chunk_text in page_chunks:
                    chunks.append({
                        'content': chunk_text,
                        'metadata': {
                            'source': document['metadata'].get('source', 'Unknown'),
                            'page': page_num,
                            'method': document.get('method', 'unknown')
                        }
                    })
        else:
            # Simple chunking on full text
            text_chunks = self.splitter.split_text(document['text'])

            for chunk_text in text_chunks:
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'source': document['metadata'].get('source', 'Unknown')
                    }
                })

        # Enrich with chunk IDs and total count
        for i, chunk in enumerate(chunks):
            chunk['metadata']['chunk_id'] = i
            chunk['metadata']['total_chunks'] = len(chunks)

        return chunks
