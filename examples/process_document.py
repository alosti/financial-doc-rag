"""
Example script to process a PDF document.

Usage:
    python examples/process_document.py <pdf_path>

Example:
    python examples/process_document.py data/sample_docs/annual_report.pdf
"""

import sys
from pathlib import Path

from rag import DocumentLoader, DocumentChunker

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def process_document(pdf_path: str, output_file: str = "data/output/chunks.txt"):
    """
    Process a PDF document and save chunked output.

    Args:
        pdf_path: Path to input PDF
        output_file: Where to save results
    """
    print(f"\n{'=' * 80}")
    print(f"PROCESSING DOCUMENT")
    print(f"{'=' * 80}\n")
    print(f"Input: {pdf_path}")
    print(f"Output: {output_file}\n")

    # Initialize components
    loader = DocumentLoader(min_text_threshold=100)
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

    # Load PDF
    print("Loading PDF...")
    try:
        document = loader.load_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return

    print(f"✓ Loaded: {document['metadata']['num_pages']} pages")
    print(f"  Method: {document['method']}")
    print(f"  Total chars: {len(document['text']):,}")

    # Chunk document
    print("\nChunking document...")
    chunks = chunker.chunk_document(document, add_page_numbers=True)

    print(f"✓ Created {len(chunks)} chunks")

    if chunks:
        avg_size = sum(len(c['content']) for c in chunks) / len(chunks)
        print(f"  Average chunk size: {avg_size:.0f} chars")
        print(f"  Size range: {min(len(c['content']) for c in chunks)}-{max(len(c['content']) for c in chunks)} chars")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"DOCUMENT PROCESSING RESULTS\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Source: {document['metadata']['source']}\n")
        f.write(f"Pages: {document['metadata']['num_pages']}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Extraction method: {document['method']}\n\n")
        f.write(f"{'=' * 80}\n")

        for chunk in chunks:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"CHUNK {chunk['metadata']['chunk_id']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Page: {chunk['metadata'].get('page', 'N/A')}\n")
            f.write(f"Length: {len(chunk['content'])} chars\n")
            f.write(f"\nCONTENT:\n{'-' * 80}\n")
            f.write(chunk['content'])
            f.write(f"\n{'-' * 80}\n")

    print(f"\n✓ Results saved to: {output_path}")

    # Show preview
    print(f"\n{'=' * 80}")
    print(f"PREVIEW: First chunk")
    print(f"{'=' * 80}\n")
    first_chunk = chunks[0]
    print(f"Page: {first_chunk['metadata'].get('page', 'N/A')}")
    print(f"Length: {len(first_chunk['content'])} chars\n")
    preview = first_chunk['content'][:400]
    print(preview + "..." if len(first_chunk['content']) > 400 else preview)
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/process_document.py <pdf_path>")
        print("\nExample:")
        print("  python examples/process_document.py data/sample_docs/report.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_document(pdf_path)
