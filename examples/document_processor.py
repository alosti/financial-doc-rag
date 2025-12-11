from rag import DocumentProcessor


def test_document_processor(pdf_path: str, output_file: str = "chunks_output.txt"):
    """
    Test document processor on a real PDF and save results.

    Args:
        pdf_path: Path to PDF to test
        output_file: Where to save chunk analysis
    """
    print(f"\n{'=' * 80}")
    print(f"TESTING DOCUMENT PROCESSOR")
    print(f"{'=' * 80}\n")

    # Initialize processor
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        min_text_threshold=100
    )

    # Process PDF
    try:
        chunks = processor.process_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        avg_chunk_size = sum(len(c['content']) for c in chunks) / len(chunks)
        print(f"Average chunk size: {avg_chunk_size:.0f} characters")
        print(f"Smallest chunk: {min(len(c['content']) for c in chunks)} chars")
        print(f"Largest chunk: {max(len(c['content']) for c in chunks)} chars")

    # Write detailed output to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"DOCUMENT PROCESSING RESULTS\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Source: {chunks[0]['metadata']['source']}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Method used: {chunks[0]['metadata']['method']}\n\n")
        f.write(f"{'=' * 80}\n\n")

        for i, chunk in enumerate(chunks):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Page: {chunk['metadata'].get('page', 'N/A')}\n")
            f.write(f"Length: {len(chunk['content'])} chars\n")
            f.write(f"\nCONTENT:\n{'-' * 80}\n")
            f.write(chunk['content'])
            f.write(f"\n{'-' * 80}\n")

    print(f"\n✓ Detailed output saved to: {output_file}")

    # Print first 3 chunks as preview
    print(f"\n{'=' * 80}")
    print(f"FIRST 3 CHUNKS PREVIEW")
    print(f"{'=' * 80}\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- CHUNK {i} (Page {chunk['metadata'].get('page', 'N/A')}) ---")
        print(chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'])
        print(f"\n(Full length: {len(chunk['content'])} chars)")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <pdf_path>")
        print("\nExample:")
        print("  python document_processor.py annual_report_2023.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    test_document_processor(pdf_path)
