"""
Financial PDF processor - handles balance sheets properly.
Extracts tables in readable format, keeps narrative text.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber


@dataclass
class ProcessedSection:
    """Section of the processed document."""
    text: str
    page: int
    section_type: str  # "text", "financial_table", "other_table"


class FinancialPDFProcessor:
    """
    Process financial PDFs with proper table handling.
    Replaces financial tables with readable format, keeps everything else.
    """

    def __init__(self):
        self.financial_keywords = [
            "stato patrimoniale",
            "conto economico",
            "rendiconto finanziario",
            "cash flow"
        ]

    def process(self, pdf_path: str) -> str:
        """
        Process entire PDF.
        Returns complete document with tables in readable format.
        """
        sections = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""

                # Check if this page has financial tables
                has_financial = self._has_financial_tables(page_text)

                if has_financial:
                    # Extract tables in readable format
                    section = self._process_financial_page(page, page_num, page_text)
                    sections.append(section)
                else:
                    # Keep text as-is
                    if page_text.strip():
                        sections.append(ProcessedSection(
                            text=page_text,
                            page=page_num,
                            section_type="text"
                        ))

        # Combine all sections
        output = []
        for section in sections:
            output.append(f"[PAGINA {section.page}]")
            output.append(section.text)
            output.append("")  # Empty line between sections

        return "\n".join(output)

    def _has_financial_tables(self, text: str) -> bool:
        """Check if page contains financial tables."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.financial_keywords)

    def _process_financial_page(self, page, page_num: int, page_text: str) -> ProcessedSection:
        """
        Process page with financial tables.
        Extracts tables and converts to readable format.
        """
        parts = []

        # Determine section type
        text_lower = page_text.lower()
        if "stato patrimoniale" in text_lower:
            section_name = "STATO PATRIMONIALE"
        elif "conto economico" in text_lower:
            section_name = "CONTO ECONOMICO"
        elif "rendiconto finanziario" in text_lower or "cash flow" in text_lower:
            section_name = "RENDICONTO FINANZIARIO"
        else:
            section_name = "TABELLA FINANZIARIA"

        parts.append("=" * 70)
        parts.append(section_name)
        parts.append("=" * 70)
        parts.append("")

        # Extract tables with proper settings
        tables = page.extract_tables({
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "intersection_tolerance": 3,
        })

        if tables:
            for i, table in enumerate(tables):
                if self._is_valid_table(table):
                    readable = self._table_to_readable_format(table)
                    if readable:
                        parts.append(readable)
                        parts.append("")
        else:
            # Fallback: if no tables extracted, use raw text
            parts.append(page_text)

        return ProcessedSection(
            text="\n".join(parts),
            page=page_num,
            section_type="financial_table"
        )

    def _is_valid_table(self, table: List[List]) -> bool:
        """Check if table is valid (not empty, has structure)."""
        if not table or len(table) < 2:
            return False

        # Check if has data
        has_data = False
        for row in table[1:]:
            if row and any(cell for cell in row):
                has_data = True
                break

        return has_data

    def _table_to_readable_format(self, table: List[List]) -> str:
        """
        Convert financial table to readable format.
        Each row becomes a natural language line.
        """
        lines = []

        # Get headers (first row)
        headers = [str(cell or "").strip() for cell in table[0]]

        # Skip if no headers
        if not any(headers):
            return ""

        # Process data rows
        for row in table[1:]:
            if not row:
                continue

            # First column is the label/description
            label = str(row[0] or "").strip()

            # Skip empty labels or separator rows
            if not label or len(label) < 2 or label.startswith("---"):
                continue

            # Build readable line: "Label | Header1: Value1 | Header2: Value2"
            parts = [label]

            for i, value in enumerate(row[1:], 1):
                if not value:
                    continue

                val_str = str(value).strip()
                if not val_str:
                    continue

                # Get header for this column
                if i < len(headers):
                    header = headers[i]
                else:
                    header = f"Col{i}"

                # Clean header
                header = header.replace("\n", " ").strip()

                # Format value
                if header:
                    parts.append(f"{header}: {val_str}")
                else:
                    parts.append(val_str)

            # Only add if we have values
            if len(parts) > 1:
                lines.append(" | ".join(parts))

        return "\n".join(lines)

    def process_and_save(self, pdf_path: str, output_path: str = None) -> str:
        """
        Process PDF and save to text file.
        Returns path to output file.
        """
        print(f"Processing financial PDF: {pdf_path}")

        # Process
        processed_text = self.process(pdf_path)

        # Determine output path
        if output_path is None:
            pdf_name = Path(pdf_path).stem
            output_path = f"{pdf_name}_processed.txt"

        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_text)

        print(f"✓ Processed document saved: {output_path}")
        print(f"  Size: {len(processed_text):,} characters")
        print(f"  Lines: {processed_text.count(chr(10)):,}")

        return output_path
