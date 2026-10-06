"""
Local shim replacing the missing od-parse GitHub package.
Uses pdfplumber for reliable text + table extraction.
"""
import pdfplumber


def parse_pdf(file_path, pipeline_type="default", use_deep_learning=False):
    """Extract text and tables from a PDF file."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages.append({"text": text, "tables": tables})
    return pages


def convert_to_markdown(parsed_data, include_images=False, include_tables=True,
                        include_forms=True, include_handwritten=True):
    """Convert parsed PDF data to a Markdown string."""
    output = []
    for i, page in enumerate(parsed_data, 1):
        output.append(f"## Page {i}\n")

        if page.get("text"):
            output.append(page["text"].strip())
            output.append("")

        if include_tables and page.get("tables"):
            for table in page["tables"]:
                if not table:
                    continue
                # Header row
                header = table[0]
                output.append("| " + " | ".join(str(c or "") for c in header) + " |")
                output.append("| " + " | ".join("---" for _ in header) + " |")
                for row in table[1:]:
                    output.append("| " + " | ".join(str(c or "") for c in row) + " |")
                output.append("")

    return "\n".join(output)
