from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    """
    Safely extract text from a PDF.
    Skips pages with broken fonts instead of crashing.
    """
    reader = PdfReader(file_path)
    extracted_pages = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                extracted_pages.append(text)
        except Exception:
            # Skip pages with font / bbox issues
            continue

    return "\n".join(extracted_pages)
