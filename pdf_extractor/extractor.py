"""
PDF Text Extraction Module
Supports multiple extraction methods for robust text extraction from CVs
"""

import io
import re
from typing import Optional, Union
from pathlib import Path


class PDFExtractor:
    """
    Extracts text from PDF files using multiple methods.
    Falls back to alternative methods if primary method fails.
    """

    def __init__(self):
        self.extraction_method = None
        self._check_available_libraries()

    def _check_available_libraries(self):
        """Check which PDF libraries are available"""
        self.has_pymupdf = False
        self.has_pdfplumber = False
        self.has_pypdf2 = False

        try:
            import fitz  # PyMuPDF
            self.has_pymupdf = True
        except ImportError:
            pass

        try:
            import pdfplumber
            self.has_pdfplumber = True
        except ImportError:
            pass

        try:
            import PyPDF2
            self.has_pypdf2 = True
        except ImportError:
            pass

    def extract(self, pdf_source: Union[str, Path, bytes, io.BytesIO]) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_source: Can be a file path (str/Path), bytes, or BytesIO object

        Returns:
            Extracted text as a string
        """
        # Convert to bytes if needed
        if isinstance(pdf_source, (str, Path)):
            with open(pdf_source, 'rb') as f:
                pdf_bytes = f.read()
        elif isinstance(pdf_source, bytes):
            pdf_bytes = pdf_source
        elif isinstance(pdf_source, io.BytesIO):
            pdf_bytes = pdf_source.getvalue()
        else:
            raise ValueError(f"Unsupported PDF source type: {type(pdf_source)}")

        # Try extraction methods in order of preference
        text = None

        if self.has_pymupdf:
            text = self._extract_with_pymupdf(pdf_bytes)
            self.extraction_method = 'pymupdf'

        if not text and self.has_pdfplumber:
            text = self._extract_with_pdfplumber(pdf_bytes)
            self.extraction_method = 'pdfplumber'

        if not text and self.has_pypdf2:
            text = self._extract_with_pypdf2(pdf_bytes)
            self.extraction_method = 'pypdf2'

        if text is None:
            raise RuntimeError(
                "No PDF extraction library available. "
                "Please install one of: PyMuPDF (fitz), pdfplumber, or PyPDF2"
            )

        # Clean and normalize the extracted text
        text = self._clean_text(text)

        return text

    def _extract_with_pymupdf(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text using PyMuPDF (fitz)"""
        try:
            import fitz

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []

            for page in doc:
                text_parts.append(page.get_text())

            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
            return None

    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text using pdfplumber"""
        try:
            import pdfplumber

            pdf_file = io.BytesIO(pdf_bytes)
            text_parts = []

            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return "\n".join(text_parts)
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            return None

    def _extract_with_pypdf2(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text using PyPDF2"""
        try:
            import PyPDF2

            pdf_file = io.BytesIO(pdf_bytes)
            reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []

            for page in reader.pages:
                text_parts.append(page.extract_text())

            return "\n".join(text_parts)
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""

        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def get_available_methods(self) -> list:
        """Return list of available extraction methods"""
        methods = []
        if self.has_pymupdf:
            methods.append('pymupdf')
        if self.has_pdfplumber:
            methods.append('pdfplumber')
        if self.has_pypdf2:
            methods.append('pypdf2')
        return methods


def extract_text_from_pdf(pdf_source: Union[str, Path, bytes, io.BytesIO]) -> str:
    """
    Convenience function to extract text from a PDF.

    Args:
        pdf_source: Can be a file path (str/Path), bytes, or BytesIO object

    Returns:
        Extracted text as a string
    """
    extractor = PDFExtractor()
    return extractor.extract(pdf_source)
