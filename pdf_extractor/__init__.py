"""
PDF Extractor Module
Extracts text from PDF files for CV processing
"""

from .extractor import PDFExtractor, extract_text_from_pdf

__all__ = ['PDFExtractor', 'extract_text_from_pdf']
