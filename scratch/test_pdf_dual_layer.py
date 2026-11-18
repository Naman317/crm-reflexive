from pypdf import PdfReader, PdfWriter
import pypdfium2 as pdfium
import fitz # PyMuPDF if available, or pypdf/reportlab
import os
import urllib.request
import json

# Check available libraries
print("Testing PDF creation mechanisms...")
try:
    import fitz
    print("fitz (PyMuPDF) is available!")
except ImportError:
    print("fitz not available, will check reportlab or pypdf")

try:
    import reportlab
    print("reportlab is available!")
except ImportError:
    print("reportlab not available")
