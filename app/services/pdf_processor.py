import hashlib
from pathlib import Path
from loguru import logger
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import get_settings

settings = get_settings()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract raw text from every page of a PDF.
    fitz is PyMuPDF — fastest PDF parser available in Python.
    """
    logger.info(f"Extracting text from {pdf_path.name}")
    doc = fitz.open(str(pdf_path))

    text_pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # skip blank pages
            text_pages.append(f"[Page {page_num + 1}]\n{text}")

    doc.close()
    full_text = "\n\n".join(text_pages)
    logger.info(f"Extracted {len(full_text)} characters from {len(text_pages)} pages")
    return full_text


def chunk_document(text: str, filename: str) -> list[Document]:
    """
    Split text into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on:
    1. Paragraphs (\n\n) first
    2. Then sentences (\n)
    3. Then words (" ")
    4. Then characters as last resort

    This keeps semantic units together as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": filename}],
    )

    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def get_document_id(file_content: bytes) -> str:
    """
    Stable unique ID based on file content, not filename.
    Two files with the same name but different content
    get different IDs. Same file always gets the same ID.
    """
    return hashlib.md5(file_content).hexdigest()


def process_pdf(pdf_path: Path, file_content: bytes) -> tuple[list[Document], str]:
    """
    Full pipeline: PDF file → chunks ready for embedding.
    Returns chunks and the document ID.
    """
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError(
            f"No text could be extracted from {pdf_path.name}. "
            "The PDF may be scanned or image-based."
        )

    chunks = chunk_document(text, pdf_path.name)
    doc_id = get_document_id(file_content)  

    return chunks, doc_id