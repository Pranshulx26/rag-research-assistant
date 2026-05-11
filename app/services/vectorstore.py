from pathlib import Path 
from loguru import logger 
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from app.core.config import get_settings

settings = get_settings()

# Embedding model — converts text to vectors
# This same model is used for BOTH indexing and querying
_embeddings = None 

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings 
    if _embeddings is None:
        logger.info('Initialising Google embedding model')
        _embeddings = GoogleGenerativeAIEmbeddings(
            model='models/gemini-embedding-001',
            google_api_key=settings.gemini_api_key
        )
    return _embeddings

def get_vectorstore() -> Chroma:
    """
    Retuns a ChromaDB instance that presists to disk.
    All documents across all sessions live here.
    """
    return Chroma(
        collection_name='research_documents',
        embedding_function=get_embeddings(),
        persist_directory=str(settings.vectorstore_dir),
    )


def add_documents(chunks: list[Document], doc_id: str) -> int:
    """
    Embed chunks and store in ChromaDB.
    Tags each chunk with the document ID so we can
    filter or deleted by document later.
    """

    # Add doc_di to every chunk's metadata
    for chunk in chunks:
        chunk.metadata['doc_id'] = doc_id

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    logger.info(f'Stored {len(chunks)} chunks for doc_id={doc_id}')
    return len(chunks)


def similarity_search(
        query: str,
        doc_id: str | None = None,
) -> list[Document]:
    """
    Find the top-k most relevant chunks for a query.
    If doc_id is provided, search only within that document.
    """
    vectorstore = get_vectorstore()

    # Filter to a specific document if requested
    where_filter = {'doc_id': doc_id} if doc_id else None 

    results = vectorstore.similarity_search(
        query=query,
        k=settings.top_k_results,
        filter=where_filter,
    )

    logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
    return results 


def document_exists(doc_id: str) -> bool:
    """Check if a document is already indexed."""
    vectorstore = get_vectorstore()
    results = vectorstore.get(where={'doc_id': doc_id})
    return len(results['ids']) > 0 

