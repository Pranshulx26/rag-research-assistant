from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import get_settings
from app.services.vectorstore import similarity_search

settings = get_settings()

# System prompt — this is prompt engineering as code
# It tells the LLM exactly how to behave
SYSTEM_PROMPT = """You are a precise research assistant. Your job is to answer
questions based ONLY on the provided document context.

Rules you must follow:
- Only use information from the context provided below
- If the answer is not in the context, say "I cannot find this information in the document"
- Always cite which page your answer comes from using [Page X] references
- Be concise and factual
- Do not make up information or use outside knowledge

Context from document:
{context}
"""

_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        logger.info(f"Initialising Gemini model: {settings.gemini_model}")
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,  # low = more factual, less creative
        )
    return _llm


def format_context(chunks: list[Document]) -> str:
    """
    Format retrieved chunks into a single context string
    that gets injected into the prompt.
    """
    formatted = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        formatted.append(
            f"--- Chunk {i+1} (from {source}) ---\n{chunk.page_content}"
        )
    return "\n\n".join(formatted)


async def answer_question(
    question: str,
    doc_id: str | None = None,
) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question
    2. Retrieve relevant chunks
    3. Format context
    4. Generate grounded answer
    """
    logger.info(f"Processing question: '{question[:60]}...'")

    # Step 1 + 2 — retrieve
    chunks = similarity_search(question, doc_id=doc_id)

    if not chunks:
        return {
            "answer": "No relevant information found in the document.",
            "sources": [],
            "chunks_used": 0,
        }

    # Step 3 — format context
    context = format_context(chunks)

    # Step 4 — generate
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = prompt | llm

    response = await chain.ainvoke({
        "context": context,
        "question": question,
    })

    # Extract source pages from metadata
    sources = list({
        chunk.metadata.get("source", "unknown")
        for chunk in chunks
    })

    logger.info(f"Generated answer using {len(chunks)} chunks")

    return {
        "answer": response.content,
        "sources": sources,
        "chunks_used": len(chunks),
    }