from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models.schemas import QuestionRequest, AnswerResponse
from app.services.rag_chain import answer_question

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=AnswerResponse)
async def chat(payload: QuestionRequest):
    """
    Ask a question. Optionally restrict to a specific document
    by passing doc_id. If no doc_id, searches all documents.
    """
    logger.info(f"Question received: '{payload.question[:60]}'")

    try:
        result = await answer_question(
            question=payload.question,
            doc_id=payload.doc_id,
        )

        return AnswerResponse(
            question=payload.question,
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
        )

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )