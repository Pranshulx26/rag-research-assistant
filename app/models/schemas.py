from pydantic import BaseModel, Field
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    doc_id: str 
    filename: str 
    chunks_created: int
    message: str 

class DocumentInfo(BaseModel):
    doc_id: str 
    filename: str 
    uploaded_at: datetime 

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description='Question to ask about the document',
        examples=['What are the main findings of this paper?'],
    )
    doc_id: str | None = Field(
        default=None,
        description='Optional: restrict search to a specific document',
    )

class AnswerResponse(BaseModel):
    question: str 
    answer: str 
    sources: list[str]
    chunks_used: int 

    