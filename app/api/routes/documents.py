import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from app.core.config import get_settings
from app.models.schemas import DocumentUploadResponse
from app.services.pdf_processor import process_pdf
from app.services.vectorstore import add_documents, document_exists

router = APIRouter(prefix="/documents", tags=["Documents"])
settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB = 20


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF and index it into ChromaDB.
    Idempotent — uploading the same file twice skips re-indexing.
    """
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported. Got: {suffix}"
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Max: {MAX_FILE_SIZE_MB}MB"
        )

    # Save to disk
    save_path = settings.upload_dir / file.filename
    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"Saved {file.filename} ({size_mb:.1f}MB)")

    # Process and index
    try:
        chunks, doc_id = process_pdf(save_path)

        # Idempotency check — don't re-index if already exists
        if document_exists(doc_id):
            logger.info(f"Document {doc_id} already indexed, skipping")
            return DocumentUploadResponse(
                doc_id=doc_id,
                filename=file.filename,
                chunks_created=0,
                message="Document already indexed. Ready to query.",
            )

        chunks_created = add_documents(chunks, doc_id)

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunks_created=chunks_created,
            message=f"Successfully indexed {chunks_created} chunks.",
        )

    except ValueError as e:
        # Clean up saved file if processing failed
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"Failed to process {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document")


@router.get("/", response_model=list[str])
async def list_documents():
    """List all uploaded PDF filenames."""
    pdfs = list(settings.upload_dir.glob("*.pdf"))
    return [p.name for p in pdfs]