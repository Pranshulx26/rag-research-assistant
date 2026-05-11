import pytest
import io
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
from app.main import app


MOCK_CHUNKS = [MagicMock(page_content="test content", metadata={"source": "test.pdf"})]
MOCK_DOC_ID = "abc123"


@pytest.fixture(autouse=True)
def mock_pdf_services():
    """Mock all external services so tests run without real PDFs or ChromaDB."""
    with patch("app.api.routes.documents.process_pdf",
               return_value=(MOCK_CHUNKS, MOCK_DOC_ID)), \
         patch("app.api.routes.documents.document_exists",
               return_value=False), \
         patch("app.api.routes.documents.add_documents",
               return_value=1):
        yield


@pytest.mark.asyncio
async def test_upload_valid_pdf():
    """Valid PDF upload should return 200 with doc_id."""
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
    fake_pdf.name = "test.pdf"

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/documents/upload",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == MOCK_DOC_ID
    assert data["filename"] == "test.pdf"
    assert data["chunks_created"] == 1


@pytest.mark.asyncio
async def test_upload_non_pdf_rejected():
    """Non-PDF files should return 400."""
    fake_file = io.BytesIO(b"not a pdf")

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/documents/upload",
            files={"file": ("malware.exe", fake_file, "application/octet-stream")},
        )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_already_indexed():
    """Uploading the same document twice should skip re-indexing."""
    with patch("app.api.routes.documents.document_exists", return_value=True):
        fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/documents/upload",
                files={"file": ("test.pdf", fake_pdf, "application/pdf")},
            )

    assert response.status_code == 200
    assert response.json()["chunks_created"] == 0
    assert "already indexed" in response.json()["message"]


@pytest.mark.asyncio
async def test_list_documents():
    """List endpoint should return 200."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/documents/")

    assert response.status_code == 200
    assert isinstance(response.json(), list)