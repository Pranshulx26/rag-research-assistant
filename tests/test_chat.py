import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from app.main import app


MOCK_ANSWER = {
    "answer": "The document discusses attention mechanisms.",
    "sources": ["vaswani_at_el.pdf"],
    "chunks_used": 4,
}


@pytest.mark.asyncio
async def test_chat_valid_question():
    """Valid question should return 200 with answer."""
    with patch("app.api.routes.chat.answer_question",
               new_callable=AsyncMock,
               return_value=MOCK_ANSWER):

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/chat/",
                json={"question": "What is the main topic?"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == MOCK_ANSWER["answer"]
    assert data["chunks_used"] == 4
    assert len(data["sources"]) == 1


@pytest.mark.asyncio
async def test_chat_question_too_short():
    """Questions under 5 chars should return 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat/",
            json={"question": "Hi"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_with_doc_id():
    """Question with doc_id should pass it to answer_question."""
    with patch("app.api.routes.chat.answer_question",
               new_callable=AsyncMock,
               return_value=MOCK_ANSWER) as mock_answer:

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/chat/",
                json={
                    "question": "What is the main topic?",
                    "doc_id": "abc123",
                },
            )

        # Verify doc_id was passed through correctly
        mock_answer.assert_called_once_with(
            question="What is the main topic?",
            doc_id="abc123",
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_check():
    """Health endpoint should return model name."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model" in response.json()