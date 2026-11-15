"""Pruebas offline del endpoint /chat y del flujo LangGraph."""

import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


class ChatApiTests(unittest.TestCase):
    def setUp(self) -> None:
        # No entrar al lifespan: estas pruebas no necesitan enviar telemetría.
        self.client = TestClient(app)

    def test_chat_retrieves_context_then_generates_answer(self) -> None:
        question = "Is the PUT method idempotent?"
        chunks = [
            {
                "content": "The PUT method is idempotent.",
                "metadata": {
                    "filename": "rfc9110-http-semantics.txt",
                    "section": "9.3.4. PUT",
                },
                "similarity": 0.91,
            }
        ]
        expected_answer = "Yes, PUT is idempotent (RFC 9110, Section 9.3.4)."
        session_id = f"telegram:chat-1:user-1:{uuid4().hex}"

        with (
            patch("app.services.rag_service.LANGFUSE_ENABLED", False),
            patch(
                "app.services.rag_service.retrieve",
                new_callable=AsyncMock,
                return_value=chunks,
            ) as retrieve,
            patch(
                "app.services.rag_service.generate_response",
                new_callable=AsyncMock,
                return_value=expected_answer,
            ) as generate_response,
        ):
            response = self.client.post(
                "/chat",
                json={"question": question, "session_id": session_id},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"answer": expected_answer, "session_id": session_id},
        )
        retrieve.assert_awaited_once_with(question)
        generate_response.assert_awaited_once_with(question, chunks, [])

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_rejects_empty_question(self) -> None:
        response = self.client.post("/chat", json={"question": ""})

        self.assertEqual(response.status_code, 422)

    def test_missing_gemini_configuration_returns_service_unavailable(self) -> None:
        with (
            patch("app.services.rag_service.LANGFUSE_ENABLED", False),
            patch(
                "app.services.rag_service.retrieve",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.rag_service.generate_response",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Falta configurar GEMINI_API_KEY"),
            ),
        ):
            response = self.client.post("/chat", json={"question": "Hola"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("GEMINI_API_KEY", response.json()["detail"])

    def test_unexpected_llm_error_is_not_leaked(self) -> None:
        with (
            patch("app.services.rag_service.LANGFUSE_ENABLED", False),
            patch(
                "app.services.rag_service.retrieve",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.rag_service.generate_response",
                new_callable=AsyncMock,
                side_effect=ValueError("internal secret detail"),
            ),
        ):
            response = self.client.post("/chat", json={"question": "Hola"})

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("internal secret detail", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
