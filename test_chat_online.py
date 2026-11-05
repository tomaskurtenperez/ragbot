"""Prueba online del flujo real Chroma -> Gemini a través de LangGraph.

Ejecución deliberada:
    RUN_ONLINE_RAG_TESTS=1 python -m unittest test_chat_online -v

Requiere GEMINI_API_KEY y una colección Chroma ``documents`` ya poblada.
"""

import asyncio
import os
import unittest
from uuid import uuid4

from app.core.config import GEMINI_API_KEY
from app.services.rag_service import rag_graph


ONLINE_TESTS_ENABLED = os.getenv("RUN_ONLINE_RAG_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


@unittest.skipUnless(
    ONLINE_TESTS_ENABLED,
    "Prueba online desactivada; define RUN_ONLINE_RAG_TESTS=1 para ejecutarla.",
)
class OnlineRAGTests(unittest.TestCase):
    def test_chroma_retrieval_and_gemini_answer(self) -> None:
        if not GEMINI_API_KEY:
            self.fail("Configura GEMINI_API_KEY en .env para la prueba online.")

        question = os.getenv(
            "RAG_ONLINE_TEST_QUESTION",
            "quien es tomas.",
        )
        result = asyncio.run(
            rag_graph.ainvoke(
                {
                    "question": question,
                    "history": [{"role": "user", "content": question}],
                },
                config={"configurable": {"thread_id": f"online-test:{uuid4().hex}"}},
            )
        )
        chunks = result.get("chunks", [])
        answer = result.get("answer", "")

        self.assertTrue(
            chunks,
            "Chroma no recuperó fragmentos. Verifica que data/chroma_db "
            "contenga documentos en la colección 'documents'.",
        )
        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip(), "Gemini devolvió una respuesta vacía.")

        sources = sorted(
            {
                chunk.get("metadata", {}).get("filename", "desconocida")
                for chunk in chunks
            }
        )
        print(f"\nFragmentos recuperados: {len(chunks)}; fuentes: {sources}")
        print(f"Respuesta Gemini: {answer}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
