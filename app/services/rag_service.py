from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.core.config import GEMINI_MODEL, LANGFUSE_ENABLED
from app.services.llm_service import generate_response
from app.services.retrieval_service import retrieve


MAX_HISTORY_MESSAGES = 12


def merge_history(
    previous: list[dict[str, str]],
    updates: list[dict[str, str]],
) -> list[dict[str, str]]:
    return (previous + updates)[-MAX_HISTORY_MESSAGES:]


class RAGState(TypedDict):
    question: str
    chunks: list[dict]
    answer: str
    history: Annotated[list[dict[str, str]], merge_history]


async def retrieve_node(state: RAGState) -> dict:
    prior_messages = state.get("history", [])[:-1]
    recent_context = [
        message["content"] for message in prior_messages[-4:]
        if message.get("content")
    ]
    retrieval_query = "\n".join([*recent_context, state["question"]])
    chunks = await retrieve(retrieval_query)
    return {"chunks": chunks}


async def generate_node(state: RAGState) -> dict:
    history = state.get("history", [])[:-1]
    answer = await generate_response(state["question"], state["chunks"], history)
    return {
        "answer": answer,
        "history": [{"role": "assistant", "content": answer}],
    }


graph_builder = StateGraph(RAGState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
checkpointer = MemorySaver()
rag_graph = graph_builder.compile(checkpointer=checkpointer)


async def generate_answer(question: str, session_id: str) -> str:
    graph_config = {"configurable": {"thread_id": session_id}}
    if not LANGFUSE_ENABLED:
        result = await rag_graph.ainvoke(
            {
                "question": question,
                "history": [{"role": "user", "content": question}],
            },
            config=graph_config,
        )
        return result["answer"]

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Langfuse está configurado, pero no está instalado. "
            "Instala las dependencias con: python -m pip install -r requirements.txt"
        ) from exc

    langfuse = get_client()
    handler = CallbackHandler()
    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-api-chat",
    ) as trace:
        trace.update(
            input={"question": question},
            metadata={"model": GEMINI_MODEL, "pipeline": "chroma-e5-bm25-gemini"},
        )
        result = await rag_graph.ainvoke(
            {
                "question": question,
                "history": [{"role": "user", "content": question}],
            },
            config={
                **graph_config,
                "callbacks": [handler],
                "run_name": "rag-chat",
            },
        )
        trace.update(
            output={"answer": result["answer"]},
            metadata={"retrieved_chunks": len(result.get("chunks", []))},
        )

    return result["answer"]
