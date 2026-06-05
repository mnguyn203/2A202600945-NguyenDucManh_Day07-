from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        context_parts = []
        for i, res in enumerate(results, 1):
            context_parts.append(f"Chunk {i}:\n{res['content']}")
        
        context = "\n\n".join(context_parts)
        prompt = (
            f"Dựa vào các thông tin sau đây:\n\n"
            f"{context}\n\n"
            f"Hãy trả lời câu hỏi: {question}"
        )
        return self.llm_fn(prompt)
