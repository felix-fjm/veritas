"""
Prompt assembly for RAG queries.

Selects the top TOP_K chunks from the retrieved results and formats them
into a prompt that instructs the LLM to answer only from the provided context.
"""

TOP_K = 5  # chunks passed to the LLM (k=10 retrieved from Qdrant, trimmed here)

_INSTRUCTION = (
    "You are a factual assistant. Answer the user's question using ONLY "
    "the context passages provided below. For every claim you make, cite "
    "the source passage using the format [Title — Section]. "
    "If the context does not contain enough information to answer the question, "
    "say so explicitly rather than guessing."
)


def assemble_prompt(question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Build the LLM prompt from the top TOP_K chunks.

    Args:
        question: the raw user question.
        chunks:   list of chunk dicts (title, section, url, text, …).
                  Expected to be ordered by relevance (best first).

    Returns:
        prompt  — full prompt string ready for the LLM.
        sources — list of {title, section, url} dicts to return in the API response.
    """
    top = chunks[:TOP_K]

    context_blocks: list[str] = []
    for i, chunk in enumerate(top, 1):
        header = f"[{i}] {chunk['title']} — {chunk['section']}"
        context_blocks.append(f"{header}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        f"{_INSTRUCTION}\n\n"
        f"=== Context ===\n{context}\n\n"
        f"=== Question ===\n{question}"
    )

    sources = [
        {"title": c["title"], "section": c["section"], "url": c["url"]}
        for c in top
    ]

    return prompt, sources
