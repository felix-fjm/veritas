"""
Embed a single query string via the Ollama nomic-embed-text container.
Returns an L2-normalised list of 768 floats — the same vector space used
during ingestion, so cosine distances are meaningful.
"""

import numpy as np
import httpx


def embed_query(text: str, embedder_url: str, model: str) -> list[float]:
    """
    POST to /api/embed, L2-normalise the result, return as a Python list.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    response = httpx.post(
        f"{embedder_url}/api/embed",
        json={"model": model, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    vector = np.array(response.json()["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()
