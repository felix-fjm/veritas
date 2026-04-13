"""
LLM connector supporting OpenAI, Anthropic, and Ollama.

Provider is inferred from the model string:
  gpt-* / o1* / o3*  → OpenAI   (requires api_key)
  claude-*            → Anthropic (requires api_key)
  anything else       → Ollama local (no api_key needed)

The caller passes the user's API key; it is used only for the duration of
this call and never stored.
"""

import httpx
import anthropic as anthropic_lib
import openai


def _provider(model: str) -> str:
    if model.startswith(("gpt-", "o1", "o3", "text-")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    return "ollama"


def complete(
    prompt: str,
    model: str,
    api_key: str | None,
    ollama_url: str,
    max_tokens: int = 1024,
) -> str:
    """
    Send prompt to the appropriate LLM provider and return the answer string.
    Raises on API or network errors — callers should catch and return HTTP 502.
    """
    provider = _provider(model)

    if provider == "openai":
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    if provider == "anthropic":
        client = anthropic_lib.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    # Ollama — local model, no API key required
    response = httpx.post(
        f"{ollama_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
