"""
Redis cache helpers for chunk results and LLM answers.

Key scheme:
  chunks:{question_hash}         → JSON list of chunk dicts    TTL 24 h
  answer:{question_hash}:{model} → JSON { answer, sources }    TTL  6 h

The question_hash is SHA-256 of (normalised_question + "|" + model),
so different models for the same question get separate answer cache entries
but share the same chunk cache entry.
"""

import hashlib
import json
import unicodedata

import redis as redis_lib


def _normalise(question: str) -> str:
    """Lowercase, NFC-normalise, collapse internal whitespace."""
    q = unicodedata.normalize("NFC", question).lower().strip()
    return " ".join(q.split())


def question_hash(question: str, model: str) -> str:
    """SHA-256 of normalised_question + "|" + model."""
    key = _normalise(question) + "|" + model
    return hashlib.sha256(key.encode()).hexdigest()


# ── Answer cache ──────────────────────────────────────────────────────────────

def get_answer(r: redis_lib.Redis, q_hash: str, model: str) -> dict | None:
    raw = r.get(f"answer:{q_hash}:{model}")
    return json.loads(raw) if raw is not None else None


def set_answer(
    r: redis_lib.Redis,
    q_hash: str,
    model: str,
    data: dict,
    ttl: int,
) -> None:
    r.setex(f"answer:{q_hash}:{model}", ttl, json.dumps(data))


# ── Chunk cache ───────────────────────────────────────────────────────────────

def get_chunks(r: redis_lib.Redis, q_hash: str) -> list[dict] | None:
    raw = r.get(f"chunks:{q_hash}")
    return json.loads(raw) if raw is not None else None


def set_chunks(
    r: redis_lib.Redis,
    q_hash: str,
    chunks: list[dict],
    ttl: int,
) -> None:
    r.setex(f"chunks:{q_hash}", ttl, json.dumps(chunks))
