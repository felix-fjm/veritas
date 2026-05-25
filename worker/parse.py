"""
Article filtering, parsing, cleaning, and chunking.

Pipeline per article:
  extract_sections → clean_text → merge short sections → chunk_section

Token counting uses a whitespace-split approximation (1 word ≈ 1 token),
which is accurate enough for the 50/300/600-token thresholds used here.
"""

import html
import logging
import re
from typing import Generator

logger = logging.getLogger(__name__)

# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Approximate token count: split on whitespace."""
    return len(text.split())


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Strip wikitext markup, HTML, citation markers, and table artefacts;
    normalise whitespace.  Safe to call on both body text and section names.
    """
    if not text:
        return ""

    # Strip [N] citation / footnote markers
    text = re.sub(r"\[\d+\]", "", text)

    # Remove <ref>...</ref> footnote blocks entirely (multiline)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref[^/]*/?>", "", text, flags=re.IGNORECASE)

    # Strip all remaining HTML tags (keep text content between tags)
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities (&amp; &lt; &nbsp; etc.) — after tag removal
    text = html.unescape(text)

    # Strip wikitext templates {{...}}, handling nesting by iterating inward-out
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Strip File / Image links entirely [[File:...]] [[Image:...]]
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    # Strip internal links: [[target|label]] → label,  [[target]] → target
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    # Strip external links: [url label] → label,  [url] → empty
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)

    # Strip bold/italic markers (''' before '' to avoid leaving stray apostrophes)
    text = re.sub(r"'{3}", "", text)
    text = re.sub(r"'{2}", "", text)

    # Remove wiki table markup remnants ({| ... |})
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Remove table cell / header lines (lines starting with | or !)
    text = re.sub(r"(?m)^\s*[|!].*$", "", text)

    # Remove wikitext section heading lines == ... ==
    text = re.sub(r"(?m)^={2,}.*={2,}$", "", text)
    text = re.sub(r"(?m)^\*+\s*$", "", text)  # stray bullet lines

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs within a line
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ── Section extraction ────────────────────────────────────────────────────────

def extract_sections(article: dict) -> list[tuple[str, str]]:
    """
    Return a list of (section_name, raw_text) tuples.

    PATH 1 — source_text present (raw wikitext):
      - "Introduction" from opening_text (exact, reliable)
      - Real named sections parsed from wikitext heading markers (== ... ==)
      - Text before the first heading is discarded; opening_text covers it.

    PATH 2 — source_text absent (typical Cirrus record):
      - "Introduction" from opening_text (exact, reliable)
      - "Body" from the flat Cirrus text field (entire article body as one section)
      - heading[] is NOT used for segmentation (offsets are unknown); it is
        stored separately as available_headings in process_article().
    """
    opening_text: str = article.get("opening_text") or ""
    source_text: str = article.get("source_text") or ""

    sections: list[tuple[str, str]] = []

    intro = opening_text.strip()
    if intro:
        sections.append(("Introduction", intro))

    if source_text:
        # PATH 1: parse real section boundaries from wikitext heading markers
        heading_pattern = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)
        matches = list(heading_pattern.finditer(source_text))

        for i, match in enumerate(matches):
            heading_name = clean_text(match.group(2).strip())
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(source_text)
            section_text = source_text[start:end].strip()
            if section_text:
                sections.append((heading_name, section_text))
    else:
        # PATH 2: flat Cirrus text field → single honest "Body" section
        body = (article.get("text") or "").strip()
        if body:
            sections.append(("Body", body))

    return sections


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_by_words(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Fallback splitter for blocks that have no paragraph boundaries but still
    exceed max_tokens.  Slices by word count and carries an overlap window into
    each successive sub-chunk.
    """
    words = text.split()
    result: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        result.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_tokens  # carry overlap into next sub-chunk
    return result


def chunk_section(
    text: str,
    max_tokens: int = 600,
    overlap_tokens: int = 50,
) -> list[str]:
    """
    Split section text into chunks according to the spec:
      < 50 tok  → return [] (caller merges with previous section)
      ≤ 600 tok → return [text] as a single chunk
      > 600 tok → split at paragraph boundaries with 50-tok overlap

    When splitting, overlap is achieved by carrying the last N tokens of the
    previous chunk into the start of the next chunk.

    Individual paragraphs that exceed max_tokens are expanded via word-level
    splitting before the paragraph-accumulation loop runs.  Without this, a
    single large paragraph (no internal double-newline) would bypass the
    max_tokens guard because the `and current_paras` condition is False when
    the accumulator is empty, letting the whole paragraph through as-is.
    """
    n = count_tokens(text)

    if n < 50:
        return []

    if n <= max_tokens:
        return [text]

    # Split at paragraph boundaries, then expand any paragraph that is itself
    # larger than max_tokens using word-level splitting.
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        if count_tokens(para) > max_tokens:
            paragraphs.extend(_split_by_words(para, max_tokens, overlap_tokens))
        else:
            paragraphs.append(para)

    chunks: list[str] = []
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_paras:
            # Emit current chunk
            chunks.append("\n\n".join(current_paras))

            # Build overlap tail from end of the just-emitted chunk
            overlap_paras: list[str] = []
            overlap_count = 0
            for p in reversed(current_paras):
                pt = count_tokens(p)
                if overlap_count + pt <= overlap_tokens:
                    overlap_paras.insert(0, p)
                    overlap_count += pt
                else:
                    break

            current_paras = overlap_paras
            current_tokens = overlap_count

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


# ── Article processing ────────────────────────────────────────────────────────

def process_article(article: dict, pageview_rank: int) -> list[dict]:
    """
    Full per-article pipeline: extract → clean → merge short sections → chunk.

    Returns a list of chunk dicts with all required metadata fields:
      text, title, section, url, last_modified, pageview_rank, chunk_index
    For PATH 2 articles (no source_text), also includes available_headings.
    """
    title: str = article.get("title") or ""
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    # Cirrus timestamp is ISO-8601; store date portion only
    raw_ts: str = article.get("timestamp") or ""
    last_modified = raw_ts[:10] if raw_ts else ""

    # PATH 2 only: preserve heading[] as payload context (not used for segmentation)
    available_headings: list[str] | None = None
    if not article.get("source_text"):
        raw_headings = article.get("heading") or []
        clean_h = [h.strip() for h in raw_headings if h.strip()]
        if clean_h:
            available_headings = clean_h

    sections = extract_sections(article)

    # Clean each section
    cleaned: list[tuple[str, str]] = []
    for section_name, section_text in sections:
        c = clean_text(section_text)
        if c:
            cleaned.append((section_name, c))

    # Merge sections with < 50 tokens into the preceding section
    merged: list[tuple[str, str]] = []
    for section_name, section_text in cleaned:
        if count_tokens(section_text) < 50 and merged:
            prev_name, prev_text = merged[-1]
            merged[-1] = (prev_name, prev_text + "\n\n" + section_text)
        else:
            merged.append((section_name, section_text))

    # Chunk and build output dicts
    result: list[dict] = []
    chunk_index = 0

    for section_name, section_text in merged:
        chunk_texts = chunk_section(section_text)
        for chunk_text in chunk_texts:
            chunk: dict = {
                "text": chunk_text,
                "title": title,
                "section": section_name,
                "url": url,
                "last_modified": last_modified,
                "pageview_rank": pageview_rank,
                "chunk_index": chunk_index,
            }
            if available_headings is not None:
                chunk["available_headings"] = available_headings
            result.append(chunk)
            chunk_index += 1

    return result


# ── Popularity filtering ──────────────────────────────────────────────────────

def compute_popularity_threshold_from_scores(
    score_pairs: list[tuple[float, str]],
    top_fraction: float,
) -> tuple[float, dict[str, int]]:
    """
    Same semantics as compute_popularity_threshold but takes pre-collected
    (popularity_score, title) pairs instead of full article dicts.

    Used by the two-pass streaming ingestion path so that only lightweight
    score data needs to be held in memory during the ranking step (~1 GB
    for 6.7M articles vs ~15 GB for full article content).
    """
    scored = [(s, t) for s, t in score_pairs if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)

    keep_n = max(1, int(len(scored) * top_fraction))
    threshold = scored[keep_n - 1][0] if scored else 0.0

    rank_map: dict[str, int] = {
        title: rank + 1
        for rank, (_, title) in enumerate(scored[:keep_n])
    }

    logger.info(
        "Popularity threshold: %.6f (keeping top %d / %d articles)",
        threshold,
        keep_n,
        len(scored),
    )
    return threshold, rank_map


def compute_popularity_threshold(
    articles: list[dict],
    top_fraction: float,
) -> tuple[float, dict[str, int]]:
    """
    Given a list of articles (already loaded from a first streaming pass),
    compute the popularity_score threshold that retains the top `top_fraction`
    of articles, and return a title→rank mapping (1 = most popular).

    Only articles with a non-zero popularity_score are considered.
    """
    scored = [
        (a.get("popularity_score") or 0.0, a.get("title") or "")
        for a in articles
        if (a.get("popularity_score") or 0.0) > 0
    ]

    scored.sort(key=lambda x: x[0], reverse=True)

    keep_n = max(1, int(len(scored) * top_fraction))
    threshold = scored[keep_n - 1][0] if scored else 0.0

    rank_map: dict[str, int] = {
        title: rank + 1
        for rank, (_, title) in enumerate(scored[:keep_n])
    }

    logger.info(
        "Popularity threshold: %.6f (keeping top %d / %d articles)",
        threshold,
        keep_n,
        len(scored),
    )
    return threshold, rank_map


def filter_articles(
    articles: list[dict],
    top_fraction: float,
    min_tokens: int,
) -> Generator[tuple[dict, int], None, None]:
    """
    Yield (article, pageview_rank) for articles that pass both filters:
      1. popularity_score in the top `top_fraction`
      2. opening_text + text has at least `min_tokens` tokens

    Uses compute_popularity_threshold to determine the score cutoff.
    """
    threshold, rank_map = compute_popularity_threshold(articles, top_fraction)

    passed = dropped_pop = dropped_stub = 0

    for article in articles:
        score = article.get("popularity_score") or 0.0
        title = article.get("title") or ""

        if score < threshold or title not in rank_map:
            dropped_pop += 1
            continue

        full_text = (article.get("opening_text") or "") + " " + (article.get("text") or "")
        if count_tokens(full_text) < min_tokens:
            dropped_stub += 1
            continue

        passed += 1
        yield article, rank_map[title]

    logger.info(
        "Filter results: %d passed, %d dropped (popularity), %d dropped (stub)",
        passed,
        dropped_pop,
        dropped_stub,
    )
