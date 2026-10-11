"""Check, in code, that each quoted passage really appears on the page the answer cites."""
from functools import lru_cache

import pymupdf
from rapidfuzz import fuzz

from common import ROOT, normalize, page_text

MIN_SCORE = 90
MIN_QUOTE_CHARS = 20  # very short quotes would match almost any page


@lru_cache(maxsize=256)
def cited_page_text(file, page):
    with pymupdf.open(ROOT / file) as doc:
        if not 1 <= page <= doc.page_count:
            return None
        return normalize(page_text(doc[page - 1]))


def verify_citation(rec, chunks):
    """Return {verified, score, reason, file, page} for one recommendation."""
    try:
        chunk = chunks[int(rec.get("excerpt")) - 1]  # the excerpt tells us which PDF file was cited
        page = int(rec.get("page"))
    except (TypeError, ValueError, IndexError):
        return {"verified": False, "score": 0, "reason": "citation does not point to a retrieved excerpt"}

    result = {"file": chunk["file"], "page": page, "verified": False, "score": 0}
    quote = normalize(str(rec.get("quote", "")))
    text = cited_page_text(chunk["file"], page)
    if text is None:
        return {**result, "reason": f"page {page} does not exist in {chunk['document']}"}
    if len(quote) < MIN_QUOTE_CHARS:
        return {**result, "reason": "quote too short to verify"}

    score = round(fuzz.partial_ratio(quote, text))
    ok = score >= MIN_SCORE
    reason = "verified on page" if ok else f"quote not found on page {page} (best match {score}%)"
    return {**result, "verified": ok, "score": score, "reason": reason}
