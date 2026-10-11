"""Run once: read the guideline PDFs, chunk them page by page, embed, and save a FAISS index.

    python ingest.py
"""
import json
from collections import Counter

import pdfplumber
import pymupdf
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import ROOT, CHUNKS_FILE, EMBED_MODEL, embedding_model, INDEX_DIR, doc_title, list_pdfs, page_lines

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
LOW_TEXT_CHARS = 100  # pages with less text than this are probably images or scans


def repeated_lines(doc):
    """Lines that appear on many pages (running headers/footers) are never section headings."""
    counts = Counter(t for page in doc for t in {t for t, _, _ in page_lines(page)})
    return {t for t, n in counts.items() if doc.page_count > 2 and n >= doc.page_count * 0.3}


def is_heading(text, size, bold, body_size, skip):
    if text in skip or not (4 <= len(text) <= 90) or sum(c.isalpha() for c in text) < 3:
        return False
    if text.endswith((".", ",", ";")) or text[0].islower():
        return False
    return size >= body_size + 1.5 or (bold and size >= body_size - 1.5)


def is_bibliography(section, doc):
    """Reference lists are not guidance and crowd out real content in search.
    (One-page posters are skipped: their reading order puts real content after 'REFERENCES'.)"""
    return section.lower() == "references" and doc.page_count > 2


def table_to_markdown(table):
    rows = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in table]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def chunk_pdf(guideline, path):
    """Return (chunks, page_count, low_text_pages) for one PDF."""
    doc, plumber = pymupdf.open(path), pdfplumber.open(path)
    skip = repeated_lines(doc)
    title, heading = doc_title(path), ""
    chunks, low_text = [], []

    for i, page in enumerate(doc):
        page_no = i + 1  # page number as shown in a PDF viewer
        lines = page_lines(page)
        text = "\n".join(t for t, _, _ in lines)
        if len(text.strip()) < LOW_TEXT_CHARS:
            low_text.append(page_no)

        # Character offset where each heading starts, so each chunk gets the heading above it.
        sizes = Counter()
        for t, size, _ in lines:
            sizes[round(size)] += len(t)  # body text = the font size covering the most characters
        body_size = sizes.most_common(1)[0][0] if lines else 0
        headings, offset = [], 0
        for t, size, bold in lines:
            if is_heading(t, size, bold, body_size, skip):
                headings.append((offset, t))
            offset += len(t) + 1

        start = 0
        for piece in splitter.split_text(text):
            pos = text.find(piece, start)
            start = max(pos, 0)
            section = next((h for o, h in reversed(headings) if o <= start), heading)
            if is_bibliography(section, doc):
                continue
            chunks.append(dict(text=piece, page=page_no, section=section, type="text"))
        if headings:
            heading = headings[-1][1]  # carries over to the next page

        for table in [] if is_bibliography(heading, doc) else plumber.pages[i].extract_tables():
            md = table_to_markdown(table)
            if len(md) > 40:
                for piece in splitter.split_text(md):
                    chunks.append(dict(text=piece, page=page_no, section=heading, type="table"))

    for n, c in enumerate(chunks):
        c.update(guideline=guideline, document=title, file=str(path.relative_to(ROOT)),
                 chunk_id=f"{path.stem}-p{c['page']}-{n}")
    return chunks, doc.page_count, low_text


def main():
    all_chunks = []
    print(f"{'Guideline':9} {'Document':62} {'Pages':>5} {'Chunks':>6} {'Tables':>6}  Low-text pages")
    for guideline, path in list_pdfs():
        chunks, pages, low = chunk_pdf(guideline, path)
        tables = sum(c["type"] == "table" for c in chunks)
        print(f"{guideline:9} {doc_title(path)[:62]:62} {pages:5} {len(chunks):6} {tables:6}  {low or ''}")
        all_chunks += chunks

    for g in sorted({c["guideline"] for c in all_chunks}):
        print(f"Total {g}: {sum(c['guideline'] == g for c in all_chunks)} chunks")
    print(f"Total: {len(all_chunks)} chunks. Embedding with {EMBED_MODEL} ...")

    docs = [Document(page_content=c["text"], metadata={"chunk_id": c["chunk_id"], "guideline": c["guideline"]})
            for c in all_chunks]
    embeddings = embedding_model()
    INDEX_DIR.mkdir(exist_ok=True)
    FAISS.from_documents(docs, embeddings).save_local(str(INDEX_DIR))
    CHUNKS_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=1))
    print(f"Saved FAISS index and {CHUNKS_FILE.name} to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
