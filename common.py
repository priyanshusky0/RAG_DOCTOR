"""Shared paths, document names and text helpers used by every other module."""
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).parent
GUIDELINE_DIR = ROOT / "guidelines"
INDEX_DIR = ROOT / "index"            # FAISS index + chunks.json live here
CHUNKS_FILE = INDEX_DIR / "chunks.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction in front of search queries (not in front of passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Folder name -> guideline name shown to the doctor.
GUIDELINES = {"ada": "ADA 2026", "icmr": "ICMR"}

# Readable titles for each PDF (PDF metadata titles are messy or empty).
DOC_TITLES = {
    "dc26srev.pdf": "Summary of Revisions",
    "dc26s002.pdf": "2. Diagnosis and Classification of Diabetes",
    "dc26s004.pdf": "4. Comprehensive Medical Evaluation and Assessment of Comorbidities",
    "dc26s006.pdf": "6. Glycemic Goals, Hypoglycemia, and Hyperglycemic Crises",
    "dc26s008.pdf": "8. Obesity and Weight Management",
    "dc26s009.pdf": "9. Pharmacologic Approaches to Glycemic Treatment",
    "dc26s010.pdf": "10. Cardiovascular Disease and Risk Management",
    "dc26s011.pdf": "11. Chronic Kidney Disease and Risk Management",
    "dc26s012.pdf": "12. Retinopathy, Neuropathy, and Foot Care",
    "ICMR_T2D_Guidelines_2018.pdf": "ICMR Guidelines for Management of Type 2 Diabetes 2018",
    "ICMR_STW_T2DM.pdf": "STW: Diabetes Mellitus Type 2",
    "ICMR_STW_CKD.pdf": "STW: Chronic Kidney Disease",
    "ICMR_STW_Diabetic_Foot.pdf": "STW: Diabetic Foot",
    "ICMR_STW_Diabetic_Retinopathy.pdf": "STW: Diabetic Retinopathy",
    "ICMR_STW_Hypertension_Adults.pdf": "STW: Hypertension in Adults",
}

# Expand ligatures (ﬁ -> fi) and join words hyphenated across line breaks.
TEXT_FLAGS = (pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_DEHYPHENATE) & ~pymupdf.TEXT_PRESERVE_LIGATURES


def list_pdfs():
    """Yield (guideline_name, pdf_path) for every guideline PDF."""
    for folder, name in GUIDELINES.items():
        for path in sorted((GUIDELINE_DIR / folder).glob("*.pdf")):
            yield name, path


def doc_title(path):
    return DOC_TITLES.get(Path(path).name, Path(path).stem.replace("_", " "))


def page_lines(page):
    """Return the page as a list of lines: (text, font_size, is_bold)."""
    lines = []
    for block in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if text == "z":  # ICMR bullets are a Wingdings "z" glyph
                continue
            size = max(s["size"] for s in spans)
            bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
            if lines and lines[-1][0].endswith("\xad"):  # soft hyphen: word continues on this line
                prev, size, bold = lines.pop()
                text = prev[:-1] + text
            lines.append((text, size, bold))
    return [(t.replace("\xad", ""), size, bold) for t, size, bold in lines]


def page_text(page):
    """Plain text of one page, exactly as chunked and as verified."""
    return "\n".join(text for text, _, _ in page_lines(page))


def embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True},
                                 query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_INSTRUCTION})


def normalize(text):
    """Loose form used for fuzzy quote matching: lowercase, plain quotes/dashes, single spaces."""
    text = text.lower()
    for a, b in {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-", "≥": ">=", "≤": "<=", "|": " "}.items():
        text = text.replace(a, b)
    text = re.sub(r"-{3,}", " ", text)  # markdown table separators
    return re.sub(r"\s+", " ", text).strip()
