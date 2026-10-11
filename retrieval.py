"""Hybrid retrieval: BM25 keyword search + FAISS semantic search, fused with Reciprocal Rank Fusion."""
import json
import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from common import CHUNKS_FILE, GUIDELINES, INDEX_DIR, embedding_model

PER_RETRIEVER_K = 8
FINAL_K = 6          # chunks kept per search query
TOTAL_K = 8          # chunks sent to the LLM
# With "Both guidelines", each guideline gets its own slots: ADA has 5x more chunks than ICMR
# and would otherwise crowd ICMR out of a shared top list.
BOTH_SLOTS = {"ADA 2026": 5, "ICMR": 3}
WEIGHTS = [0.5, 0.5]
RRF_C = 60  # EnsembleRetriever's default constant
STOPWORDS = set("a an and are as at be by can do does for from has have how i in is it of on or should "
                "the their this to was what when which who will with patient guidelines say".split())
FILTERS = ["Both guidelines"] + [f"{name} only" for name in GUIDELINES.values()]


def tokenize(text):
    """BM25 tokens: lowercase words/numbers, without punctuation and common question words."""
    return [t for t in re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


class HybridRetriever:
    def __init__(self):
        chunks = json.loads(CHUNKS_FILE.read_text())
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.docs = [Document(page_content=c["text"], metadata={"chunk_id": c["chunk_id"], "guideline": c["guideline"]})
                     for c in chunks]
        embeddings = embedding_model()
        self.faiss = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        self.ensembles = {f: self._build(f) for f in FILTERS}

    def _build(self, filter_name):
        """One BM25 + FAISS pair per filter option, so filtering happens before ranking."""
        guideline = None if filter_name == FILTERS[0] else filter_name.removesuffix(" only")
        docs = [d for d in self.docs if guideline in (None, d.metadata["guideline"])]
        bm25 = BM25Retriever.from_documents(docs, k=PER_RETRIEVER_K, preprocess_func=tokenize)
        search_kwargs = {"k": PER_RETRIEVER_K}
        if guideline:  # fetch_k = all chunks so the metadata filter never runs out of candidates
            search_kwargs.update(filter={"guideline": guideline}, fetch_k=len(self.docs))
        semantic = self.faiss.as_retriever(search_kwargs=search_kwargs)
        return EnsembleRetriever(retrievers=[bm25, semantic], weights=WEIGHTS, id_key="chunk_id")

    def search(self, question, filter_name=FILTERS[0], k=FINAL_K):
        """Return the top chunks for one query as dicts with text, metadata, found_by and rrf_score."""
        ensemble = self.ensembles[filter_name]
        fused = ensemble.invoke(question)[:k]

        # EnsembleRetriever does not return scores, so recompute RRF from each retriever's ranking.
        keyword, semantic = (r.invoke(question) for r in ensemble.retrievers)
        ranks = [{d.metadata["chunk_id"]: i + 1 for i, d in enumerate(lst)} for lst in (keyword, semantic)]
        results = []
        for doc in fused:
            cid = doc.metadata["chunk_id"]
            hits = [cid in r for r in ranks]
            score = sum(w / (RRF_C + r[cid]) for w, r in zip(WEIGHTS, ranks) if cid in r)
            found_by = "both" if all(hits) else "keyword" if hits[0] else "semantic"
            results.append({**self.chunks[cid], "found_by": found_by,
                            "keyword_rank": ranks[0].get(cid), "semantic_rank": ranks[1].get(cid),
                            "rrf_score": round(score, 5)})
        return results

    def search_many(self, queries, filter_name=FILTERS[0]):
        """Search several queries (the question + focused rewrites) and fuse them with RRF.
        "Both guidelines" fills separate ADA and ICMR slots; a single guideline gets all TOTAL_K."""
        if filter_name == FILTERS[0]:
            targets = [(f"{g} only", n) for g, n in BOTH_SLOTS.items()]
        else:
            targets = [(filter_name, TOTAL_K)]
        results = []
        for target, slots in targets:
            fused = {}
            for query in queries:
                for rank, chunk in enumerate(self.search(query, target), 1):
                    best = fused.setdefault(chunk["chunk_id"], {**chunk, "rrf_score": 0.0, "queries": []})
                    best["rrf_score"] += 1 / (RRF_C + rank)
                    best["queries"].append(query)
                    if best["found_by"] != chunk["found_by"]:
                        best["found_by"] = "both"
                    for key in ("keyword_rank", "semantic_rank"):
                        ranks = [r for r in (best[key], chunk[key]) if r]
                        best[key] = min(ranks) if ranks else None
            top = sorted(fused.values(), key=lambda c: -c["rrf_score"])[:slots]
            results += [{**c, "rrf_score": round(c["rrf_score"], 5)} for c in top]
        return results
