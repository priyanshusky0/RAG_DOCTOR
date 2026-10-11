"""Question -> search queries -> hybrid retrieval -> LLM answer -> quote verification.
Used by app.py and run_tests.py."""
import time

from llm import LLM, NOT_FOUND
from retrieval import FILTERS, HybridRetriever
from verify import verify_citation


class Assistant:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = LLM()

    def ask(self, question, filter_name=FILTERS[0]):
        t0 = time.perf_counter()
        # A patient description mixes several issues; focused queries find the matching recommendations.
        rewrites, rewrite_tokens = self.llm.search_queries(question)
        queries = [question] + [q for q in rewrites if q.lower() != question.lower()]
        t1 = time.perf_counter()
        chunks = self.retriever.search_many(queries, filter_name)
        t2 = time.perf_counter()
        answer, answer_tokens = self.llm.answer(question, chunks)
        t3 = time.perf_counter()

        for rec in answer["recommendations"]:
            rec["verification"] = verify_citation(rec, chunks)
        if answer["not_found"] and not answer["recommendations"]:
            answer["answer_summary"] = NOT_FOUND

        tokens = {k: rewrite_tokens.get(k, 0) + answer_tokens.get(k, 0) for k in ("prompt", "completion", "total")}
        return {
            "question": question,
            "filter": filter_name,
            "queries": queries,
            "answer": answer,
            "retrieved": chunks,
            "timing": {"rewrite_s": round(t1 - t0, 2), "retrieval_s": round(t2 - t1, 2),
                       "llm_s": round(t3 - t2, 2), "total_s": round(time.perf_counter() - t0, 2)},
            "tokens": tokens,
            "model": self.llm.model,
        }
