"""Run the test questions end to end and print the answers:  python run_tests.py
Full results are saved to test_results.json."""
import json
import time

from pipeline import Assistant

QUESTIONS = [
    "Patient with T2DM and established cardiovascular disease: which drug class is preferred after metformin?",
    "Patient with eGFR 25: can metformin be continued?",
    "When should insulin be started in a newly diagnosed patient?",
    "What do the guidelines say about SGLT2 inhibitors in heart failure?",
    "What is the recommended dose of pembrolizumab for metastatic melanoma?",
]


def show(result):
    a = result["answer"]
    print(f"\n{'=' * 100}\nQ: {result['question']}\n{'=' * 100}")
    print("Searched for:", " | ".join(result["queries"]))
    print("Retrieved:", ", ".join(f"{c['guideline']} {c['document'][:20]} p{c['page']} ({c['found_by']})"
                                   for c in result["retrieved"]))
    print(f"\nSUMMARY (not_found={a['not_found']}): {a['answer_summary']}")
    for i, rec in enumerate(a["recommendations"], 1):
        v = rec["verification"]
        mark = "✔ verified on page" if v["verified"] else f"⚠ not verified ({v['reason']})"
        print(f"\n {i}. {rec.get('text')}\n    [{rec.get('guideline')}, {rec.get('document')}, p.{rec.get('page')}]"
              f"\n    \"{rec.get('quote')}\"\n    {mark} [{v['score']}%]")
    for d in a["differences_between_guidelines"]:
        print(f"\n DIFFERENCE: {d}")
    t, tok = result["timing"], result["tokens"]
    print(f"\n {t['total_s']} s (LLM {t['llm_s']} s) · tokens {tok}")


if __name__ == "__main__":
    assistant = Assistant()
    results = []
    for i, q in enumerate(QUESTIONS):
        if i:
            time.sleep(13)  # Gemini free tier allows 5 requests per minute
        results.append(assistant.ask(q))
    for r in results:
        show(r)
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
