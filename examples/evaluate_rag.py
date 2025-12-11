"""
Quick RAG evaluation - check if the system actually works
Run after indexing docs to see retrieval quality
"""

import json
import os

from dotenv import load_dotenv

from rag.complete_rag_system import CompleteRAGSystem
from rag.providers import AnthropicProvider


def test_rag(rag_system, queries):
    """Run queries and see what happens"""

    results = []
    total_cost = 0

    print("\n" + "=" * 60)
    print("RAG SYSTEM TEST")
    print("=" * 60 + "\n")

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {q}")
        print("-" * 60)

        resp = rag_system.query(question=q, k=5, min_score=0.5, stream=False)

        # Check quality
        best = max([s.score for s in resp.sources]) if resp.sources else 0
        avg = sum([s.score for s in resp.sources]) / len(resp.sources) if resp.sources else 0

        # Simple rating
        if best >= 0.7:
            rating = "GOOD"
            emoji = "✓"
        elif best >= 0.5:
            rating = "OK"
            emoji = "~"
        else:
            rating = "BAD"
            emoji = "✗"

        print(f"{emoji} {rating} - score: {best:.2f} (avg {avg:.2f})")
        print(f"Found {len(resp.sources)} sources, cost ${resp.cost_usd:.4f}")
        print(f"Answer: {resp.answer[:150]}...")

        if resp.sources:
            top = resp.sources[0]
            print(f"Top match: {top.metadata['source']} p.{top.metadata['page']}")

        total_cost += resp.cost_usd

        results.append({
            "query": q,
            "score": best,
            "avg_score": avg,
            "sources": len(resp.sources),
            "cost": resp.cost_usd
        })

    # Stats
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print('=' * 60)

    good = sum(1 for r in results if r['score'] >= 0.7)
    ok = sum(1 for r in results if 0.5 <= r['score'] < 0.7)
    bad = sum(1 for r in results if r['score'] < 0.5)

    print(f"\nQueries: {len(queries)}")
    print(f"  Good (>=0.7): {good}")
    print(f"  OK (0.5-0.7): {ok}")
    print(f"  Bad (<0.5): {bad}")

    avg_score = sum(r['score'] for r in results) / len(results)
    avg_sources = sum(r['sources'] for r in results) / len(results)

    print(f"\nAvg score: {avg_score:.2f}")
    print(f"Avg sources: {avg_sources:.1f}")
    print(f"Total cost: ${total_cost:.4f} (${total_cost / len(results):.4f} per query)")

    # What to do
    print(f"\n{'-' * 60}")
    if avg_score < 0.5:
        print("⚠ Low scores - try:")
        print("  - Reduce chunk_size (1000 -> 700)")
        print("  - Lower min_score threshold (0.5 -> 0.3)")
    elif avg_score >= 0.7:
        print("✓ Looks good, system working well")
    else:
        print("~ Decent, maybe tweak chunk_overlap")

    # Save for later
    with open("test_results.json", 'w') as f:
        json.dump({
            "avg_score": avg_score,
            "good": good,
            "ok": ok,
            "bad": bad,
            "total_cost": total_cost,
            "queries": results
        }, f, indent=2)

    print(f"\nSaved to test_results.json")
    print('=' * 60 + "\n")


# Test queries - change based on your docs
TEST_QUERIES = [
    "Qual è stata la crescita dei ricavi?",
    "Come stanno i margini?",
    "Ci sono espansioni geografiche?",
    "Investimenti in R&D?",
    "Posizione debitoria?",
    "Soddisfazione clienti?",
]

if __name__ == "__main__":
    # Load system
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Need ANTHROPIC_API_KEY env var")
        exit(1)

    # Assumes you already saved the system with rag.save("rag_system_test")
    print("Loading RAG system...")
    provider = AnthropicProvider(api_key=api_key, model="claude-sonnet-4-20250514")
    rag = CompleteRAGSystem.load("rag_system_test", llm_provider=provider)

    print(f"Loaded: {rag.vector_store.index.ntotal} chunks\n")

    # Run test
    test_rag(rag, TEST_QUERIES)
