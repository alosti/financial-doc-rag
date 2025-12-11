import os

from dotenv import load_dotenv

from rag.complete_rag_system import CompleteRAGSystem
from rag.providers import AnthropicProvider


def test_complete_rag_system():
    """Test complete RAG system end-to-end."""

    print("\n" + "=" * 60)
    print("TEST COMPLETE RAG SYSTEM")
    print("=" * 60 + "\n")

    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("!!!  Please set the ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        return

    llm_provider = AnthropicProvider(
        api_key=api_key,
        model="claude-sonnet-4-20250514"
    )

    # Option 2: Use OpenAI (uncomment to use)
    # from rag.providers import OpenAIProvider
    # api_key = os.getenv("OPENAI_API_KEY")
    # llm_provider = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")

    # Create RAG system
    print("Initializing RAG system...")
    rag = CompleteRAGSystem(llm_provider=llm_provider)
    print("✓ RAG system initialized\n")

    # Simulate indexed documents (in real scenario, use rag.index_documents())
    print("Simulating indexed documents...")

    test_docs = [
        "I ricavi del Q4 2023 sono stati pari a €50M, in crescita del 15% rispetto al Q4 2022.",
        "L'azienda ha espanso le operazioni nella regione Asia-Pacifico, aprendo tre nuove sedi a Singapore, Tokyo e Sydney.",
        "Le spese operative sono diminuite dell'8% su base annua grazie a ottimizzazioni di processo e automazione.",
        "Il lancio di nuovi prodotti ha contribuito significativamente alla crescita del fatturato, con particolare successo nella linea premium.",
        "I livelli di soddisfazione dei clienti hanno raggiunto il massimo storico del 94%, superando l'obiettivo dell'anno.",
    ]

    test_metadata = [
        {"source": "relazione_annuale_2023.pdf", "page": 5, "section": "risultati_finanziari"},
        {"source": "relazione_annuale_2023.pdf", "page": 12, "section": "espansione_geografica"},
        {"source": "relazione_annuale_2023.pdf", "page": 7, "section": "efficienza_operativa"},
        {"source": "relazione_annuale_2023.pdf", "page": 15, "section": "innovazione_prodotto"},
        {"source": "relazione_annuale_2023.pdf", "page": 20, "section": "customer_satisfaction"},
    ]

    # Generate embeddings and add to vector store
    embeddings = rag.embedding_generator.generate(test_docs, show_progress=False)
    rag.vector_store.add_documents(test_docs, embeddings, test_metadata)
    print(f"✓ {len(test_docs)} documents indexed\n")

    # Test queries
    queries = [
        "Qual è stata la crescita dei ricavi nel Q4 2023?",
        "Dove si è espansa l'azienda?",
        "Come sta la soddisfazione dei clienti?",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60 + "\n")

        # Query with streaming
        response = rag.query(
            question=query,
            k=3,
            min_score=0.5,
            stream=True  # Enable streaming for better UX
        )

        # Print metadata
        print(f"\n{'─' * 60}")
        print(f"Model: {response.model}")
        print(f"Tokens used: {response.tokens_used}")
        print(f"Cost: ${response.cost_usd:.6f}")
        print(f"Sources used: {len(response.sources)}")

        if response.sources:
            print("\nSOURCES:")
            for i, source in enumerate(response.sources, 1):
                print(f"  [{i}] {source.metadata['source']}, page {source.metadata['page']} "
                      f"(score: {source.score:.4f})")

    # Test save/load
    print(f"\n\n{'=' * 60}")
    print("TEST SAVE/LOAD")
    print('=' * 60)

    save_dir = "rag_system_test"
    rag.save(save_dir)
    print(f"✓ System saved to '{save_dir}/'\n")

    # Load
    loaded_rag = CompleteRAGSystem.load(save_dir, llm_provider=llm_provider)
    print(f"✓ System loaded ({loaded_rag.vector_store.index.ntotal} documents)\n")

    print("=" * 60)
    print("✓ ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    test_complete_rag_system()
