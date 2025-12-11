from rag.embedding_and_vectorstore import EmbeddingGenerator, VectorStore


def test_embedding_and_search():
    """Test completo: embeddings + ricerca"""

    print("\n" + "=" * 60)
    print("TEST EMBEDDINGS & VECTOR STORE")
    print("=" * 60 + "\n")

    # 1. Crea alcuni documenti di test
    test_docs = [
        "Revenue in Q4 2023 was €50M, up 15% from Q4 2022.",
        "The company expanded operations in Asia Pacific region.",
        "Operating expenses decreased by 8% year-over-year.",
        "New product launches contributed significantly to growth.",
        "Customer satisfaction ratings reached all-time high of 94%."
    ]

    test_metadata = [
        {"source": "annual_report.pdf", "page": 5, "section": "financials"},
        {"source": "annual_report.pdf", "page": 12, "section": "operations"},
        {"source": "annual_report.pdf", "page": 7, "section": "financials"},
        {"source": "annual_report.pdf", "page": 15, "section": "products"},
        {"source": "annual_report.pdf", "page": 20, "section": "customer_feedback"}
    ]

    # 2. Genera embeddings
    print("Generazione embeddings...")
    generator = EmbeddingGenerator()
    embeddings = generator.generate(test_docs, show_progress=True)
    print(f"✓ Embeddings generati: shape {embeddings.shape}\n")

    # 3. Crea vector store e aggiungi documenti
    print("Creazione vector store...")
    store = VectorStore(dimension=generator.dimension)
    store.add_documents(test_docs, embeddings, test_metadata)
    print(f"✓ Vector store creato: {store.index.ntotal} documenti\n")

    # 4. Test ricerca
    queries = [
        "What was the revenue growth?",
        "Tell me about customer satisfaction",
        "Did the company expand internationally?"
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")
        print('=' * 60)

        query_emb = generator.generate_query_embedding(query)
        results = store.search(query_emb, k=3)

        for result in results:
            print(f"\n[Rank {result.rank}] Score: {result.score:.4f}")
            print(f"Text: {result.chunk_text}")
            print(f"Metadata: page {result.metadata['page']}, section '{result.metadata['section']}'")

    # 5. Test save/load
    print(f"\n\n{'=' * 60}")
    print("TEST SAVE/LOAD")
    print('=' * 60)

    store.save("test_vectorstore")
    print("✓ Vector store salvato in 'test_vectorstore/'\n")

    loaded_store = VectorStore.load("test_vectorstore")
    print(f"✓ Vector store caricato: {loaded_store.index.ntotal} documenti\n")

    # Verifica che funzioni dopo il load
    query_emb = generator.generate_query_embedding("revenue")
    results = loaded_store.search(query_emb, k=1)
    print(f"Test ricerca post-load: '{results[0].chunk_text[:60]}...'")
    print(f"Score: {results[0].score:.4f}\n")

    print("=" * 60)
    print("✓ TUTTI I TEST COMPLETATI")
    print("=" * 60)


def test_embedding_and_search_italiano():
    """Test completo con documenti finanziari in italiano"""

    print("\n" + "=" * 60)
    print("TEST EMBEDDINGS & VECTOR STORE - DOCUMENTI ITALIANI")
    print("=" * 60 + "\n")

    # Simulate chunks from Italian financial documents
    test_docs = [
        "I ricavi del Q4 2023 sono stati pari a €50M, in crescita del 15% rispetto al Q4 2022.",
        "L'azienda ha espanso le operazioni nella regione Asia-Pacifico, aprendo tre nuove sedi.",
        "Le spese operative sono diminuite dell'8% su base annua grazie a ottimizzazioni di processo.",
        "Il lancio di nuovi prodotti ha contribuito significativamente alla crescita del fatturato.",
        "I livelli di soddisfazione dei clienti hanno raggiunto il massimo storico del 94%.",
        "Il margine EBITDA si è attestato al 22%, superando le previsioni degli analisti.",
        "Gli investimenti in R&D sono aumentati del 12%, con focus su intelligenza artificiale.",
        "Il debito netto è stato ridotto di €15M, migliorando la solidità patrimoniale.",
        "Il board ha approvato un dividendo di €1.20 per azione, in aumento del 10%.",
        "La strategia ESG ha portato a una riduzione del 18% delle emissioni di CO2."
    ]

    test_metadata = [
        {"source": "relazione_annuale.pdf", "page": 5, "section": "risultati_finanziari"},
        {"source": "relazione_annuale.pdf", "page": 12, "section": "espansione_geografica"},
        {"source": "relazione_annuale.pdf", "page": 7, "section": "efficienza_operativa"},
        {"source": "relazione_annuale.pdf", "page": 15, "section": "innovazione_prodotto"},
        {"source": "relazione_annuale.pdf", "page": 20, "section": "customer_satisfaction"},
        {"source": "relazione_annuale.pdf", "page": 6, "section": "margini"},
        {"source": "relazione_annuale.pdf", "page": 18, "section": "ricerca_sviluppo"},
        {"source": "relazione_annuale.pdf", "page": 8, "section": "debito"},
        {"source": "relazione_annuale.pdf", "page": 25, "section": "dividendi"},
        {"source": "relazione_annuale.pdf", "page": 22, "section": "sostenibilita"}
    ]

    # Generate embeddings
    print("Generazione embeddings...")
    generator = EmbeddingGenerator()
    embeddings = generator.generate(test_docs, show_progress=True)
    print(f"✓ Embeddings generati: shape {embeddings.shape}\n")

    # Create vector store and add documents
    print("Creazione vector store...")
    store = VectorStore(dimension=generator.dimension)
    store.add_documents(test_docs, embeddings, test_metadata)
    print(f"✓ Vector store creato: {store.index.ntotal} documenti\n")

    # Test queries in Italian
    queries = [
        "Qual è stata la crescita dei ricavi?",
        "Dimmi qualcosa sulla soddisfazione dei clienti",
        "L'azienda si è espansa a livello internazionale?",
        "Quali sono i margini di profitto?",
        "Ci sono investimenti in innovazione?",
        "Come stanno i debiti dell'azienda?"
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")
        print('=' * 60)

        query_emb = generator.generate_query_embedding(query)
        results = store.search(query_emb, k=3)

        for result in results:
            print(f"\n[Rank {result.rank}] Score: {result.score:.4f}")
            print(f"Testo: {result.chunk_text}")
            print(f"Metadata: pagina {result.metadata['page']}, sezione '{result.metadata['section']}'")

    # Test save/load
    print(f"\n\n{'=' * 60}")
    print("TEST SAVE/LOAD")
    print('=' * 60)

    store.save("test_vectorstore_italiano")
    print("✓ Vector store salvato in 'test_vectorstore_italiano/'\n")

    loaded_store = VectorStore.load("test_vectorstore_italiano")
    print(f"✓ Vector store caricato: {loaded_store.index.ntotal} documenti\n")

    # Verify it works after loading
    query_emb = generator.generate_query_embedding("ricavi")
    results = loaded_store.search(query_emb, k=1)
    print(f"Test ricerca post-load: '{results[0].chunk_text[:60]}...'")
    print(f"Score: {results[0].score:.4f}\n")

    print("=" * 60)
    print("✓ TUTTI I TEST COMPLETATI CON SUCCESSO")
    print("=" * 60)

    # Print some statistics
    print(f"\n{'=' * 60}")
    print("STATISTICHE")
    print('=' * 60)
    print(f"Documenti indicizzati: {store.index.ntotal}")
    print(f"Dimensione embeddings: {generator.dimension}")
    print(f"Memoria occupata (stima): ~{store.index.ntotal * generator.dimension * 4 / 1024:.2f} KB")
    print(f"Query testate: {len(queries)}")


if __name__ == "__main__":
    # test_embedding_and_search()
    test_embedding_and_search_italiano()
