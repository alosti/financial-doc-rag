# Financial Document RAG

Retrieval-Augmented Generation system for analyzing financial documents 
(annual reports, earnings, etc.) using embeddings and vector search.

Built with: SentenceTransformers, FAISS, FastAPI

## Architecture Choices

### Vector Store: FAISS
- **Why FAISS**: For this scope (single-user, <1000 docs), in-memory FAISS 
  provides optimal performance with minimal overhead
- **Trade-off**: If scaling to enterprise (multi-tenant, millions of docs), 
  migration path to Qdrant/Pinecone is straightforward
- **RAM footprint**: ~100MB per 10k documents (384-dim embeddings)

### Text Splitting: LangChain RecursiveCharacterTextSplitter
- **Why not custom**: Mature, tested solution for text chunking
- **Minimal dependency**: Only `langchain-text-splitters` package, not full framework

### Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- **Why local**: Privacy, zero cost, low latency
- **Quality**: Sufficient for 90% use cases, upgrade path to OpenAI if needed


## License

MIT License - see [LICENSE](LICENSE) file for details
