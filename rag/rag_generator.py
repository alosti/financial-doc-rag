"""
RAG Generator: combines retrieval with LLM generation.
Uses provider abstraction for LLM flexibility.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Generator, Dict

from .embedding_and_vectorstore import SearchResult
from .providers import BaseLLMProvider, Message

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete RAG response with answer and sources."""
    answer: str
    sources: List[SearchResult]
    tokens_used: int
    cost_usd: float
    model: str


class RAGGenerator:
    """
    Generates answers using retrieved context and LLM.
    Includes source citations in responses.
    """

    # Default system prompt for financial Q&A
    DEFAULT_SYSTEM_PROMPT = """Sei un assistente esperto in analisi finanziaria e documenti aziendali.

Il tuo compito è rispondere alle domande dell'utente basandoti ESCLUSIVAMENTE sui documenti forniti come contesto.

REGOLE IMPORTANTI:
1. Rispondi SOLO usando informazioni presenti nel contesto fornito
2. Se la risposta non è nel contesto, dì chiaramente "Non ho trovato questa informazione nei documenti forniti"
3. Cita sempre le fonti usando il formato [Fonte: nome_documento.pdf, pagina X]
4. Se ci sono numeri o dati specifici, citali esattamente come appaiono nei documenti
5. Rispondi in modo conciso ma completo
6. Usa un tono professionale ma accessibile

FORMATO CITAZIONI:
Quando menzioni un'informazione, aggiungi immediatamente la citazione:
"I ricavi del Q4 2023 sono stati €50M [Fonte: relazione_annuale.pdf, pagina 5]"
"""

    def __init__(
            self,
            llm_provider: BaseLLMProvider,
            system_prompt: Optional[str] = None,
            temperature: float = 0.3,  # Low temperature for factual responses
            max_tokens: int = 2000
    ):
        """
        Args:
            llm_provider: LLM provider instance (Anthropic, OpenAI, etc)
            system_prompt: Custom system prompt (uses default if None)
            temperature: LLM temperature (0.0-1.0, lower = more focused)
            max_tokens: Maximum response length
        """
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"RAG Generator initialized with {llm_provider.__class__.__name__}")

    def _build_context(self, search_results: List[SearchResult]) -> str:
        """
        Build context string from search results.

        Args:
            search_results: List of retrieved chunks

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(search_results, 1):
            source = result.metadata.get("source", "unknown")
            page = result.metadata.get("page", "?")

            context_parts.append(
                f"[DOCUMENTO {i}]\n"
                f"Fonte: {source}, Pagina: {page}\n"
                f"Contenuto: {result.chunk_text}\n"
            )

        return "\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build user prompt with query and context.

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        return f"""CONTESTO:
{context}

DOMANDA DELL'UTENTE:
{query}

Rispondi alla domanda basandoti SOLO sul contesto fornito. Includi le citazioni delle fonti."""

    def generate(
            self,
            query: str,
            search_results: List[SearchResult],
            stream: bool = False
    ) -> RAGResponse:
        """
        Generate answer for query using retrieved context.

        Args:
            query: User question
            search_results: Retrieved chunks from vector search
            stream: If True, print response as it's generated (for UX)

        Returns:
            RAGResponse with answer and metadata
        """
        if not search_results:
            logger.warning("No search results provided, returning empty response")
            return RAGResponse(
                answer="Non ho trovato documenti rilevanti per rispondere a questa domanda.",
                sources=[],
                tokens_used=0,
                cost_usd=0.0,
                model=self.llm_provider.model
            )

        # Build context from search results
        context = self._build_context(search_results)
        logger.info(f"Context built from {len(search_results)} chunks")

        # Build messages
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=self._build_prompt(query, context))
        ]

        # Generate response
        if stream:
            logger.info("Generating streaming response...")
            print("\nRisposta: ", end="", flush=True)

            full_response = ""
            stream_generator = self.llm_provider.stream(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Consume the generator
            for chunk in stream_generator:
                print(chunk, end="", flush=True)
                full_response += chunk

            print("\n")  # Newline after streaming

            # Get the completion response from the provider's attribute
            final_response = self.llm_provider._last_completion_response
        else:
            logger.info("Generating complete response...")
            final_response = self.llm_provider.complete(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        logger.info(
            f"Response generated: {final_response.tokens_total} tokens, "
            f"${final_response.cost_usd:.6f}"
        )

        return RAGResponse(
            answer=final_response.content,
            sources=search_results,
            tokens_used=final_response.tokens_total,
            cost_usd=final_response.cost_usd,
            model=final_response.model
        )

    def generate_with_threshold(
            self,
            query: str,
            search_results: List[SearchResult],
            min_score: float = 0.5,
            stream: bool = False
    ) -> RAGResponse:
        """
        Generate answer filtering search results by minimum score.

        Args:
            query: User question
            search_results: Retrieved chunks
            min_score: Minimum similarity score to include (0.0-1.0)
            stream: If True, stream response

        Returns:
            RAGResponse
        """
        # Filter by score
        filtered_results = [
            r for r in search_results
            if r.score >= min_score
        ]

        if not filtered_results:
            logger.warning(
                f"No results above threshold {min_score}. "
                f"Best score was {search_results[0].score:.4f}"
            )
            return RAGResponse(
                answer=f"Non ho trovato informazioni sufficientemente rilevanti per rispondere a questa domanda (soglia di similarità: {min_score}).",
                sources=[],
                tokens_used=0,
                cost_usd=0.0,
                model=self.llm_provider.model
            )

        logger.info(
            f"Using {len(filtered_results)}/{len(search_results)} results "
            f"above threshold {min_score}"
        )

        return self.generate(query, filtered_results, stream=stream)

    def generate_stream(
            self,
            query: str,
            search_results: List[SearchResult]
    ) -> Generator[Dict, None, None]:
        """
        Generate streaming answer for query using retrieved context.
        Yields dictionaries with streaming events.

        Args:
            query: User question
            search_results: Retrieved chunks from vector search

        Yields:
            Dictionaries with structure:
            - {"type": "token", "text": "..."}  # Text chunks as they arrive
            - {"type": "error", "message": "..."}  # If error occurs

        Returns:
            Generator yielding token dictionaries
        """
        try:
            if not search_results:
                logger.warning("No search results provided, returning empty response")
                yield {
                    "type": "token",
                    "text": "Non ho trovato documenti rilevanti per rispondere a questa domanda."
                }
                return

            # Build context from search results
            context = self._build_context(search_results)
            logger.info(f"Context built from {len(search_results)} chunks")

            # Build messages
            messages = [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=self._build_prompt(query, context))
            ]

            # Generate streaming response
            logger.info("Generating streaming response...")

            stream_generator = self.llm_provider.stream(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Stream tokens to caller
            for chunk in stream_generator:
                yield {
                    "type": "token",
                    "text": chunk
                }

            # Get final response metadata
            final_response = self.llm_provider._last_completion_response

            logger.info(
                f"Stream completed: {final_response.tokens_total} tokens, "
                f"${final_response.cost_usd:.6f}"
            )

        except Exception as e:
            logger.error(f"Error during streaming generation: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "message": f"Errore durante la generazione: {str(e)}"
            }
