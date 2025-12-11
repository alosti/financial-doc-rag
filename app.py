"""
Streamlit interface for Financial Document RAG system.
Beautiful demo for portfolio presentation.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.complete_rag_system import CompleteRAGSystem
from rag.providers import AnthropicProvider, OpenAIProvider

# Page config
st.set_page_config(
    page_title="Financial RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .source-card {
        background: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system(provider_type: str, api_key: str, model: str):
    """Initialize RAG system with caching."""
    if provider_type == "Anthropic":
        provider = AnthropicProvider(api_key=api_key, model=model)
    else:
        provider = OpenAIProvider(api_key=api_key, model=model)

    return CompleteRAGSystem(llm_provider=provider)


def main():
    # Header
    load_dotenv()

    st.markdown('<h1 class="main-header">📊 Financial Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("""
    Sistema RAG (Retrieval Augmented Generation) per l'analisi di documenti finanziari.
    Carica i tuoi PDF e fai domande con risposte basate sul contenuto effettivo.
    """)

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configurazione")

        # Provider selection
        provider_type = st.selectbox(
            "LLM Provider",
            ["Anthropic", "OpenAI"],
            help="Seleziona il provider LLM"
        )

        # Model selection
        if provider_type == "Anthropic":
            model_options = {
                "Claude Sonnet 4": "claude-sonnet-4-20250514",
                "Claude Sonnet 3.5": "claude-sonnet-3-5-20241022",
                "Claude Haiku 3.5": "claude-haiku-3-5-20241022"
            }
        else:
            model_options = {
                "GPT-4o": "gpt-4o",
                "GPT-4o Mini": "gpt-4o-mini",
                "GPT-4 Turbo": "gpt-4-turbo"
            }

        selected_model = st.selectbox(
            "Modello",
            list(model_options.keys())
        )
        model = model_options[selected_model]

        # API Key
        api_key_env = "ANTHROPIC_API_KEY" if provider_type == "Anthropic" else "OPENAI_API_KEY"
        api_key = st.text_input(
            f"API Key ({provider_type})",
            type="password",
            value=os.getenv(api_key_env, ""),
            help=f"La tua {provider_type} API key"
        )

        st.divider()

        # RAG Parameters
        st.subheader("Parametri RAG")

        k_results = st.slider(
            "Numero di chunk da recuperare",
            min_value=1,
            max_value=10,
            value=5,
            help="Quanti chunk del documento usare come contesto"
        )

        min_score = st.slider(
            "Soglia similarità minima",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Score minimo per includere un chunk (0-1)"
        )

        enable_streaming = st.checkbox(
            "Abilita streaming",
            value=True,
            help="Mostra la risposta man mano che viene generata"
        )

        st.divider()

        # System stats
        if 'rag_system' in st.session_state and st.session_state.rag_system:
            st.subheader("📈 Statistiche Sistema")
            total_docs = st.session_state.rag_system.vector_store.index.ntotal
            st.metric("Chunk indicizzati", total_docs)

            if 'total_cost' in st.session_state:
                st.metric("Costo totale", f"${st.session_state.total_cost:.4f}")

            if 'total_queries' in st.session_state:
                st.metric("Query totali", st.session_state.total_queries)

    # Main area - Two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Caricamento Documenti")

        # File uploader
        uploaded_files = st.file_uploader(
            "Carica PDF",
            type=['pdf'],
            accept_multiple_files=True,
            help="Carica uno o più file PDF da indicizzare"
        )

        if uploaded_files and st.button("🚀 Indicizza Documenti", type="primary"):
            if not api_key:
                st.error(f"⚠️ Inserisci la tua {provider_type} API key nella sidebar!")
            else:
                with st.spinner("Inizializzazione sistema RAG..."):
                    # Initialize RAG system
                    rag_system = initialize_rag_system(provider_type, api_key, model)

                    # Save temporary files and index
                    pdf_paths = []
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for uploaded_file in uploaded_files:
                            file_path = Path(tmpdir) / uploaded_file.name
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            pdf_paths.append(str(file_path))

                        # Index documents
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, pdf_path in enumerate(pdf_paths):
                            status_text.text(f"Indicizzazione {Path(pdf_path).name}...")
                            progress_bar.progress((i + 1) / len(pdf_paths))

                        rag_system.index_documents(pdf_paths)

                    # Store in session state
                    st.session_state.rag_system = rag_system
                    st.session_state.total_cost = 0.0
                    st.session_state.total_queries = 0

                    st.success(f"✅ {len(uploaded_files)} documenti indicizzati con successo!")
                    st.rerun()

        # Show indexed files
        if 'rag_system' in st.session_state and st.session_state.rag_system:
            st.divider()
            st.subheader("📚 Documenti Indicizzati")

            # Extract unique sources from metadata
            if st.session_state.rag_system.vector_store.metadata:
                sources = set(
                    m['source']
                    for m in st.session_state.rag_system.vector_store.metadata
                )
                for source in sources:
                    st.text(f"📄 {source}")

    with col2:
        st.header("💬 Fai una Domanda")

        if 'rag_system' not in st.session_state or not st.session_state.rag_system:
            st.info("👈 Carica prima dei documenti PDF nella colonna a sinistra")
        else:
            # Query input
            query = st.text_area(
                "La tua domanda",
                placeholder="Es: Qual è stata la crescita dei ricavi nel Q4 2023?",
                height=100
            )

            if st.button("🔍 Cerca Risposta", type="primary", disabled=not query):
                with st.spinner("Ricerca in corso..."):
                    # Query the system
                    response = st.session_state.rag_system.query(
                        question=query,
                        k=k_results,
                        min_score=min_score,
                        stream=False  # Handle streaming separately for Streamlit
                    )

                    # Update stats
                    st.session_state.total_cost += response.cost_usd
                    st.session_state.total_queries += 1

                    # Display response
                    st.divider()
                    st.subheader("✨ Risposta")
                    st.markdown(response.answer)

                    # Display metadata
                    st.divider()
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Token utilizzati", response.tokens_used)
                    with cols[1]:
                        st.metric("Costo query", f"${response.cost_usd:.6f}")
                    with cols[2]:
                        st.metric("Fonti utilizzate", len(response.sources))

                    # Display sources
                    if response.sources:
                        st.divider()
                        st.subheader("📖 Fonti")

                        for i, source in enumerate(response.sources, 1):
                            with st.expander(
                                    f"Fonte {i}: {source.metadata.get('source', 'unknown')} - Pagina {source.metadata.get('page', '-')} "
                                    f"(Score: {source.score:.4f})"
                            ):
                                st.text(source.chunk_text)

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 2rem;'>
        <p><strong>Financial Document RAG System</strong></p>
        <p>Powered by SentenceTransformers, FAISS, and {provider}</p>
        <p>Built by Alessandro Osti | <a href='https://github.com/alosti/financial-doc-rag'>GitHub</a></p>
    </div>
    """.format(provider=provider_type), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
