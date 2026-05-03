"""
app.py
──────
Main Streamlit entry point for the RAG Chatbot.

Responsibilities:
  • Page config & session-state initialisation
  • Render the chat-focused sidebar → receive cfg dict
  • Build (and cache) the RAG / CoRAG / Self-RAG chain
  • Render the chat view

Document management (upload, embed, rebuild DB) lives entirely in
pages/chunk-manager.py. The embedding provider/model used here is
read directly from `settings` to stay consistent with the DB that
Chunk Manager built.
"""

from __future__ import annotations

import os
import warnings

from core.hybrid_retriever import HybridRetriever
from core.vectorstore import get_all_documents

warnings.filterwarnings("ignore", message="accessing __path__ from.*zoedepth")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import settings, get_prompt
from core import (
    CoRAGChain,
    SelfRAGChain,
    build_rag_chain,
    embedding_factory,
    get_retriever,
    llm_factory,
    load_vectorstore,
)
from ui import (
    clear_chat_button,
    model_badge,
    render_chat,
    render_sidebar,
    status_banner,
)

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Initialise message history once per browser session
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _gemini_ready() -> bool:
    """True when a Gemini API key is available in env or session state."""
    return bool(
        st.session_state.get("gemini_api_key_input", "").strip()
        or settings.google_api_key.strip()
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
cfg = render_sidebar()


# ── Chain factory (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang khởi tạo chain...")
def get_chain(
    # LLM knobs — sidebar-driven
    llm_provider:             str,
    llm_model:                str,
    temperature:              float,
    # Retriever knobs — sidebar-driven
    prompt_mode:              str,
    retriever_type:           str,   # "vector" | "hybrid"
    retriever_k:              int,
    score_threshold:          float,
    # Chain-type knobs — sidebar-driven
    chain_type:               str,
    corag_max_iter:           int,
    selfrag_max_retrieval:    int,
    selfrag_max_generation:   int,
    selfrag_quality_threshold: int,
    # DB path — from settings (not user-configurable at runtime)
    db_path:                  str,
    # Embedding config — from settings, must match how the DB was built
    embed_provider:           str,
    embed_model:              str,
):
    """
    Builds and returns (chain, retriever).

    Why embedding params are sourced from settings, not the sidebar:
      The FAISS index was built with a specific embedding model (configured in
      Chunk Manager). Using a different model here would produce vectors in a
      different space, making retrieval meaningless. By reading from settings
      we guarantee the same model is always used for both build and query.

    Returns (None, None) when the DB does not exist yet.
    """
    embeddings = embedding_factory(embed_provider, embed_model)
    vs         = load_vectorstore(db_path, embeddings)

    if vs is None:
        return None, None

    llm              = llm_factory(llm_provider, llm_model, temperature)
    vector_retriever = get_retriever(vs, k=retriever_k, score_threshold=score_threshold)

    # ── Choose retriever based on user selection ─────────────────────────
    if retriever_type == "hybrid":
        documents = get_all_documents(vs)
        retriever = HybridRetriever(vector_retriever, documents, k=retriever_k)
    else:
        # Pure vector retriever: wrap to expose .invoke() uniformly
        retriever = vector_retriever

    if chain_type == "corag":
        chain = CoRAGChain(
            llm=llm,
            retriever=retriever,
            prompt_mode=prompt_mode,
            max_iterations=corag_max_iter,
        )
    elif chain_type == "selfrag":
        chain = SelfRAGChain(
            llm=llm,
            retriever=retriever,
            prompt_mode=prompt_mode,
            max_retrieval_attempts=selfrag_max_retrieval,
            max_generation_attempts=selfrag_max_generation,
            quality_threshold=selfrag_quality_threshold,
        )
    else:
        chain = build_rag_chain(llm, retriever, get_prompt(prompt_mode))

    return chain, retriever


# ── Build chain ────────────────────────────────────────────────────────────────
rag_chain     = None
rag_retriever = None

if _gemini_ready() or cfg["llm_provider"] == "ollama":
    _result = get_chain(
        # LLM
        llm_provider=cfg["llm_provider"],
        llm_model=cfg["llm_model"],
        temperature=cfg["temperature"],
        # Retriever
        prompt_mode=cfg["prompt_mode"],
        retriever_type=cfg["retriever_type"],
        retriever_k=cfg["retriever_k"],
        score_threshold=cfg["score_threshold"],
        # Chain type
        chain_type=cfg["chain_type"],
        corag_max_iter=cfg["corag_max_iter"],
        selfrag_max_retrieval=cfg["selfrag_max_retrieval"],
        selfrag_max_generation=cfg["selfrag_max_generation"],
        selfrag_quality_threshold=cfg["selfrag_quality_threshold"],
        # DB (immutable at chat runtime)
        db_path=settings.faiss_db_folder_path,
        # Embedding — sourced from settings, not from cfg
        embed_provider=settings.default_embedding_provider,
        embed_model=settings.default_embedding_model,
    )
    if _result is not None:
        rag_chain, rag_retriever = _result


# ── Page layout ────────────────────────────────────────────────────────────────
col_title, col_clear = st.columns([5, 1])
with col_title:
    st.title("🤖 RAG Chatbot")
with col_clear:
    clear_chat_button()

model_badge(cfg["llm_provider"], cfg["llm_model"])
st.divider()

# Guard: Gemini selected but no key entered
if cfg["llm_provider"] == "gemini" and not _gemini_ready():
    st.warning("🔑 Nhập Google API Key trong sidebar để bắt đầu chat với Gemini.")
    st.stop()

# Guard: DB not built yet
status_banner(db_exists=rag_chain is not None)

# ── Chat ───────────────────────────────────────────────────────────────────────
render_chat(
    rag_chain=rag_chain,
    provider=cfg["llm_provider"],
    model=cfg["llm_model"],
    retriever=rag_retriever,
    chain_type=cfg["chain_type"],
    memory_window=cfg["memory_window"],
)