"""
ui/sidebar.py
─────────────
Chat-focused sidebar. Responsibilities:
  • Google API key input
  • Chat history navigation (placeholder for multi-session support)
  • LLM provider / model / temperature
  • Chain type selector  (RAG · CoRAG · Self-RAG) + per-chain knobs
  • Prompt mode + retriever tuning (vector vs hybrid)
  • DB status indicator (read-only — management lives in Chunk Manager page)

All document upload, embedding selection, chunk settings, and
DB rebuild logic have been moved to pages/chunk-manager.py.
"""

from __future__ import annotations

import os
import streamlit as st

from config.settings import settings
from config.prompts import PROMPT_MODES
from models import GEMINI_LLM_MODELS, OLLAMA_LLM_MODELS

# ── Constants ──────────────────────────────────────────────────────────────────
PROVIDERS = ["gemini", "ollama"]

_CHAIN_LABELS: dict[str, str] = {
    "rag":     "🔗 RAG",
    "corag":   "🔄 CoRAG",
    "selfrag": "🪞 Self-RAG",
}


# ── Private helpers ────────────────────────────────────────────────────────────

def _has_gemini_key() -> bool:
    """True when a Gemini API key is available (from .env or manual input)."""
    return bool(
        st.session_state.get("gemini_api_key_input", "").strip()
        or settings.google_api_key.strip()
    )


def _apply_api_key() -> None:
    """on_change callback: push manually entered key into os.environ."""
    key = st.session_state.get("gemini_api_key_input", "").strip()
    if key:
        os.environ["GOOGLE_API_KEY"] = key


def _llm_model_selector(provider: str) -> str:
    """
    Renders a selectbox for the LLM model list of the chosen provider,
    plus a free-text override input for custom / unlisted model names.
    Returns the effective model string.
    """
    locked     = (provider == "gemini") and not _has_gemini_key()
    model_list = GEMINI_LLM_MODELS if provider == "gemini" else OLLAMA_LLM_MODELS

    selected = st.selectbox(
        "Model",
        options=model_list,
        key="llm_model",
        disabled=locked,
        help="Nhập Google API Key để chọn model Gemini." if locked else None,
    )
    custom = st.text_input(
        "Hoặc nhập model tùy chỉnh",
        value="",
        key="llm_model_custom",
        placeholder="Bỏ trống để dùng lựa chọn trên",
        disabled=locked,
    )
    return custom.strip() or selected


# ── Chat history section ───────────────────────────────────────────────────────
def _render_chat_history() -> None:
    """
    Renders the "🕒 Lịch sử Chat" section.

    Currently shows:
      • A "New conversation" button that clears the active session.
      • The active session as a highlighted card (when messages exist).
      • Greyed-out placeholder cards previewing the future multi-session UI.

    TODO: Replace placeholder cards with real persisted sessions once a
          session-storage backend (SQLite / Redis) is integrated.
    """
    st.markdown("#### 🕒 Lịch sử Chat")

    # ── "New conversation" button ────────────────────────────────────────────
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Active session card ──────────────────────────────────────────────────
    messages    = st.session_state.get("messages", [])
    first_user  = next((m["content"] for m in messages if m["role"] == "user"), None)
    turn_count  = sum(1 for m in messages if m["role"] == "user")

    active_title    = (first_user[:38] + "…" if first_user and len(first_user) > 38
                       else first_user or "Cuộc trò chuyện hiện tại")
    active_subtitle = f"{turn_count} tin nhắn" if turn_count else "Chưa có tin nhắn"

    st.markdown(
        f"""
        <div style="
            background  : rgba(79,142,247,0.12);
            border      : 1px solid rgba(79,142,247,0.35);
            border-left : 3px solid #4f8ef7;
            border-radius: 8px;
            padding     : 9px 12px;
            margin-bottom: 5px;
            color       : var(--text-color);
        ">
            <div style="
                font-size   : 0.82rem;
                font-weight : 600;
                white-space : nowrap;
                overflow    : hidden;
                text-overflow: ellipsis;
            ">💬 {active_title}</div>
            <div style="
                font-size  : 0.70rem;
                opacity    : 0.7;
                margin-top : 2px;
            ">{active_subtitle} · Phiên hiện tại</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Placeholder history cards (future multi-session preview) ─────────────
    _PLACEHOLDERS = [
        ("Phân tích báo cáo Q3 2024…",     "8 tin nhắn · hôm qua"),
        ("Tổng hợp tài liệu nghiên cứu…",  "5 tin nhắn · 2 ngày trước"),
        ("So sánh phương pháp RAG và…",    "12 tin nhắn · tuần trước"),
    ]
    for p_title, p_meta in _PLACEHOLDERS:
        st.markdown(
            f"""
            <div style="
                background    : transparent;
                border        : 1px solid rgba(128,128,128,0.2);
                border-radius : 8px;
                padding       : 8px 12px;
                margin-bottom : 4px;
                color         : var(--text-color);
                opacity       : 0.65;
            ">
                <div style="
                    font-size     : 0.80rem;
                    white-space   : nowrap;
                    overflow      : hidden;
                    text-overflow : ellipsis;
                ">💬 {p_title}</div>
                <div style="
                    font-size  : 0.68rem;
                    opacity    : 0.7;
                    margin-top : 2px;
                ">{p_meta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("_Lưu trữ đa phiên sẽ được bổ sung trong phiên bản tới._")

# ── Public render function ─────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Renders the chat-focused sidebar and returns a lean configuration dict.

    Returned keys
    ─────────────
    llm_provider              : "gemini" | "ollama"
    llm_model                 : str
    temperature               : float
    chain_type                : "rag" | "corag" | "selfrag"
    corag_max_iter            : int
    selfrag_max_retrieval     : int
    selfrag_max_generation    : int
    selfrag_quality_threshold : int
    prompt_mode               : str  (key of PROMPT_MODES)
    retriever_k               : int
    score_threshold           : float
    """
    with st.sidebar:

        # ── 1. GOOGLE API KEY ─────────────────────────────────────────────────
        with st.expander("🔑 Google API Key", expanded=not _has_gemini_key()):
            if settings.google_api_key.strip():
                st.success("API Key đã load từ `.env`", icon="✅")
            else:
                st.text_input(
                    "Nhập API Key",
                    type="password",
                    key="gemini_api_key_input",
                    placeholder="AIza...",
                    help="Chỉ lưu trong session, không ghi vào file.",
                    on_change=_apply_api_key,
                )
                if not _has_gemini_key():
                    st.caption("Không có key → chỉ dùng được Ollama (local).")

        st.divider()

        # ── 2. CHAT HISTORY ───────────────────────────────────────────────────
        _render_chat_history()

        st.divider()

        # ── 3. LLM ────────────────────────────────────────────────────────────
        st.subheader("💬 LLM")

        llm_provider = st.radio(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(settings.default_llm_provider),
            horizontal=True,
            key="llm_provider",
        )

        if llm_provider == "gemini" and not _has_gemini_key():
            st.warning("Cần API Key để dùng Gemini.", icon="⚠️")

        llm_model   = _llm_model_selector(llm_provider)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="temperature")

        st.divider()

        # ── 4. CHAIN TYPE ─────────────────────────────────────────────────────
        st.subheader("⛓️ Chế độ Chain")

        chain_type = st.radio(
            "Loại chain",
            list(_CHAIN_LABELS.keys()),
            format_func=_CHAIN_LABELS.get,
            horizontal=True,
            key="chain_type",
        )

        # Defaults — overridden by the per-chain widget blocks below
        corag_max_iter            = 3
        selfrag_max_retrieval     = 2
        selfrag_max_generation    = 2
        selfrag_quality_threshold = 3

        if chain_type == "corag":
            corag_max_iter = st.slider(
                "Số vòng truy xuất tối đa", 1, 5, 3,
                key="corag_max_iter",
                help="CoRAG lặp tối đa N vòng để bổ sung context trước khi trả lời.",
            )

        elif chain_type == "selfrag":
            st.caption(
                "Self-RAG tự quyết định có cần retrieve không, "
                "chấm điểm từng chunk, kiểm tra faithfulness và chất lượng."
            )
            selfrag_max_retrieval = st.slider(
                "Số lần retrieve tối đa", 1, 4, 2,
                key="selfrag_max_retrieval",
                help="Số lần tối đa viết lại query nếu không tìm được doc liên quan.",
            )
            selfrag_max_generation = st.slider(
                "Số lần sinh câu trả lời tối đa", 1, 3, 2,
                key="selfrag_max_generation",
                help="Số lần tối đa regenerate khi câu trả lời không faithful.",
            )
            selfrag_quality_threshold = st.slider(
                "Ngưỡng chất lượng tối thiểu (1–5)", 1, 5, 3,
                key="selfrag_quality_threshold",
                help="Câu trả lời phải đạt điểm ≥ ngưỡng này thì mới được chấp nhận.",
            )

        st.divider()

        # ── 5. MEMORY ─────────────────────────────────────────────────────────
        st.subheader("🧠 Bộ nhớ ngắn hạn")

        memory_window = st.slider(
            "Số lượt nhớ (memory window)",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            key="memory_window",
            help=(
                "Số lượt hỏi-đáp gần nhất được đưa vào ngữ cảnh hội thoại.\n\n"
                "**0** = tắt bộ nhớ (mỗi câu hỏi độc lập).\n\n"
                "Tăng giá trị để chatbot hiểu câu hỏi liên quan đến lượt trước."
            ),
        )

        if memory_window == 0:
            st.caption("🔕 Bộ nhớ đang tắt — mỗi câu hỏi độc lập.")
        else:
            st.caption(f"💬 Nhớ **{memory_window}** lượt hỏi-đáp gần nhất.")

        st.divider()

        # ── 6. PROMPT & RETRIEVER ─────────────────────────────────────────────
        st.subheader("📝 Prompt & Retriever")

        prompt_mode = st.radio(
            "Prompt Mode",
            list(PROMPT_MODES.keys()),
            key="prompt_mode",
            horizontal=True,
        )

        # ── Retriever type ────────────────────────────────────────────────────
        _RETRIEVER_LABELS = {
            "vector": "🔍 Vector",
            "hybrid": "⚡ Hybrid (Vector + BM25)",
        }
        retriever_type = st.radio(
            "Retriever",
            list(_RETRIEVER_LABELS.keys()),
            format_func=_RETRIEVER_LABELS.get,
            index=1,          # default: hybrid
            horizontal=True,
            key="retriever_type",
            help=(
                "**Vector**: tìm kiếm dựa trên embedding (semantic search).\n\n"
                "**Hybrid**: kết hợp Vector + BM25 (keyword) qua Reciprocal Rank "
                "Fusion — cân bằng giữa ngữ nghĩa và từ khóa chính xác."
            ),
        )

        if retriever_type == "hybrid":
            st.caption(
                "⚡ Hybrid dùng Reciprocal Rank Fusion để kết hợp "
                "semantic search và BM25 keyword search."
            )
        else:
            st.caption("🔍 Vector search thuần túy dựa trên embedding.")

        retriever_k = st.slider("Số chunks (k)", 1, 10, 3, key="retriever_k")
        score_threshold = st.slider(
            "Ngưỡng tương đồng", 0.0, 1.0, 0.0, 0.05,
            key="score_threshold",
        )

        st.divider()

        # ── 6. DB STATUS (read-only indicator) ────────────────────────────────
        db_exists = os.path.exists(settings.faiss_db_folder_path)
        if db_exists:
            st.success("✅ Database sẵn sàng", icon="🗄️")
        else:
            st.error("❌ Chưa có Database")
            st.caption(
                "Vào trang **📦 Chunk Manager** để upload tài liệu "
                "và xây dựng database."
            )

    # ── Return chat-relevant config only ──────────────────────────────────────
    return {
        "llm_provider":               llm_provider,
        "llm_model":                  llm_model,
        "temperature":                temperature,
        "chain_type":                 chain_type,
        "corag_max_iter":             corag_max_iter,
        "selfrag_max_retrieval":      selfrag_max_retrieval,
        "selfrag_max_generation":     selfrag_max_generation,
        "selfrag_quality_threshold":  selfrag_quality_threshold,
        "memory_window":              memory_window,
        "prompt_mode":                prompt_mode,
        "retriever_type":             retriever_type,
        "retriever_k":                retriever_k,
        "score_threshold":            score_threshold,
    }