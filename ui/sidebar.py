import os
import streamlit as st
from config.settings import settings
from config.prompts import PROMPT_MODES
from models import (
    GEMINI_LLM_MODELS, GEMINI_EMBEDDING_MODELS,
    OLLAMA_LLM_MODELS, OLLAMA_EMBEDDING_MODELS,
)

UPLOAD_DIR = "./papers"
PROVIDERS = ["gemini", "ollama"]


def _has_gemini_key() -> bool:
    """Kiểm tra API key Gemini — từ .env hoặc nhập tay trong session."""
    return bool(
        st.session_state.get("gemini_api_key_input", "").strip()
        or settings.google_api_key.strip()
    )


def _model_selectbox(label: str, key: str, gemini_models: list, ollama_models: list, provider: str) -> str:
    gemini_locked = (provider == "gemini") and not _has_gemini_key()
    model_list = gemini_models if provider == "gemini" else ollama_models

    selected = st.selectbox(
        label,
        options=model_list,
        key=key,
        disabled=gemini_locked,
        help="Nhập Google API Key bên dưới để chọn model Gemini." if gemini_locked else None,
    )

    custom = st.text_input(
        "Hoặc nhập model tùy chỉnh",
        value="",
        key=f"{key}_custom",
        placeholder="Bỏ trống để dùng lựa chọn trên",
        disabled=gemini_locked,
    )
    return custom.strip() if custom.strip() else selected


def _papers_manager():
    """Render phần quản lý file trong thư mục papers."""
    with st.expander("🗂️ Quản lý file trong thư mục"):
        existing_files = sorted([
            f for f in os.listdir(UPLOAD_DIR)
            if f.lower().endswith(".pdf")
        ]) if os.path.exists(UPLOAD_DIR) else []

        if not existing_files:
            st.caption("Thư mục trống.")
            return

        st.caption(f"{len(existing_files)} file PDF")

        if st.button("🗑️ Xóa tất cả", use_container_width=True, key="delete_all"):
            for f in existing_files:
                os.remove(os.path.join(UPLOAD_DIR, f))
            st.toast("Đã xóa tất cả file.", icon="🗑️")
            st.rerun()

        st.divider()

        for filename in existing_files:
            col_name, col_btn = st.columns([4, 1])
            with col_name:
                st.caption(f"📄 {filename}")
            with col_btn:
                if st.button("✕", key=f"del_{filename}", help=f"Xóa {filename}"):
                    os.remove(os.path.join(UPLOAD_DIR, filename))
                    st.toast(f"Đã xóa: {filename}", icon="🗑️")
                    st.rerun()


def render_sidebar() -> dict:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with st.sidebar:

        # ── API KEY INPUT ─────────────────────────────────────────────
        with st.expander("🔑 Google API Key", expanded=not _has_gemini_key()):
            if settings.google_api_key.strip():
                st.success("API Key đã được load từ `.env`", icon="✅")
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

        # ── LLM CONFIG ────────────────────────────────────────────────
        st.subheader("💬 LLM")
        llm_provider = st.radio(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(settings.default_llm_provider),
            horizontal=True,
            key="llm_provider",
        )

        if llm_provider == "gemini" and not _has_gemini_key():
            st.warning("⚠️ Cần API Key để dùng Gemini LLM.", icon="⚠️")

        llm_model = _model_selectbox(
            label="Model",
            key="llm_model",
            gemini_models=GEMINI_LLM_MODELS,
            ollama_models=OLLAMA_LLM_MODELS,
            provider=llm_provider,
        )

        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="temperature")

        st.divider()

        # ── CHAIN TYPE ────────────────────────────────────────────────
        st.subheader("⛓️ Chế độ Chain")
        chain_type = st.radio(
            "Loại chain",
            ["rag", "corag"],
            format_func=lambda x: "🔗 RAG (1 lần truy xuất)" if x == "rag" else "🔄 CoRAG (nhiều vòng)",
            horizontal=True,
            key="chain_type",
        )

        corag_max_iter = 3
        if chain_type == "corag":
            corag_max_iter = st.slider(
                "Số vòng truy xuất tối đa", 1, 5, 3, key="corag_max_iter",
                help="CoRAG sẽ lặp tối đa N vòng để bổ sung context trước khi trả lời."
            )

        st.divider()

        # ── PROMPT & RETRIEVER ────────────────────────────────────────
        st.subheader("📝 Prompt & Retriever")
        prompt_mode = st.radio(
            "Prompt Mode",
            list(PROMPT_MODES.keys()),
            key="prompt_mode",
            horizontal=True,
        )
        k = st.slider("Số chunks (k)", 1, 10, 3, key="retriever_k")
        score_threshold = st.slider(
            "Ngưỡng tương đồng", 0.0, 1.0, 0.0, 0.05, key="score_threshold"
        )

        st.divider()

        # ── EMBEDDING CONFIG ──────────────────────────────────────────
        st.subheader("🔢 Embedding")
        embed_provider = st.radio(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(settings.default_embedding_provider),
            horizontal=True,
            key="embed_provider",
        )

        if embed_provider == "gemini" and not _has_gemini_key():
            st.warning("⚠️ Cần API Key để dùng Gemini Embedding.", icon="⚠️")

        embed_model = _model_selectbox(
            label="Model",
            key="embed_model",
            gemini_models=GEMINI_EMBEDDING_MODELS,
            ollama_models=OLLAMA_EMBEDDING_MODELS,
            provider=embed_provider,
        )

        st.caption("⚠️ Đổi embedding model → cần **Build lại DB**.")

        st.divider()

        

        # ── FILE UPLOAD ───────────────────────────────────────────────
        st.subheader("📂 Tài liệu")
        uploaded_files = st.file_uploader(
            "Chọn file PDF", type=["pdf"], accept_multiple_files=True
        )
        c_size = st.slider("Chunk Size", 500, 3000, 1200, key="c_size")
        c_overlap = st.slider("Chunk Overlap", 0, 500, 200, key="c_overlap")

        upload_result = None
        if uploaded_files:
            can_embed = not (embed_provider == "gemini" and not _has_gemini_key())
            if st.button(
                "🚀 Thêm vào Database",
                type="primary",
                use_container_width=True,
                disabled=not can_embed,
                help=None if can_embed else "Cần API Key Gemini để embed với provider này.",
            ):
                upload_result = {
                    "files": uploaded_files,
                    "chunk_size": c_size,
                    "chunk_overlap": c_overlap,
                }

        # ── QUẢN LÝ FILE PAPERS ───────────────────────────────────────
        _papers_manager()

        # ── REBUILD ───────────────────────────────────────────────────
        with st.expander("🛠️ Công cụ nâng cao"):
            st.warning("Build lại sẽ tính phí embedding toàn bộ tài liệu.")
            rebuild_triggered = st.button("🔄 Build lại TOÀN BỘ Database")

        st.divider()

        # ── DB STATUS ─────────────────────────────────────────────────
        db_path = settings.faiss_db_folder_path
        if os.path.exists(db_path):
            st.success("✅ Database sẵn sàng")
        else:
            st.error("❌ Chưa có Database")

    return {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "temperature": temperature,
        "embed_provider": embed_provider,
        "embed_model": embed_model,
        "prompt_mode": prompt_mode,
        "retriever_k": k,
        "score_threshold": score_threshold,
        "upload_result": upload_result,
        "rebuild_triggered": rebuild_triggered,
        "chunk_size": c_size,
        "chunk_overlap": c_overlap,
        "chain_type": chain_type,
        "corag_max_iter": corag_max_iter,
    }


def _apply_api_key():
    """Callback: set GOOGLE_API_KEY vào env khi user nhập tay."""
    key = st.session_state.get("gemini_api_key_input", "").strip()
    if key:
        os.environ["GOOGLE_API_KEY"] = key