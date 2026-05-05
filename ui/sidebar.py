import os
import streamlit as st
from config.settings import settings
from config.prompts import PROMPT_MODES
from core.multimodal_loader import LOADER_DISPLAY_NAMES
from models import (
    GEMINI_LLM_MODELS, GEMINI_EMBEDDING_MODELS,
    OLLAMA_LLM_MODELS, OLLAMA_EMBEDDING_MODELS,
)

UPLOAD_DIR = "./papers"
PROVIDERS = ["gemini", "ollama"]


def _model_selectbox(label: str, key: str, gemini_models: list, ollama_models: list, provider: str) -> str:
    model_list = gemini_models if provider == "gemini" else ollama_models

    selected = st.selectbox(label, options=model_list, key=key)

    custom = st.text_input(
        "Hoặc nhập model tùy chỉnh",
        value="",
        key=f"{key}_custom",
        placeholder="Bỏ trống để dùng lựa chọn trên",
    )
    return custom.strip() if custom.strip() else selected


def _papers_manager():
    """Render file list inside the papers folder."""
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

        # ── LLM CONFIG ────────────────────────────────────────────────
        st.subheader("💬 LLM")
        llm_provider = st.radio(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(settings.default_llm_provider),
            horizontal=True,
            key="llm_provider",
        )

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
                help="CoRAG sẽ lặp tối đa N vòng để bổ sung context trước khi trả lời.",
            )

        st.divider()

        # ── MEMORY ────────────────────────────────────────────────────
        st.subheader("🧠 Bộ nhớ hội thoại")
        memory_window = st.slider(
            "Số lượt nhớ (turns)",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            key="memory_window",
            help=(
                "0 = tắt bộ nhớ (mỗi câu hỏi độc lập).\n"
                "N > 0 = nhớ N lượt hỏi-đáp gần nhất. "
                "Câu hỏi follow-up sẽ được viết lại thành câu độc lập trước khi truy xuất."
            ),
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

        # Retriever type selector — NEW
        retriever_type = st.radio(
            "Retriever",
            ["vector", "hybrid"],
            format_func=lambda x: (
                "🔵 Vector (Semantic)" if x == "vector"
                else "🟣 Hybrid (BM25 + Vector)"
            ),
            horizontal=True,
            key="retriever_type",
            help=(
                "Vector: pure embedding similarity search.\n"
                "Hybrid: fuses vector + BM25 keyword ranking via Reciprocal Rank Fusion — "
                "better recall when the question contains specific terms."
            ),
        )

        k = st.slider("Số chunks (k)", 1, 10, 5, key="retriever_k")
        score_threshold = st.slider(
            "Ngưỡng tương đồng", 0.0, 1.0, 0.0, 0.05, key="score_threshold",
            help="BM25 ignores this threshold (no score); only affects Vector retriever.",
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

        # PDF loader strategy — NEW
        loader_label = st.radio(
            "PDF Loader",
            list(LOADER_DISPLAY_NAMES.keys()),
            key="pdf_loader_label",
            help=(
                "Standard: fast, plain-text extraction (UnstructuredFileLoader).\n"
                "Enhanced: pymupdf4llm converts pages to Markdown — tables and "
                "multi-column layouts are preserved correctly."
            ),
        )
        loader_strategy = LOADER_DISPLAY_NAMES[loader_label]

        uploaded_files = st.file_uploader(
            "Chọn file PDF", type=["pdf"], accept_multiple_files=True
        )
        c_size = st.slider("Chunk Size", 500, 3000, 1200, key="c_size")
        c_overlap = st.slider("Chunk Overlap", 0, 500, 200, key="c_overlap")

        upload_result = None
        if uploaded_files:
            if st.button(
                "🚀 Thêm vào Database",
                type="primary",
                use_container_width=True,
            ):
                upload_result = {
                    "files": uploaded_files,
                    "chunk_size": c_size,
                    "chunk_overlap": c_overlap,
                    "loader_strategy": loader_strategy,
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
        "retriever_type": retriever_type,
        "retriever_k": k,
        "score_threshold": score_threshold,
        "upload_result": upload_result,
        "rebuild_triggered": rebuild_triggered,
        "chunk_size": c_size,
        "chunk_overlap": c_overlap,
        "chain_type": chain_type,
        "corag_max_iter": corag_max_iter,
        "loader_strategy": loader_strategy,
        "memory_window": memory_window,
    }