"""
sidebar.py
==========
Giao diện thanh bên quản lý cấu hình LLM, Embedding và các cuộc trò chuyện.
"""

import os
import streamlit as st
from config.settings import settings
from config.prompts import PROMPT_MODES
from core.multimodal_loader import LOADER_DISPLAY_NAMES
from models import (
    GEMINI_LLM_MODELS, GEMINI_EMBEDDING_MODELS,
    OLLAMA_LLM_MODELS, OLLAMA_EMBEDDING_MODELS,
)

# ── Conversation store ────────────────────────────────────────────────────────
try:
    from core.conversation_store import ConversationStore
    _DB_PATH = getattr(settings, "conversations_db_path", "./data/conversations.db")
    _store = ConversationStore(_DB_PATH)
    _STORE_AVAILABLE = True
except Exception:
    _store = None
    _STORE_AVAILABLE = False

UPLOAD_DIR = "./papers"
PROVIDERS = ["gemini", "ollama"]


def get_store() -> "ConversationStore | None":
    return _store if _STORE_AVAILABLE else None


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
    with st.expander("Quản lý tệp tin"):
        existing_files = sorted([
            f for f in os.listdir(UPLOAD_DIR)
            if f.lower().endswith(".pdf")
        ]) if os.path.exists(UPLOAD_DIR) else []

        if not existing_files:
            st.caption("Thư mục trống.")
            return

        st.caption(f"Đang có {len(existing_files)} tệp PDF")

        if st.button("Xóa tất cả tài liệu", use_container_width=True, key="delete_all"):
            for f in existing_files:
                os.remove(os.path.join(UPLOAD_DIR, f))
            # Đã xóa tham số icon chứa emoji
            st.toast("Đã dọn dẹp thư mục lưu trữ.")
            st.rerun()

        st.divider()

        for filename in existing_files:
            col_name, col_btn = st.columns([4, 1])
            with col_name:
                st.caption(f"Tệp: {filename}")
            with col_btn:
                # Nút xóa text thông thường
                if st.button("X", key=f"del_{filename}", help=f"Xóa {filename}"):
                    os.remove(os.path.join(UPLOAD_DIR, filename))
                    st.toast(f"Đã xóa: {filename}")
                    st.rerun()


def _conversation_selector():
    if not _STORE_AVAILABLE or _store is None:
        return 

    st.subheader("Danh sách hội thoại")

    # Nút tạo mới với icon Add
    if st.button("Tạo cuộc hội thoại mới", icon=":material/add:", use_container_width=True, key="sidebar_new_conv"):
        new_id = _store.create_conversation()
        st.session_state.messages = []
        st.session_state["active_conv_id"] = new_id
        st.session_state["active_conv_title"] = "Cuộc trò chuyện mới"
        st.rerun()

    convs = _store.list_conversations()

    if not convs:
        st.caption("Lịch sử trống.")
        st.divider()
        return

    active_id = st.session_state.get("active_conv_id")
    MAX_SHOW = 10
    shown = convs[:MAX_SHOW]

    for conv in shown:
        cid = conv["id"]
        is_active = (cid == active_id)
        
        raw_title = conv['title']
        display_label = (raw_title[:25] + '...') if len(raw_title) > 25 else raw_title
        
        # Chia tỉ lệ cột: 7 phần cho tên, 1.5 phần cho mỗi icon thao tác
        col_load, col_edit, col_del = st.columns([6.5, 2, 1.5], gap="xsmall")
        
        with col_load:
            if st.button(
                display_label, 
                key=f"sb_load_{cid}", 
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                msgs = _store.load_messages_into_session_format(cid)
                st.session_state.messages = msgs
                st.session_state["active_conv_id"] = cid
                st.session_state["active_conv_title"] = raw_title
                st.rerun()
        
        with col_edit:
            # Sử dụng icon Edit (SVG) thay cho chữ "Sửa"
            with st.popover("", icon=":material/edit:", help="Đổi tên hội thoại"):
                st.markdown("**Đổi tên hội thoại**")
                new_title = st.text_input("Tiêu đề mới", value=raw_title, key=f"ren_input_{cid}", label_visibility="collapsed")
                if st.button("Cập nhật", key=f"ren_btn_{cid}", use_container_width=True):
                    _store.rename_conversation(cid, new_title)
                    if is_active:
                        st.session_state["active_conv_title"] = new_title
                    st.rerun()

        with col_del:
            # Sử dụng icon Delete (SVG) thay cho chữ "Xóa"
            if st.button("", icon=":material/delete:", key=f"sb_del_{cid}", help="Xóa hội thoại này"):
                _store.delete_conversation(cid)
                if active_id == cid:
                    st.session_state["active_conv_id"] = None
                    st.session_state.messages = []
                st.rerun()

    if len(convs) > MAX_SHOW:
        st.caption(f"Còn {len(convs) - MAX_SHOW} cuộc trò chuyện khác.")
        st.page_link("pages/conversations.py", label="Xem toàn bộ lịch sử", icon=":material/history:")

    st.divider()

def render_sidebar() -> dict:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with st.sidebar:
        _conversation_selector()

        st.subheader("Cấu hình mô hình (LLM)")
        llm_provider = st.radio(
            "Nhà cung cấp",
            PROVIDERS,
            index=PROVIDERS.index(settings.default_llm_provider),
            horizontal=True,
            key="llm_provider",
        )

        llm_model = _model_selectbox(
            label="Mô hình",
            key="llm_model",
            gemini_models=GEMINI_LLM_MODELS,
            ollama_models=OLLAMA_LLM_MODELS,
            provider=llm_provider,
        )

        temperature = st.slider("Độ sáng tạo (Temperature)", 0.0, 1.0, 0.1, 0.05, key="temperature")

        st.divider()

        st.subheader("Phương thức truy xuất")
        chain_type = st.radio(
            "Loại Chain",
            ["rag", "corag"],
            format_func=lambda x: "RAG (Truy xuất chuẩn)" if x == "rag" else "CoRAG (Truy xuất lặp)",
            horizontal=True,
            key="chain_type",
        )

        corag_max_iter = 3
        if chain_type == "corag":
            corag_max_iter = st.slider(
                "Số vòng lặp tối đa", 1, 5, 3, key="corag_max_iter",
                help="CoRAG sẽ tìm kiếm bổ sung thông tin tối đa N lần nếu câu trả lời chưa đủ ý.",
            )

        st.divider()

        st.subheader("Cấu hình bộ nhớ")
        memory_window = st.slider(
            "Số lượt hội thoại ghi nhớ",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            key="memory_window",
            help="Số lượng cặp câu hỏi-đáp gần nhất mà AI sẽ nhớ để hiểu ngữ cảnh.",
        )

        st.divider()

        st.subheader("Tối ưu hóa truy vấn")
        prompt_mode = st.radio(
            "Chế độ Prompt",
            list(PROMPT_MODES.keys()),
            key="prompt_mode",
            horizontal=True,
        )

        retriever_type = st.radio(
            "Cơ chế tìm kiếm",
            ["vector", "hybrid"],
            format_func=lambda x: (
                "Vector (Ngữ nghĩa)" if x == "vector"
                else "Hybrid (Từ khóa + Ngữ nghĩa)"
            ),
            horizontal=True,
            key="retriever_type",
        )

        k = st.slider("Số đoạn văn bản lấy ra (k)", 1, 10, 5, key="retriever_k")
        score_threshold = st.slider(
            "Ngưỡng tin cậy", 0.0, 1.0, 0.0, 0.05, key="score_threshold",
        )

        st.divider()

        st.subheader("Mô hình nhúng (Embedding)")
        embed_provider = st.radio(
            "Nhà cung cấp Embedding",
            PROVIDERS,
            index=PROVIDERS.index(settings.default_embedding_provider),
            horizontal=True,
            key="embed_provider",
        )

        embed_model = _model_selectbox(
            label="Mô hình nhúng",
            key="embed_model",
            gemini_models=GEMINI_EMBEDDING_MODELS,
            ollama_models=OLLAMA_EMBEDDING_MODELS,
            provider=embed_provider,
        )

        st.caption("Lưu ý: Thay đổi embedding cần xây dựng lại Database.")

        st.divider()

        st.subheader("Cơ sở dữ liệu tài liệu")
        loader_label = st.radio(
            "Trình đọc PDF",
            list(LOADER_DISPLAY_NAMES.keys()),
            key="pdf_loader_label",
        )
        loader_strategy = LOADER_DISPLAY_NAMES[loader_label]

        uploaded_files = st.file_uploader(
            "Tải lên tệp PDF", type=["pdf"], accept_multiple_files=True
        )
        c_size = st.slider("Kích thước đoạn (Chunk)", 500, 3000, 1200, key="c_size")
        c_overlap = st.slider("Độ chồng lấp (Overlap)", 0, 500, 200, key="c_overlap")

        upload_result = None
        if uploaded_files:
            if st.button("Cập nhật Database", type="primary", use_container_width=True):
                upload_result = {
                    "files": uploaded_files,
                    "chunk_size": c_size,
                    "chunk_overlap": c_overlap,
                    "loader_strategy": loader_strategy,
                }

        _papers_manager()

        with st.expander("Công cụ hệ thống"):
            st.warning("Hành động này sẽ xóa và nạp lại toàn bộ dữ liệu.")
            rebuild_triggered = st.button("Xây dựng lại toàn bộ Database")

        st.divider()

        db_path = settings.faiss_db_folder_path
        if os.path.exists(db_path):
            st.info("Trạng thái: Database đã sẵn sàng")
        else:
            st.error("Trạng thái: Chưa có dữ liệu")

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
        "active_conv_id": st.session_state.get("active_conv_id"),
    }