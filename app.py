import os
import warnings

from core.retriever import build_retriever
# Ignore the specific warning regarding __path__ from zoedepth
warnings.filterwarnings("ignore", message="accessing __path__ from.*zoedepth")
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import settings, get_prompt
from core import (
    llm_factory, embedding_factory,
    load_vectorstore, add_documents_to_db, rebuild_db,
    get_retriever, build_rag_chain, CoRAGChain,
)
from ui import render_sidebar, render_chat, clear_chat_button, model_badge, status_banner

UPLOAD_DIR = "./papers"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
cfg = render_sidebar()

# ── GUARD: provider Gemini mà thiếu key → không làm gì cả ────────────────────
def _gemini_ready() -> bool:
    # Sidebar đã xóa phần nhập key, nên chỉ kiểm tra key từ settings/env
    return bool(settings.google_api_key.strip())

def _check_provider(provider: str, action: str) -> bool:
    if provider == "gemini" and not _gemini_ready():
        st.error(f"Cần Google API Key để {action} với Gemini.")
        return False
    return True

# ── XỬ LÝ UPLOAD ──────────────────────────────────────────────────────────────
if cfg["upload_result"] and _check_provider(cfg["embed_provider"], "nhúng tài liệu"):
    upload = cfg["upload_result"]
    with st.spinner("Đang xử lý và nhúng file..."):
        saved_paths = []
        for f in upload["files"]:
            path = os.path.join(UPLOAD_DIR, f.name)
            with open(path, "wb") as fp:
                fp.write(f.getbuffer())
            saved_paths.append(path)

        embeddings = embedding_factory(cfg["embed_provider"], cfg["embed_model"])
        ok, msg = add_documents_to_db(
            file_paths=saved_paths,
            db_path=settings.faiss_db_folder_path,
            embeddings=embeddings,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        st.toast(msg)
        if ok:
            st.cache_resource.clear()

# ── XỬ LÝ REBUILD ─────────────────────────────────────────────────────────────
if cfg["rebuild_triggered"] and _check_provider(cfg["embed_provider"], "build DB"):
    with st.spinner("Đang build lại toàn bộ database..."):
        embeddings = embedding_factory(cfg["embed_provider"], cfg["embed_model"])
        ok, msg = rebuild_db(
            papers_dir=UPLOAD_DIR,
            db_path=settings.faiss_db_folder_path,
            embeddings=embeddings,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        st.toast(msg)
        if ok:
            st.cache_resource.clear()

# ── BUILD RAG CHAIN ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang khởi tạo chain...")
def get_chain(
    llm_provider, llm_model, temperature,
    embed_provider, embed_model,
    prompt_mode, retriever_type, retriever_k, score_threshold,
    db_path,
    chain_type: str = "rag",
    corag_max_iter: int = 3,
):
    embeddings = embedding_factory(embed_provider, embed_model)
    vs = load_vectorstore(db_path, embeddings)
    if vs is None:
        return None, None
    llm = llm_factory(llm_provider, llm_model, temperature)
    
    retriever = build_retriever(vs, retriever_type, retriever_k, score_threshold)
    
    if chain_type == "corag":
        chain = CoRAGChain(llm=llm, retriever=retriever, prompt_mode=prompt_mode, max_iterations=corag_max_iter)
    else:
        from config import get_prompt
        prompt = get_prompt(prompt_mode)
        chain = build_rag_chain(llm, retriever, prompt)
    return chain, retriever


# Chỉ build chain nếu provider đang chọn có đủ điều kiện
llm_ready = _check_provider(cfg["llm_provider"], "chat") if False else True  # lazy — kiểm tra khi chat
rag_chain = None
rag_retriever = None

if _gemini_ready() or cfg["llm_provider"] == "ollama":
    _result = get_chain(
        llm_provider=cfg["llm_provider"],
        llm_model=cfg["llm_model"],
        temperature=cfg["temperature"],
        embed_provider=cfg["embed_provider"],
        embed_model=cfg["embed_model"],
        prompt_mode=cfg["prompt_mode"],
        retriever_type=cfg["retriever_type"],  # Truyền biến retriever_type vào đây
        retriever_k=cfg["retriever_k"],
        score_threshold=cfg["score_threshold"],
        db_path=settings.faiss_db_folder_path,
        chain_type=cfg["chain_type"],
        corag_max_iter=cfg["corag_max_iter"],
    )
    if _result is not None:
        rag_chain, rag_retriever = _result

# ── UI ─────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    st.title("RAG Chatbot")
with col2:
    clear_chat_button()

model_badge(cfg["llm_provider"], cfg["llm_model"])
st.divider()

# Nếu Gemini được chọn mà thiếu key → báo lỗi rõ ràng thay vì crash
if cfg["llm_provider"] == "gemini" and not _gemini_ready():
    st.warning("Cần cấu hình Google API Key để bắt đầu chat với Gemini (vui lòng kiểm tra cài đặt môi trường).")
    st.stop()

status_banner(db_exists=rag_chain is not None)

# Thay sidebar_config thành cfg

from core.conversation_store import ConversationStore
store = ConversationStore("./data/conversations.db")

render_chat(
    rag_chain=rag_chain,
    provider=cfg["llm_provider"],
    model=cfg["llm_model"],
    llm=llm_factory("ollama", "qwen2.5:3b", temperature=0),
    retriever=rag_retriever,
    chain_type=cfg["chain_type"],
    memory_window=cfg["memory_window"],
    store=store,
)