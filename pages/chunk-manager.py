"""
Chunk Manager with Routing
Quản lý Vector Database với hệ thống routing cho các tính năng khác nhau:
  - 📤 DB Management: Upload files, Rebuild DB
  - 🔍 Browse: Duyệt & tìm kiếm chunk
  - 📊 Table: Xem dạng bảng
  - 📈 Stats: Thống kê
  - 🗑️ Delete: Xóa theo nguồn
"""

import os
import math
import json
import pandas as pd
import warnings
# Ignore the specific warning regarding __path__ from zoedepth
warnings.filterwarnings("ignore", message="accessing __path__ from.*zoedepth")
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

from config import settings
from core import (
    embedding_factory, load_vectorstore, rebuild_db,
    add_documents_to_db,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chunk Manager — RAG Chatbot",
    page_icon="📦",
    layout="wide",
)

UPLOAD_DIR = "./papers"
CHUNKS_PER_PAGE = 20
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Session State Init ──────────────────────────────────────────────────────────
if "cm_page" not in st.session_state:
    st.session_state.cm_page = "db_management"

if "cm_confirm_rebuild" not in st.session_state:
    st.session_state.cm_confirm_rebuild = False

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Chunk card */
.chunk-card {
    background: var(--background-color, #1e1e2e);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid #4f8ef7;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    transition: border-left-color 0.2s;
    color: white;
}
.chunk-card:hover { border-left-color: #a78bfa; }

/* Metadata pill */
.meta-pill {
    display: inline-block;
    background: rgba(79,142,247,0.15);
    border: 1px solid rgba(79,142,247,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #4f8ef7;
    margin-right: 6px;
    margin-bottom: 6px;
    font-family: monospace;
}

/* Highlighted search match */
mark {
    background: rgba(250,204,21,0.35);
    color: inherit;
    border-radius: 3px;
    padding: 0 2px;
}

/* Stats card */
.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.stat-num {
    font-size: 2rem;
    font-weight: 700;
    color: #4f8ef7;
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _has_gemini_key() -> bool:
    """Check if Gemini API key is available."""
    return bool(
        st.session_state.get("gemini_api_key_input", "").strip()
        or settings.google_api_key.strip()
    )


@st.cache_resource(show_spinner="Đang load vector database...")
def _load_vs(embed_provider: str, embed_model: str, db_path: str):
    embeddings = embedding_factory(embed_provider, embed_model)
    return load_vectorstore(db_path, embeddings)


def _get_all_chunks(vs) -> pd.DataFrame:
    """Extract every document from FAISS docstore into a DataFrame."""
    raw: dict = vs.docstore._dict
    rows = []
    for doc_id, doc in raw.items():
        meta = doc.metadata or {}
        src = meta.get("source", "unknown")
        rows.append({
            "id": doc_id,
            "source": os.path.basename(src),
            "source_full": src,
            "page": meta.get("page", "—"),
            "start_index": meta.get("start_index", 0),
            "char_count": len(doc.page_content),
            "content": doc.page_content,
        })
    if not rows:
        return pd.DataFrame(columns=["id", "source", "source_full", "page", "start_index", "char_count", "content"])
    df = pd.DataFrame(rows).sort_values(["source", "start_index"]).reset_index(drop=True)
    df.index += 1
    return df


def _highlight(text: str, query: str, max_chars: int = 400) -> str:
    """Return HTML-escaped text with <mark> around query matches, truncated."""
    import html, re
    if len(text) > max_chars:
        idx = text.lower().find(query.lower()) if query else -1
        start = max(0, idx - max_chars // 2) if idx != -1 else 0
        text = ("…" if start > 0 else "") + text[start:start + max_chars] + "…"
    safe = html.escape(text)
    if query:
        pattern = re.compile(re.escape(html.escape(query)), re.IGNORECASE)
        safe = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", safe)
    return safe


def _source_color(source: str, sources: list[str]) -> str:
    colors = ["#4f8ef7", "#a78bfa", "#34d399", "#f59e0b", "#f87171", "#38bdf8", "#fb923c"]
    idx = sources.index(source) % len(colors) if source in sources else 0
    return colors[idx]


def _delete_by_source(source_basename: str, embed_provider: str, embed_model: str):
    """Rebuild DB excluding all chunks from the specified source file."""
    db_path = settings.faiss_db_folder_path
    embeddings = embedding_factory(embed_provider, embed_model)
    vs = load_vectorstore(db_path, embeddings)
    if vs is None:
        return False, "Không tìm thấy DB."

    keep_docs = [
        doc for doc in vs.docstore._dict.values()
        if os.path.basename(doc.metadata.get("source", "")) != source_basename
    ]

    if not keep_docs:
        return False, "Không thể xóa: đây là file duy nhất trong DB. Hãy dùng 'Build lại DB' sau khi xóa file."

    from langchain_community.vectorstores import FAISS
    new_vs = FAISS.from_documents(documents=keep_docs, embedding=embeddings)
    new_vs.save_local(db_path)
    st.cache_resource.clear()
    return True, f"✅ Đã xóa tất cả chunk của **{source_basename}** và lưu lại DB."


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER & ROUTING NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 📦 Chunk Manager")

# Navigation buttons
col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns(5)

with col_nav1:
    if st.button("📤 DB Management", use_container_width=True,
                 type="primary" if st.session_state.cm_page == "db_management" else "secondary"):
        st.session_state.cm_page = "db_management"
        st.rerun()

with col_nav2:
    if st.button("🔍 Browse", use_container_width=True,
                 type="primary" if st.session_state.cm_page == "browse" else "secondary"):
        st.session_state.cm_page = "browse"
        st.rerun()

with col_nav3:
    if st.button("📊 Table", use_container_width=True,
                 type="primary" if st.session_state.cm_page == "table" else "secondary"):
        st.session_state.cm_page = "table"
        st.rerun()

with col_nav4:
    if st.button("📈 Stats", use_container_width=True,
                 type="primary" if st.session_state.cm_page == "stats" else "secondary"):
        st.session_state.cm_page = "stats"
        st.rerun()

with col_nav5:
    if st.button("🗑️ Delete", use_container_width=True,
                 type="primary" if st.session_state.cm_page == "delete" else "secondary"):
        st.session_state.cm_page = "delete"
        st.rerun()

st.divider()

# ── Sidebar Config (available on all pages) ────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình Embedding")
    
    # API Key
    if not _has_gemini_key():
        api_key = st.text_input("Google API Key", type="password", key="gemini_api_key_input",
                                placeholder="AIza…", help="Để sử dụng Gemini Embedding")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.success("Google API Key ✅")
    
    # Embedding provider & model
    embed_provider = st.selectbox("Embedding Provider", ["gemini", "ollama"], 
                                 key="cm_embed_provider",
                                 help="Chọn provider để nhúng documents")
    embed_model_map = {
        "gemini": ["gemini-embedding-001"],
        "ollama": ["nomic-embed-text"],
    }
    embed_model = st.selectbox("Embedding Model",
                              embed_model_map[embed_provider],
                              key="cm_embed_model")
    
    st.divider()
    
    
    # DB Status
    db_path = settings.faiss_db_folder_path
    if os.path.exists(db_path):
        st.success("✅ Database sẵn sàng")
    else:
        st.warning("❌ Database chưa tồn tại")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DB MANAGEMENT (Upload & Rebuild)
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.cm_page == "db_management":
    st.markdown("## 📤 Quản lý Vector Database")
    st.caption("Upload tài liệu mới hoặc rebuild toàn bộ database.")
    
    # ── Upload Files ───────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 📁 Thêm tài liệu mới")
        
        uploaded_files = st.file_uploader(
            "Chọn file PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="db_uploader",
        )
        
        col_chunk1, col_chunk2 = st.columns(2)
        with col_chunk1:
            upload_chunk_size = st.slider("Chunk Size", 500, 3000, 
                                         int(st.session_state.get("cm_chunk_size", 1200)),
                                         step=100, key="upload_chunk_size")
        with col_chunk2:
            upload_chunk_overlap = st.slider("Chunk Overlap", 0, 500,
                                            int(st.session_state.get("cm_chunk_overlap", 200)),
                                            step=50, key="upload_chunk_overlap")
        
        if uploaded_files:
            col_upload_btn, col_upload_info = st.columns([3, 1])
            with col_upload_btn:
                can_upload = _has_gemini_key() or embed_provider == "ollama"
                if st.button("🚀 Thêm vào Database", type="primary", 
                            use_container_width=True, disabled=not can_upload):
                    if not can_upload:
                        st.error("Cần Google API Key để upload với Gemini provider.")
                    else:
                        with st.spinner("Đang xử lý và nhúng file..."):
                            embeddings = embedding_factory(embed_provider, embed_model)
                            saved_paths = []
                            for f in uploaded_files:
                                path = os.path.join(UPLOAD_DIR, f.name)
                                with open(path, "wb") as fp:
                                    fp.write(f.getbuffer())
                                saved_paths.append(path)
                            
                            ok, msg = add_documents_to_db(
                                file_paths=saved_paths,
                                db_path=settings.faiss_db_folder_path,
                                embeddings=embeddings,
                                chunk_size=upload_chunk_size,
                                chunk_overlap=upload_chunk_overlap,
                            )
                            if ok:
                                st.success(msg)
                                st.cache_resource.clear()
                                st.balloons()
                            else:
                                st.error(msg)
            with col_upload_info:
                st.caption(f"📄 {len(uploaded_files)} file")
    
    st.divider()
    
    # ── Rebuild Database ───────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 🔨 Rebuild toàn bộ Database")
        st.warning(
            "Xóa DB cũ, đọc lại tất cả PDF trong `./papers`, nhúng lại từ đầu. "
            "**Sẽ tốn API call embedding!**",
            icon="⚠️",
        )
        
        # Existing papers
        existing_papers = sorted([
            f for f in os.listdir(UPLOAD_DIR) 
            if f.lower().endswith(".pdf")
        ]) if os.path.exists(UPLOAD_DIR) else []
        
        if existing_papers:
            st.info(f"📚 Tìm thấy **{len(existing_papers)}** file PDF trong `./papers`")
            with st.expander("📋 Danh sách file"):
                for fname in existing_papers:
                    st.caption(f"📄 {fname}")
        else:
            st.warning("📁 Thư mục `./papers` đang trống.")
        
        col_rebuild_size, col_rebuild_overlap = st.columns(2)
        with col_rebuild_size:
            rebuild_chunk_size = st.slider("Chunk Size", 500, 3000,
                                          int(st.session_state.get("cm_chunk_size", 1200)),
                                          step=100, key="rebuild_chunk_size")
        with col_rebuild_overlap:
            rebuild_chunk_overlap = st.slider("Chunk Overlap", 0, 500,
                                             int(st.session_state.get("cm_chunk_overlap", 200)),
                                             step=50, key="rebuild_chunk_overlap")
        
        st.divider()
        
        # Two-step confirmation
        if not st.session_state.get("db_mgmt_confirm_rebuild", False):
            if st.button("🔨 Rebuild DB", type="secondary", use_container_width=True):
                st.session_state["db_mgmt_confirm_rebuild"] = True
                st.rerun()
        else:
            st.error("⚠️ **CẢNH BÁO:** Bạn sắp xóa toàn bộ DB cũ và rebuild từ đầu. Không thể hoàn tác!",
                    icon="⚠️")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Xác nhận REBUILD", type="primary", use_container_width=True):
                    if not (_has_gemini_key() or embed_provider == "ollama"):
                        st.error("Cần Google API Key để rebuild với Gemini provider.")
                    else:
                        with st.spinner("Đang rebuild DB..."):
                            embeddings = embedding_factory(embed_provider, embed_model)
                            ok, msg = rebuild_db(
                                papers_dir=UPLOAD_DIR,
                                db_path=settings.faiss_db_folder_path,
                                embeddings=embeddings,
                                chunk_size=rebuild_chunk_size,
                                chunk_overlap=rebuild_chunk_overlap,
                            )
                            if ok:
                                st.success(msg)
                                st.cache_resource.clear()
                                st.balloons()
                            else:
                                st.error(msg)
                    st.session_state["db_mgmt_confirm_rebuild"] = False
                    st.rerun()
            with col_no:
                if st.button("❌ Hủy", use_container_width=True):
                    st.session_state["db_mgmt_confirm_rebuild"] = False
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES: BROWSE, TABLE, STATS, DELETE (require existing DB)
# ═══════════════════════════════════════════════════════════════════════════════

# Check if DB exists and load data
if st.session_state.cm_page in ["browse", "table", "stats", "delete"]:
    if not os.path.exists(settings.faiss_db_folder_path):
        st.warning("**Chưa có Vector Database.** Hãy tạo từ trang **📤 DB Management**.", icon="🚨")
        st.stop()
    
    vs = _load_vs(embed_provider, embed_model, settings.faiss_db_folder_path)
    if vs is None:
        st.error("Không thể load Vector Database. Kiểm tra API key và đường dẫn.")
        st.stop()
    
    df_all = _get_all_chunks(vs)
    if df_all.empty:
        st.warning("Database đang trống. Chưa có chunk nào.", icon="ℹ️")
        st.stop()
    
    sources_list = sorted(df_all["source"].unique().tolist())
    
    # Stats bar
    total_chars = df_all["char_count"].sum()
    avg_chars = int(df_all["char_count"].mean())
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-num">{len(df_all):,}</div>
            <div class="stat-label">Tổng Chunk</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-num">{len(sources_list)}</div>
            <div class="stat-label">File Nguồn</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-num">{total_chars:,}</div>
            <div class="stat-label">Tổng Ký Tự</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-num">{avg_chars:,}</div>
            <div class="stat-label">TB Ký Tự/Chunk</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE: BROWSE
    # ─────────────────────────────────────────────────────────────────────────────
    if st.session_state.cm_page == "browse":
        st.markdown("## 🔍 Duyệt & Tìm Kiếm")
        
        col_filter, col_main = st.columns([1, 3])
        
        with col_filter:
            st.markdown("### 🔧 Bộ Lọc")
            
            source_options = ["— Tất cả —"] + sources_list
            selected_source = st.selectbox("File nguồn", source_options, key="browse_source")
            
            search_query = st.text_input("🔍 Tìm trong nội dung", placeholder="Nhập từ khóa…",
                                        key="browse_search")
            
            min_c, max_c = int(df_all["char_count"].min()), int(df_all["char_count"].max())
            char_range = st.slider("Độ dài chunk (ký tự)", min_value=min_c, max_value=max_c,
                                  value=(min_c, max_c), key="browse_char_range")
            
            pages_available = [p for p in df_all["page"].unique() if p != "—"]
            if pages_available:
                selected_page = st.selectbox("Trang",
                    ["— Tất cả —"] + sorted(set(str(p) for p in pages_available)),
                    key="browse_page")
            else:
                selected_page = "— Tất cả —"
            
            st.divider()
            
            sort_by = st.selectbox("Sắp xếp theo",
                ["source + vị trí", "độ dài (dài nhất)", "độ dài (ngắn nhất)"],
                key="browse_sort")
        
        with col_main:
            # Apply filters
            df = df_all.copy()
            
            if selected_source != "— Tất cả —":
                df = df[df["source"] == selected_source]
            
            if search_query.strip():
                mask = df["content"].str.contains(search_query.strip(), case=False, na=False)
                df = df[mask]
            
            df = df[(df["char_count"] >= char_range[0]) & (df["char_count"] <= char_range[1])]
            
            if selected_page != "— Tất cả —":
                df = df[df["page"].astype(str) == selected_page]
            
            # Sort
            if sort_by == "độ dài (dài nhất)":
                df = df.sort_values("char_count", ascending=False)
            elif sort_by == "độ dài (ngắn nhất)":
                df = df.sort_values("char_count", ascending=True)
            
            # Result header
            total_filtered = len(df)
            col_res1, col_res2 = st.columns([3, 1])
            with col_res1:
                if search_query:
                    st.markdown(f"**{total_filtered:,}** kết quả cho `{search_query}`" +
                               (f" trong **{selected_source}**" if selected_source != "— Tất cả —" else ""))
                else:
                    st.markdown(f"Hiển thị **{total_filtered:,}** chunk" +
                               (f" từ **{selected_source}**" if selected_source != "— Tất cả —" else ""))
            
            with col_res2:
                if not df.empty:
                    csv_bytes = df[["id", "source", "page", "start_index", "char_count", "content"]]\
                                  .to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Export CSV", data=csv_bytes,
                                      file_name="chunks_export.csv", mime="text/csv",
                                      use_container_width=True)
            
            if df.empty:
                st.info("Không tìm thấy chunk nào khớp với bộ lọc.")
            else:
                # Pagination
                total_pages = max(1, math.ceil(total_filtered / CHUNKS_PER_PAGE))
                page_num = st.number_input(f"Trang (/{total_pages})", min_value=1, max_value=total_pages,
                                          value=1, step=1, key="browse_page_num")
                
                start = (page_num - 1) * CHUNKS_PER_PAGE
                end = start + CHUNKS_PER_PAGE
                df_page = df.iloc[start:end]
                
                st.caption(f"Chunk {start+1}–{min(end, total_filtered)} / {total_filtered}")
                
                # Render chunks
                for idx, (_, row) in enumerate(df_page.iterrows()):
                    src_color = _source_color(row["source"], sources_list)
                    
                    pills_html = (
                        f'<span class="meta-pill" style="border-color:{src_color}40;'
                        f'color:{src_color};background:{src_color}15;">📄 {row["source"]}</span>'
                        f'<span class="meta-pill">Trang {row["page"]}</span>'
                        f'<span class="meta-pill">Vị trí {row["start_index"]}</span>'
                        f'<span class="meta-pill">{row["char_count"]:,} ký tự</span>'
                    )
                    
                    with st.expander(
                        f"#{start + idx + 1} · {row['source']} · trang {row['page']} · {row['char_count']:,} ký tự",
                        expanded=False):
                        st.markdown(pills_html, unsafe_allow_html=True)
                        st.markdown(f'<div class="chunk-card">{_highlight(row["content"], search_query.strip(), max_chars=9999)}</div>',
                                   unsafe_allow_html=True)
                        st.caption(f"ID: `{row['id']}`")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE: TABLE
    # ─────────────────────────────────────────────────────────────────────────────
    elif st.session_state.cm_page == "table":
        st.markdown("## 📊 Dữ Liệu Dạng Bảng")
        
        col_a, col_b = st.columns([3, 1])
        with col_a:
            tbl_search = st.text_input("🔍 Lọc nhanh theo nội dung hoặc nguồn", key="tbl_search")
        with col_b:
            tbl_source = st.selectbox("Nguồn", ["— Tất cả —"] + sources_list, key="tbl_source")
        
        df_tbl = df_all.copy()
        if tbl_source != "— Tất cả —":
            df_tbl = df_tbl[df_tbl["source"] == tbl_source]
        if tbl_search.strip():
            mask = (df_tbl["content"].str.contains(tbl_search, case=False, na=False) |
                   df_tbl["source"].str.contains(tbl_search, case=False, na=False))
            df_tbl = df_tbl[mask]
        
        st.caption(f"{len(df_tbl):,} chunk")
        
        df_display = df_tbl[["source", "page", "start_index", "char_count", "content"]].copy()
        df_display["content"] = df_display["content"].str.slice(0, 120) + "…"
        
        st.dataframe(df_display, use_container_width=True, height=520,
                    column_config={
                        "source": st.column_config.TextColumn("Nguồn", width="medium"),
                        "page": st.column_config.TextColumn("Trang", width="small"),
                        "start_index": st.column_config.NumberColumn("Vị trí", width="small"),
                        "char_count": st.column_config.NumberColumn("Ký tự", width="small", format="%d"),
                        "content": st.column_config.TextColumn("Nội dung (rút gọn)", width="large"),
                    })
        
        csv_all = df_all[["id", "source", "page", "start_index", "char_count", "content"]]\
                    .to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export toàn bộ CSV", data=csv_all, file_name="all_chunks.csv",
                          mime="text/csv")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE: STATS
    # ─────────────────────────────────────────────────────────────────────────────
    elif st.session_state.cm_page == "stats":
        st.markdown("## 📈 Thống Kê Phân Bố")
        
        # Per-source chunk count
        per_source = (df_all.groupby("source")
                     .agg(chunks=("id", "count"), total_chars=("char_count", "sum"),
                          avg_chars=("char_count", "mean"), min_chars=("char_count", "min"),
                          max_chars=("char_count", "max"))
                     .reset_index()
                     .sort_values("chunks", ascending=False))
        per_source["avg_chars"] = per_source["avg_chars"].round(0).astype(int)
        
        st.markdown("**Số chunk theo nguồn**")
        st.dataframe(per_source, use_container_width=True,
                    column_config={
                        "source": st.column_config.TextColumn("File nguồn"),
                        "chunks": st.column_config.NumberColumn("Số chunk", format="%d"),
                        "total_chars": st.column_config.NumberColumn("Tổng ký tự", format="%d"),
                        "avg_chars": st.column_config.NumberColumn("TB ký tự/chunk", format="%d"),
                        "min_chars": st.column_config.NumberColumn("Min", format="%d"),
                        "max_chars": st.column_config.NumberColumn("Max", format="%d"),
                    }, hide_index=True)
        
        st.divider()
        st.markdown("**Phân bố độ dài chunk (ký tự)**")
        
        bins = pd.cut(df_all["char_count"], bins=20)
        hist_df = (df_all.groupby(bins, observed=True)["char_count"].count().reset_index())
        hist_df.columns = ["Khoảng ký tự", "Số chunk"]
        hist_df["Khoảng ký tự"] = hist_df["Khoảng ký tự"].astype(str)
        st.bar_chart(hist_df.set_index("Khoảng ký tự")["Số chunk"])
        
        st.divider()
        st.markdown("**Top 10 chunk dài nhất**")
        top10 = df_all.nlargest(10, "char_count")[["source", "page", "start_index", "char_count", "content"]].copy()
        top10["content"] = top10["content"].str.slice(0, 100) + "…"
        st.dataframe(top10, use_container_width=True, hide_index=True)
        
        st.markdown("**Top 10 chunk ngắn nhất**")
        bot10 = df_all.nsmallest(10, "char_count")[["source", "page", "start_index", "char_count", "content"]].copy()
        bot10["content"] = bot10["content"].str.slice(0, 100) + "…"
        st.dataframe(bot10, use_container_width=True, hide_index=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE: DELETE
    # ─────────────────────────────────────────────────────────────────────────────
    elif st.session_state.cm_page == "delete":
        st.markdown("## 🗑️ Xóa Chunk Theo Nguồn")
        st.warning("Thao tác này sẽ **rebuild lại DB** sau khi loại bỏ tất cả chunk của file được chọn. "
                  "**Không thể hoàn tác.**", icon="⚠️")
        
        per_source_del = (df_all.groupby("source")["id"].count().reset_index()
                         .rename(columns={"id": "chunks"})
                         .sort_values("chunks", ascending=False))
        
        st.markdown("**Danh sách file đang có trong DB:**")
        st.dataframe(per_source_del, use_container_width=True,
                    column_config={
                        "source": st.column_config.TextColumn("File nguồn"),
                        "chunks": st.column_config.NumberColumn("Số chunk", format="%d"),
                    }, hide_index=True, height=min(38 + len(per_source_del) * 35, 400))
        
        st.divider()
        
        selected_del = st.selectbox("Chọn file muốn xóa khỏi DB", sources_list, key="del_source_select")
        
        if selected_del:
            count_to_del = int(per_source_del.loc[per_source_del["source"] == selected_del, "chunks"].values[0])
            st.info(f"Sẽ xóa **{count_to_del:,} chunk** của file **{selected_del}** "
                   f"(còn lại: {len(df_all) - count_to_del:,} chunk từ {len(sources_list) - 1} file).",
                   icon="ℹ️")
            
            confirm_key = f"confirm_del_{selected_del}"
            if not st.session_state.get(confirm_key, False):
                if st.button(f"🗑️ Xóa tất cả chunk của '{selected_del}'", type="secondary",
                           use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()
            else:
                st.error(f"⚠️ **Xác nhận lần 2:** Bạn chắc chắn muốn xóa **{count_to_del} chunk** của `{selected_del}`?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Xác nhận XÓA", type="primary", use_container_width=True):
                        with st.spinner("Đang xóa và rebuild DB..."):
                            ok, msg = _delete_by_source(selected_del, embed_provider, embed_model)
                        st.session_state[confirm_key] = False
                        if ok:
                            st.success(msg)
                            st.balloons()
                        else:
                            st.error(msg)
                        st.rerun()
                with col_no:
                    if st.button("❌ Hủy", use_container_width=True):
                        st.session_state[confirm_key] = False
                        st.rerun()
