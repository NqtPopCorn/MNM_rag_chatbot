"""
pages/benchmark.py
==================
Compare two RAG pipelines side-by-side.

UX improvements
---------------
• Both pipelines stream tokens in real-time, side-by-side
• Per-pipeline live progress indicator (spinner / CoRAG iteration badge)
• Comparison + evaluation only fires AFTER both streams finish
• Parallel streaming via threading.Thread + queue.Queue
• All original metrics, history, export preserved
"""

import os
import time
import queue
import threading
import warnings
import json
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore", message="accessing __path__ from.*zoedepth")

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import settings, get_prompt
from config.prompts import PROMPT_MODES
from core import (
    llm_factory, embedding_factory,
    load_vectorstore, build_retriever,
    build_rag_chain, CoRAGChain,
    history_manager,
)
from models import (
    GEMINI_LLM_MODELS, GEMINI_EMBEDDING_MODELS,
    OLLAMA_LLM_MODELS, OLLAMA_EMBEDDING_MODELS,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Benchmark — RAG Chatbot",
    page_icon="⚡",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
CHAIN_OPTIONS     = {"🔗 RAG": "rag", "🔄 CoRAG": "corag"}
RETRIEVER_OPTIONS = {"🔵 Vector (Semantic)": "vector", "🟣 Hybrid (BM25 + Vector)": "hybrid"}
PROVIDERS         = ["gemini", "ollama"]

# Sentinel tokens used in the stream queue
_DONE   = object()   # pipeline finished successfully
_ERROR  = object()   # pipeline raised an exception

# CoRAG trace sentinel (same as corag.py)
_TRACE_START = "\x00CORAG_TRACE\x00"
_TRACE_END   = "\x00END\x00"

# ── Session state ──────────────────────────────────────────────────────────────
if "bm_history"       not in st.session_state:
    st.session_state.bm_history = []
if "bm_chat_messages" not in st.session_state:
    st.session_state.bm_chat_messages = []


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _approx_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def _history_block(history: str) -> str:
    return f"Conversation History:\n{history}\n\n" if history.strip() else ""


@st.cache_resource(show_spinner="Đang load DB và LLM…")
def _load_shared_resources(
    llm_provider, llm_model, temperature,
    embed_provider, embed_model, db_path,
):
    embeddings = embedding_factory(embed_provider, embed_model)
    vs  = load_vectorstore(db_path, embeddings)
    llm = llm_factory(llm_provider, llm_model, temperature)
    return vs, llm


def _make_pipeline(vs, llm, cfg: dict):
    retriever = build_retriever(
        vs,
        retriever_type=cfg["retriever_type"],
        k=cfg["k"],
        score_threshold=cfg["score_threshold"],
    )
    if cfg["chain_type"] == "corag":
        chain = CoRAGChain(
            llm=llm,
            retriever=retriever,
            prompt_mode=cfg["prompt_mode"],
            max_iterations=cfg["corag_max_iter"],
        )
    else:
        chain = build_rag_chain(llm, retriever, get_prompt(cfg["prompt_mode"]))
    return chain, retriever


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING WORKERS — run in background threads, push chunks into a queue
# ══════════════════════════════════════════════════════════════════════════════

def _stream_worker(chain, retriever, cfg: dict, question: str, history: str, q: queue.Queue):
    """
    Background thread: streams the pipeline and pushes chunks into `q`.
    Pushes _DONE (with metadata dict) or _ERROR (with str) as final item.
    """
    t0 = time.perf_counter()
    answer_parts = []
    trace = None
    chunks_retrieved = 0
    corag_iterations = None

    try:
        if cfg["chain_type"] == "corag":
            # CoRAG streams: first a trace sentinel, then answer tokens
            trace_buf = ""
            in_trace = False
            for chunk in chain.stream(question, chat_history=history):
                if _TRACE_START in chunk or in_trace:
                    trace_buf += chunk
                    in_trace = True
                    if _TRACE_END in trace_buf:
                        start = trace_buf.index(_TRACE_START) + len(_TRACE_START)
                        end   = trace_buf.index(_TRACE_END)
                        try:
                            trace = json.loads(trace_buf[start:end])
                            corag_iterations = len(trace)
                            chunks_retrieved = trace[-1].get("total_docs", 0) if trace else 0
                        except json.JSONDecodeError:
                            trace = []
                        # push the trace signal so UI can update the badge
                        q.put(("__trace__", trace))
                        # anything after the end sentinel is answer text
                        remainder = trace_buf[end + len(_TRACE_END):]
                        if remainder:
                            answer_parts.append(remainder)
                            q.put(("chunk", remainder))
                        in_trace = False
                        trace_buf = ""
                    continue

                answer_parts.append(chunk)
                q.put(("chunk", chunk))

        else:
            # Standard RAG — retrieve first (fast), then stream LLM
            docs = retriever.invoke(question)
            chunks_retrieved = len(docs)
            q.put(("__retrieved__", chunks_retrieved))   # signal: retrieval done

            for chunk in chain.stream({
                "question": question,
                "chat_history_block": _history_block(history),
            }):
                answer_parts.append(chunk)
                q.put(("chunk", chunk))

    except Exception as exc:
        q.put((_ERROR, str(exc)))
        return

    elapsed = time.perf_counter() - t0
    answer  = "".join(answer_parts)
    context_chars = chunks_retrieved * cfg["k"] * 400
    input_text    = question + history + "x" * context_chars
    input_tokens  = _approx_tokens(input_text)
    output_tokens = _approx_tokens(answer)

    q.put((_DONE, {
        "answer":           answer,
        "elapsed":          elapsed,
        "input_tokens":     input_tokens,
        "output_tokens":    output_tokens,
        "total_tokens":     input_tokens + output_tokens,
        "chunks_retrieved": chunks_retrieved,
        "corag_iterations": corag_iterations,
        "trace":            trace,
        "error":            None,
    }))


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING RENDER — drain both queues and update two Streamlit placeholders
# ══════════════════════════════════════════════════════════════════════════════

def _render_streaming_results(q_a: queue.Queue, q_b: queue.Queue):
    """
    Drain q_a and q_b in a tight loop, updating per-pipeline placeholders.
    Returns (res_a, res_b) once both are done.
    """
    # ── Layout ────────────────────────────────────────────────────────────────
    col_a, col_sep, col_b = st.columns([10, 1, 10])

    with col_sep:
        st.markdown(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:100%;font-size:1.8rem;font-weight:700;color:#555;padding-top:60px;'>"
            "VS</div>",
            unsafe_allow_html=True,
        )

    # Per-pipeline containers
    with col_a:
        st.markdown(
            "<div style='background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.18);"
            "border-radius:10px;padding:14px 18px;margin-bottom:8px;'>"
            "<h3 style='margin:0;'>🅰️ Pipeline A</h3></div>",
            unsafe_allow_html=True,
        )
        status_a  = st.empty()
        stream_a  = st.empty()

    with col_b:
        st.markdown(
            "<div style='background:rgba(167,139,250,0.06);border:1px solid rgba(167,139,250,0.18);"
            "border-radius:10px;padding:14px 18px;margin-bottom:8px;'>"
            "<h3 style='margin:0;'>🅱️ Pipeline B</h3></div>",
            unsafe_allow_html=True,
        )
        status_b  = st.empty()
        stream_b  = st.empty()

    # ── Drain loop ────────────────────────────────────────────────────────────
    text_a: list[str] = []
    text_b: list[str] = []
    res_a = res_b = None
    done_a = done_b = False

    status_a.info("⏳ Đang truy xuất & sinh câu trả lời…")
    status_b.info("⏳ Đang truy xuất & sinh câu trả lời…")

    while not (done_a and done_b):
        # Poll A
        if not done_a:
            try:
                item = q_a.get(timeout=0.05)
                kind, payload = item
                if kind == "chunk":
                    text_a.append(payload)
                    stream_a.markdown("".join(text_a) + "▌")
                elif kind == "__retrieved__":
                    status_a.success(f"✅ Đã lấy {payload} chunks — đang sinh câu trả lời…")
                elif kind == "__trace__":
                    n = len(payload) if payload else 0
                    status_a.info(f"🔄 CoRAG hoàn tất {n} vòng — đang sinh câu trả lời…")
                elif kind is _DONE:
                    res_a = payload
                    done_a = True
                    stream_a.markdown("".join(text_a))
                    status_a.success(f"✅ Xong — {payload['elapsed']:.2f}s")
                elif kind is _ERROR:
                    done_a = True
                    stream_a.error(f"❌ {payload}")
                    status_a.error("Pipeline A gặp lỗi.")
                    res_a = {
                        "answer": f"❌ {payload}", "elapsed": 0,
                        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "chunks_retrieved": 0, "corag_iterations": None,
                        "trace": None, "error": payload,
                    }
            except queue.Empty:
                pass

        # Poll B
        if not done_b:
            try:
                item = q_b.get(timeout=0.05)
                kind, payload = item
                if kind == "chunk":
                    text_b.append(payload)
                    stream_b.markdown("".join(text_b) + "▌")
                elif kind == "__retrieved__":
                    status_b.success(f"✅ Đã lấy {payload} chunks — đang sinh câu trả lời…")
                elif kind == "__trace__":
                    n = len(payload) if payload else 0
                    status_b.info(f"🔄 CoRAG hoàn tất {n} vòng — đang sinh câu trả lời…")
                elif kind is _DONE:
                    res_b = payload
                    done_b = True
                    stream_b.markdown("".join(text_b))
                    status_b.success(f"✅ Xong — {payload['elapsed']:.2f}s")
                elif kind is _ERROR:
                    done_b = True
                    stream_b.error(f"❌ {payload}")
                    status_b.error("Pipeline B gặp lỗi.")
                    res_b = {
                        "answer": f"❌ {payload}", "elapsed": 0,
                        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "chunks_retrieved": 0, "corag_iterations": None,
                        "trace": None, "error": payload,
                    }
            except queue.Empty:
                pass

    return res_a, res_b


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — global config
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Cấu hình chung")

    st.subheader("💬 LLM")
    g_llm_provider = st.radio("Provider", PROVIDERS, horizontal=True, key="bm_llm_provider")
    g_llm_model    = st.selectbox(
        "Model",
        GEMINI_LLM_MODELS if g_llm_provider == "gemini" else OLLAMA_LLM_MODELS,
        key="bm_llm_model",
    )
    g_temperature  = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="bm_temperature")
    st.divider()

    st.subheader("🔢 Embedding")
    g_embed_provider = st.radio("Provider", PROVIDERS, horizontal=True, key="bm_embed_provider")
    g_embed_model    = st.selectbox(
        "Model",
        GEMINI_EMBEDDING_MODELS if g_embed_provider == "gemini" else OLLAMA_EMBEDDING_MODELS,
        key="bm_embed_model",
    )
    st.caption("⚠️ Phải khớp với DB đã build.")
    st.divider()

    st.subheader("🧠 Bộ nhớ hội thoại")
    g_memory_window = st.slider(
        "Số lượt nhớ (turns)", 0, 10, 0, key="bm_memory_window",
        help="0 = tắt. Cả 2 pipeline dùng chung lịch sử này.",
    )
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True, key="bm_clear_chat"):
        st.session_state.bm_chat_messages = []
        st.toast("Đã xóa lịch sử chat.", icon="🗑️")
    st.divider()

    if os.path.exists(settings.faiss_db_folder_path):
        st.success("✅ Database sẵn sàng")
    else:
        st.error("❌ Chưa có Database")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# ⚡ Benchmark — So sánh Pipeline")
st.caption("Chạy **2 pipeline song song** — stream cả hai cùng lúc, so sánh sau khi cả hai xong.")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIG UI
# ══════════════════════════════════════════════════════════════════════════════

def _pipeline_ui(col, title: str, kp: str, default_chain: str = "rag") -> dict:
    with col:
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.1);"
            f"border-radius:10px;padding:16px 18px;margin-bottom:4px;'>"
            f"<h3 style='margin:0 0 12px 0;'>{title}</h3>",
            unsafe_allow_html=True,
        )
        chain_label = st.radio(
            "Chain type", list(CHAIN_OPTIONS.keys()),
            index=list(CHAIN_OPTIONS.values()).index(default_chain),
            horizontal=True, key=f"{kp}_chain",
        )
        chain_type = CHAIN_OPTIONS[chain_label]

        corag_max_iter = 3
        if chain_type == "corag":
            corag_max_iter = st.slider(
                "Vòng lặp tối đa", 1, 5, 3, key=f"{kp}_corag_iter",
                help="CoRAG sẽ lặp tối đa N vòng để bổ sung context.",
            )

        prompt_mode = st.radio(
            "Prompt mode", list(PROMPT_MODES.keys()),
            horizontal=True, key=f"{kp}_prompt",
        )
        ret_label = st.radio(
            "Retriever", list(RETRIEVER_OPTIONS.keys()),
            horizontal=True, key=f"{kp}_ret",
        )
        retriever_type = RETRIEVER_OPTIONS[ret_label]
        k = st.slider("Chunks (k)", 1, 10, 5, key=f"{kp}_k")
        score_threshold = st.slider(
            "Score threshold", 0.0, 1.0, 0.0, 0.05, key=f"{kp}_score",
            help="Chỉ áp dụng cho Vector retriever.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return {
            "title": title, "chain_type": chain_type,
            "prompt_mode": prompt_mode, "retriever_type": retriever_type,
            "k": k, "score_threshold": score_threshold,
            "corag_max_iter": corag_max_iter,
        }


col_a_cfg, col_vs_cfg, col_b_cfg = st.columns([10, 1, 10])
cfg_a = _pipeline_ui(col_a_cfg, "🅰️ Pipeline A", "pa", default_chain="rag")
with col_vs_cfg:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:100%;font-size:1.8rem;font-weight:700;color:#555;padding-top:120px;'>"
        "VS</div>", unsafe_allow_html=True,
    )
cfg_b = _pipeline_ui(col_b_cfg, "🅱️ Pipeline B", "pb", default_chain="corag")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# QUERY INPUT + RUN BUTTON
# ══════════════════════════════════════════════════════════════════════════════

query = st.text_area(
    "📝 Câu hỏi benchmark",
    placeholder="Nhập câu hỏi để so sánh 2 pipeline…",
    height=90, key="bm_query",
)

btn_col, info_col = st.columns([3, 7])
with btn_col:
    run_btn = st.button(
        "▶️  Chạy Benchmark", type="primary", use_container_width=True,
        disabled=not query.strip() or not os.path.exists(settings.faiss_db_folder_path),
    )
with info_col:
    if not os.path.exists(settings.faiss_db_folder_path):
        st.warning("Database chưa sẵn sàng — hãy upload tài liệu ở trang chính trước.")
    elif not query.strip():
        st.caption("Nhập câu hỏi để bắt đầu so sánh.")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# RUN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

if run_btn and query.strip():

    vs, llm = _load_shared_resources(
        g_llm_provider, g_llm_model, g_temperature,
        g_embed_provider, g_embed_model,
        settings.faiss_db_folder_path,
    )
    if vs is None:
        st.error("❌ Không load được Database.")
        st.stop()

    chain_a, ret_a = _make_pipeline(vs, llm, cfg_a)
    chain_b, ret_b = _make_pipeline(vs, llm, cfg_b)

    # ── Chat history & contextualize ────────────────────────────────────────
    history = history_manager.format(st.session_state.bm_chat_messages, window=g_memory_window)
    ctx_question = query
    if history and llm is not None:
        with st.spinner("🔁 Contextualize câu hỏi…"):
            ctx_question = history_manager.contextualize(query, history, llm)

    if ctx_question != query:
        with st.expander("🔁 Câu hỏi đã được viết lại", expanded=False):
            st.caption(f"**Gốc:** {query}")
            st.caption(f"**Viết lại:** {ctx_question}")

    # ── Kick off both streaming threads ─────────────────────────────────────
    q_a: queue.Queue = queue.Queue()
    q_b: queue.Queue = queue.Queue()

    t_a = threading.Thread(
        target=_stream_worker,
        args=(chain_a, ret_a, cfg_a, ctx_question, history, q_a),
        daemon=True,
    )
    t_b = threading.Thread(
        target=_stream_worker,
        args=(chain_b, ret_b, cfg_b, ctx_question, history, q_b),
        daemon=True,
    )
    t_a.start()
    t_b.start()

    # ── Stream both into side-by-side UI, block until done ──────────────────
    res_a, res_b = _render_streaming_results(q_a, q_b)

    t_a.join()
    t_b.join()


    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON + METRICS (only after both done)
    # ══════════════════════════════════════════════════════════════════════

    st.divider()
    st.markdown("### 📊 So sánh chi tiết")

    def _better_color(val_a, val_b, lower_is_better=True):
        if val_a == val_b:
            return "#9ca3af", "#9ca3af"
        if (val_a < val_b) == lower_is_better:
            return "#34d399", "#f87171"
        return "#f87171", "#34d399"

    clr_time  = _better_color(res_a["elapsed"],       res_b["elapsed"],       lower_is_better=True)
    clr_tok   = _better_color(res_a["total_tokens"],  res_b["total_tokens"],  lower_is_better=True)
    clr_chunk = _better_color(res_a["chunks_retrieved"], res_b["chunks_retrieved"], lower_is_better=False)

    def _card(label, value, color, sub=""):
        sub_html = f"<div style='font-size:0.7rem;color:#6b7280;margin-top:2px;'>{sub}</div>" if sub else ""
        return (
            f"<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);"
            f"border-left:4px solid {color};border-radius:8px;padding:10px 14px;'>"
            f"<div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;'>{label}</div>"
            f"<div style='font-size:1.35rem;font-weight:700;color:{color};line-height:1.2;'>{value}</div>"
            f"{sub_html}</div>"
        )

    def _render_metric_row(res, cfg, clrs, side):
        tc  = clr_time[0]  if side == "a" else clr_time[1]
        tkc = clr_tok[0]   if side == "a" else clr_tok[1]
        cc  = clr_chunk[0] if side == "a" else clr_chunk[1]
        chain_tag = "🔗 RAG" if cfg["chain_type"] == "rag" else "🔄 CoRAG"
        ret_tag   = "🔵 Vector" if cfg["retriever_type"] == "vector" else "🟣 Hybrid"
        label     = "🅰️ Pipeline A" if side == "a" else "🅱️ Pipeline B"

        st.markdown(
            f"<div style='font-size:0.75rem;color:#9ca3af;margin-bottom:8px;'>"
            f"**{label}** · {chain_tag} · {ret_tag} · k={cfg['k']} · prompt={cfg['prompt_mode'][:12]}…</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(_card("⏱️ Latency", f"{res['elapsed']:.2f}s", tc), unsafe_allow_html=True)
        with m2:
            st.markdown(_card(
                "🔢 Tokens", f"{res['total_tokens']:,}", tkc,
                sub=f"in {res['input_tokens']:,} · out {res['output_tokens']:,}",
            ), unsafe_allow_html=True)
        with m3:
            iter_sub = f"{res['corag_iterations']} vòng lặp" if res["corag_iterations"] is not None else ""
            st.markdown(_card("📄 Chunks", str(res["chunks_retrieved"]), cc, sub=iter_sub), unsafe_allow_html=True)

    metric_col_a, metric_col_b = st.columns(2)
    with metric_col_a:
        _render_metric_row(res_a, cfg_a, clr_time, "a")
    with metric_col_b:
        _render_metric_row(res_b, cfg_b, clr_time, "b")

    # ── Winner summary bar ──────────────────────────────────────────────────
    st.divider()
    faster     = "A" if res_a["elapsed"]      < res_b["elapsed"]      else "B"
    cheaper    = "A" if res_a["total_tokens"] < res_b["total_tokens"] else "B"
    time_diff  = abs(res_a["elapsed"]      - res_b["elapsed"])
    token_diff = abs(res_a["total_tokens"] - res_b["total_tokens"])

    faster_color  = "#34d399" if faster  == "A" else "#a78bfa"
    cheaper_color = "#34d399" if cheaper == "A" else "#a78bfa"

    st.markdown(
        f"<div style='display:flex;gap:20px;flex-wrap:wrap;'>"
        f"<div style='background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);"
        f"border-radius:8px;padding:10px 18px;'>"
        f"🏆 Pipeline <b style='color:{faster_color};'>{faster}</b> nhanh hơn <b>{time_diff:.2f}s</b></div>"
        f"<div style='background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);"
        f"border-radius:8px;padding:10px 18px;'>"
        f"💰 Pipeline <b style='color:{cheaper_color};'>{cheaper}</b> dùng ít token hơn <b>~{token_diff:,}</b></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── CoRAG trace expanders (after comparison) ────────────────────────────
    trace_col_a, trace_col_b = st.columns(2)
    for tcol, res, side_label in [
        (trace_col_a, res_a, "A"),
        (trace_col_b, res_b, "B"),
    ]:
        if res.get("trace"):
            with tcol:
                with st.expander(f"🔄 CoRAG Trace — Pipeline {side_label} ({len(res['trace'])} vòng)", expanded=False):
                    for step in res["trace"]:
                        i = step["iteration"]
                        st.markdown(f"**Vòng {i}** — query: `{step['query']}`")
                        c1, c2 = st.columns(2)
                        c1.caption(f"📄 {step['docs_retrieved']} chunk mới · tổng {step['total_docs']}")
                        reasoning = step.get("reasoning", "")
                        if reasoning:
                            icon = "✅" if step.get("sufficient") else "🔄"
                            c2.caption(f"{icon} {reasoning}")
                        follow_up = step.get("follow_up_query", "")
                        if follow_up:
                            st.caption(f"➡️ Follow-up: `{follow_up}`")
                        if i < len(res["trace"]):
                            st.divider()

    # ── Save to run history ─────────────────────────────────────────────────
    st.session_state.bm_history.append({
        "Giờ":          datetime.now().strftime("%H:%M:%S"),
        "Câu hỏi":      query[:55] + ("…" if len(query) > 55 else ""),
        "A Chain":      cfg_a["chain_type"].upper(),
        "A Retriever":  cfg_a["retriever_type"],
        "A k":          cfg_a["k"],
        "A ⏱️ (s)":     round(res_a["elapsed"], 2),
        "A 🔢 tokens":  res_a["total_tokens"],
        "A 📄 chunks":  res_a["chunks_retrieved"],
        "B Chain":      cfg_b["chain_type"].upper(),
        "B Retriever":  cfg_b["retriever_type"],
        "B k":          cfg_b["k"],
        "B ⏱️ (s)":     round(res_b["elapsed"], 2),
        "B 🔢 tokens":  res_b["total_tokens"],
        "B 📄 chunks":  res_b["chunks_retrieved"],
        "🏆 Nhanh hơn": f"Pipeline {faster} ({time_diff:.2f}s)",
    })

    st.session_state.bm_chat_messages.append({"role": "user",      "content": query})
    st.session_state.bm_chat_messages.append({"role": "assistant", "content": res_a["answer"]})


# ══════════════════════════════════════════════════════════════════════════════
# RUN HISTORY TABLE
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.bm_history:
    st.divider()
    n = len(st.session_state.bm_history)
    with st.expander(f"📊 Lịch sử benchmark — {n} lần chạy", expanded=n <= 5):
        df = pd.DataFrame(st.session_state.bm_history)
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Giờ":          st.column_config.TextColumn(width="small"),
                "Câu hỏi":      st.column_config.TextColumn(width="large"),
                "A Chain":      st.column_config.TextColumn(width="small"),
                "A Retriever":  st.column_config.TextColumn(width="small"),
                "A k":          st.column_config.NumberColumn(width="small"),
                "A ⏱️ (s)":     st.column_config.NumberColumn(format="%.2f", width="small"),
                "A 🔢 tokens":  st.column_config.NumberColumn(format="%d",   width="small"),
                "A 📄 chunks":  st.column_config.NumberColumn(format="%d",   width="small"),
                "B Chain":      st.column_config.TextColumn(width="small"),
                "B Retriever":  st.column_config.TextColumn(width="small"),
                "B k":          st.column_config.NumberColumn(width="small"),
                "B ⏱️ (s)":     st.column_config.NumberColumn(format="%.2f", width="small"),
                "B 🔢 tokens":  st.column_config.NumberColumn(format="%d",   width="small"),
                "B 📄 chunks":  st.column_config.NumberColumn(format="%d",   width="small"),
                "🏆 Nhanh hơn": st.column_config.TextColumn(width="medium"),
            },
        )

        if n > 1:
            st.markdown("**📈 Trung bình qua các lần chạy**")
            avg_col_a, avg_col_b = st.columns(2)
            with avg_col_a:
                st.metric("Pipeline A — TB latency", f"{df['A ⏱️ (s)'].mean():.2f}s")
                st.metric("Pipeline A — TB tokens",  f"{int(df['A 🔢 tokens'].mean()):,}")
            with avg_col_b:
                st.metric("Pipeline B — TB latency", f"{df['B ⏱️ (s)'].mean():.2f}s")
                st.metric("Pipeline B — TB tokens",  f"{int(df['B 🔢 tokens'].mean()):,}")

        ecol, ccol = st.columns([4, 1])
        with ecol:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export CSV", data=csv,
                file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True,
            )
        with ccol:
            if st.button("🗑️ Xóa", use_container_width=True, key="bm_clear_hist"):
                st.session_state.bm_history = []
                st.rerun()