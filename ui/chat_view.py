"""
chat_view.py
============
Renders the chat UI and auto-saves every turn to SQLite.

Auto-save contract
──────────────────
• render_chat() requires `store` (ConversationStore) and reads
  `st.session_state.active_conv_id` directly — no need to pass conv_id.
• Every completed user+assistant turn is written atomically:
    1. user   message saved  (plain content)
    2. assistant message saved (content + metadata_json)
    3. conversation title auto-set from 1st user message (idempotent)
• Persistence errors are caught and shown as st.toast — they never crash UI.
"""

from __future__ import annotations

import html
import json
import os
from typing import Optional

import streamlit as st

from core.conversation_store import ConversationStore
from core.memory import history_manager

# CoRAG stream sentinels
_TRACE_START = "\x00CORAG_TRACE\x00"
_TRACE_END   = "\x00END\x00"


# ── Rendering helpers (unchanged) ─────────────────────────────────────────────

def _render_corag_trace(trace: list[dict]):
    if not trace:
        return
    with st.expander(f"🔄 CoRAG — {len(trace)} vòng truy xuất", expanded=False):
        for step in trace:
            i = step["iteration"]
            st.markdown(f"**Vòng {i}**")
            c1, c2 = st.columns(2)
            c1.markdown(f"🔍 **Query:** `{step['query']}`")
            c2.markdown(
                f"📄 Tìm được **{step['docs_retrieved']}** chunk mới "
                f"(tổng: {step['total_docs']})"
            )
            reasoning = step.get("reasoning", "")
            if reasoning:
                icon = "✅" if step.get("sufficient") else "🔄"
                st.markdown(f"{icon} *{reasoning}*")
            if step.get("follow_up_query"):
                st.markdown(f"➡️ **Follow-up:** `{step['follow_up_query']}`")
            if i < len(trace):
                st.divider()


def _render_context_docs(docs: list):
    if not docs:
        return
    colors = ["#4f8ef7", "#a78bfa", "#34d399", "#f59e0b", "#f87171", "#38bdf8", "#fb923c"]
    with st.expander(f"📚 Context — {len(docs)} chunk đã dùng", expanded=False):
        sources_seen: list[str] = []
        for doc in docs:
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            if src not in sources_seen:
                sources_seen.append(src)
        for doc in docs:
            meta  = doc.metadata or {}
            src   = os.path.basename(meta.get("source", "unknown"))
            page  = meta.get("page", "—")
            start = meta.get("start_index", 0)
            color = colors[sources_seen.index(src) % len(colors)]
            preview = doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else "")
            st.markdown(
                f'<span style="background:{color}20;border:1px solid {color}50;'
                f'border-radius:20px;padding:2px 10px;font-size:.72rem;color:{color};'
                f'font-family:monospace;margin-right:6px;">📄 {src}</span>'
                f'<span style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);'
                f'border-radius:20px;padding:2px 10px;font-size:.72rem;color:rgba(255,255,255,.6);'
                f'font-family:monospace;margin-right:6px;">Trang {page}</span>'
                f'<span style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);'
                f'border-radius:20px;padding:2px 10px;font-size:.72rem;color:rgba(255,255,255,.6);'
                f'font-family:monospace;">Vị trí {start}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);'
                f'border-left:3px solid {color};border-radius:6px;padding:10px 14px;'
                f'font-family:monospace;font-size:.8rem;line-height:1.6;'
                f'white-space:pre-wrap;word-break:break-word;color:grey;margin-bottom:10px;">'
                f'{html.escape(preview)}</div>',
                unsafe_allow_html=True,
            )


def _render_history_badge(contextualized: str, original: str):
    if contextualized != original:
        with st.expander("🔁 Câu hỏi đã được viết lại", expanded=False):
            st.caption(f"**Gốc:** {original}")
            st.caption(f"**Viết lại:** {contextualized}")


# ── CoRAG streaming ───────────────────────────────────────────────────────────

def _stream_corag(chain, question: str, chat_history: str = ""):
    trace: list[dict] = []
    parts: list[str]  = []
    trace_buf = ""
    in_trace  = False
    placeholder = st.empty()

    for chunk in chain.stream(question, chat_history):
        if _TRACE_START in chunk or in_trace:
            trace_buf += chunk
            in_trace   = True
            if _TRACE_END in trace_buf:
                s = trace_buf.index(_TRACE_START) + len(_TRACE_START)
                e = trace_buf.index(_TRACE_END)
                try:
                    trace = json.loads(trace_buf[s:e])
                except json.JSONDecodeError:
                    trace = []
                placeholder.markdown(
                    f"*🔄 Hoàn tất {len(trace)} vòng — đang sinh câu trả lời...*"
                )
                remainder = trace_buf[e + len(_TRACE_END):]
                if remainder:
                    parts.append(remainder)
                    placeholder.markdown("".join(parts) + "▌")
                in_trace  = False
                trace_buf = ""
            continue
        parts.append(chunk)
        placeholder.markdown("".join(parts) + "▌")

    placeholder.empty()
    return "".join(parts), trace


# ── Persistence helper ────────────────────────────────────────────────────────

def _save_turn(
    store: ConversationStore,
    conv_id: int,
    user_input: str,
    response: str,
    model_info: str,
    chain_type: str,
    memory_window: int,
    trace,
) -> None:
    """
    Write one user+assistant turn to SQLite.
    Called after both messages are already in session_state.
    Errors are swallowed so they never crash the UI.
    """
    try:
        # 1. User message — no metadata needed
        store.add_message(conv_id, "user", user_input)

        # 2. Assistant message — full metadata
        store.add_message(
            conv_id, 
            "assistant", 
            response,
            metadata={
                "model_info":    model_info,
                "chain_type":    chain_type,
                "memory_window": memory_window,
                "trace":         trace,          # None for RAG, list for CoRAG
            },
        )

        # 3. Auto-title from first user message (idempotent, no-op after 1st turn)
        store.auto_title_from_first_message(conv_id)

    except Exception as exc:
        st.toast(f"⚠️ Lỗi lưu hội thoại: {exc}", icon="⚠️")


# ── Main render ───────────────────────────────────────────────────────────────

def render_chat(
    rag_chain,
    provider: str,
    model: str,
    store: ConversationStore,
    chain_type: str = "rag",
    retriever=None,
    llm=None,
    memory_window: int = 3,
):
    """
    Render chat UI.  Reads active conv_id from st.session_state.active_conv_id.

    Parameters
    ----------
    rag_chain     : RAG Runnable or CoRAGChain
    provider      : "gemini" | "ollama" (for caption badge)
    model         : model name string
    store         : ConversationStore — required for auto-save
    chain_type    : "rag" | "corag"
    retriever     : VectorStoreRetriever | HybridRetriever (RAG only)
    llm           : BaseChatModel — required for contextualize()
    memory_window : number of past turns to include in history (0 = off)
    """
    if llm is None:
        raise ValueError("llm is required (used for question contextualization).")

    conv_id: Optional[int] = st.session_state.get("active_conv_id")

    # ── Replay history ─────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and "model_info" in msg:
                mode_tag = " · CoRAG" if msg.get("chain_type") == "corag" else ""
                mem_tag  = f" · 🧠 {msg.get('memory_window', 0)}t" if msg.get("memory_window", 0) > 0 else ""
                st.caption(f"⚡ **Trả lời bởi:** `{msg['model_info']}`{mode_tag}{mem_tag}")
            if msg.get("chain_type") == "corag" and msg.get("trace"):
                _render_corag_trace(msg["trace"])
            if msg["role"] == "assistant" and msg.get("context_docs"):
                _render_context_docs(msg["context_docs"])
            st.markdown(msg["content"])

    # ── New input ──────────────────────────────────────────────────────────────
    if not (user_input := st.chat_input("Hỏi gì đó về tài liệu...")):
        return

    # ── Guard: ensure we have an active conversation ───────────────────────────
    if conv_id is None:
        conv_id = store.create_conversation()
        st.session_state.active_conv_id = conv_id

    model_info_str = f"{provider.upper()} - {model}"

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        mode_label = " · CoRAG" if chain_type == "corag" else ""
        mem_label  = f" · 🧠 {memory_window}t" if memory_window > 0 else ""
        st.caption(f"⚡ **Trả lời bởi:** `{model_info_str}`{mode_label}{mem_label}")

        # ── Build history string ───────────────────────────────────────────────
        prior = st.session_state.messages[:-1]
        chat_history = history_manager.format(prior, window=memory_window)

        # ── Contextualize question ─────────────────────────────────────────────
        ctx_question = user_input
        if chat_history:
            with st.spinner("🔁 Đang phân tích ngữ cảnh hội thoại..."):
                ctx_question = history_manager.contextualize(
                    question=user_input, history=chat_history, llm=llm
                )
        _render_history_badge(ctx_question, user_input)

        # ── Retrieve context docs (RAG only) ───────────────────────────────────
        context_docs = []
        if retriever is not None and chain_type != "corag":
            with st.spinner("Đang truy xuất context..."):
                try:
                    context_docs = retriever.invoke(ctx_question)
                except Exception:
                    context_docs = []

        # ── Run chain ──────────────────────────────────────────────────────────
        if chain_type == "corag":
            response, trace = _stream_corag(rag_chain, ctx_question, chat_history)
            _render_corag_trace(trace)
            st.markdown(response)
        else:
            response = st.write_stream(
                rag_chain.stream({"question": ctx_question, "chat_history": chat_history})
            )
            trace = None

        if context_docs:
            _render_context_docs(context_docs)

    # ── Append to session_state ────────────────────────────────────────────────
    assistant_msg = {
        "role":          "assistant",
        "content":       response,
        "model_info":    model_info_str,
        "chain_type":    chain_type,
        "trace":         trace,
        "context_docs":  context_docs,
        "memory_window": memory_window,
    }
    st.session_state.messages.append(assistant_msg)

    # ── Auto-save to SQLite ────────────────────────────────────────────────────
    _save_turn(
        store=store,
        conv_id=conv_id,
        user_input=user_input,
        response=response,
        model_info=model_info_str,
        chain_type=chain_type,
        memory_window=memory_window,
        trace=trace,
    )


def clear_chat_button():
    if st.button("🗑️ Làm mới cuộc trò chuyện", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()