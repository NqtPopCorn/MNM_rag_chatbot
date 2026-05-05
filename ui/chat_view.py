import os
import json
import streamlit as st
from langchain_core.runnables import Runnable
from core.memory import history_manager

# Sentinel prefix emitted by CoRAGChain.stream()
_TRACE_START = "\x00CORAG_TRACE\x00"
_TRACE_END   = "\x00END\x00"


def _render_corag_trace(trace: list[dict]):
    """Render CoRAG iteration trace inside an expander."""
    num = len(trace)
    label = f"🔄 CoRAG — {num} vòng truy xuất"
    with st.expander(label, expanded=False):
        for step in trace:
            i = step["iteration"]
            st.markdown(f"**Vòng {i}**")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"🔍 **Query:** `{step['query']}`")
            with col2:
                st.markdown(
                    f"📄 Tìm được **{step['docs_retrieved']}** chunk mới "
                    f"(tổng: {step['total_docs']})"
                )

            reasoning = step.get("reasoning", "")
            if reasoning:
                icon = "✅" if step.get("sufficient") else "🔄"
                st.markdown(f"{icon} *{reasoning}*",
                            help="Đánh giá của LLM về mức độ đủ thông tin")

            follow_up = step.get("follow_up_query", "")
            if follow_up:
                st.markdown(f"➡️ **Follow-up query:** `{follow_up}`")

            if i < num:
                st.divider()


def _render_context_docs(docs: list):
    """Hiển thị các chunk context đã dùng để trả lời trong expander."""
    if not docs:
        return
    with st.expander(f"📚 Context — {len(docs)} chunk đã dùng", expanded=False):
        colors = ["#4f8ef7", "#a78bfa", "#34d399", "#f59e0b", "#f87171", "#38bdf8", "#fb923c"]
        sources_seen: list[str] = []
        for doc in docs:
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            if src not in sources_seen:
                sources_seen.append(src)

        for i, doc in enumerate(docs):
            meta = doc.metadata or {}
            src = os.path.basename(meta.get("source", "unknown"))
            page = meta.get("page", "—")
            start = meta.get("start_index", 0)
            color = colors[sources_seen.index(src) % len(colors)] if src in sources_seen else colors[0]

            header = (
                f'<span style="display:inline-block;background:{color}20;border:1px solid {color}50;'
                f'border-radius:20px;padding:2px 10px;font-size:0.72rem;color:{color};'
                f'font-family:monospace;margin-right:6px;">📄 {src}</span>'
                f'<span style="display:inline-block;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);'
                f'border-radius:20px;padding:2px 10px;font-size:0.72rem;color:rgba(255,255,255,0.6);'
                f'font-family:monospace;margin-right:6px;">Trang {page}</span>'
                f'<span style="display:inline-block;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);'
                f'border-radius:20px;padding:2px 10px;font-size:0.72rem;color:rgba(255,255,255,0.6);'
                f'font-family:monospace;">Vị trí {start}</span>'
            )
            content_preview = doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else "")
            import html
            safe_content = html.escape(content_preview)

            st.markdown(header, unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
                f'border-left:3px solid {color};border-radius:6px;padding:10px 14px;'
                f'font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;line-height:1.6;'
                f'white-space:pre-wrap;word-break:break-word;color:grey;margin-bottom:10px;">'
                f'{safe_content}</div>',
                unsafe_allow_html=True,
            )


def _render_history_badge(contextualized: str, original: str):
    """Show a small badge when the query was rewritten by contextualize()."""
    if contextualized != original:
        with st.expander("🔁 Câu hỏi đã được viết lại (contextualize)", expanded=False):
            st.caption(f"**Gốc:** {original}")
            st.caption(f"**Viết lại:** {contextualized}")


def render_chat(
    rag_chain,
    provider: str,
    model: str,
    chain_type: str = "rag",
    retriever=None,
    llm=None,
    memory_window: int = 3,
):
    """
    Render the chat area: display history + handle new user input.

    Parameters
    ----------
    rag_chain      : RAG chain (Runnable) or CoRAGChain instance
    provider       : LLM provider string (for the caption badge)
    model          : LLM model string
    chain_type     : "rag" | "corag"
    retriever      : used for context-doc retrieval on RAG turns
    llm            : BaseChatModel — needed for contextualize() LLM call
    memory_window  : how many past turns to include (0 = disabled)
    """

    if llm is None:
        raise ValueError("LLM instance is required for chat rendering (for contextualize function).")

    # ── Display history ────────────────────────────────────────────────────────
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
    if user_input := st.chat_input("Hỏi gì đó về tài liệu..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        model_info_str = f"{provider.upper()} - {model}"

        with st.chat_message("assistant", avatar="🤖"):
            mode_label = " · CoRAG" if chain_type == "corag" else ""
            mem_label  = f" · 🧠 {memory_window}t" if memory_window > 0 else ""
            st.caption(f"⚡ **Trả lời bởi:** `{model_info_str}`{mode_label}{mem_label}")

            # ── Build chat history string ──────────────────────────────────────
            # Exclude the message we just appended (it hasn't been answered yet)
            prior_messages = st.session_state.messages[:-1]
            chat_history = history_manager.format(prior_messages, window=memory_window)

            # ── Contextualize question (rewrite follow-ups into standalone) ────
            contextualized_question = user_input
            if chat_history and llm is not None:
                print("Contextualizing question with LLM...")
                with st.spinner("🔁 Đang phân tích ngữ cảnh hội thoại..."):
                    contextualized_question = history_manager.contextualize(
                        question=user_input,
                        history=chat_history,
                        llm=llm,
                    )
            print(f"Contextualized question: '{contextualized_question}'")
            _render_history_badge(contextualized_question, user_input)

            # ── Retrieve context docs (RAG only) ──────────────────────────────
            context_docs = []
            if retriever is not None and chain_type != "corag":
                with st.spinner("Đang truy xuất context..."):
                    try:
                        context_docs = retriever.invoke(contextualized_question)
                    except Exception:
                        context_docs = []

            # ── Run chain ─────────────────────────────────────────────────────
            if chain_type == "corag":
                response, trace = _stream_corag(
                    rag_chain,
                    question=contextualized_question,
                    chat_history=chat_history,
                )
                _render_corag_trace(trace)
                st.markdown(response)
            else:
                response = st.write_stream(
                    rag_chain.stream({
                        "question": contextualized_question,
                        "chat_history": chat_history,
                    })
                )
                trace = None

            if context_docs:
                _render_context_docs(context_docs)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "model_info": model_info_str,
            "chain_type": chain_type,
            "trace": trace,
            "context_docs": context_docs,
            "memory_window": memory_window,
        })


def _stream_corag(chain, question: str, chat_history: str = ""):
    """
    Drive CoRAGChain.stream(), intercept the trace marker, collect remaining
    text into a string, and return (answer_text, trace_list).
    """
    trace: list[dict] = []
    answer_parts: list[str] = []
    trace_buf = ""
    in_trace = False
    answer_placeholder = st.empty()

    for chunk in chain.stream(question, chat_history):
        if _TRACE_START in chunk or in_trace:
            trace_buf += chunk
            if not in_trace:
                in_trace = True

            if _TRACE_END in trace_buf:
                start = trace_buf.index(_TRACE_START) + len(_TRACE_START)
                end   = trace_buf.index(_TRACE_END)
                try:
                    trace = json.loads(trace_buf[start:end])
                except json.JSONDecodeError:
                    trace = []

                num_iter = len(trace)
                answer_placeholder.markdown(
                    f"*🔄 Hoàn tất {num_iter} vòng truy xuất — đang sinh câu trả lời...*"
                )

                remainder = trace_buf[end + len(_TRACE_END):]
                if remainder:
                    answer_parts.append(remainder)
                    answer_placeholder.markdown("".join(answer_parts) + "▌")

                in_trace = False
                trace_buf = ""
            continue

        answer_parts.append(chunk)
        answer_placeholder.markdown("".join(answer_parts) + "▌")

    final_answer = "".join(answer_parts)
    answer_placeholder.empty()
    return final_answer, trace


def clear_chat_button():
    if st.button("🗑️ Làm mới cuộc trò chuyện", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()