import os
import json
import streamlit as st
from langchain_core.runnables import Runnable

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
                color = "green" if step.get("sufficient") else "orange"
                icon  = "✅" if step.get("sufficient") else "🔄"
                st.markdown(
                    f"{icon} *{reasoning}*",
                    help="Đánh giá của LLM về mức độ đủ thông tin",
                )

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


def render_chat(
    rag_chain,
    provider: str,
    model: str,
    chain_type: str = "rag",   # "rag" | "corag"
    retriever=None,
):
    """Render vùng chat: hiển thị history + xử lý câu hỏi mới."""

    # ── Hiển thị lịch sử ──────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and "model_info" in msg:
                mode_tag = " · CoRAG" if msg.get("chain_type") == "corag" else ""
                st.caption(f"⚡ **Trả lời bởi:** `{msg['model_info']}`{mode_tag}")

            # Re-render CoRAG trace nếu có
            if msg.get("chain_type") == "corag" and msg.get("trace"):
                _render_corag_trace(msg["trace"])

            # Re-render context chunks nếu có
            if msg["role"] == "assistant" and msg.get("context_docs"):
                _render_context_docs(msg["context_docs"])

            st.markdown(msg["content"])

    # ── Input mới ─────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Hỏi gì đó về tài liệu..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        model_info_str = f"{provider.upper()} - {model}"

        with st.chat_message("assistant", avatar="🤖"):
            mode_label = " · CoRAG" if chain_type == "corag" else ""
            st.caption(f"⚡ **Trả lời bởi:** `{model_info_str}`{mode_label}")

            # Lấy context chunks trước khi stream (chỉ với RAG; CoRAG retrieve nội bộ)
            context_docs = []
            if retriever is not None and chain_type != "corag":
                with st.spinner("Đang truy xuất context..."):
                    try:
                        context_docs = retriever.invoke(user_input)
                    except Exception:
                        context_docs = []

            if chain_type == "corag":
                response, trace = _stream_corag(rag_chain, user_input)
                _render_corag_trace(trace)
                st.markdown(response)
            else:
                response = st.write_stream(rag_chain.stream(user_input))
                trace = None

            # Hiển thị context chunks sau câu trả lời
            if context_docs:
                _render_context_docs(context_docs)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "model_info": model_info_str,
            "chain_type": chain_type,
            "trace": trace,
            "context_docs": context_docs,
        })


def _stream_corag(chain, question: str):
    """
    Drive CoRAGChain.stream(), intercept the trace marker, collect remaining
    text into a string, and return (answer_text, trace_list).
    """
    trace: list[dict] = []
    answer_parts: list[str] = []
    trace_buf = ""
    in_trace = False
    answer_placeholder = st.empty()

    for chunk in chain.stream(question):
        # ── Detect / accumulate trace marker ──────────────────────────────────
        if _TRACE_START in chunk or in_trace:
            trace_buf += chunk
            if not in_trace:
                in_trace = True

            if _TRACE_END in trace_buf:
                # Extract JSON between sentinels
                start = trace_buf.index(_TRACE_START) + len(_TRACE_START)
                end   = trace_buf.index(_TRACE_END)
                try:
                    trace = json.loads(trace_buf[start:end])
                except json.JSONDecodeError:
                    trace = []

                # Show a status badge once trace is ready
                num_iter = len(trace)
                answer_placeholder.markdown(
                    f"*🔄 Hoàn tất {num_iter} vòng truy xuất — đang sinh câu trả lời...*"
                )

                # Remainder after sentinel (might contain first answer tokens)
                remainder = trace_buf[end + len(_TRACE_END):]
                if remainder:
                    answer_parts.append(remainder)
                    answer_placeholder.markdown("".join(answer_parts) + "▌")

                in_trace = False
                trace_buf = ""
            continue

        # ── Normal answer tokens ───────────────────────────────────────────────
        answer_parts.append(chunk)
        answer_placeholder.markdown("".join(answer_parts) + "▌")

    final_answer = "".join(answer_parts)
    answer_placeholder.empty()   # Clear streaming placeholder — caller renders markdown
    return final_answer, trace


def clear_chat_button():
    if st.button("🗑️ Làm mới cuộc trò chuyện", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()