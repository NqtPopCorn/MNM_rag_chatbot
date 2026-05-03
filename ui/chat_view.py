import os
import json
import streamlit as st
from langchain_core.runnables import Runnable

# ── Sentinel constants ─────────────────────────────────────────────────────────
_CORAG_START   = "\x00CORAG_TRACE\x00"
_CORAG_END     = "\x00END\x00"
_SELFRAG_START = "\x00SELFRAG_TRACE\x00"
_SELFRAG_END   = "\x00END\x00"


# ── CoRAG trace renderer ───────────────────────────────────────────────────────

def _render_corag_trace(trace: list[dict]):
    num = len(trace)
    with st.expander(f"🔄 CoRAG — {num} vòng truy xuất", expanded=False):
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
                st.markdown(
                    f"{'✅' if step.get('sufficient') else '🔄'} *{reasoning}*"
                )

            follow_up = step.get("follow_up_query", "")
            if follow_up:
                st.markdown(f"➡️ **Follow-up query:** `{follow_up}`")

            if i < num:
                st.divider()


# ── Self-RAG trace renderer ────────────────────────────────────────────────────

def _render_selfrag_trace(trace: dict):
    """Render a full Self-RAG pipeline trace in a structured expander."""
    if not trace:
        return

    retrieval_rounds = trace.get("retrieval_rounds", [])
    gen_attempts     = trace.get("generation_attempts", [])
    needed           = trace.get("retrieval_needed", True)
    final_source     = trace.get("final_source", "retrieval")
    total_relevant   = trace.get("total_relevant_docs", 0)

    # ── Header badge ──────────────────────────────────────────────────────────
    source_icon = {"retrieval": "📚", "direct": "🧠", "fallback": "⚠️"}.get(final_source, "📚")
    source_label = {
        "retrieval": "Từ tài liệu",
        "direct":    "Kiến thức nền (không cần retrieve)",
        "fallback":  "Fallback (không tìm được doc liên quan)",
    }.get(final_source, final_source)

    label = (
        f"🪞 Self-RAG · {source_icon} {source_label} · "
        f"{len(retrieval_rounds)} vòng retrieve · "
        f"{len(gen_attempts)} lần sinh"
    )

    with st.expander(label, expanded=False):

        # ── Retrieval decision ─────────────────────────────────────────────
        st.markdown("#### 1️⃣ Quyết định truy xuất")
        if needed:
            st.success(
                f"✅ Cần retrieve — {trace.get('retrieval_reason', '')}",
                icon="🔍",
            )
        else:
            st.info(
                f"🧠 Bỏ qua retrieve — {trace.get('retrieval_reason', '')}",
                icon="💡",
            )

        # ── Retrieval rounds ───────────────────────────────────────────────
        if retrieval_rounds:
            st.divider()
            st.markdown("#### 2️⃣–3️⃣ Truy xuất & Chấm điểm tài liệu")

            for rnd in retrieval_rounds:
                rnd_num   = rnd.get("round", "?")
                query     = rnd.get("query", "")
                fetched   = rnd.get("docs_fetched", 0)
                relevant  = rnd.get("docs_relevant", 0)
                grades    = rnd.get("grades", [])
                rewrite   = rnd.get("rewrite_reason", "")

                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.markdown(f"**Vòng {rnd_num}** · 🔍 `{query}`")
                with cols[1]:
                    st.metric("Tổng fetch", fetched)
                with cols[2]:
                    st.metric("Relevant", relevant, delta=None)

                if grades:
                    for g in grades:
                        color  = "🟢" if g.get("relevant") else "🔴"
                        src    = g.get("source", "?")
                        page   = g.get("page", "—")
                        reason = g.get("reason", "")
                        st.markdown(
                            f"{color} **Doc {g.get('doc_index', '?')+1}** "
                            f"`{src}` p.{page} — *{reason}*"
                        )

                if rewrite:
                    st.warning(f"⟳ Query rewrite: *{rewrite}*", icon="✏️")

                if rnd_num < len(retrieval_rounds):
                    st.divider()

        # ── Generation attempts ────────────────────────────────────────────
        if gen_attempts:
            st.divider()
            st.markdown("#### 4️⃣–6️⃣ Sinh câu trả lời & Kiểm tra")

            for att in gen_attempts:
                num        = att.get("attempt", "?")
                faithful   = att.get("faithful", True)
                f_reason   = att.get("faithfulness_reason", "")
                quality    = att.get("quality_score", 0)
                q_reason   = att.get("quality_reason", "")

                cols = st.columns([1, 2, 1, 2])
                with cols[0]:
                    st.markdown(f"**Lần {num}**")
                with cols[1]:
                    icon = "✅" if faithful else "❌"
                    st.markdown(f"{icon} Faithful — *{f_reason}*")
                with cols[2]:
                    stars = "⭐" * quality + "☆" * (5 - quality)
                    st.markdown(f"**{quality}/5** {stars}")
                with cols[3]:
                    st.markdown(f"*{q_reason}*")

        # ── Summary ────────────────────────────────────────────────────────
        st.divider()
        st.caption(
            f"📊 Tổng kết: {total_relevant} chunk relevant · "
            f"{len(retrieval_rounds)} vòng retrieve · "
            f"{len(gen_attempts)} lần sinh · nguồn: **{final_source}**"
        )


# ── Context doc renderer ───────────────────────────────────────────────────────

def _render_context_docs(docs: list):
    if not docs:
        return
    with st.expander(f"📚 Context — {len(docs)} chunk đã dùng", expanded=False):
        colors = ["#4f8ef7", "#a78bfa", "#34d399", "#f59e0b", "#f87171", "#38bdf8", "#fb923c"]
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
            import html
            safe_content = html.escape(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))

            st.markdown(header, unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
                f'border-left:3px solid {color};border-radius:6px;padding:10px 14px;'
                f'font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;line-height:1.6;'
                f'white-space:pre-wrap;word-break:break-word;color:grey;margin-bottom:10px;">'
                f'{safe_content}</div>',
                unsafe_allow_html=True,
            )


# ── Stream helpers ─────────────────────────────────────────────────────────────

def _stream_corag(chain, question: str):
    """
    Drive CoRAGChain.stream(), intercept trace marker, return (answer, trace).
    """
    trace: list[dict] = []
    answer_parts: list[str] = []
    trace_buf = ""
    in_trace = False
    placeholder = st.empty()

    for chunk in chain.stream(question):
        if _CORAG_START in chunk or in_trace:
            trace_buf += chunk
            in_trace = True
            if _CORAG_END in trace_buf:
                start = trace_buf.index(_CORAG_START) + len(_CORAG_START)
                end   = trace_buf.index(_CORAG_END)
                try:
                    trace = json.loads(trace_buf[start:end])
                except json.JSONDecodeError:
                    trace = []
                num_iter = len(trace)
                placeholder.markdown(
                    f"*🔄 Hoàn tất {num_iter} vòng truy xuất — đang sinh câu trả lời...*"
                )
                remainder = trace_buf[end + len(_CORAG_END):]
                if remainder:
                    answer_parts.append(remainder)
                    placeholder.markdown("".join(answer_parts) + "▌")
                in_trace = False
                trace_buf = ""
            continue
        answer_parts.append(chunk)
        placeholder.markdown("".join(answer_parts) + "▌")

    placeholder.empty()
    return "".join(answer_parts), trace


def _stream_selfrag(chain, question: str):
    """
    Drive SelfRAGChain.stream(), intercept trace sentinel, return (answer, trace_dict).
    """
    trace: dict = {}
    answer_parts: list[str] = []
    trace_buf = ""
    in_trace = False
    placeholder = st.empty()

    for chunk in chain.stream(question):
        if _SELFRAG_START in chunk or in_trace:
            trace_buf += chunk
            in_trace = True
            if _SELFRAG_END in trace_buf:
                start = trace_buf.index(_SELFRAG_START) + len(_SELFRAG_START)
                end   = trace_buf.index(_SELFRAG_END)
                try:
                    trace = json.loads(trace_buf[start:end])
                except json.JSONDecodeError:
                    trace = {}

                # Show status while final answer streams
                rounds = len(trace.get("retrieval_rounds", []))
                relevant = trace.get("total_relevant_docs", 0)
                placeholder.markdown(
                    f"*🪞 Self-RAG: {rounds} vòng retrieve · "
                    f"{relevant} chunk relevant — đang sinh câu trả lời...*"
                )

                remainder = trace_buf[end + len(_SELFRAG_END):]
                if remainder:
                    answer_parts.append(remainder)
                    placeholder.markdown("".join(answer_parts) + "▌")
                in_trace = False
                trace_buf = ""
            continue

        answer_parts.append(chunk)
        placeholder.markdown("".join(answer_parts) + "▌")

    placeholder.empty()
    return "".join(answer_parts), trace


# ── Main render ────────────────────────────────────────────────────────────────

def render_chat(
    rag_chain,
    provider: str,
    model: str,
    chain_type: str = "rag",
    retriever=None,
):
    # ── History ───────────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and "model_info" in msg:
                mode_tag = {
                    "corag":   " · CoRAG",
                    "selfrag": " · Self-RAG",
                }.get(msg.get("chain_type", ""), "")
                st.caption(f"⚡ **Trả lời bởi:** `{msg['model_info']}`{mode_tag}")

            if msg.get("chain_type") == "corag" and msg.get("trace"):
                _render_corag_trace(msg["trace"])

            if msg.get("chain_type") == "selfrag" and msg.get("trace"):
                _render_selfrag_trace(msg["trace"])

            if msg["role"] == "assistant" and msg.get("context_docs"):
                _render_context_docs(msg["context_docs"])

            st.markdown(msg["content"])

    # ── New input ─────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Hỏi gì đó về tài liệu..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        model_info_str = f"{provider.upper()} - {model}"

        with st.chat_message("assistant", avatar="🤖"):
            mode_label = {
                "corag":   " · CoRAG",
                "selfrag": " · Self-RAG",
            }.get(chain_type, "")
            st.caption(f"⚡ **Trả lời bởi:** `{model_info_str}`{mode_label}")

            context_docs = []
            trace = None

            if chain_type == "corag":
                response, trace = _stream_corag(rag_chain, user_input)
                _render_corag_trace(trace)
                st.markdown(response)

            elif chain_type == "selfrag":
                response, trace = _stream_selfrag(rag_chain, user_input)
                _render_selfrag_trace(trace)
                st.markdown(response)

            else:
                # Classic RAG — pull context docs for display
                if retriever is not None:
                    with st.spinner("Đang truy xuất context..."):
                        try:
                            context_docs = retriever.invoke(user_input)
                        except Exception:
                            context_docs = []
                response = st.write_stream(rag_chain.stream(user_input))

            if context_docs:
                _render_context_docs(context_docs)

        st.session_state.messages.append({
            "role":         "assistant",
            "content":      response,
            "model_info":   model_info_str,
            "chain_type":   chain_type,
            "trace":        trace,
            "context_docs": context_docs,
        })


def clear_chat_button():
    if st.button("🗑️ Làm mới cuộc trò chuyện", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()