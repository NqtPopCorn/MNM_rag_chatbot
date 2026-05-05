from langchain_core.prompts import ChatPromptTemplate

# ── RAG prompts (single-pass) ──────────────────────────────────────────────────

RAG_STRICT_TEMPLATE = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) If applicable, cite sources as (source: page:start_index) using the metadata.\n"
    "5) Answer in the SAME language as the question. Do NOT translate unless asked.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

RAG_BALANCED_TEMPLATE = (
    "You are a helpful assistant. Use the provided context as your primary source.\n"
    "If the context is insufficient, you may supplement with your general knowledge "
    "but clearly label it as: [General knowledge].\n"
    "Always cite sources from metadata when using context.\n"
    "Answer in the SAME language as the question.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

PROMPT_MODES = {
    "🔒 Strict (chỉ dùng tài liệu)": RAG_STRICT_TEMPLATE,
    "⚖️ Balanced (ưu tiên tài liệu + kiến thức nền)": RAG_BALANCED_TEMPLATE,
}


def get_prompt(mode: str = "🔒 Strict (chỉ dùng tài liệu)") -> ChatPromptTemplate:
    template = PROMPT_MODES.get(mode, RAG_STRICT_TEMPLATE)
    return ChatPromptTemplate.from_template(template)