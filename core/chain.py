"""
core/chain.py
─────────────
Builds the standard (single-pass) RAG chain.

Pipeline
────────
  question + history
       │
       ▼
  [contextualize]  ← rewrites follow-up questions into standalone queries
       │                before they hit the vector store
       ▼
  [retrieve]       ← FAISS / hybrid search on the contextualized query
       │
       ▼
  [generate]       ← LLM reads retrieved context + original question
       │
       ▼
     answer (str, streaming)

Why contextualize BEFORE retrieve (not just prepend history to the prompt)?
  The prompt already receives the history for the LLM to read. But retrieval
  happens *before* the LLM sees anything. Sending "nó có nhược điểm gì?" to
  the vector store matches nothing useful; "Nhược điểm của Attention pooling
  là gì?" matches relevant chunks. Contextualization fixes the retrieval step.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from .memory import HistoryManager, history_manager as _default_hm


def build_rag_chain(
    llm: BaseChatModel,
    retriever,
    prompt: ChatPromptTemplate,
    hm: HistoryManager | None = None,
) -> Runnable:
    """
    Assemble a RAG chain using LCEL.

    Parameters
    ----------
    llm       : any LangChain BaseChatModel
    retriever : vector / hybrid retriever with .invoke(query) → List[Document]
    prompt    : ChatPromptTemplate with {context}, {question}, {chat_history}
    hm        : HistoryManager instance (uses module singleton if None)

    Input dict
    ----------
    {
      "question":     str,
      "chat_history": str   # pre-formatted by HistoryManager.format()
    }
    """
    _hm = hm or _default_hm

    def _get_context(x: dict) -> list:
        """Contextualize → retrieve.  Called once per user turn."""
        question = x["question"]
        history  = x.get("chat_history", "")

        # Rewrite follow-up question to standalone before hitting FAISS
        standalone_q = _hm.contextualize(question, history, llm)
        return retriever.invoke(standalone_q)

    return (
        {
            "context":      RunnableLambda(_get_context),
            "question":     lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", ""),
        }
        | prompt
        | llm
        | StrOutputParser()
    )