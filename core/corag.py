"""
CoRAG — Chain-of-Retrieval Augmented Generation (history-aware)
===============================================================
Adds conversation memory on top of the iterative retrieval loop:

  1. contextualize()  — rewrite follow-up question into standalone query
                        (one fast LLM call, skipped on first turn)
  2. Iterative retrieve / evaluate loop (unchanged)
  3. Final answer generation with chat_history injected into the prompt

Public API (unchanged from callers' perspective):
  CoRAGChain.invoke(question, chat_history="") → {"answer", "trace"}
  CoRAGChain.stream(question, chat_history="") → generator[str]
"""

import json
import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever


# ── Internal prompts ───────────────────────────────────────────────────────────

_EVAL_TEMPLATE = """\
You are a research assistant performing iterative document retrieval.
{chat_history_block}
Original question: {question}

Context retrieved so far (iteration {iteration}/{max_iterations}):
{context}

Task: Decide whether the context above is sufficient to answer the question fully and accurately.
If there is a conversation history above, use it to understand any pronouns or references in the question.

Respond ONLY with a valid JSON object — no markdown fences, no extra text:
{{
  "sufficient": <true | false>,
  "reasoning": "<one-sentence explanation>",
  "follow_up_query": "<concise search query to fill the gap, or empty string if sufficient>"
}}
"""

_FINAL_STRICT_TEMPLATE = """\
You are a strict, citation-focused assistant for a private knowledge base.

RULES:
1) Use ONLY the provided context to answer.
2) If the answer is not clearly contained in the context, say exactly:
   "I don't know based on the provided documents."
3) Do NOT use outside knowledge or guessing.
4) Cite sources as (source: <filename>, page: <page>) using metadata when applicable.
5) Answer in the SAME language as the question.
6) If there is conversation history, use it to understand the question intent — but still answer ONLY from the context.

[Retrieved via {num_iterations} CoRAG iteration(s)]

{chat_history_block}
=== Retrieved Context ===
{context}
=== End of Context ===

Question: {question}
Answer:"""

_FINAL_BALANCED_TEMPLATE = """\
You are a helpful assistant. Use the provided context as your primary source.
If the context is insufficient, you may supplement with general knowledge but label it [General knowledge].
Always cite sources from metadata when using context.
Answer in the SAME language as the question.
If there is conversation history, use it to understand intent and maintain coherent dialogue.

[Retrieved via {num_iterations} CoRAG iteration(s)]

{chat_history_block}
=== Retrieved Context ===
{context}
=== End of Context ===

Question: {question}
Answer:"""

CORAG_FINAL_PROMPTS = {
    "🔒 Strict (chỉ dùng tài liệu)": _FINAL_STRICT_TEMPLATE,
    "⚖️ Balanced (ưu tiên tài liệu + kiến thức nền)": _FINAL_BALANCED_TEMPLATE,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(No documents retrieved yet.)"
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(f"[Doc {i}] (source: {src}, page: {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"sufficient": False, "reasoning": "Parse error", "follow_up_query": ""}


def _history_block(history: str) -> str:
    return f"Conversation History:\n{history}\n\n" if history.strip() else ""


# ── CoRAGChain ─────────────────────────────────────────────────────────────────

class CoRAGChain:
    """
    Parameters
    ----------
    llm             : any LangChain BaseChatModel
    retriever       : configured VectorStoreRetriever or HybridRetriever
    prompt_mode     : one of CORAG_FINAL_PROMPTS keys
    max_iterations  : maximum retrieval rounds (1–5 recommended)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        retriever,
        prompt_mode: str = "🔒 Strict (chỉ dùng tài liệu)",
        max_iterations: int = 3,
    ):
        self.llm = llm
        self.retriever = retriever
        self.max_iterations = max(1, max_iterations)

        self._eval_prompt = ChatPromptTemplate.from_template(_EVAL_TEMPLATE)

        final_tpl = CORAG_FINAL_PROMPTS.get(prompt_mode, _FINAL_STRICT_TEMPLATE)
        self._final_prompt = ChatPromptTemplate.from_template(final_tpl)
        self._parser = StrOutputParser()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _evaluate(
        self, question: str, context_str: str, iteration: int, chat_history: str = ""
    ) -> Tuple[bool, str, str]:
        chain = self._eval_prompt | self.llm | self._parser
        raw = chain.invoke({
            "question": question,
            "context": context_str,
            "iteration": iteration,
            "max_iterations": self.max_iterations,
            "chat_history_block": _history_block(chat_history),
        })
        parsed = _extract_json(raw)
        return (
            bool(parsed.get("sufficient", False)),
            parsed.get("reasoning", ""),
            parsed.get("follow_up_query", ""),
        )

    def _run_iterations(self, question: str, chat_history: str = "") -> Tuple[List[Document], List[dict]]:
        """Core iterative retrieval loop. Returns (all_docs, trace)."""
        all_docs: List[Document] = []
        seen: set = set()
        trace: List[dict] = []
        current_query = question

        for iteration in range(1, self.max_iterations + 1):
            new_docs = self.retriever.invoke(current_query)
            unique = [d for d in new_docs if d.page_content not in seen]
            for d in unique:
                seen.add(d.page_content)
            all_docs.extend(unique)

            step: dict = {
                "iteration": iteration,
                "query": current_query,
                "docs_retrieved": len(unique),
                "total_docs": len(all_docs),
            }

            if iteration < self.max_iterations:
                sufficient, reasoning, follow_up = self._evaluate(
                    question, _format_docs(all_docs), iteration, chat_history
                )
                step["sufficient"] = sufficient
                step["reasoning"] = reasoning
                if not sufficient and follow_up:
                    step["follow_up_query"] = follow_up
                    current_query = follow_up
                trace.append(step)
                if sufficient:
                    break
            else:
                step["sufficient"] = True
                step["reasoning"] = "Max iterations reached — using accumulated context."
                trace.append(step)

        return all_docs, trace

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(self, question: str, chat_history: str = "") -> dict:
        """
        Parameters
        ----------
        question     : current user question (already contextualized upstream)
        chat_history : formatted history string from HistoryManager.format()

        Returns {"answer": str, "trace": list[dict]}
        """
        all_docs, trace = self._run_iterations(question, chat_history)
        context_str = _format_docs(all_docs)

        answer_chain = self._final_prompt | self.llm | self._parser
        answer = answer_chain.invoke({
            "question": question,
            "context": context_str,
            "num_iterations": len(trace),
            "chat_history_block": _history_block(chat_history),
        })
        return {"answer": answer, "trace": trace}

    def stream(self, question: str, chat_history: str = ""):
        """
        Generator compatible with st.write_stream.

        Yields
        ------
        • One special marker chunk carrying the trace as JSON
        • Then text chunks of the final answer (streamed token-by-token)
        """
        all_docs, trace = self._run_iterations(question, chat_history)
        context_str = _format_docs(all_docs)

        yield f"\x00CORAG_TRACE\x00{json.dumps(trace, ensure_ascii=False)}\x00END\x00"

        answer_chain = self._final_prompt | self.llm | self._parser
        for chunk in answer_chain.stream({
            "question": question,
            "context": context_str,
            "num_iterations": len(trace),
            "chat_history_block": _history_block(chat_history),
        }):
            yield chunk