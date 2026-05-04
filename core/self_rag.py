"""
Self-RAG — Self-Reflective Retrieval Augmented Generation
=========================================================
Adaptive pipeline where the model critiques its own retrieval and generation.

Change vs. original
───────────────────
The question fed to the retrieval decision + first retrieval round is now
contextualized via HistoryManager.  The original (user-visible) question is
preserved for grading, quality scoring, and the final answer prompt so that
the user's phrasing is never altered in the output.

  retrieval_q  = contextualize(question, history)   ← used for vector search
  display_q    = question                            ← used in prompts/trace

Public API:
  SelfRAGChain.invoke(question, chat_history) → {"answer": str, "trace": dict}
  SelfRAGChain.stream(question, chat_history) → generator[str]
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from .memory import HistoryManager, history_manager as _default_hm

# ── Prompt Templates ───────────────────────────────────────────────────────────

_RETRIEVAL_DECISION_TEMPLATE = """\
You are a routing assistant. Decide whether the question below requires \
searching a private knowledge base to answer accurately.

Chat History:
{chat_history}

Question: {question}

Rules:
- Answer "yes" if the question asks about specific facts, data, or content \
that likely exists in specialized documents (research papers, reports, manuals).
- Answer "no" if the question is general knowledge, a simple calculation, \
a greeting, or something you can answer confidently without documents.

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{"retrieve": <true | false>, "reason": "<one sentence>"}}
"""

_DOCUMENT_GRADE_TEMPLATE = """\
You are a strict relevance grader. Assess whether the retrieved document \
contains information that is DIRECTLY useful for answering the question.

Chat History:
{chat_history}

Question: {question}

Document:
{document}

Criteria:
- "relevant" if the document contains facts, data, or context that directly \
help answer the question.
- "irrelevant" if the document is off-topic or contains only tangential info.

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{"relevant": <true | false>, "reason": "<one sentence>"}}
"""

_FAITHFULNESS_TEMPLATE = """\
You are a faithfulness auditor. Check whether the answer is fully grounded \
in (supported by) the provided context. Every key claim in the answer must \
be traceable to the context.

Chat History:
{chat_history}

Context:
{context}

Answer:
{answer}

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{"faithful": <true | false>, "reason": "<one sentence explaining any unsupported claims>"}}
"""

_ANSWER_QUALITY_TEMPLATE = """\
You are an answer quality evaluator. Rate how well the answer addresses \
the original question on a scale of 1 to 5.

Scoring guide:
  5 — Complete, accurate, directly answers the question with appropriate detail.
  4 — Good answer, minor gaps or slight imprecision.
  3 — Partially answers, missing key aspects or somewhat vague.
  2 — Superficial or largely off-target.
  1 — Does not answer the question or is factually wrong.

Chat History:
{chat_history}

Question: {question}
Answer: {answer}

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{"score": <1-5>, "reason": "<one sentence>"}}
"""

_QUERY_REWRITE_TEMPLATE = """\
You are a query optimization assistant. The original query failed to retrieve \
useful documents or produced a low-quality answer. Rewrite the query to be \
more specific, use different keywords, and better target the relevant information.

Chat History:
{chat_history}

Original question: {question}
Previous query used: {previous_query}
Reason for rewrite: {reason}

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{"rewritten_query": "<improved search query, concise and specific>"}}
"""

_DIRECT_ANSWER_TEMPLATE = """\
You are a knowledgeable assistant. Answer the following question using your \
general knowledge. Be concise and accurate.
Answer in the SAME language as the question.

Chat History:
{chat_history}

Question: {question}
Answer:"""

_FINAL_STRICT_TEMPLATE = """\
You are a strict, citation-focused assistant for a private knowledge base.

RULES:
1) Use ONLY the provided context to answer.
2) If the answer is not clearly in the context, say exactly: \
"I don't know based on the provided documents."
3) Do NOT use outside knowledge or guessing.
4) Cite sources as (source: <filename>, page: <page>) using metadata.
5) Answer in the SAME language as the question.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
Answer:"""

_FINAL_BALANCED_TEMPLATE = """\
You are a helpful assistant. Use the provided context as your primary source.
If context is insufficient, supplement with general knowledge but label it \
[General knowledge].
Always cite sources from metadata when using context.
Answer in the SAME language as the question.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
Answer:"""

SELFRAG_FINAL_PROMPTS = {
    "🔒 Strict (chỉ dùng tài liệu)": _FINAL_STRICT_TEMPLATE,
    "⚖️ Balanced (ưu tiên tài liệu + kiến thức nền)": _FINAL_BALANCED_TEMPLATE,
}


# ── Data Classes (Trace) ───────────────────────────────────────────────────────

@dataclass
class DocGrade:
    doc_index: int
    source: str
    page: object
    relevant: bool
    reason: str


@dataclass
class RetrievalRound:
    round: int
    query: str
    docs_fetched: int
    docs_relevant: int
    grades: List[DocGrade] = field(default_factory=list)
    rewrite_reason: str = ""


@dataclass
class GenerationAttempt:
    attempt: int
    faithful: bool
    faithfulness_reason: str
    quality_score: int
    quality_reason: str


@dataclass
class SelfRAGTrace:
    question: str
    retrieval_needed: bool
    retrieval_reason: str
    retrieval_rounds: List[RetrievalRound]     = field(default_factory=list)
    generation_attempts: List[GenerationAttempt] = field(default_factory=list)
    final_source: str = "retrieval"
    total_relevant_docs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(No relevant documents retrieved.)"
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        src  = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(f"[Doc {i}] (source: {src}, page: {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _doc_meta(doc: Document) -> Tuple[str, object]:
    import os
    meta = doc.metadata or {}
    return os.path.basename(meta.get("source", "unknown")), meta.get("page", "—")


# ── SelfRAGChain ───────────────────────────────────────────────────────────────

class SelfRAGChain:
    """
    Parameters
    ----------
    llm                     : LangChain BaseChatModel
    retriever               : configured VectorStoreRetriever
    prompt_mode             : key of SELFRAG_FINAL_PROMPTS
    max_retrieval_attempts  : max query rewrites / retrieval rounds (default 2)
    max_generation_attempts : max regeneration cycles on faithfulness failure (default 2)
    quality_threshold       : min answer quality score 1–5 (default 3)
    hm                      : HistoryManager instance (uses module singleton if None)
    """

    TRACE_START = "\x00SELFRAG_TRACE\x00"
    TRACE_END   = "\x00END\x00"

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: VectorStoreRetriever,
        prompt_mode: str = "🔒 Strict (chỉ dùng tài liệu)",
        max_retrieval_attempts: int = 2,
        max_generation_attempts: int = 2,
        quality_threshold: int = 3,
        hm: HistoryManager | None = None,
    ):
        self.llm                     = llm
        self.retriever               = retriever
        self.max_retrieval_attempts  = max(1, max_retrieval_attempts)
        self.max_generation_attempts = max(1, max_generation_attempts)
        self.quality_threshold       = max(1, min(5, quality_threshold))
        self._parser                 = StrOutputParser()
        self._hm                     = hm or _default_hm

        self._p_decide   = ChatPromptTemplate.from_template(_RETRIEVAL_DECISION_TEMPLATE)
        self._p_grade    = ChatPromptTemplate.from_template(_DOCUMENT_GRADE_TEMPLATE)
        self._p_faithful = ChatPromptTemplate.from_template(_FAITHFULNESS_TEMPLATE)
        self._p_quality  = ChatPromptTemplate.from_template(_ANSWER_QUALITY_TEMPLATE)
        self._p_rewrite  = ChatPromptTemplate.from_template(_QUERY_REWRITE_TEMPLATE)
        self._p_direct   = ChatPromptTemplate.from_template(_DIRECT_ANSWER_TEMPLATE)

        final_tpl        = SELFRAG_FINAL_PROMPTS.get(prompt_mode, _FINAL_STRICT_TEMPLATE)
        self._p_final    = ChatPromptTemplate.from_template(final_tpl)

    # ── Step helpers ─────────────────────────────────────────────────────────

    def _decide_retrieval(self, question: str, chat_history: str = "") -> Tuple[bool, str]:
        chain  = self._p_decide | self.llm | self._parser
        raw    = chain.invoke({"question": question, "chat_history": chat_history})
        parsed = _extract_json(raw)
        return bool(parsed.get("retrieve", True)), parsed.get("reason", "")

    def _grade_documents(
        self, question: str, docs: List[Document], chat_history: str = ""
    ) -> List[Tuple[Document, DocGrade]]:
        chain   = self._p_grade | self.llm | self._parser
        results = []
        for i, doc in enumerate(docs):
            raw    = chain.invoke({
                "question":     question,
                "document":     doc.page_content[:2000],
                "chat_history": chat_history,
            })
            parsed = _extract_json(raw)
            src, page = _doc_meta(doc)
            results.append((doc, DocGrade(
                doc_index = i,
                source    = src,
                page      = page,
                relevant  = bool(parsed.get("relevant", False)),
                reason    = parsed.get("reason", ""),
            )))
        return results

    def _check_faithfulness(
        self, context: str, answer: str, chat_history: str = ""
    ) -> Tuple[bool, str]:
        chain  = self._p_faithful | self.llm | self._parser
        raw    = chain.invoke({"context": context, "answer": answer, "chat_history": chat_history})
        parsed = _extract_json(raw)
        return bool(parsed.get("faithful", True)), parsed.get("reason", "")

    def _score_quality(
        self, question: str, answer: str, chat_history: str = ""
    ) -> Tuple[int, str]:
        chain  = self._p_quality | self.llm | self._parser
        raw    = chain.invoke({"question": question, "answer": answer, "chat_history": chat_history})
        parsed = _extract_json(raw)
        return max(1, min(5, int(parsed.get("score", 3)))), parsed.get("reason", "")

    def _rewrite_query(
        self, question: str, previous_query: str, reason: str, chat_history: str = ""
    ) -> str:
        chain  = self._p_rewrite | self.llm | self._parser
        raw    = chain.invoke({
            "question":       question,
            "previous_query": previous_query,
            "reason":         reason,
            "chat_history":   chat_history,
        })
        return _extract_json(raw).get("rewritten_query", question)

    def _generate_answer(self, question: str, context: str, chat_history: str = "") -> str:
        return (self._p_final | self.llm | self._parser).invoke(
            {"question": question, "context": context, "chat_history": chat_history}
        )

    def _generate_direct(self, question: str, chat_history: str = "") -> str:
        return (self._p_direct | self.llm | self._parser).invoke(
            {"question": question, "chat_history": chat_history}
        )

    def _stream_answer(self, question: str, context: str, chat_history: str = ""):
        yield from (self._p_final | self.llm | self._parser).stream(
            {"question": question, "context": context, "chat_history": chat_history}
        )

    def _stream_direct(self, question: str, chat_history: str = ""):
        yield from (self._p_direct | self.llm | self._parser).stream(
            {"question": question, "chat_history": chat_history}
        )

    # ── Core pipeline ────────────────────────────────────────────────────────

    def _run_pipeline(
        self, question: str, chat_history: str = ""
    ) -> Tuple[List[Document], SelfRAGTrace, bool]:
        """
        Execute the full Self-RAG pipeline.

        Two query forms are maintained:
          retrieval_q — contextualized, used for vector search
          question    — original user question, used in prompts & trace
        Returns (relevant_docs, trace, is_direct_answer).
        """
        trace = SelfRAGTrace(
            question          = question,
            retrieval_needed  = True,
            retrieval_reason  = "",
        )

        # ── Step 1: Retrieval Decision ────────────────────────────────────────
        # Contextualize before deciding — the LLM needs full context to route correctly
        retrieval_q = self._hm.contextualize(question, chat_history, self.llm)

        retrieve, reason = self._decide_retrieval(retrieval_q, chat_history)
        trace.retrieval_needed = retrieve
        trace.retrieval_reason = reason

        if not retrieve:
            trace.final_source = "direct"
            return [], trace, True

        # ── Steps 2–3: Retrieval + Grading loop ──────────────────────────────
        current_query = retrieval_q          # first query is contextualized
        all_relevant: List[Document] = []
        seen_content: set = set()

        for attempt in range(1, self.max_retrieval_attempts + 1):
            raw_docs = self.retriever.invoke(current_query)
            graded   = self._grade_documents(question, raw_docs, chat_history)

            round_info = RetrievalRound(
                round          = attempt,
                query          = current_query,
                docs_fetched   = len(raw_docs),
                docs_relevant  = 0,
                grades         = [g for _, g in graded],
            )

            new_relevant = [
                doc for doc, grade in graded
                if grade.relevant and doc.page_content not in seen_content
            ]
            for doc in new_relevant:
                seen_content.add(doc.page_content)

            round_info.docs_relevant = len(new_relevant)
            all_relevant.extend(new_relevant)
            trace.retrieval_rounds.append(round_info)

            if all_relevant:
                break

            if attempt < self.max_retrieval_attempts:
                rewrite_reason = "No relevant documents found — trying a different query."
                current_query  = self._rewrite_query(
                    question, current_query, rewrite_reason, chat_history
                )
                round_info.rewrite_reason = rewrite_reason

        trace.total_relevant_docs = len(all_relevant)
        if not all_relevant:
            trace.final_source = "fallback"

        return all_relevant, trace, False

    def _run_generation(
        self,
        question: str,
        relevant_docs: List[Document],
        trace: SelfRAGTrace,
        is_direct: bool,
        chat_history: str = "",
    ) -> str:
        if is_direct:
            return self._generate_direct(question, chat_history)

        context = _format_docs(relevant_docs)

        for attempt in range(1, self.max_generation_attempts + 1):
            answer = self._generate_answer(question, context, chat_history)

            if relevant_docs:
                faithful, faithful_reason = self._check_faithfulness(context, answer, chat_history)
            else:
                faithful, faithful_reason = True, "No context to check."

            quality_score, quality_reason = self._score_quality(question, answer, chat_history)

            trace.generation_attempts.append(GenerationAttempt(
                attempt             = attempt,
                faithful            = faithful,
                faithfulness_reason = faithful_reason,
                quality_score       = quality_score,
                quality_reason      = quality_reason,
            ))

            if faithful and quality_score >= self.quality_threshold:
                break
            if attempt == self.max_generation_attempts:
                break

        trace.final_source = (
            "direct"   if is_direct
            else "fallback" if not relevant_docs
            else "retrieval"
        )
        return answer

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(self, question: str, chat_history: str = "") -> dict:
        relevant_docs, trace, is_direct = self._run_pipeline(question, chat_history)
        answer = self._run_generation(question, relevant_docs, trace, is_direct, chat_history)
        return {"answer": answer, "trace": trace.to_dict()}

    def stream(self, question: str, chat_history: str = ""):
        relevant_docs, trace, is_direct = self._run_pipeline(question, chat_history)
        context = _format_docs(relevant_docs)

        for attempt in range(1, self.max_generation_attempts + 1):
            if is_direct:
                answer_for_grading = self._generate_direct(question, chat_history)
            else:
                answer_for_grading = self._generate_answer(question, context, chat_history)

            if relevant_docs and not is_direct:
                faithful, faithful_reason = self._check_faithfulness(
                    context, answer_for_grading, chat_history
                )
            else:
                faithful, faithful_reason = True, "No context to check."

            quality_score, quality_reason = self._score_quality(
                question, answer_for_grading, chat_history
            )

            trace.generation_attempts.append(GenerationAttempt(
                attempt             = attempt,
                faithful            = faithful,
                faithfulness_reason = faithful_reason,
                quality_score       = quality_score,
                quality_reason      = quality_reason,
            ))

            if faithful and quality_score >= self.quality_threshold:
                break
            if attempt == self.max_generation_attempts:
                break

        trace.final_source = (
            "direct"    if is_direct
            else "fallback" if not relevant_docs
            else "retrieval"
        )

        yield (
            f"{self.TRACE_START}"
            f"{json.dumps(trace.to_dict(), ensure_ascii=False)}"
            f"{self.TRACE_END}"
        )

        if is_direct:
            yield from self._stream_direct(question, chat_history)
        else:
            yield from self._stream_answer(question, context, chat_history)