"""
Self-RAG — Self-Reflective Retrieval Augmented Generation
=========================================================
An adaptive pipeline where the model critiques its own retrieval and generation:

  Step 1 · [Retrieval Decision]
      Does this question actually need document retrieval?
      → "no"  → answer directly from LLM knowledge (skip to Step 5)
      → "yes" → proceed

  Step 2 · [Retrieval]
      Fetch top-k chunks from FAISS with the current query.

  Step 3 · [Document Grading]
      Score each retrieved chunk: relevant | irrelevant.
      → Keep only relevant chunks.
      → If zero relevant docs AND attempts < max_retrieval_attempts:
          rewrite query → back to Step 2.
      → If still empty after max attempts: answer with empty context.

  Step 4 · [Answer Generation]
      Produce a candidate answer from the relevant context.

  Step 5 · [Faithfulness Check]
      Is the answer grounded in (supported by) the retrieved docs?
      → Not faithful AND attempts < max_generation_attempts → regenerate (Step 4).

  Step 6 · [Answer Quality Score]
      Rate usefulness 1–5.
      → Score < quality_threshold AND retrieval_rounds < max_retrieval_attempts:
          rewrite query → back to Step 2.
      → Otherwise → emit final answer.

Public API:
  SelfRAGChain.invoke(question) → {"answer": str, "trace": dict}
  SelfRAGChain.stream(question) → generator[str]   (st.write_stream compatible)

Trace sentinel (stream mode):
  \\x00SELFRAG_TRACE\\x00<json>\\x00END\\x00
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


# ── Prompt Templates ───────────────────────────────────────────────────────────

_RETRIEVAL_DECISION_TEMPLATE = """\
You are a routing assistant. Decide whether the question below requires \
searching a private knowledge base to answer accurately.

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

Question: {question}
Answer: {answer}

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{"score": <1-5>, "reason": "<one sentence>"}}
"""

_QUERY_REWRITE_TEMPLATE = """\
You are a query optimization assistant. The original query failed to retrieve \
useful documents or produced a low-quality answer. Rewrite the query to be \
more specific, use different keywords, and better target the relevant information.

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
    retrieval_rounds: List[RetrievalRound] = field(default_factory=list)
    generation_attempts: List[GenerationAttempt] = field(default_factory=list)
    final_source: str = "retrieval"   # "retrieval" | "direct" | "fallback"
    total_relevant_docs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Robustly extract JSON — strips markdown fences if present."""
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
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(
            f"[Doc {i}] (source: {src}, page: {page})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _doc_meta(doc: Document) -> Tuple[str, object]:
    import os
    meta = doc.metadata or {}
    src = os.path.basename(meta.get("source", "unknown"))
    page = meta.get("page", "—")
    return src, page


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
    quality_threshold       : min answer quality score (1–5) to accept (default 3)
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
    ):
        self.llm = llm
        self.retriever = retriever
        self.max_retrieval_attempts = max(1, max_retrieval_attempts)
        self.max_generation_attempts = max(1, max_generation_attempts)
        self.quality_threshold = max(1, min(5, quality_threshold))
        self._parser = StrOutputParser()

        # Build prompts
        self._p_decide   = ChatPromptTemplate.from_template(_RETRIEVAL_DECISION_TEMPLATE)
        self._p_grade    = ChatPromptTemplate.from_template(_DOCUMENT_GRADE_TEMPLATE)
        self._p_faithful = ChatPromptTemplate.from_template(_FAITHFULNESS_TEMPLATE)
        self._p_quality  = ChatPromptTemplate.from_template(_ANSWER_QUALITY_TEMPLATE)
        self._p_rewrite  = ChatPromptTemplate.from_template(_QUERY_REWRITE_TEMPLATE)
        self._p_direct   = ChatPromptTemplate.from_template(_DIRECT_ANSWER_TEMPLATE)

        final_tpl = SELFRAG_FINAL_PROMPTS.get(prompt_mode, _FINAL_STRICT_TEMPLATE)
        self._p_final = ChatPromptTemplate.from_template(final_tpl)

    # ── Step helpers ─────────────────────────────────────────────────────────

    def _decide_retrieval(self, question: str) -> Tuple[bool, str]:
        chain = self._p_decide | self.llm | self._parser
        raw = chain.invoke({"question": question})
        parsed = _extract_json(raw)
        return bool(parsed.get("retrieve", True)), parsed.get("reason", "")

    def _grade_documents(
        self, question: str, docs: List[Document]
    ) -> List[Tuple[Document, DocGrade]]:
        chain = self._p_grade | self.llm | self._parser
        results = []
        for i, doc in enumerate(docs):
            raw = chain.invoke({
                "question": question,
                "document": doc.page_content[:2000],  # cap to save tokens
            })
            parsed = _extract_json(raw)
            src, page = _doc_meta(doc)
            grade = DocGrade(
                doc_index=i,
                source=src,
                page=page,
                relevant=bool(parsed.get("relevant", False)),
                reason=parsed.get("reason", ""),
            )
            results.append((doc, grade))
        return results

    def _check_faithfulness(self, context: str, answer: str) -> Tuple[bool, str]:
        chain = self._p_faithful | self.llm | self._parser
        raw = chain.invoke({"context": context, "answer": answer})
        parsed = _extract_json(raw)
        return bool(parsed.get("faithful", True)), parsed.get("reason", "")

    def _score_quality(self, question: str, answer: str) -> Tuple[int, str]:
        chain = self._p_quality | self.llm | self._parser
        raw = chain.invoke({"question": question, "answer": answer})
        parsed = _extract_json(raw)
        score = int(parsed.get("score", 3))
        return max(1, min(5, score)), parsed.get("reason", "")

    def _rewrite_query(
        self, question: str, previous_query: str, reason: str
    ) -> str:
        chain = self._p_rewrite | self.llm | self._parser
        raw = chain.invoke({
            "question": question,
            "previous_query": previous_query,
            "reason": reason,
        })
        parsed = _extract_json(raw)
        return parsed.get("rewritten_query", question)

    def _generate_answer(self, question: str, context: str) -> str:
        chain = self._p_final | self.llm | self._parser
        return chain.invoke({"question": question, "context": context})

    def _generate_direct(self, question: str) -> str:
        chain = self._p_direct | self.llm | self._parser
        return chain.invoke({"question": question})

    def _stream_answer(self, question: str, context: str):
        chain = self._p_final | self.llm | self._parser
        yield from chain.stream({"question": question, "context": context})

    def _stream_direct(self, question: str):
        chain = self._p_direct | self.llm | self._parser
        yield from chain.stream({"question": question})

    # ── Core pipeline ────────────────────────────────────────────────────────

    def _run_pipeline(
        self, question: str
    ) -> Tuple[List[Document], SelfRAGTrace, bool]:
        """
        Execute the full Self-RAG pipeline.
        Returns (relevant_docs, trace, is_direct_answer).
        """
        trace = SelfRAGTrace(
            question=question,
            retrieval_needed=True,
            retrieval_reason="",
        )

        # ── Step 1: Retrieval Decision ────────────────────────────────────────
        retrieve, reason = self._decide_retrieval(question)
        trace.retrieval_needed = retrieve
        trace.retrieval_reason = reason

        if not retrieve:
            trace.final_source = "direct"
            return [], trace, True

        # ── Steps 2–3: Retrieval + Grading loop ──────────────────────────────
        current_query = question
        all_relevant: List[Document] = []
        seen_content: set = set()

        for attempt in range(1, self.max_retrieval_attempts + 1):
            raw_docs = self.retriever.invoke(current_query)

            graded = self._grade_documents(question, raw_docs)

            round_info = RetrievalRound(
                round=attempt,
                query=current_query,
                docs_fetched=len(raw_docs),
                docs_relevant=0,
                grades=[g for _, g in graded],
            )

            new_relevant = []
            for doc, grade in graded:
                if grade.relevant and doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    new_relevant.append(doc)

            round_info.docs_relevant = len(new_relevant)
            all_relevant.extend(new_relevant)
            trace.retrieval_rounds.append(round_info)

            if all_relevant:
                break  # Have relevant docs — proceed to generation

            # No relevant docs — try rewriting unless this is the last attempt
            if attempt < self.max_retrieval_attempts:
                rewrite_reason = (
                    "No relevant documents found. "
                    "Trying a different query to surface better results."
                )
                current_query = self._rewrite_query(
                    question, current_query, rewrite_reason
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
    ) -> str:
        """
        Run generation + faithfulness + quality loop.
        Returns the final answer string.
        """
        if is_direct:
            return self._generate_direct(question)

        context = _format_docs(relevant_docs)

        for attempt in range(1, self.max_generation_attempts + 1):
            answer = self._generate_answer(question, context)

            # ── Step 5: Faithfulness ──────────────────────────────────────────
            if relevant_docs:
                faithful, faithful_reason = self._check_faithfulness(context, answer)
            else:
                faithful, faithful_reason = True, "No context to check."

            # ── Step 6: Quality ───────────────────────────────────────────────
            quality_score, quality_reason = self._score_quality(question, answer)

            gen_attempt = GenerationAttempt(
                attempt=attempt,
                faithful=faithful,
                faithfulness_reason=faithful_reason,
                quality_score=quality_score,
                quality_reason=quality_reason,
            )
            trace.generation_attempts.append(gen_attempt)

            # Accept if faithful and quality is sufficient
            if faithful and quality_score >= self.quality_threshold:
                break

            # On last attempt, keep whatever we have
            if attempt == self.max_generation_attempts:
                break

            # Otherwise regenerate (prompt implicitly varies due to LLM temperature)

        trace.final_source = (
            "direct" if is_direct
            else ("fallback" if not relevant_docs else "retrieval")
        )
        return answer

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(self, question: str) -> dict:
        """
        Returns
        -------
        {
          "answer": str,
          "trace":  dict  (SelfRAGTrace serialized)
        }
        """
        relevant_docs, trace, is_direct = self._run_pipeline(question)
        answer = self._run_generation(question, relevant_docs, trace, is_direct)
        return {"answer": answer, "trace": trace.to_dict()}

    def stream(self, question: str):
        """
        Generator compatible with st.write_stream.

        Yields
        ------
        1. A sentinel chunk carrying the trace as JSON (ui intercepts & renders it).
        2. Text chunks of the final answer streamed token-by-token.
        """
        relevant_docs, trace, is_direct = self._run_pipeline(question)

        # Run generation (non-streaming) to get faithfulness / quality grades
        context = _format_docs(relevant_docs)

        for attempt in range(1, self.max_generation_attempts + 1):
            if is_direct:
                answer_for_grading = self._generate_direct(question)
            else:
                answer_for_grading = self._generate_answer(question, context)

            if relevant_docs and not is_direct:
                faithful, faithful_reason = self._check_faithfulness(
                    context, answer_for_grading
                )
            else:
                faithful, faithful_reason = True, "No context to check."

            quality_score, quality_reason = self._score_quality(
                question, answer_for_grading
            )

            trace.generation_attempts.append(GenerationAttempt(
                attempt=attempt,
                faithful=faithful,
                faithfulness_reason=faithful_reason,
                quality_score=quality_score,
                quality_reason=quality_reason,
            ))

            if faithful and quality_score >= self.quality_threshold:
                break
            if attempt == self.max_generation_attempts:
                break

        trace.final_source = (
            "direct" if is_direct
            else ("fallback" if not relevant_docs else "retrieval")
        )

        # Emit trace sentinel
        yield (
            f"{self.TRACE_START}"
            f"{json.dumps(trace.to_dict(), ensure_ascii=False)}"
            f"{self.TRACE_END}"
        )

        # Stream the final answer
        if is_direct:
            yield from self._stream_direct(question)
        else:
            yield from self._stream_answer(question, context)