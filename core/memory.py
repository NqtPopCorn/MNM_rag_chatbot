"""
core/memory.py
──────────────
Centralized conversation-history utilities for the RAG pipeline.

HistoryManager:
  • format()        — token-budget-aware history string (no per-message char cap)
  • contextualize() — rewrite a follow-up question into a self-contained query
                      BEFORE it hits the vector store (key RAG retrieval boost)
"""
from __future__ import annotations

from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ── Constants ──────────────────────────────────────────────────────────────────

# Vietnamese + English mix: ~4 chars / token on average
_CHARS_PER_TOKEN: int = 4

# Token budget for the history block inside the prompt.
# Keep this well below your LLM's context window minus expected answer length.
_DEFAULT_TOKEN_BUDGET: int = 600   # ≈ 2 400 chars

_CONTEXTUALIZE_TMPL = """\
Given the conversation history and a follow-up question, rewrite the follow-up \
question as a fully self-contained, standalone question that includes every \
piece of context needed to answer it — without access to the history.

Rules:
- If the follow-up question is already standalone, return it UNCHANGED.
- If the follow-up question introduces a completely new topic unrelated to the conversation history, return it UNCHANGED.
- Only use the conversation history when the question clearly depends on it (e.g., pronouns, ellipsis, references).
- Do NOT answer the question. Only rewrite it.

Conversation History:
{chat_history}

Follow-up Question: {question}
Standalone Question:"""


class HistoryManager:
    """
    Manages short-term conversation memory for all chain types (RAG / CoRAG / Self-RAG).

    Parameters
    ----------
    token_budget : int
        Approximate token budget for the formatted history block.
        Newer turns are kept verbatim; older turns are trimmed to fit.
    """

    def __init__(self, token_budget: int = _DEFAULT_TOKEN_BUDGET) -> None:
        self._char_budget: int = token_budget * _CHARS_PER_TOKEN
        self._prompt = ChatPromptTemplate.from_template(_CONTEXTUALIZE_TMPL)

    # ── format ─────────────────────────────────────────────────────────────────

    def format(self, messages: List[Dict], window: int) -> str:
        """
        Build a compact history string from the last `window` user-assistant pairs.

        Key differences from the original _format_history:
          • Single global char budget instead of a per-message 300-char hard cap.
            Recent turns get full text; older turns are progressively compressed.
          • Allocates 35% of budget to human turns, 65% to assistant turns —
            answers carry more retrieval-relevant information.
          • No verbose "[Conversation history …]" wrapper — saves tokens.
          • Handles edge cases (mismatched roles, empty messages) gracefully.
        """
        if window <= 0 or not messages:
            return ""

        # ── Collect (human, ai) pairs, newest first ────────────────────────
        pairs: list[tuple[str, str]] = []
        i = len(messages) - 1
        while i >= 0 and len(pairs) < window:
            msg = messages[i]
            if msg["role"] == "assistant":
                if i > 0 and messages[i - 1]["role"] == "user":
                    pairs.append((messages[i - 1]["content"], msg["content"]))
                    i -= 2
                else:
                    i -= 1
            elif msg["role"] == "user":
                pairs.append((msg["content"], ""))
                i -= 1
            else:
                i -= 1

        if not pairs:
            return ""

        # ── Apply budget, most-recent pairs first ──────────────────────────
        budget = self._char_budget
        blocks: list[str] = []

        for human, ai in pairs:          # newest → oldest
            h_cap = max(80, int(budget * 0.35))
            a_cap = max(120, int(budget * 0.65))

            h = human[:h_cap] + ("…" if len(human) > h_cap else "")
            a = ai[:a_cap]   + ("…" if len(ai)    > a_cap else "")

            block = f"Human: {h}\nAssistant: {a}" if ai else f"Human: {h}"
            blocks.append(block)

            budget -= len(block)
            if budget <= 0:
                break

        # ── Reverse to chronological (latest first) order  ─────────────────────────────────
        blocks.reverse()
        return "\n\n".join(blocks)

    # ── contextualize ──────────────────────────────────────────────────────────

    def contextualize(self, question: str, history: str, llm: BaseChatModel) -> str:
        """
        Rewrite a follow-up question into a standalone query using history.

        Short-circuits immediately (no LLM call) when history is empty —
        avoids latency on the first turn or when memory is disabled.

        Returns the original question if the LLM returns an empty string
        (graceful fallback).
        """

        print(f"Contextualizing question: '{question}' with history:\n{history}\n---")

        if not history.strip():
            print("No history provided, skipping contextualization.")
            return question

        chain = self._prompt | llm | StrOutputParser()
        rewritten = chain.invoke({"question": question, "chat_history": history})
        print(f"Rewritten question: '{rewritten}'")
        return rewritten.strip() or question


# ── Module-level singleton ─────────────────────────────────────────────────────
history_manager = HistoryManager()