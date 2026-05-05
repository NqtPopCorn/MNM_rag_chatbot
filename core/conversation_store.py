"""
core/conversation_store.py
==========================
SQLite-backed persistence layer for chat conversations.

Schema
------
  conversations(id, title, created_at, updated_at, model_info, chain_type)
  messages(id, conversation_id, role, content, metadata_json, created_at)

Public API
----------
  ConversationStore(db_path)
    .create_conversation(title, model_info, chain_type)  -> conv_id (int)
    .list_conversations()                                -> list[dict]
    .get_conversation(conv_id)                           -> dict | None
    .get_messages(conv_id)                               -> list[dict]
    .add_message(conv_id, role, content, metadata)       -> msg_id (int)
    .rename_conversation(conv_id, new_title)
    .delete_conversation(conv_id)
    .export_conversation_json(conv_id)                   -> str (JSON)
    .export_conversation_markdown(conv_id)               -> str (Markdown)
    .auto_title_from_first_message(conv_id)              -> str

Usage example
-------------
  from core.conversation_store import ConversationStore
  store = ConversationStore("./data/conversations.db")
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL DEFAULT 'Cuộc trò chuyện mới',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    model_info  TEXT    NOT NULL DEFAULT '',
    chain_type  TEXT    NOT NULL DEFAULT 'rag'
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content         TEXT    NOT NULL DEFAULT '',
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
"""


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


class ConversationStore:
    """
    Thread-safe (check_same_thread=False) SQLite wrapper.
    Creates the database file and schema on first use.
    """

    def __init__(self, db_path: str = "./data/conversations.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    # ── Internal ───────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ── Conversations ──────────────────────────────────────────────────────────

    def create_conversation(
        self,
        title: str = "Cuộc trò chuyện mới",
        model_info: str = "",
        chain_type: str = "rag",
    ) -> int:
        """Create a new conversation row. Returns the new conversation id."""
        now = _now()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at, model_info, chain_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (title, now, now, model_info, chain_type),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def list_conversations(self) -> List[Dict[str, Any]]:
        """Return all conversations ordered newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT c.*, COUNT(m.id) AS message_count "
                "FROM conversations c "
                "LEFT JOIN messages m ON m.conversation_id = c.id "
                "GROUP BY c.id "
                "ORDER BY c.updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conv_id: int) -> Optional[Dict[str, Any]]:
        """Return a single conversation dict or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
        return dict(row) if row else None

    def rename_conversation(self, conv_id: int, new_title: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (new_title.strip() or "Cuộc trò chuyện mới", _now(), conv_id),
            )

    def delete_conversation(self, conv_id: int) -> None:
        """Delete conversation and all its messages (CASCADE)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    def touch_conversation(self, conv_id: int, model_info: str = "", chain_type: str = "") -> None:
        """Update updated_at (and optionally model_info / chain_type)."""
        with self._conn() as conn:
            if model_info or chain_type:
                conn.execute(
                    "UPDATE conversations "
                    "SET updated_at = ?, model_info = COALESCE(NULLIF(?, ''), model_info), "
                    "    chain_type = COALESCE(NULLIF(?, ''), chain_type) "
                    "WHERE id = ?",
                    (_now(), model_info, chain_type, conv_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (_now(), conv_id),
                )

    # ── Messages ───────────────────────────────────────────────────────────────

    def add_message(
        self,
        conv_id: int,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append a message and update the parent conversation's updated_at."""
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        now = _now()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, role, content, meta_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_messages(self, conv_id: int) -> List[Dict[str, Any]]:
        """Return all messages for a conversation in chronological order."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conv_id,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
            except (json.JSONDecodeError, KeyError):
                d["metadata"] = {}
            result.append(d)
        return result

    def load_messages_into_session_format(self, conv_id: int) -> List[Dict[str, Any]]:
        """
        Convert DB messages into the st.session_state.messages format
        used by chat_view.py:
          {"role": ..., "content": ..., "model_info": ..., "chain_type": ..., ...}
        """
        rows = self.get_messages(conv_id)
        out = []
        for r in rows:
            meta = r.get("metadata") or {}
            msg = {
                "role": r["role"],
                "content": r["content"],
                **meta,
            }
            out.append(msg)
        return out

    # ── Auto-title ─────────────────────────────────────────────────────────────

    def auto_title_from_first_message(self, conv_id: int) -> str:
        """
        Derive a title from the first user message (≤ 50 chars).
        Updates the conversation row and returns the new title.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content FROM messages "
                "WHERE conversation_id = ? AND role = 'user' "
                "ORDER BY id ASC LIMIT 1",
                (conv_id,),
            ).fetchone()
        if not row:
            return "Cuộc trò chuyện mới"
        raw = row["content"].strip().replace("\n", " ")
        title = raw[:50] + ("…" if len(raw) > 50 else "")
        self.rename_conversation(conv_id, title)
        return title

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_conversation_json(self, conv_id: int) -> str:
        """Return the full conversation (metadata + messages) as a JSON string."""
        conv = self.get_conversation(conv_id)
        if not conv:
            return json.dumps({}, ensure_ascii=False, indent=2)
        messages = self.get_messages(conv_id)
        payload = {
            "conversation": conv,
            "messages": messages,
            "exported_at": _now(),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    def export_conversation_markdown(self, conv_id: int) -> str:
        """Return the conversation formatted as Markdown."""
        conv = self.get_conversation(conv_id)
        if not conv:
            return "# Không tìm thấy cuộc trò chuyện"
        messages = self.get_messages(conv_id)

        lines: List[str] = [
            f"# {conv['title']}",
            f"",
            f"- **Tạo lúc:** {conv['created_at']}",
            f"- **Cập nhật:** {conv['updated_at']}",
            f"- **Model:** {conv['model_info']}",
            f"- **Chain:** {conv['chain_type'].upper()}",
            f"",
            f"---",
            f"",
        ]
        for msg in messages:
            role_label = "🧑‍💻 **Bạn**" if msg["role"] == "user" else "🤖 **Trợ lý**"
            lines.append(f"### {role_label}")
            lines.append(msg["content"])
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)