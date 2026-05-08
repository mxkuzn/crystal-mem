"""LangChain integration — `CrystalMessageHistory` for langchain-core >= 1.0.

Implements `BaseChatMessageHistory` so it works with `RunnableWithMessageHistory`,
modern LangChain runnable patterns, and any chain expecting message history.

Each user / assistant message is added as a CrystalMem entry. `messages`
property returns the chronological list. `clear` does GDPR-clean exact forget
across all entries.

Optional: pass `top_k=...` to enable retrieval-style memory — only the most
relevant past messages are surfaced for the current query.

Usage:
    from crystal_mem import CrystalMem
    from crystal_mem.integrations.langchain import CrystalMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    mem = CrystalMem(user_id="alice", dim=384, embedder=...)

    def get_session_history(session_id):
        return CrystalMessageHistory(mem, session_id=session_id, top_k=5)

    chain_with_memory = RunnableWithMessageHistory(
        chain, get_session_history,
        input_messages_key="input", history_messages_key="history",
    )
"""
from __future__ import annotations

from typing import List


def _import_base():
    try:
        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.messages import (
            AIMessage, BaseMessage, HumanMessage, SystemMessage,
        )
        return BaseChatMessageHistory, BaseMessage, HumanMessage, AIMessage, SystemMessage
    except ImportError as e:
        raise ImportError(
            "langchain-core not installed. Run: pip install langchain-core"
        ) from e


def make_message_history():
    BaseChatMessageHistory, BaseMessage, HumanMessage, AIMessage, SystemMessage = _import_base()

    class CrystalMessageHistory(BaseChatMessageHistory):
        """LangChain message history backed by CrystalMem."""

        def __init__(self, memory, session_id: str = "default",
                     top_k: int | None = None):
            self._mem = memory
            self.session_id = session_id
            self.top_k = top_k

        @property
        def messages(self) -> List[BaseMessage]:
            entries = sorted(
                (e for e in self._mem.get_all()
                 if e.metadata.get("session_id") == self.session_id),
                key=lambda e: e.timestamp,
            )
            out: List[BaseMessage] = []
            for e in entries:
                role = e.metadata.get("role", "user")
                content = e.content if isinstance(e.content, str) else str(e.content)
                if role == "user":
                    out.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    out.append(AIMessage(content=content))
                else:
                    out.append(SystemMessage(content=content))
            return out

        def add_message(self, message: BaseMessage) -> None:
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            role = role_map.get(message.type, message.type)
            self._mem.add(
                message.content,
                tags={f"session:{self.session_id}", role},
                metadata={"role": role, "session_id": self.session_id},
            )

        def add_user_message(self, message: str) -> None:
            self._mem.add(
                message,
                tags={f"session:{self.session_id}", "user"},
                metadata={"role": "user", "session_id": self.session_id},
            )

        def add_ai_message(self, message: str) -> None:
            self._mem.add(
                message,
                tags={f"session:{self.session_id}", "assistant"},
                metadata={"role": "assistant", "session_id": self.session_id},
            )

        def search_relevant(self, query: str) -> List[BaseMessage]:
            """Retrieval-style memory — only top-k relevant past messages."""
            k = self.top_k or 5
            results = self._mem.search(query, top_k=k)
            out: List[BaseMessage] = []
            for entry, _ in results:
                if entry.metadata.get("session_id") != self.session_id:
                    continue
                role = entry.metadata.get("role", "user")
                content = entry.content if isinstance(entry.content, str) else str(entry.content)
                if role == "user":
                    out.append(HumanMessage(content=content))
                elif role == "assistant":
                    out.append(AIMessage(content=content))
                else:
                    out.append(SystemMessage(content=content))
            return out

        def clear(self) -> None:
            for mid in [
                mid for mid, e in self._mem.entries.items()
                if e.metadata.get("session_id") == self.session_id
            ]:
                self._mem.forget(mid)

    return CrystalMessageHistory


def CrystalMessageHistory(*args, **kwargs):
    cls = make_message_history()
    return cls(*args, **kwargs)


# Legacy alias (for users on older LangChain)
def CrystalMemoryAdapter(*args, **kwargs):
    """Deprecated: use CrystalMessageHistory with RunnableWithMessageHistory."""
    return CrystalMessageHistory(*args, **kwargs)
