"""LlamaIndex memory adapter.

Provides a ChatMemoryBuffer-compatible interface backed by CrystalMem.

Usage:
    from crystal_mem import CrystalMem
    from crystal_mem.integrations.llamaindex import CrystalMemBuffer
    from llama_index.core.agent import ReActAgent

    mem = CrystalMem(user_id="alice")
    buf = CrystalMemBuffer.from_crystal_mem(mem)
    agent = ReActAgent.from_tools(tools=[...], memory=buf)
"""
from __future__ import annotations


def _import_base():
    try:
        from llama_index.core.memory import ChatMemoryBuffer
        from llama_index.core.llms import ChatMessage
        return ChatMemoryBuffer, ChatMessage
    except ImportError as e:
        raise ImportError(
            "llama-index not installed. Run: pip install llama-index-core"
        ) from e


def make_buffer():
    ChatMemoryBuffer, ChatMessage = _import_base()

    class CrystalMemBuffer(ChatMemoryBuffer):
        @classmethod
        def from_crystal_mem(cls, memory, top_k: int = 5):
            inst = cls.from_defaults()
            inst._crystal = memory
            inst._top_k = top_k
            return inst

        def put(self, message) -> None:
            super().put(message)
            text = getattr(message, "content", str(message))
            if text:
                self._crystal.add(
                    text,
                    tags={getattr(message, "role", "msg")},
                )

        def get(self, input: str | None = None, **_kwargs):
            if input is None:
                return super().get()
            results = self._crystal.search(input, top_k=self._top_k)
            return [ChatMessage(role="system", content=e.content)
                    for e, _ in results]

    return CrystalMemBuffer


def CrystalMemBuffer(*args, **kwargs):
    cls = make_buffer()
    return cls(*args, **kwargs)
