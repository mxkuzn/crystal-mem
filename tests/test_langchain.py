"""LangChain CrystalMessageHistory e2e — verify add/messages/clear flow.

Doesn't call a real LLM, but validates that the BaseChatMessageHistory
contract works with langchain-core message types.
"""
from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from crystal_mem import CrystalMem
from crystal_mem.integrations.langchain import CrystalMessageHistory


def test_add_user_and_ai_messages():
    m = CrystalMem(dim=384, n_heads=2)
    h = CrystalMessageHistory(m, session_id="conv1")
    h.add_user_message("Hello, I'm Alice")
    h.add_ai_message("Nice to meet you Alice")
    h.add_user_message("My favorite color is azure")

    msgs = h.messages
    assert len(msgs) == 3
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert isinstance(msgs[2], HumanMessage)
    assert msgs[0].content == "Hello, I'm Alice"


def test_add_message_via_base_method():
    m = CrystalMem(dim=384, n_heads=2)
    h = CrystalMessageHistory(m, session_id="x")
    h.add_message(HumanMessage(content="user msg"))
    h.add_message(AIMessage(content="assistant msg"))
    h.add_message(SystemMessage(content="system note"))
    assert len(h.messages) == 3


def test_session_isolation():
    m = CrystalMem(dim=384, n_heads=2)
    a = CrystalMessageHistory(m, session_id="alice")
    b = CrystalMessageHistory(m, session_id="bob")
    a.add_user_message("alice's message")
    b.add_user_message("bob's message")
    assert len(a.messages) == 1
    assert len(b.messages) == 1
    assert "alice" in a.messages[0].content


def test_clear_only_clears_session():
    m = CrystalMem(dim=384, n_heads=2)
    a = CrystalMessageHistory(m, session_id="alice")
    b = CrystalMessageHistory(m, session_id="bob")
    a.add_user_message("alice m1")
    a.add_user_message("alice m2")
    b.add_user_message("bob m1")
    a.clear()
    assert len(a.messages) == 0
    assert len(b.messages) == 1


def test_search_relevant_returns_topk():
    m = CrystalMem(dim=384, n_heads=2)
    h = CrystalMessageHistory(m, session_id="x", top_k=2)
    for msg in [
        "I prefer Python for backend",
        "Just had lunch — soup",
        "Python is my favorite language",
        "Working on the proposal",
        "What time is it?",
    ]:
        h.add_user_message(msg)

    relevant = h.search_relevant("programming languages I like")
    assert len(relevant) <= 2
    contents = " ".join(m.content.lower() for m in relevant)
    # At least one of the python messages should be in top-2
    assert "python" in contents


def test_message_history_persists_across_instance():
    """Verify history survives even if you make a new CrystalMessageHistory wrapper."""
    m = CrystalMem(dim=384, n_heads=2)
    h1 = CrystalMessageHistory(m, session_id="conv2")
    h1.add_user_message("Persistent message")

    # New instance pointing at the same memory
    h2 = CrystalMessageHistory(m, session_id="conv2")
    msgs = h2.messages
    assert len(msgs) == 1
    assert msgs[0].content == "Persistent message"
