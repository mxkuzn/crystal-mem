"""End-to-end LangChain test using FakeListLLM (no API key needed).

Verifies that CrystalMessageHistory works inside the modern langchain-core
runnable chain pattern via RunnableWithMessageHistory.
"""
from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models import FakeListLLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from crystal_mem import CrystalMem
from crystal_mem.integrations.langchain import CrystalMessageHistory


def test_runnable_with_message_history_e2e():
    """Build a chain: prompt → fake LLM, wrap with RunnableWithMessageHistory."""
    mem = CrystalMem(dim=384, n_heads=2)
    sessions: dict[str, CrystalMessageHistory] = {}

    def get_history(session_id: str) -> CrystalMessageHistory:
        if session_id not in sessions:
            sessions[session_id] = CrystalMessageHistory(mem, session_id=session_id)
        return sessions[session_id]

    fake = FakeListLLM(responses=[
        "Hello Alice, nice to meet you.",
        "Got it. Python is your preference.",
        "I remember — your favorite color is azure.",
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt | fake
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "alice-conv"}}

    r1 = chain_with_history.invoke({"input": "Hi, I'm Alice."}, config=cfg)
    assert "alice" in r1.lower() or "hello" in r1.lower()

    r2 = chain_with_history.invoke({"input": "I prefer Python over Go."}, config=cfg)
    assert isinstance(r2, str)

    r3 = chain_with_history.invoke({"input": "My favorite color is azure."}, config=cfg)
    assert isinstance(r3, str)

    # History contains all 6 messages (3 human + 3 ai)
    h = get_history("alice-conv")
    msgs = h.messages
    assert len(msgs) == 6
    # Roles alternate user / ai
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)


def test_session_isolation_in_runnable():
    """Two different session_ids should keep independent history."""
    mem = CrystalMem(dim=384, n_heads=2)

    def get_history(session_id):
        return CrystalMessageHistory(mem, session_id=session_id)

    fake = FakeListLLM(responses=["alice resp 1", "alice resp 2", "bob resp 1"])
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt | fake
    chain_h = RunnableWithMessageHistory(
        chain, get_history,
        input_messages_key="input", history_messages_key="history",
    )

    chain_h.invoke({"input": "alice msg 1"}, config={"configurable": {"session_id": "alice"}})
    chain_h.invoke({"input": "alice msg 2"}, config={"configurable": {"session_id": "alice"}})
    chain_h.invoke({"input": "bob msg 1"},   config={"configurable": {"session_id": "bob"}})

    alice_hist = CrystalMessageHistory(mem, session_id="alice").messages
    bob_hist   = CrystalMessageHistory(mem, session_id="bob").messages
    assert len(alice_hist) == 4   # 2 user + 2 ai
    assert len(bob_hist) == 2     # 1 user + 1 ai


def test_history_persists_across_chain_calls():
    """After multiple chain.invoke calls, history is monotonically growing."""
    mem = CrystalMem(dim=384, n_heads=2)
    fake = FakeListLLM(responses=["r1", "r2", "r3"])
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = (prompt | fake)
    chain_h = RunnableWithMessageHistory(
        chain,
        lambda sid: CrystalMessageHistory(mem, session_id=sid),
        input_messages_key="input", history_messages_key="history",
    )
    cfg = {"configurable": {"session_id": "growing"}}

    for q in ["q1", "q2", "q3"]:
        chain_h.invoke({"input": q}, config=cfg)

    final = CrystalMessageHistory(mem, session_id="growing").messages
    assert len(final) == 6
