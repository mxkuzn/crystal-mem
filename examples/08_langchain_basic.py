"""08 — use CrystalMem inside a LangChain runnable chain.

Drop-in for langchain-core >= 1.0 via `RunnableWithMessageHistory`. Replaces
ConversationBufferMemory etc. without modifying chain logic.

    pip install crystal-mem[langchain]

Uses FakeListLLM here so the example runs without API keys. Replace with
your real LLM (`ChatAnthropic`, `ChatOpenAI`, etc.) in production.
"""
from langchain_core.language_models import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from crystal_mem import CrystalMem
from crystal_mem.integrations.langchain import CrystalMessageHistory


# One CrystalMem instance per user / per agent (or shared with session_id keying)
mem = CrystalMem(dim=384, n_heads=4)


def get_history(session_id: str) -> CrystalMessageHistory:
    return CrystalMessageHistory(mem, session_id=session_id)


fake_llm = FakeListLLM(responses=["Got it.", "Noted.", "Understood."])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | fake_llm
chain_with_memory = RunnableWithMessageHistory(
    chain, get_history,
    input_messages_key="input",
    history_messages_key="history",
)

cfg = {"configurable": {"session_id": "alice"}}

chain_with_memory.invoke({"input": "Hi, I'm Alice."},                   config=cfg)
chain_with_memory.invoke({"input": "I prefer Python over Go."},         config=cfg)
chain_with_memory.invoke({"input": "My favourite color is azure."},     config=cfg)

# History persists in CrystalMem and survives across chain.invoke() calls
hist = CrystalMessageHistory(mem, session_id="alice")
print(f"messages so far: {len(hist.messages)}")
for msg in hist.messages:
    print(f"  [{msg.type}] {msg.content}")
