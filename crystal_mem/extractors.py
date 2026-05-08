"""LLM-based fact extraction (mode='auto', Path A).

Mode 'auto' is opt-in. Mode 'explicit' (default) does NOT call LLMs.

Provide your own extractor via:

    def my_extract(messages: list[dict]) -> list[str]:
        ...
    m = CrystalMem(mode="auto", llm_extractor=my_extract)

Or use one of the presets below. Presets require their respective SDKs to be
installed.
"""
from __future__ import annotations

import json
from typing import Callable


SYSTEM_PROMPT = (
    "You extract durable, atomic facts from a conversation. "
    "Return JSON array of short factual statements about the user, their "
    "preferences, decisions, or context. Skip ephemeral details. "
    "Each fact should be self-contained and worth remembering across sessions."
)


def openai_extractor(model: str = "gpt-4o-mini") -> Callable[[list[dict]], list[str]]:
    """Returns an extractor backed by OpenAI Chat Completions."""
    def _extract(messages: list[dict]) -> list[str]:
        from openai import OpenAI
        client = OpenAI()
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": joined},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict) and "facts" in data:
            return list(data["facts"])
        if isinstance(data, list):
            return [str(x) for x in data]
        return []
    return _extract


def anthropic_extractor(model: str = "claude-haiku-4-5-20251001") -> Callable[[list[dict]], list[str]]:
    """Returns an extractor backed by Anthropic Messages API."""
    def _extract(messages: list[dict]) -> list[str]:
        import anthropic
        client = anthropic.Anthropic()
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT + " Return JSON: {\"facts\": [...]}",
            messages=[{"role": "user", "content": joined}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        try:
            data = json.loads(text)
            if "facts" in data:
                return list(data["facts"])
        except json.JSONDecodeError:
            return [line.strip("- ").strip() for line in text.splitlines()
                    if line.strip()]
        return []
    return _extract
