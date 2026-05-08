"""MCP server smoke test — invoke build_server() in-process and call tools."""
from __future__ import annotations

import asyncio
import json

import pytest

mcp = pytest.importorskip("mcp")

from crystal_mem import CrystalMem  # noqa: E402
from crystal_mem.mcp_server import build_server  # noqa: E402


@pytest.mark.asyncio
async def _run_smoke():
    m = CrystalMem(dim=512, n_heads=2)
    server = build_server(m)

    # Inspect server has list_tools/call_tool registered.
    # We can't invoke the stdio protocol here, but we can call the request handlers
    # registered on the Server directly.
    list_handler = server.request_handlers[mcp.types.ListToolsRequest]
    call_handler = server.request_handlers[mcp.types.CallToolRequest]

    # list_tools
    res = await list_handler(
        mcp.types.ListToolsRequest(method="tools/list", params=None)
    )
    tools = res.root.tools
    names = {t.name for t in tools}
    assert "memory_add" in names
    assert "memory_search" in names
    assert "memory_forget" in names
    assert "memory_export" in names
    assert "memory_merge_file" in names
    assert "memory_unmerge_file" in names

    # add a fact
    res = await call_handler(
        mcp.types.CallToolRequest(
            method="tools/call",
            params=mcp.types.CallToolRequestParams(
                name="memory_add",
                arguments={"content": "hello world", "tags": ["test"]},
            ),
        )
    )
    payload = json.loads(res.root.content[0].text)
    assert "memory_id" in payload

    # search
    res = await call_handler(
        mcp.types.CallToolRequest(
            method="tools/call",
            params=mcp.types.CallToolRequestParams(
                name="memory_search",
                arguments={"query": "hello", "top_k": 3},
            ),
        )
    )
    results = json.loads(res.root.content[0].text)
    assert len(results) >= 1
    assert "content" in results[0]

    # stats
    res = await call_handler(
        mcp.types.CallToolRequest(
            method="tools/call",
            params=mcp.types.CallToolRequestParams(
                name="memory_stats",
                arguments={},
            ),
        )
    )
    stats = json.loads(res.root.content[0].text)
    assert stats["n_entries"] >= 1


def test_mcp_smoke():
    asyncio.run(_run_smoke())
