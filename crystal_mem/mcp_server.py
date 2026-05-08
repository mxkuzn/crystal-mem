"""MCP server for CrystalMem.

Exposes memory operations to MCP-compatible clients (Claude Desktop, Claude Code,
Cursor, custom IDE plugins). Run via:

    python -m crystal_mem.mcp_server --user-id alice --dim 1024

Tools exposed:
    memory_add(content, tags, metadata)        — write fact
    memory_search(query, top_k)                — retrieve top-k
    memory_get(memory_id)                      — fetch by id
    memory_forget(memory_id)                   — exact removal (GDPR)
    memory_list()                              — all entries
    memory_export(path)                        — write portable crystal file
    memory_merge_file(path)                    — import + merge another crystal
    memory_stats()                             — capacity / health
    memory_unmerge_file(path)                  — undo a previous merge

Requires `mcp` package: pip install mcp
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def build_server(memory):
    if not MCP_AVAILABLE:
        raise RuntimeError(
            "mcp package not installed. Run: pip install mcp"
        )
    server = Server("crystal-mem")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="memory_add",
                description="Write a fact to long-term memory. Returns memory_id.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Fact to remember"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "metadata": {"type": "object"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="memory_search",
                description="Retrieve top-k relevant memories matching query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="memory_get",
                description="Fetch one memory by id.",
                inputSchema={
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}},
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="memory_forget",
                description="GDPR-grade exact removal of a memory entry. Math-clean Δ=4×10⁻⁶.",
                inputSchema={
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}},
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="memory_list",
                description="List all memory entries for current user.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="memory_export",
                description="Export memory as portable .crystal file. Use for cross-instance sync.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            Tool(
                name="memory_merge_file",
                description="Import + merge another crystal file into current memory. Mathematically identical to centralized build.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            Tool(
                name="memory_unmerge_file",
                description="Undo a previous merge — subtract the contents of a crystal file from current memory.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            Tool(
                name="memory_stats",
                description="Capacity, recall estimate, crystal norms.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "memory_add":
            mid = memory.add(
                arguments["content"],
                tags=arguments.get("tags", []),
                metadata=arguments.get("metadata", {}),
            )
            return [TextContent(type="text", text=json.dumps({"memory_id": mid}))]

        if name == "memory_search":
            results = memory.search(arguments["query"], top_k=arguments.get("top_k", 5))
            payload = [
                {"id": e.id, "content": e.content, "score": s,
                 "tags": list(e.tags), "metadata": e.metadata}
                for e, s in results
            ]
            return [TextContent(type="text", text=json.dumps(payload))]

        if name == "memory_get":
            e = memory.get(arguments["memory_id"])
            return [TextContent(type="text",
                                text=json.dumps(e.to_dict() if e else None))]

        if name == "memory_forget":
            memory.forget(arguments["memory_id"])
            return [TextContent(type="text", text=json.dumps({"ok": True}))]

        if name == "memory_list":
            payload = [e.to_dict() for e in memory.get_all()]
            return [TextContent(type="text", text=json.dumps(payload))]

        if name == "memory_export":
            memory.export_file(arguments["path"])
            return [TextContent(type="text",
                                text=json.dumps({"ok": True, "path": arguments["path"]}))]

        if name == "memory_merge_file":
            memory.merge_file(arguments["path"])
            return [TextContent(type="text",
                                text=json.dumps({"ok": True, "stats": memory.stats()}))]

        if name == "memory_unmerge_file":
            from .portability import Crystal
            memory.unmerge(Crystal.from_file(arguments["path"]))
            return [TextContent(type="text",
                                text=json.dumps({"ok": True, "stats": memory.stats()}))]

        if name == "memory_stats":
            return [TextContent(type="text", text=json.dumps(memory.stats()))]

        return [TextContent(type="text",
                            text=json.dumps({"error": f"unknown tool: {name}"}))]

    return server


async def _amain(args):
    from . import CrystalMem

    state_path = Path(args.state) if args.state else None
    if state_path and state_path.exists():
        memory = CrystalMem.from_file(state_path, user_id=args.user_id)
    else:
        memory = CrystalMem(user_id=args.user_id, dim=args.dim, n_heads=args.heads)

    server = build_server(memory)

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    p = argparse.ArgumentParser(prog="crystal-mem-mcp")
    p.add_argument("--user-id", default=os.environ.get("CRYSTAL_MEM_USER", "default"))
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--state", default=os.environ.get("CRYSTAL_MEM_STATE"),
                   help="Path to .crystal state file (loaded on start).")
    args = p.parse_args()

    if not MCP_AVAILABLE:
        raise SystemExit(
            "mcp package not installed. Run: pip install mcp\n"
            "Or skip MCP and use the library directly."
        )

    import asyncio
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
