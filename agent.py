import argparse
import asyncio
import os
from typing import Any
from uuid import uuid4
from dotenv import load_dotenv

from acp import (
    Agent as ACPAgent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message,
    start_tool_call,
    start_read_tool_call,
    start_edit_tool_call,
    update_tool_call,
    tool_content,
)
from acp.interfaces import Client
from acp.schema import (
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    TextContentBlock,
)
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import Runnable, RunnableConfig

load_dotenv()


class ACPDeepAgent(ACPAgent):
    _conn: Client

    _deepagent: Runnable
    _root_dir: str

    def __init__(self, deepagent: Runnable, root_dir: str):
        self._deepagent = deepagent
        self._root_dir = root_dir
        super().__init__()

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        **kwargs: Any,
    ) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        # Convert ACP content blocks to LangChain multimodal content format
        content_blocks = []
        for block in prompt:
            if isinstance(block, TextContentBlock):
                content_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContentBlock):
                # Image blocks contain visual data
                # URIs are local file paths, provide as text so agent can read with tools
                if block.uri:
                    # Truncate root_dir from path while preserving file:// prefix
                    uri = block.uri
                    has_file_prefix = uri.startswith("file://")
                    if has_file_prefix:
                        path = uri[7:]  # Remove "file://" temporarily
                    else:
                        path = uri

                    # Remove root_dir prefix to get path relative to agent's working directory
                    if path.startswith(self._root_dir):
                        path = path[len(self._root_dir) :].lstrip("/")

                    # Restore file:// prefix if it was present
                    uri = f"file://{path}" if has_file_prefix else path
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": f"[Image file at path: {uri}, MIME type: {block.mime_type}]",
                        }
                    )
                else:
                    # Use inline base64 data
                    data_uri = f"data:{block.mime_type};base64,{block.data}"
                    content_blocks.append({"type": "image_url", "image_url": data_uri})
            elif isinstance(block, AudioContentBlock):
                # Audio blocks - represent as text with data URI for tools to access
                data_uri = f"data:{block.mime_type};base64,{block.data}"
                content_blocks.append(
                    {"type": "text", "text": f"[Audio file available at: {data_uri}]"}
                )
            elif isinstance(block, ResourceContentBlock):
                # Resource blocks reference external resources
                resource_text = f"[Resource: {block.name}"
                if block.uri:
                    # Truncate root_dir from path while preserving file:// prefix
                    uri = block.uri
                    has_file_prefix = uri.startswith("file://")
                    if has_file_prefix:
                        path = uri[7:]  # Remove "file://" temporarily
                    else:
                        path = uri

                    # Remove root_dir prefix to get path relative to agent's working directory
                    if path.startswith(self._root_dir):
                        path = path[len(self._root_dir) :].lstrip("/")

                    # Restore file:// prefix if it was present
                    uri = f"file://{path}" if has_file_prefix else path
                    resource_text += f"\nURI: {uri}"
                if block.description:
                    resource_text += f"\nDescription: {block.description}"
                if block.mime_type:
                    resource_text += f"\nMIME type: {block.mime_type}"
                resource_text += "]"
                content_blocks.append({"type": "text", "text": resource_text})
            elif isinstance(block, EmbeddedResourceContentBlock):
                # Embedded resource blocks contain the resource data inline
                resource = block.resource
                if hasattr(resource, "text"):
                    # Text resource
                    content_blocks.append({"type": "text", "text": resource.text})
                elif hasattr(resource, "blob"):
                    # Binary resource - provide as data URI
                    mime_type = getattr(
                        resource, "mime_type", "application/octet-stream"
                    )
                    data_uri = f"data:{mime_type};base64,{resource.blob}"
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": f"[Embedded resource available at: {data_uri}]",
                        }
                    )

        # Stream the deep agent response with multimodal content
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        # Track active tool calls
        active_tool_calls = {}

        # Stream messages which include LLM output and tool calls
        async for message_chunk, metadata in self._deepagent.astream(
            {"messages": [{"role": "user", "content": content_blocks}]},
            config=config,
            stream_mode="messages",
        ):
            # Check for tool calls in the message
            if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                for tool_call in message_chunk.tool_calls:
                    if isinstance(tool_call, dict):
                        tool_id = tool_call.get("id", None)
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                    else:
                        tool_id = getattr(tool_call, "id", None)
                        tool_name = getattr(tool_call, "name", "unknown")
                        tool_args = getattr(tool_call, "args", {})

                    # Skip if we don't have a valid tool_id (streaming will provide it later)
                    if not tool_id:
                        continue

                    # Start a new tool call if we haven't seen this one
                    if tool_id not in active_tool_calls:
                        active_tool_calls[tool_id] = {"name": tool_name}

                        # Use specialized tool call functions for read/edit operations
                        if tool_name == "read_file" and isinstance(tool_args, dict):
                            path = tool_args.get("file_path")
                            update = start_read_tool_call(
                                tool_call_id=tool_id,
                                title=f"Read {path}",
                                path=path or "",
                            )
                        elif tool_name in ["edit_file", "write_file"] and isinstance(
                            tool_args, dict
                        ):
                            path = tool_args.get("file_path")
                            content = tool_args.get("content", "")
                            update = start_edit_tool_call(
                                tool_call_id=tool_id,
                                title=f"Edit {path}"
                                if tool_name == "edit_file"
                                else f"Write {path}",
                                path=path or "",
                                content=content,
                            )
                        else:
                            # Generic tool call for other tools (ls, glob, grep, etc.)
                            kind_map: dict = {
                                "ls": "search",
                                "glob": "search",
                                "grep": "search",
                                "delete": "delete",
                                "move": "move",
                                "execute": "execute",
                                "bash": "execute",
                            }
                            tool_kind = kind_map.get(tool_name, "other")
                            update = start_tool_call(
                                tool_call_id=tool_id,
                                title=tool_name,
                                kind=tool_kind,
                                status="pending",
                            )

                        await self._conn.session_update(
                            session_id=session_id, update=update, source="DeepAgent"
                        )

            # Check for tool results (ToolMessage responses)
            if hasattr(message_chunk, "type") and message_chunk.type == "tool":
                # This is a tool result message
                tool_call_id = getattr(message_chunk, "tool_call_id", None)
                if tool_call_id and tool_call_id in active_tool_calls:
                    content = getattr(message_chunk, "content", "")
                    # Update the tool call with completion status and result
                    update = update_tool_call(
                        tool_call_id=tool_call_id,
                        status="completed",
                        content=[tool_content(text_block(str(content)))],
                    )
                    await self._conn.session_update(
                        session_id=session_id, update=update, source="DeepAgent"
                    )

            elif message_chunk.content:
                # content can be a string or a list of content blocks
                if isinstance(message_chunk.content, str):
                    text = message_chunk.content
                elif isinstance(message_chunk.content, list):
                    # Extract text from content blocks
                    text = ""
                    for block in message_chunk.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")
                        elif isinstance(block, str):
                            text += block
                else:
                    text = str(message_chunk.content)

                if text:
                    update = update_agent_message(text_block(text))
                    await self._conn.session_update(
                        session_id=session_id, update=update, source="DeepAgent"
                    )

        return PromptResponse(stop_reason="end_turn")


async def main(root_dir: str) -> None:
    checkpointer = MemorySaver()
    deepagent = create_deep_agent(
        # tools=[...],
        # interrupt_on={...},
        checkpointer=checkpointer,
        backend=FilesystemBackend(root_dir=root_dir, virtual_mode=True),
    )
    acp_agent = ACPDeepAgent(deepagent=deepagent, root_dir=root_dir)
    await run_agent(acp_agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ACP DeepAgent with specified root directory"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/",
        help="Root directory accessible to the agent (default: /)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.root_dir))
