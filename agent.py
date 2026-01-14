import argparse
import ast
import asyncio
import json
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
    start_edit_tool_call,
    update_tool_call,
    tool_content,
    tool_diff_content,
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

        # Track active tool calls and accumulate chunks by index
        active_tool_calls = {}
        tool_call_accumulator = {}  # index -> {id, name, args_str}

        # Stream messages which include LLM output and tool calls
        async for message_chunk, metadata in self._deepagent.astream(
            {"messages": [{"role": "user", "content": content_blocks}]},
            config=config,
            stream_mode="messages",
        ):
            # Check for tool call chunks (streaming pieces of tool calls)
            if (
                hasattr(message_chunk, "tool_call_chunks")
                and message_chunk.tool_call_chunks
            ):
                for chunk in message_chunk.tool_call_chunks:
                    chunk_id = chunk.get("id")
                    chunk_name = chunk.get("name")
                    chunk_args = chunk.get("args", "")
                    chunk_index = chunk.get("index", 0)

                    # Initialize accumulator for this index if we have id and name
                    if chunk_id and chunk_name:
                        if (
                            chunk_index not in tool_call_accumulator
                            or chunk_id != tool_call_accumulator[chunk_index]
                        ):
                            tool_call_accumulator[chunk_index] = {
                                "id": chunk_id,
                                "name": chunk_name,
                                "args_str": "",
                            }

                    # Accumulate args string chunks using index
                    if chunk_args and chunk_index in tool_call_accumulator:
                        tool_call_accumulator[chunk_index]["args_str"] += chunk_args
            # After processing chunks, try to start any tool calls with complete args
            for index, acc in tool_call_accumulator.items():
                tool_id = acc.get("id")
                tool_name = acc.get("name")
                args_str = acc.get("args_str", "")

                # Only start if we haven't started yet and have parseable args
                if tool_id and tool_id not in active_tool_calls and args_str:
                    try:
                        tool_args = json.loads(args_str)

                        # Mark as started
                        active_tool_calls[tool_id] = {"name": tool_name}

                        kind_map: dict = {
                            "read_file": "read",
                            "edit_file": "edit",
                            "write_file": "edit",
                            "ls": "search",
                            "glob": "search",
                            "grep": "search",
                        }
                        tool_kind = kind_map.get(tool_name, "other")

                        # Determine title based on tool type and args
                        if tool_name == "read_file" and isinstance(tool_args, dict):
                            path = tool_args.get("file_path")
                            title = f"Read `{path}`" if path else tool_name
                            update = start_tool_call(
                                tool_call_id=tool_id,
                                title=title,
                                kind=tool_kind,
                                status="pending",
                            )
                        elif tool_name == "edit_file" and isinstance(tool_args, dict):
                            path = tool_args.get("file_path", "")
                            old_string = tool_args.get("old_string", "")
                            new_string = tool_args.get("new_string", "")
                            title = f"Edit `{path}`" if path else tool_name

                            # Only create diff if we have both old and new strings
                            if path and old_string and new_string:
                                diff_content = tool_diff_content(
                                    path=path,
                                    new_text=new_string,
                                    old_text=old_string,
                                )
                                update = start_edit_tool_call(
                                    tool_call_id=tool_id,
                                    title=title,
                                    path=path,
                                    content=diff_content,
                                    # This is silly but for some reason content isn't passed through
                                    extra_options=[diff_content],
                                )
                            else:
                                # Fallback to generic tool call if data incomplete
                                update = start_tool_call(
                                    tool_call_id=tool_id,
                                    title=title,
                                    kind=tool_kind,
                                    status="pending",
                                )
                        elif tool_name == "write_file" and isinstance(tool_args, dict):
                            path = tool_args.get("file_path")
                            title = f"Write `{path}`" if path else tool_name
                            update = start_tool_call(
                                tool_call_id=tool_id,
                                title=title,
                                kind=tool_kind,
                                status="pending",
                            )
                        else:
                            title = tool_name
                            update = start_tool_call(
                                tool_call_id=tool_id,
                                title=title,
                                kind=tool_kind,
                                status="pending",
                            )

                        await self._conn.session_update(
                            session_id=session_id, update=update, source="DeepAgent"
                        )
                    except json.JSONDecodeError:
                        pass

            # Check for tool results (ToolMessage responses)
            if hasattr(message_chunk, "type") and message_chunk.type == "tool":
                # This is a tool result message
                tool_call_id = getattr(message_chunk, "tool_call_id", None)
                if tool_call_id and tool_call_id in active_tool_calls:
                    if active_tool_calls[tool_call_id].get("name") != "edit_file":
                        # Update the tool call with completion status and result
                        content = getattr(message_chunk, "content", "")
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
