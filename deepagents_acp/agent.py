import json
from typing import Any
from uuid import uuid4
from deepagents.graph import Checkpointer
from dotenv import load_dotenv

from acp import (
    Agent as ACPAgent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    SetSessionModeResponse,
    run_agent as run_acp_agent,
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
    AgentCapabilities,
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    PermissionOption,
    PromptCapabilities,
    ResourceContentBlock,
    SessionMode,
    SessionModeState,
    SseMcpServer,
    TextContentBlock,
    ToolCallUpdate,
)
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.graph import CompiledStateGraph

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from deepagents_acp.utils import (
    convert_audio_block_to_content_blocks,
    convert_embedded_resource_block_to_content_blocks,
    convert_resource_block_to_content_blocks,
    convert_text_block_to_content_blocks,
    convert_image_block_to_content_blocks,
)

load_dotenv()


class ACPDeepAgent(ACPAgent):
    _conn: Client

    _deepagent: CompiledStateGraph
    _root_dir: str
    _checkpointer: Checkpointer
    _mode: str

    def __init__(
        self,
        root_dir: str,
        checkpointer: Checkpointer,
        mode: str,
        interrupt_config: dict,
    ):
        self._root_dir = root_dir
        self._checkpointer = checkpointer
        self._mode = mode
        self._deepagent = create_deep_agent(
            checkpointer=checkpointer,
            backend=FilesystemBackend(root_dir=root_dir, virtual_mode=True),
            interrupt_on=interrupt_config,
        )
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
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(
                    image=True,
                    # embedded_context=True,
                )
            ),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        **kwargs: Any,
    ) -> NewSessionResponse:
        # Define available modes
        available_modes = [
            SessionMode(
                id="ask_before_edits",
                name="Ask before edits",
                description="Ask permission before edits and writes",
            ),
            SessionMode(
                id="auto",
                name="Full Auto",
                description="Never ask for permission",
            ),
        ]

        return NewSessionResponse(
            session_id=uuid4().hex,
            modes=SessionModeState(
                available_modes=available_modes,
                current_mode_id="ask_before_edits",
            ),
        )

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModeResponse:
        # Map mode IDs to interrupt configurations
        mode_to_interrupt = {
            "ask_before_edits": {
                "edit_file": {"allowed_decisions": ["approve", "reject"]},
                "write_file": {"allowed_decisions": ["approve", "reject"]},
            },
            "auto": None,  # No interrupts, full auto
        }

        interrupt_config = mode_to_interrupt.get(mode_id)

        # Recreate the deep agent with new interrupt config
        self._deepagent = create_deep_agent(
            checkpointer=self._checkpointer,
            backend=FilesystemBackend(root_dir=self._root_dir, virtual_mode=True),
            interrupt_on=interrupt_config,
        )

        self._current_mode = mode_id

        return SetSessionModeResponse()

    async def _log(self, session_id: str, text: str):
        update = update_agent_message(text_block(text))
        await self._conn.session_update(
            session_id=session_id, update=update, source="DeepAgent"
        )

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
                content_blocks.extend(convert_text_block_to_content_blocks(block))
            elif isinstance(block, ImageContentBlock):
                content_blocks.extend(
                    convert_image_block_to_content_blocks(
                        block, root_dir=self._root_dir
                    )
                )
            elif isinstance(block, AudioContentBlock):
                content_blocks.extend(convert_audio_block_to_content_blocks(block))
            elif isinstance(block, ResourceContentBlock):
                content_blocks.extend(
                    convert_resource_block_to_content_blocks(
                        block, root_dir=self._root_dir
                    )
                )
            elif isinstance(block, EmbeddedResourceContentBlock):
                content_blocks.extend(
                    convert_embedded_resource_block_to_content_blocks(block)
                )
        # Stream the deep agent response with multimodal content
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        # Track active tool calls and accumulate chunks by index
        active_tool_calls = {}
        tool_call_accumulator = {}  # index -> {id, name, args_str}

        current_state = None
        user_decisions = []

        while current_state is None or current_state.interrupts:
            async for message_chunk, metadata in self._deepagent.astream(
                Command(resume={"decisions": user_decisions})
                if user_decisions
                else {"messages": [{"role": "user", "content": content_blocks}]},
                config=config,
                stream_mode="messages",
            ):
                # Check for tool call chunks (streaming pieces of tool calls)
                if (
                    not isinstance(message_chunk, str)
                    and hasattr(message_chunk, "tool_call_chunks")
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
                                if tool_name == "read_file" and isinstance(
                                    tool_args, dict
                                ):
                                    path = tool_args.get("file_path")
                                    title = f"Read `{path}`" if path else tool_name
                                    update = start_tool_call(
                                        tool_call_id=tool_id,
                                        title=title,
                                        kind=tool_kind,
                                        status="pending",
                                    )
                                elif tool_name == "edit_file" and isinstance(
                                    tool_args, dict
                                ):
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
                                elif tool_name == "write_file" and isinstance(
                                    tool_args, dict
                                ):
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
                                    session_id=session_id,
                                    update=update,
                                    source="DeepAgent",
                                )
                            except json.JSONDecodeError:
                                pass

                if isinstance(message_chunk, str):
                    update = update_agent_message(text_block(message_chunk))
                    await self._conn.session_update(
                        session_id=session_id, update=update, source="DeepAgent"
                    )
                # Check for tool results (ToolMessage responses)
                elif hasattr(message_chunk, "type") and message_chunk.type == "tool":
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

            # Check if the agent is interrupted (waiting for HITL approval)
            current_state = await self._deepagent.aget_state(config)
            user_decisions = []
            if current_state.next and current_state.interrupts:
                # Agent is interrupted, request permission from user
                for interrupt in current_state.interrupts:
                    # Get the tool call info from the interrupt
                    tool_call_id = interrupt.id
                    interrupt_value = interrupt.value

                    # Extract tool information from interrupt value
                    tool_name = (
                        interrupt_value.get("name", "tool")
                        if isinstance(interrupt_value, dict)
                        else "tool"
                    )
                    tool_args = (
                        interrupt_value.get("args", {})
                        if isinstance(interrupt_value, dict)
                        else {}
                    )

                    # Create a title for the permission request
                    if tool_name == "edit_file" and isinstance(tool_args, dict):
                        file_path = tool_args.get("file_path", "file")
                        title = f"Edit `{file_path}`"
                    elif tool_name == "write_file" and isinstance(tool_args, dict):
                        file_path = tool_args.get("file_path", "file")
                        title = f"Write `{file_path}`"
                    else:
                        title = tool_name

                    # Create permission options
                    options = [
                        PermissionOption(
                            option_id="approve",
                            name="Approve",
                            kind="allow_once",
                        ),
                        PermissionOption(
                            option_id="reject",
                            name="Reject",
                            kind="reject_once",
                        ),
                    ]

                    # Request permission from the client
                    tool_call_update = ToolCallUpdate(
                        tool_call_id=tool_call_id,
                        title=title,
                    )
                    response = await self._conn.request_permission(
                        session_id=session_id,
                        tool_call=tool_call_update,
                        options=options,
                    )
                    # Handle the user's decision
                    if response.outcome.outcome == "selected":
                        user_decisions.append({"type": response.outcome.option_id})
                    else:
                        # User cancelled, treat as rejection
                        user_decisions.append({"type": "reject"})

        return PromptResponse(stop_reason="end_turn")


async def run_agent(root_dir: str) -> None:
    checkpointer = MemorySaver()

    # Start with ask_before_edits mode (ask before edits)
    mode_id = "ask_before_edits"

    # Configure interrupt based on mode
    mode_to_interrupt = {
        "ask_before_edits": {
            "edit_file": {"allowed_decisions": ["approve", "reject"]},
            "write_file": {"allowed_decisions": ["approve", "reject"]},
        },
        "auto": None,  # No interrupts, full auto
    }

    interrupt_config = mode_to_interrupt.get(mode_id)

    acp_agent = ACPDeepAgent(
        root_dir=root_dir,
        mode=mode_id,
        checkpointer=checkpointer,
        interrupt_config=interrupt_config,
    )
    await run_acp_agent(acp_agent)
