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
        self, cwd: str, mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio], **kwargs: Any
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
                    content_blocks.append({
                        "type": "text",
                        "text": f"[Image file at path: {block.uri}, MIME type: {block.mime_type}]"
                    })
                else:
                    # Use inline base64 data
                    data_uri = f"data:{block.mime_type};base64,{block.data}"
                    content_blocks.append({"type": "image_url", "image_url": data_uri})
            elif isinstance(block, AudioContentBlock):
                # Audio blocks - represent as text with data URI for tools to access
                data_uri = f"data:{block.mime_type};base64,{block.data}"
                content_blocks.append({
                    "type": "text",
                    "text": f"[Audio file available at: {data_uri}]"
                })
            elif isinstance(block, ResourceContentBlock):
                # Resource blocks reference external resources
                resource_text = f"[Resource: {block.name}"
                if block.uri:
                    resource_text += f"\nURI: {block.uri}"
                if block.description:
                    resource_text += f"\nDescription: {block.description}"
                if block.mime_type:
                    resource_text += f"\nMIME type: {block.mime_type}"
                resource_text += "]"
                content_blocks.append({"type": "text", "text": resource_text})
            elif isinstance(block, EmbeddedResourceContentBlock):
                # Embedded resource blocks contain the resource data inline
                resource = block.resource
                if hasattr(resource, 'text'):
                    # Text resource
                    content_blocks.append({"type": "text", "text": resource.text})
                elif hasattr(resource, 'blob'):
                    # Binary resource - provide as data URI
                    mime_type = getattr(resource, 'mime_type', 'application/octet-stream')
                    data_uri = f"data:{mime_type};base64,{resource.blob}"
                    content_blocks.append({
                        "type": "text",
                        "text": f"[Embedded resource available at: {data_uri}]"
                    })
        
        # Invoke the deep agent with multimodal content
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        res = self._deepagent.invoke({"messages": [{"role": "user", "content": content_blocks}]}, config=config)
        chunk = update_agent_message(text_block(res["messages"][-1].content + self._root_dir))
        await self._conn.session_update(session_id=session_id, update=chunk, source="DeepAgent")
        
        return PromptResponse(stop_reason="end_turn")


async def main(root_dir: str) -> None:
    checkpointer = MemorySaver()
    print(root_dir)
    deepagent = create_deep_agent(
        # tools=[...],
        # interrupt_on={...},
        checkpointer=checkpointer,
        backend=FilesystemBackend(root_dir=root_dir, virtual_mode=True)
    )
    acp_agent = ACPDeepAgent(deepagent=deepagent, root_dir=root_dir)
    await run_agent(acp_agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACP DeepAgent with specified root directory")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/",
        help="Root directory accessible to the agent (default: /)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.root_dir))
