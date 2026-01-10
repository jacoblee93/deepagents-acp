import asyncio
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
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import Runnable, RunnableConfig

load_dotenv()

class ACPDeepAgent(ACPAgent):
    _conn: Client
    
    _deepagent: Runnable
    
    def __init__(self, deepagent: Runnable):
        self._deepagent = deepagent
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
        for block in prompt:
            text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            config: RunnableConfig = {"configurable": {"thread_id": session_id}}
            res = self._deepagent.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
            chunk = update_agent_message(text_block(res["messages"][-1].content))
            await self._conn.session_update(session_id=session_id, update=chunk, source="echo_agent")
        return PromptResponse(stop_reason="end_turn")


async def main() -> None:
    checkpointer = MemorySaver()
    deepagent = create_deep_agent(
        # tools=[...],
        # interrupt_on={...},
        checkpointer=checkpointer
    )
    acp_agent = ACPDeepAgent(deepagent=deepagent)
    await run_agent(acp_agent)


if __name__ == "__main__":
    asyncio.run(main())
