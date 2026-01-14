import argparse
import asyncio

from deepagents_acp.agent import run_agent

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
    asyncio.run(run_agent(args.root_dir))
