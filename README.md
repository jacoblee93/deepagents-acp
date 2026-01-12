Add this to your Zed `settings.json`:

```json
{
  "DeepAgents": {
    "type": "custom",
    "command": "sh",
    "args": [
      "-c",
      "ORIG_PWD=\"$PWD\" && cd /Users/jacoblee/langchain/testing/deepagents-acp && uv run agent.py --root-dir \"$ORIG_PWD\""
    ]
  }
}
```
