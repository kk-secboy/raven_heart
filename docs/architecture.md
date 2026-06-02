# Architecture

`agent_core` is a base, not a Raven runtime. Keep concrete systems behind ports:

| Core area | Runtime responsibility |
| --- | --- |
| LLM provider center | OpenAI, Anthropic, local model, or Raven model adapters |
| Tool center | Concrete pentest tools, sandboxing, operator approval |
| MCP center | Server lifecycle, credentials, deployment |
| Memory center | Graphiti, document RAG, project-specific durable stores |
| Prompt buckets | Runtime task contracts and domain-specific context material |
| Harness/ReAct runner | UI events, API routes, persistence, task orchestration |

This keeps the base reusable for code, ops, research, and security agents.

