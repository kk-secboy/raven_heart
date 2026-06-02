from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core.memory import MarkdownMemoryStore, MemoryCenter, MemoryQuery, MemoryWrite, SQLiteMemoryStore
from agent_core.skills import SkillRegistry, SkillSpec, SkillsContext


async def main() -> None:
    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="code-review",
            description="Review code changes before style comments.",
            prompt="Check correctness, safety, and missing tests before style.",
            tags=("code", "review"),
            priority=10,
        )
    )
    skills = SkillsContext(registry)
    skills.load("code-review")

    root_path = Path(tempfile.mkdtemp(prefix="raven-heart-example-"))
    memory = MemoryCenter(default_store="sqlite")
    memory.register("sqlite", SQLiteMemoryStore(root_path / "memory.sqlite"), priority=10)
    memory.register("markdown", MarkdownMemoryStore(root_path / "notes"), priority=1)

    await memory.write(MemoryWrite(content="Prefer focused tests for narrow changes.", source="sqlite"))
    await memory.write(
        MemoryWrite(
            content="Code agent should preserve unrelated user changes.",
            source="note",
            metadata={"store": "markdown"},
        )
    )

    hits = await memory.search(MemoryQuery(query="focused tests unrelated changes", limit=4))
    print(json.dumps(
        {
            "loaded_skills": [item.name for item in skills.loaded()],
            "memory_hits": [hit.content for hit in hits],
            "memory_root": str(root_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    asyncio.run(main())
