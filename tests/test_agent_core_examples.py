from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_example(path: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(ROOT / path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_minimal_react_example_runs() -> None:
    payload = _run_example("examples/minimal_react.py")

    assert payload["status"] == "completed"
    assert payload["output"] == "demo-service summarized"
    assert payload["tool_calls"] == 1


def test_memory_and_skills_example_runs() -> None:
    payload = _run_example("examples/memory_and_skills.py")

    assert payload["loaded_skills"] == ["code-review"]
    assert any("focused tests" in item for item in payload["memory_hits"])