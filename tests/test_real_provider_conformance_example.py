from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_real_provider_conformance_example_dry_run_is_offline(monkeypatch) -> None:
    monkeypatch.delenv("RAVEN_HEART_LLM_API_KEY", raising=False)

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "real_provider_conformance.py"),
            "--dry-run",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(result.stdout)

    assert manifest["schema_version"] == "raven-heart-real-provider-smoke-dry-run/v1"
    assert manifest["ready_to_run"] is False
    assert manifest["config"]["api_key"] == ""
    assert manifest["config"]["model"]
