from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_core
from agent_core.providers import (
    LLMProviderError,
    OpenAICompatibleLLMProviderCodec,
    TransportLLMProvider,
)


class OpenAICompatibleHTTPTransport:
    """Example runtime-owned transport for opt-in real provider testing."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        organization: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.organization = organization

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._post_json, dict(payload, stream=False))

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        stream_payload = {
            **payload,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        chunks = await asyncio.to_thread(self._post_stream, stream_payload)
        for chunk in chunks:
            yield chunk

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "raven-heart-example-openai-compatible-http-transport/v1",
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "organization_configured": bool(self.organization),
            "api_key_configured": bool(self.api_key),
        }

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        with urllib.request.urlopen(
            self._request(payload),
            timeout=self.timeout_seconds,
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_stream(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        with urllib.request.urlopen(
            self._request(payload),
            timeout=self.timeout_seconds,
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                chunks.append(json.loads(data))
        return chunks

    def _request(self, payload: dict[str, Any]) -> urllib.request.Request:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )


def _config_from_env() -> dict[str, Any]:
    return {
        "base_url": os.getenv("RAVEN_HEART_LLM_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("RAVEN_HEART_LLM_API_KEY", ""),
        "model": os.getenv("RAVEN_HEART_LLM_MODEL", "gpt-4.1-mini"),
        "organization": os.getenv("RAVEN_HEART_LLM_ORG", ""),
        "require_streaming": _env_flag("RAVEN_HEART_LLM_REQUIRE_STREAMING", True),
        "require_json_mode": _env_flag("RAVEN_HEART_LLM_REQUIRE_JSON_MODE", True),
        "require_tool_calls": _env_flag("RAVEN_HEART_LLM_REQUIRE_TOOL_CALLS", False),
        "max_output_tokens": int(os.getenv("RAVEN_HEART_LLM_MAX_OUTPUT_TOKENS", "96")),
    }


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ("<configured>" if key == "api_key" and value else value)
        for key, value in config.items()
    }


async def _run(*, dry_run: bool) -> int:
    config = _config_from_env()
    if dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "raven-heart-real-provider-smoke-dry-run/v1",
                    "ready_to_run": bool(config["api_key"]),
                    "config": _redacted_config(config),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not config["api_key"]:
        print(
            "RAVEN_HEART_LLM_API_KEY is required for real provider conformance.",
            file=sys.stderr,
        )
        return 2

    transport = OpenAICompatibleHTTPTransport(
        base_url=str(config["base_url"]),
        api_key=str(config["api_key"]),
        organization=str(config["organization"]),
    )
    provider = TransportLLMProvider(
        transport,
        codec=OpenAICompatibleLLMProviderCodec(),
        name="real-openai-compatible",
        metadata={"example": "real_provider_conformance"},
    )
    spec = agent_core.AgentCoreProviderConformanceSpec(
        provider_name="real-openai-compatible",
        model=str(config["model"]),
        require_streaming=bool(config["require_streaming"]),
        require_json_mode=bool(config["require_json_mode"]),
        require_tool_calls=bool(config["require_tool_calls"]),
        max_output_tokens=int(config["max_output_tokens"]),
        metadata={"real_provider": True},
    )
    try:
        report = await agent_core.run_agent_core_provider_conformance(
            provider=provider,
            spec=spec,
            metadata={"example": "real_provider_conformance"},
        )
    except (LLMProviderError, urllib.error.URLError, TimeoutError) as exc:
        print(f"Provider conformance transport failed: {exc}", file=sys.stderr)
        return 1

    manifest = report.manifest()
    manifest["metadata"]["config"] = _redacted_config(config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if report.ready else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print redacted environment-derived config without making network calls",
    )
    args = parser.parse_args()
    return asyncio.run(_run(dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
