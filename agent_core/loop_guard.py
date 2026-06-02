"""Loop/stall detection for provider-neutral agent runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from agent_core.actions import ParsedAction
from agent_core.tools import ToolResult


@dataclass(frozen=True)
class LoopGuardConfig:
    enabled: bool = True
    warning_repetition_limit: int = 2
    stall_repetition_limit: int = 3
    include_tool_result: bool = True
    max_result_fingerprint_bytes: int = 4096


@dataclass(frozen=True)
class LoopGuardDecision:
    status: str = "ok"
    repetition_count: int = 1
    fingerprint: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def warning(self) -> bool:
        return self.status == "warning"

    @property
    def stalled(self) -> bool:
        return self.status == "stalled"


class LoopGuard:
    """Detect repeated action/result patterns inside one agent run.

    The guard deliberately works on core abstractions only. Runtime-specific
    adapters can reset or replace it between resumed runs, but they do not need
    to understand provider SDK internals.
    """

    def __init__(self, config: LoopGuardConfig | None = None) -> None:
        self.config = config or LoopGuardConfig()
        self._last_fingerprint = ""
        self._repeat_count = 0

    def observe(
        self,
        action: ParsedAction,
        *,
        tool_result: ToolResult | None = None,
        iteration: int = 0,
    ) -> LoopGuardDecision:
        if not self.config.enabled:
            return LoopGuardDecision()
        fingerprint = self._fingerprint(action, tool_result=tool_result)
        if fingerprint == self._last_fingerprint:
            self._repeat_count += 1
        else:
            self._last_fingerprint = fingerprint
            self._repeat_count = 1

        metadata = {
            "repetition_count": self._repeat_count,
            "fingerprint": fingerprint,
            "action": action.name,
            "iteration": iteration,
        }
        if tool_result is not None:
            metadata.update(
                {
                    "tool_name": tool_result.tool_name,
                    "tool_status": tool_result.status,
                }
            )

        if self._repeat_count >= max(1, self.config.stall_repetition_limit):
            message = (
                f"loop guard stopped repeated action/result pattern after "
                f"{self._repeat_count} repetitions: {action.name}"
            )
            return LoopGuardDecision(
                status="stalled",
                repetition_count=self._repeat_count,
                fingerprint=fingerprint,
                message=message,
                metadata=metadata,
            )
        if self._repeat_count >= max(1, self.config.warning_repetition_limit):
            message = (
                f"same action/result pattern repeated {self._repeat_count} times; "
                "change strategy or finish if enough information is available"
            )
            return LoopGuardDecision(
                status="warning",
                repetition_count=self._repeat_count,
                fingerprint=fingerprint,
                message=message,
                metadata=metadata,
            )
        return LoopGuardDecision(
            repetition_count=self._repeat_count,
            fingerprint=fingerprint,
            metadata=metadata,
        )

    def reset(self) -> None:
        self._last_fingerprint = ""
        self._repeat_count = 0

    def _fingerprint(self, action: ParsedAction, *, tool_result: ToolResult | None) -> str:
        payload: dict[str, Any] = {
            "action": action.name,
            "arguments": action.arguments,
        }
        if self.config.include_tool_result and tool_result is not None:
            content = tool_result.error if tool_result.error else tool_result.content
            raw = content.encode("utf-8")
            sample = raw[: max(0, self.config.max_result_fingerprint_bytes)]
            payload["tool_result"] = {
                "tool_name": tool_result.tool_name,
                "status": tool_result.status,
                "content_sha256": hashlib.sha256(sample).hexdigest(),
                "content_bytes": len(raw),
            }
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

