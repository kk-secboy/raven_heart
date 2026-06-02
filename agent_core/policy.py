"""Policy decision protocol for the agent core."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal, Protocol

from agent_core.actions import ParsedAction
from agent_core.tools import ToolInvocation


PolicyStatus = Literal["allow", "deny", "approval_required"]


@dataclass(frozen=True)
class ApprovalRequest:
    reason: str
    subject: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    status: PolicyStatus = "allow"
    reason: str = ""
    approval: ApprovalRequest | None = None

    @property
    def allowed(self) -> bool:
        return self.status == "allow"


class PolicyPort(Protocol):
    async def check_action(self, action: ParsedAction) -> PolicyDecision:
        """Check whether an action is allowed."""

    async def check_tool(self, invocation: ToolInvocation) -> PolicyDecision:
        """Check whether a tool invocation is allowed."""


class AllowAllPolicy:
    async def check_action(self, action: ParsedAction) -> PolicyDecision:
        return PolicyDecision()

    async def check_tool(self, invocation: ToolInvocation) -> PolicyDecision:
        return PolicyDecision()


@dataclass(frozen=True)
class PolicyRule:
    name: str
    status: PolicyStatus
    reason: str = ""
    action_names: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    tool_tags: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    mcp_transports: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def check_action(self, action: ParsedAction) -> PolicyDecision | None:
        if not self.action_names:
            return None
        if not _matches_any(action.name, self.action_names):
            return None
        return self._decision(subject=f"action:{action.name}")

    def check_tool(self, invocation: ToolInvocation) -> PolicyDecision | None:
        if not self._has_tool_predicate:
            return None
        spec = _tool_spec_metadata(invocation)
        names = {
            invocation.tool_name,
            str(spec.get("name") or ""),
            *tuple(str(alias) for alias in spec.get("aliases") or ()),
        }
        if self.tool_names and not any(_matches_any(name, self.tool_names) for name in names):
            return None
        tags = {str(tag).lower() for tag in spec.get("tags") or ()}
        if self.tool_tags and not tags.intersection({tag.lower() for tag in self.tool_tags}):
            return None
        mcp = spec.get("metadata", {}).get("mcp", {}) if isinstance(spec.get("metadata"), dict) else {}
        if self.mcp_servers and not _matches_any(str(mcp.get("server_name") or ""), self.mcp_servers):
            return None
        if self.mcp_transports and not _matches_any(str(mcp.get("transport") or ""), self.mcp_transports):
            return None
        if self.metadata and not _metadata_matches(spec.get("metadata", {}), self.metadata):
            return None
        return self._decision(subject=f"tool:{invocation.tool_name}")

    @property
    def _has_tool_predicate(self) -> bool:
        return bool(
            self.tool_names
            or self.tool_tags
            or self.mcp_servers
            or self.mcp_transports
            or self.metadata
        )

    def _decision(self, *, subject: str) -> PolicyDecision:
        reason = self.reason or f"policy rule matched: {self.name}"
        approval = None
        if self.status == "approval_required":
            approval = ApprovalRequest(
                reason=reason,
                subject=subject,
                metadata={"rule": self.name},
            )
        return PolicyDecision(status=self.status, reason=reason, approval=approval)


class RuleBasedPolicy:
    def __init__(self, rules: Iterable[PolicyRule] = ()) -> None:
        self.rules = tuple(rules)

    async def check_action(self, action: ParsedAction) -> PolicyDecision:
        return self._first_decision(rule.check_action(action) for rule in self.rules)

    async def check_tool(self, invocation: ToolInvocation) -> PolicyDecision:
        return self._first_decision(rule.check_tool(invocation) for rule in self.rules)

    @staticmethod
    def _first_decision(decisions: Iterable[PolicyDecision | None]) -> PolicyDecision:
        approval: PolicyDecision | None = None
        for decision in decisions:
            if decision is None or decision.status == "allow":
                continue
            if decision.status == "deny":
                return decision
            if decision.status == "approval_required" and approval is None:
                approval = decision
        return approval or PolicyDecision()


class CompositePolicy:
    def __init__(self, policies: Iterable[PolicyPort]) -> None:
        self.policies = tuple(policies)

    async def check_action(self, action: ParsedAction) -> PolicyDecision:
        approval: PolicyDecision | None = None
        for policy in self.policies:
            decision = await policy.check_action(action)
            if decision.status == "deny":
                return decision
            if decision.status == "approval_required" and approval is None:
                approval = decision
        return approval or PolicyDecision()

    async def check_tool(self, invocation: ToolInvocation) -> PolicyDecision:
        approval: PolicyDecision | None = None
        for policy in self.policies:
            decision = await policy.check_tool(invocation)
            if decision.status == "deny":
                return decision
            if decision.status == "approval_required" and approval is None:
                approval = decision
        return approval or PolicyDecision()


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(value, pattern) for pattern in patterns)


def _tool_spec_metadata(invocation: ToolInvocation) -> dict[str, Any]:
    spec = invocation.metadata.get("tool_spec")
    return spec if isinstance(spec, dict) else {}


def _metadata_matches(metadata: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(metadata, dict):
        return False
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if isinstance(expected_value, dict):
            if not _metadata_matches(actual, expected_value):
                return False
            continue
        if actual != expected_value:
            return False
    return True

