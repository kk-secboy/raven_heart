"""Action registry and validation for ReAct execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_core.errors import ActionError
from agent_core.schema import (
    SchemaValidationIssue,
    SchemaValidationResult,
    validate_json_schema_subset,
)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str = ""
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    terminal: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-action-spec/v1",
            "name": self.name,
            "description": self.description,
            "parameters_schema": dict(self.parameters_schema),
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class ParsedAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-parsed-action/v1",
            "name": self.name,
            "argument_keys": sorted(str(key) for key in self.arguments),
            "raw_type": type(self.raw).__name__ if self.raw is not None else "",
        }


@dataclass(frozen=True)
class ActionResult:
    action: ParsedAction
    ok: bool
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionVerification:
    ok: bool = True
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionVerifierPort(Protocol):
    async def verify(self, action: ParsedAction) -> ActionVerification:
        """Verify an action before policy/tool execution."""


class NullActionVerifier:
    async def verify(self, action: ParsedAction) -> ActionVerification:
        return ActionVerification()


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, ActionSpec] = {}

    def register(self, spec: ActionSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ActionError("action name is required")
        self._actions[name] = spec

    def get(self, name: str) -> ActionSpec | None:
        return self._actions.get(name)

    def specs(self) -> tuple[ActionSpec, ...]:
        return tuple(self._actions.values())

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-action-registry/v1",
            "actions": [
                spec.manifest()
                for spec in sorted(self._actions.values(), key=lambda item: item.name)
            ],
        }

    def render_actions(self) -> str:
        lines = []
        for spec in sorted(self._actions.values(), key=lambda item: item.name):
            terminal = " terminal=true" if spec.terminal else ""
            lines.append(f"- {spec.name}{terminal}: {spec.description}".rstrip())
        return "\n".join(lines)

    def parse(self, raw: str | dict[str, Any]) -> ParsedAction:
        data: Any = raw
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                raise ActionError("empty model response")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                data = _extract_first_json_object(text)
                if data is None:
                    raise ActionError(f"model response is not valid JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ActionError("action payload must be an object")
        nested = data.get("next_action")
        if isinstance(nested, dict) and not (data.get("action") or data.get("name")):
            data = {
                "action": nested.get("type") or nested.get("action") or nested.get("name"),
                "arguments": nested.get("arguments", nested.get("params", {})),
            }
        name = str(data.get("action") or data.get("name") or "").strip()
        if not name:
            raise ActionError("action payload missing action/name")
        arguments = data.get("arguments", data.get("args"))
        if arguments is None:
            arguments = {
                str(key): value
                for key, value in data.items()
                if key not in {"action", "name", "next_action"}
            }
        if not isinstance(arguments, dict):
            raise ActionError("action arguments must be an object")
        parsed = ParsedAction(name=name, arguments=arguments, raw=raw)
        self.validate(parsed)
        return parsed

    def validate(self, action: ParsedAction) -> None:
        result = self.validate_result(action)
        if result.ok:
            return
        raise ActionError(result.error)

    def validate_result(self, action: ParsedAction) -> SchemaValidationResult:
        spec = self.get(action.name)
        if spec is None:
            return SchemaValidationResult(
                ok=False,
                schema_name=f"action:{action.name}",
                issues=(
                    _schema_issue(
                        path="$",
                        code="unknown_action",
                        message=f"unknown action: {action.name}",
                    ),
                ),
                metadata={"action": action.manifest()},
            )
        schema = spec.parameters_schema or {}
        validation = validate_json_schema_subset(
            action.arguments,
            schema,
            schema_name=f"action:{action.name}",
            metadata={
                "action": action.manifest(),
                "action_spec": spec.manifest(),
            },
        )
        return validation


def _schema_issue(
    *,
    path: str,
    code: str,
    message: str,
) -> SchemaValidationIssue:
    return SchemaValidationIssue(path=path, code=code, message=message)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None

