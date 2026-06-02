"""Action registry and validation for ReAct execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_core.errors import ActionError


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str = ""
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    terminal: bool = False


@dataclass(frozen=True)
class ParsedAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


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
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters_schema": spec.parameters_schema,
                    "terminal": spec.terminal,
                }
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
        arguments = data.get("arguments", data.get("args", {}))
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ActionError("action arguments must be an object")
        parsed = ParsedAction(name=name, arguments=arguments, raw=raw)
        self.validate(parsed)
        return parsed

    def validate(self, action: ParsedAction) -> None:
        spec = self.get(action.name)
        if spec is None:
            raise ActionError(f"unknown action: {action.name}")
        schema = spec.parameters_schema or {}
        required = schema.get("required") or []
        for key in required:
            if key not in action.arguments:
                raise ActionError(f"action {action.name} missing required argument: {key}")
        properties = schema.get("properties") or {}
        for key, prop_schema in properties.items():
            if key in action.arguments:
                self._validate_type(action.name, key, action.arguments[key], prop_schema)

    @staticmethod
    def _validate_type(action_name: str, key: str, value: Any, prop_schema: dict[str, Any]) -> None:
        expected = prop_schema.get("type")
        if not expected:
            return
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        py_type = type_map.get(str(expected))
        if py_type and not isinstance(value, py_type):
            raise ActionError(
                f"action {action_name} argument {key} must be {expected}, "
                f"got {type(value).__name__}"
            )


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

