"""Structured output validation and repair hints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class StructuredOutputSpec:
    """Request-level output contract for the final agent answer."""

    schema: dict[str, Any]
    name: str = "structured_output"
    description: str = ""
    max_repairs: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-structured-output-spec/v1",
            "name": self.name,
            "description": self.description,
            "schema": dict(self.schema),
            "max_repairs": self.max_repairs,
            "metadata": dict(self.metadata),
        }

    def render_prompt(self) -> str:
        lines = [
            "== Structured Output ==",
            f"name: {self.name}",
            "Return the final finish.output as JSON matching this schema.",
        ]
        if self.description:
            lines.append(f"description: {self.description}")
        lines.append(json.dumps(self.schema, ensure_ascii=False, sort_keys=True))
        return "\n".join(lines)


@dataclass(frozen=True)
class StructuredOutputResult:
    ok: bool
    raw_output: str
    value: Any = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-structured-output-result/v1",
            "ok": self.ok,
            "raw_output_bytes": len(self.raw_output.encode("utf-8")),
            "value": self.value if self.ok else None,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class StructuredOutputValidatorPort(Protocol):
    async def validate(
        self,
        output: str,
        spec: StructuredOutputSpec,
    ) -> StructuredOutputResult:
        """Validate final output against a provider-neutral contract."""


class JsonStructuredOutputValidator:
    """Small deterministic JSON schema subset validator.

    It intentionally mirrors the SDK's action/tool schema subset instead of
    pulling in a heavy schema dependency. Runtime adapters can replace this
    with richer validation if they need full JSON Schema semantics.
    """

    async def validate(
        self,
        output: str,
        spec: StructuredOutputSpec,
    ) -> StructuredOutputResult:
        raw = str(output or "").strip()
        if not raw:
            return StructuredOutputResult(ok=False, raw_output=output, error="output is empty")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            extracted = _extract_json_value(raw)
            if extracted is None:
                return StructuredOutputResult(
                    ok=False,
                    raw_output=output,
                    error=f"output is not valid JSON: {exc.msg}",
                )
            value = extracted
        error = _validate_schema(value, spec.schema, path="$")
        if error:
            return StructuredOutputResult(ok=False, raw_output=output, value=value, error=error)
        return StructuredOutputResult(
            ok=True,
            raw_output=output,
            value=value,
            metadata={"schema_name": spec.name},
        )

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "agent-core-json-structured-output-validator/v1"}


def structured_output_feedback(
    result: StructuredOutputResult,
    spec: StructuredOutputSpec,
) -> str:
    return (
        "Final output failed structured validation. "
        f"Error: {result.error}. "
        "Call finish again with finish.output as JSON matching this schema: "
        f"{json.dumps(spec.schema, ensure_ascii=False, sort_keys=True)}"
    )


def _extract_json_value(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def _validate_schema(value: Any, schema: dict[str, Any], *, path: str) -> str:
    expected = schema.get("type")
    if expected:
        type_error = _validate_type(value, str(expected), path)
        if type_error:
            return type_error
    if isinstance(value, dict):
        required = schema.get("required") or ()
        for key in required:
            if key not in value:
                return f"{path}.{key} is required"
        properties = schema.get("properties") or {}
        for key, prop_schema in properties.items():
            if key in value and isinstance(prop_schema, dict):
                error = _validate_schema(value[key], prop_schema, path=f"{path}.{key}")
                if error:
                    return error
    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _validate_schema(item, item_schema, path=f"{path}[{index}]")
                if error:
                    return error
    return ""


def _validate_type(value: Any, expected: str, path: str) -> str:
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
        "null": type(None),
    }
    py_type = type_map.get(expected)
    if py_type is None:
        return ""
    if expected == "integer" and isinstance(value, bool):
        return f"{path} must be integer, got boolean"
    if expected == "number" and isinstance(value, bool):
        return f"{path} must be number, got boolean"
    if not isinstance(value, py_type):
        return f"{path} must be {expected}, got {type(value).__name__}"
    return ""
