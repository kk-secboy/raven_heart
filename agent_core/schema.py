"""Small provider-neutral JSON schema validation primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SchemaValidationIssue:
    path: str
    code: str
    message: str
    expected: str = ""
    actual: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SchemaValidationResult:
    ok: bool
    schema_name: str = ""
    issues: tuple[SchemaValidationIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error(self) -> str:
        return self.issues[0].message if self.issues else ""

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-schema-validation-result/v1",
            "ok": self.ok,
            "schema_name": self.schema_name,
            "issue_count": len(self.issues),
            "issues": [issue.manifest() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


def validate_json_schema_subset(
    value: Any,
    schema: dict[str, Any],
    *,
    schema_name: str = "",
    metadata: dict[str, Any] | None = None,
) -> SchemaValidationResult:
    issues: list[SchemaValidationIssue] = []
    _validate_schema(value, schema or {}, path="$", issues=issues)
    return SchemaValidationResult(
        ok=not issues,
        schema_name=schema_name,
        issues=tuple(issues),
        metadata=dict(metadata or {}),
    )


def _validate_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
    issues: list[SchemaValidationIssue],
) -> None:
    expected = schema.get("type")
    if expected and not _validate_type(value, expected, path, issues):
        return
    if "const" in schema and value != schema["const"]:
        issues.append(
            SchemaValidationIssue(
                path=path,
                code="const_mismatch",
                message=f"{path} must equal {schema['const']!r}",
                expected=repr(schema["const"]),
                actual=repr(value),
            )
        )
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        issues.append(
            SchemaValidationIssue(
                path=path,
                code="enum_mismatch",
                message=f"{path} must be one of {enum_values!r}",
                expected=repr(enum_values),
                actual=repr(value),
            )
        )
    if isinstance(value, dict):
        _validate_object(value, schema, path=path, issues=issues)
    if isinstance(value, list):
        _validate_array(value, schema, path=path, issues=issues)
    if isinstance(value, str):
        _validate_string(value, schema, path=path, issues=issues)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_number(value, schema, path=path, issues=issues)


def _validate_object(
    value: dict[str, Any],
    schema: dict[str, Any],
    *,
    path: str,
    issues: list[SchemaValidationIssue],
) -> None:
    required = schema.get("required") or ()
    for key in required:
        if key not in value:
            issues.append(
                SchemaValidationIssue(
                    path=f"{path}.{key}",
                    code="required_missing",
                    message=f"{path}.{key} is required",
                    expected="present",
                    actual="missing",
                )
            )
    properties = schema.get("properties") or {}
    if isinstance(properties, dict):
        for key, prop_schema in properties.items():
            if key in value and isinstance(prop_schema, dict):
                _validate_schema(value[key], prop_schema, path=f"{path}.{key}", issues=issues)
    if schema.get("additionalProperties") is False and isinstance(properties, dict):
        allowed = set(properties)
        for key in sorted(set(value) - allowed):
            issues.append(
                SchemaValidationIssue(
                    path=f"{path}.{key}",
                    code="additional_property",
                    message=f"{path}.{key} is not allowed",
                    expected="absent",
                    actual="present",
                )
            )


def _validate_array(
    value: list[Any],
    schema: dict[str, Any],
    *,
    path: str,
    issues: list[SchemaValidationIssue],
) -> None:
    min_items = schema.get("minItems")
    if min_items is not None and len(value) < int(min_items):
        issues.append(
            SchemaValidationIssue(
                path=path,
                code="min_items",
                message=f"{path} must contain at least {int(min_items)} items",
                expected=str(int(min_items)),
                actual=str(len(value)),
            )
        )
    max_items = schema.get("maxItems")
    if max_items is not None and len(value) > int(max_items):
        issues.append(
            SchemaValidationIssue(
                path=path,
                code="max_items",
                message=f"{path} must contain at most {int(max_items)} items",
                expected=str(int(max_items)),
                actual=str(len(value)),
            )
        )
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            _validate_schema(item, item_schema, path=f"{path}[{index}]", issues=issues)


def _validate_string(
    value: str,
    schema: dict[str, Any],
    *,
    path: str,
    issues: list[SchemaValidationIssue],
) -> None:
    min_length = schema.get("minLength")
    if min_length is not None and len(value) < int(min_length):
        issues.append(
            SchemaValidationIssue(
                path=path,
                code="min_length",
                message=f"{path} must contain at least {int(min_length)} characters",
                expected=str(int(min_length)),
                actual=str(len(value)),
            )
        )
    max_length = schema.get("maxLength")
    if max_length is not None and len(value) > int(max_length):
        issues.append(
            SchemaValidationIssue(
                path=path,
                code="max_length",
                message=f"{path} must contain at most {int(max_length)} characters",
                expected=str(int(max_length)),
                actual=str(len(value)),
            )
        )


def _validate_number(
    value: int | float,
    schema: dict[str, Any],
    *,
    path: str,
    issues: list[SchemaValidationIssue],
) -> None:
    minimum = schema.get("minimum")
    if minimum is not None and value < float(minimum):
        issues.append(
            SchemaValidationIssue(
                path=path,
                code="minimum",
                message=f"{path} must be at least {minimum}",
                expected=str(minimum),
                actual=str(value),
            )
        )
    maximum = schema.get("maximum")
    if maximum is not None and value > float(maximum):
        issues.append(
            SchemaValidationIssue(
                path=path,
                code="maximum",
                message=f"{path} must be at most {maximum}",
                expected=str(maximum),
                actual=str(value),
            )
        )


def _validate_type(
    value: Any,
    expected: Any,
    path: str,
    issues: list[SchemaValidationIssue],
) -> bool:
    expected_types = tuple(str(item) for item in expected) if isinstance(expected, list) else (str(expected),)
    if any(_matches_type(value, item) for item in expected_types):
        return True
    expected_label = "|".join(expected_types)
    actual = _value_type(value)
    issues.append(
        SchemaValidationIssue(
            path=path,
            code="type_mismatch",
            message=f"{path} must be {expected_label}, got {actual}",
            expected=expected_label,
            actual=actual,
        )
    )
    return False


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__
