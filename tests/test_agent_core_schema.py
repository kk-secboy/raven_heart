from __future__ import annotations

from agent_core.actions import ActionRegistry, ActionSpec, ParsedAction
from agent_core.schema import validate_json_schema_subset


def test_schema_validator_reports_multiple_prompt_safe_issues() -> None:
    result = validate_json_schema_subset(
        {"query": "", "limit": 0, "extra": True},
        {
            "type": "object",
            "required": ["query", "mode"],
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "mode": {"type": "string", "enum": ["fast", "deep"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
        },
        schema_name="tool:search",
    )
    manifest = result.manifest()
    codes = {issue.code for issue in result.issues}

    assert not result.ok
    assert result.error == "$.mode is required"
    assert {
        "required_missing",
        "additional_property",
        "min_length",
        "minimum",
    } <= codes
    assert manifest["schema_version"] == "agent-core-schema-validation-result/v1"
    assert manifest["schema_name"] == "tool:search"
    assert manifest["issue_count"] == len(result.issues)


def test_action_registry_exposes_schema_validation_result_without_throwing() -> None:
    registry = ActionRegistry()
    registry.register(
        ActionSpec(
            name="call_tool",
            parameters_schema={
                "type": "object",
                "required": ["tool_name"],
                "additionalProperties": False,
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
            },
        )
    )

    result = registry.validate_result(
        ParsedAction(
            name="call_tool",
            arguments={"tool_name": 123, "arguments": {}, "debug": True},
        )
    )
    manifest = result.manifest()

    assert not result.ok
    assert result.schema_name == "action:call_tool"
    assert {issue.code for issue in result.issues} == {
        "type_mismatch",
        "additional_property",
    }
    assert manifest["metadata"]["action"]["name"] == "call_tool"
    assert manifest["metadata"]["action_spec"]["schema_version"] == "agent-core-action-spec/v1"
