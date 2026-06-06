from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _truthy(report: dict[str, Any], path: str, issues: list[str]) -> None:
    value: Any = report
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            issues.append(f"missing {path}")
            return
        value = value[part]
    if value is not True:
        issues.append(f"{path} is not true: {value!r}")


def _semidynamic_1_stable(report: dict[str, Any], prefix: str, issues: list[str]) -> None:
    value: Any = report
    stable_path = f"{prefix}.semi_dynamic_1_stable"
    for part in stable_path.split("."):
        if not isinstance(value, dict) or part not in value:
            break
        value = value[part]
    else:
        if value is not True:
            issues.append(f"{stable_path} is not true: {value!r}")
        return

    value = report
    changed_path = f"{prefix}.semi_dynamic_1_changed"
    for part in changed_path.split("."):
        if not isinstance(value, dict) or part not in value:
            issues.append(f"missing {stable_path}")
            return
        value = value[part]
    if value is not False:
        issues.append(f"{stable_path} is not true and {changed_path} is not false: {value!r}")


def _min_int(report: dict[str, Any], path: str, minimum: int, issues: list[str]) -> None:
    value: Any = report
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            issues.append(f"missing {path}")
            return
        value = value[part]
    try:
        actual = int(value)
    except (TypeError, ValueError):
        issues.append(f"{path} is not an int: {value!r}")
        return
    if actual < minimum:
        issues.append(f"{path}={actual} < {minimum}")


def _max_int(report: dict[str, Any], path: str, maximum: int, issues: list[str]) -> None:
    value: Any = report
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            issues.append(f"missing {path}")
            return
        value = value[part]
    try:
        actual = int(value)
    except (TypeError, ValueError):
        issues.append(f"{path} is not an int: {value!r}")
        return
    if actual > maximum:
        issues.append(f"{path}={actual} > {maximum}")


def _verify_live(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if report.get("status") != "completed":
        issues.append(f"status is not completed: {report.get('status')!r}")
    if len(report.get("knowledge_documents") or ()) < 3:
        issues.append("live report has fewer than 3 knowledge documents")
    for doc in report.get("knowledge_documents") or ():
        if not doc.get("path") or not doc.get("sha256"):
            issues.append(f"knowledge document lacks path/sha256: {doc!r}")
    for path in (
        "functional_assertions.all_runs_completed",
        "functional_assertions.alpha_tool_fact_in_timeline",
        "functional_assertions.beta_tool_fact_in_timeline",
        "functional_assertions.alpha_tool_fact_in_memory",
        "functional_assertions.beta_tool_fact_in_memory",
        "functional_assertions.alpha_return_prompt_contains_memory",
        "functional_assertions.alpha_return_prompt_contains_alpha_fact",
        "functional_assertions.tool_result_visible_in_provider_request",
        "functional_assertions.skills_visible_in_prompt",
        "functional_assertions.mcp_visible_in_prompt",
        "functional_assertions.knowledge_or_midterm_used_after_topic_switch",
        "functional_assertions.embedding_used",
        "functional_assertions.perception_used",
        "prompt_stability.high_static_stable",
        "prompt_stability.frozen_capability_prefix_bounded",
        "prompt_stability.semi_dynamic_2_stable",
        "prompt_stability.timeline_open_dynamic_changed",
    ):
        _truthy(report, path, issues)
    _semidynamic_1_stable(report, "prompt_stability", issues)
    _min_int(report, "embedding.call_count", 1, issues)
    _min_int(report, "memory.record_count", 1, issues)
    _min_int(report, "timeline.item_count", 1, issues)
    _max_int(report, "prompt_stability.max_compact_delta_bytes", 2048, issues)
    _max_int(report, "prompt_stability.max_content_bytes", 20000, issues)
    return issues


def _verify_mock_soak(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if report.get("status") != "completed":
        issues.append(f"status is not completed: {report.get('status')!r}")
    for path in (
        "high_static_stable",
        "semi_dynamic_2_stable",
        "semi_dynamic_changed",
        "timeline_open_dynamic_changed",
        "alpha_return_contains_memory",
        "alpha_return_contains_alpha_fact",
        "beta_request_contains_beta_fact",
        "skill_context_visible",
        "mcp_context_visible",
        "memory_contains_alpha",
        "memory_contains_beta",
        "request_growth_bounded",
        "compact_delta_bounded",
    ):
        _truthy(report, path, issues)
    _min_int(report, "task_count", 12, issues)
    _min_int(report, "provider_request_count", 20, issues)
    _min_int(report, "tool_invocation_count", 11, issues)
    _min_int(report, "timeline_item_count", 50, issues)
    _min_int(report, "memory_record_count", 10, issues)
    _max_int(report, "max_request_bytes", 18000, issues)
    _max_int(report, "max_compact_delta_bytes", 1024, issues)
    _max_int(report, "tail_request_byte_range", 2500, issues)
    return issues


def _verify_live_soak(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if report.get("status") != "completed":
        issues.append(f"status is not completed: {report.get('status')!r}")
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    for path in (
        "aggregate.all_sessions_completed",
        "aggregate.cache_control_present",
        "aggregate.has_prompt_cache_usage",
        "aggregate.high_static_stable",
        "aggregate.frozen_capability_prefix_stable",
        "aggregate.semi_dynamic_2_stable",
        "aggregate.request_growth_bounded",
        "aggregate.compact_delta_bounded",
        "aggregate.failure_recovery_completed",
        "aggregate.alpha_recalled_after_switch",
        "aggregate.knowledge_visible",
        "aggregate.mcp_visible",
    ):
        _truthy(report, path, issues)
    _semidynamic_1_stable(report, "aggregate", issues)
    _min_int(report, "aggregate.session_count", 1, issues)
    _min_int(report, "aggregate.task_count", 20, issues)
    _min_int(report, "aggregate.provider_request_count", 20, issues)
    _min_int(report, "aggregate.perception_request_count", 1, issues)
    _min_int(report, "aggregate.embedding_call_count", 1, issues)
    _min_int(report, "aggregate.memory_hits_total", 1, issues)
    _min_int(report, "aggregate.knowledge_hits_total", 1, issues)
    _min_int(report, "aggregate.midterm_hits_total", 1, issues)
    _min_int(report, "aggregate.timeline_item_count", 1, issues)
    _min_int(report, "aggregate.memory_record_count", 1, issues)
    _min_int(report, "aggregate.failed_tool_result_count", 1, issues)
    _max_int(report, "aggregate.max_request_bytes", 18000, issues)
    _max_int(report, "aggregate.max_compact_delta_bytes", 2048, issues)
    sessions = report.get("sessions") or ()
    if len(sessions) != int(aggregate.get("session_count") or -1):
        issues.append("session_count does not match sessions length")
    for item in sessions:
        if not isinstance(item, dict):
            issues.append(f"invalid session report: {item!r}")
            continue
        if item.get("status") != "completed":
            issues.append(f"session did not complete: {item.get('session_index')}")
        if not item.get("task_reports"):
            issues.append(f"session has no task reports: {item.get('session_index')}")
    return issues


def verify_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    schema = report.get("schema_version")
    if schema == "raven-heart-live-yaklang-e2e/v1":
        issues = _verify_live(report)
    elif schema == "raven-heart-mock-yaklang-soak/v1":
        issues = _verify_mock_soak(report)
    elif schema == "raven-heart-live-yaklang-soak/v1":
        issues = _verify_live_soak(report)
    else:
        issues = [f"unsupported schema_version: {schema!r}"]
    return {
        "path": str(path),
        "schema_version": schema,
        "ok": not issues,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="Yaklang-style JSON reports to verify.")
    args = parser.parse_args()
    results = [verify_report(Path(item)) for item in args.reports]
    print(json.dumps({"schema_version": "raven-heart-yaklang-report-verifier/v1", "results": results}, indent=2))
    if not all(result["ok"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
