from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_resume_acceptance_records_checkpoint_resume_trace() -> None:
    import agent_core

    report = await agent_core.run_agent_core_resume_acceptance(metadata={"test": "resume"})
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-resume-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["resume_index"]["candidate_count"] >= 1
    assert manifest["resume_plan"]["status"] == "ready"
    assert manifest["resume_plan"]["ready"] is True
    assert (
        manifest["resume_manifest"]["checkpoint_id"]
        == manifest["run_summary"]["checkpoint_id"]
    )
    assert manifest["resume_manifest"]["state"]["observation"] == "admin UI found"
    assert manifest["run_summary"]["status"] == "completed"
    assert manifest["run_summary"]["output"] == "resumed-ok"
    assert manifest["run_summary"]["provider_request_count"] == 1
    assert manifest["run_summary"]["resume_prompt_injected"] is True
    assert manifest["trace_eval"]["ok"] is True
    assert manifest["trace_summary"]["resume_plan_ready"] is True
    assert manifest["terminal_rejection"]["ready"] is False
    assert "terminal_checkpoint_not_allowed" in manifest["terminal_rejection"]["issue_codes"]


def test_resume_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "resume_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreResumeAcceptanceHarness" in agent_core.__all__
    assert "run_agent_core_resume_acceptance" in agent_core.__all__
    assert "AgentCoreResumeAcceptanceHarness" not in readiness["matched"]["public_api"]
    assert "AgentCoreResumeAcceptanceHarness" not in stability["present_stable_api"]
