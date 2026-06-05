from __future__ import annotations


def test_agent_core_provider_contract_profile_declares_supported_adapter_formats() -> None:
    import agent_core

    profile = agent_core.agent_core_provider_contract_profile(
        metadata={"test": "provider_contract_profile"}
    )
    manifest = profile.manifest()
    formats = {item["name"]: item for item in manifest["formats"]}

    assert manifest["schema_version"] == "agent-core-provider-contract-profile/v1"
    assert manifest["format_names"] == [
        "llm_provider_port",
        "agent_core_transport",
        "chat_completions_compatible",
        "custom_codec_transport",
    ]
    assert manifest["metadata"] == {"test": "provider_contract_profile"}
    assert "LLMProviderPort" in manifest["required_core_contracts"]
    assert "TransportLLMProvider" in manifest["required_core_contracts"]
    assert "ChatCompletionsLLMProviderCodec" in manifest["required_core_contracts"]
    assert "openai" in manifest["forbidden_core_dependencies"]
    assert "httpx" in manifest["forbidden_core_dependencies"]
    assert formats["llm_provider_port"]["status"] == "stable_contract"
    assert "native_tool_calls" in formats["llm_provider_port"]["supported_features"]
    assert "multimodal_content_parts" in formats["agent_core_transport"][
        "supported_features"
    ]
    assert "tool_calls" in formats["chat_completions_compatible"][
        "supported_features"
    ]
    assert "codec implementation" in formats["custom_codec_transport"][
        "runtime_responsibilities"
    ]


def test_provider_contract_profile_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()
    sdk_manifest = agent_core.agent_core_sdk_manifest().manifest()
    capability_names = {capability["name"] for capability in sdk_manifest["capabilities"]}

    assert "provider_contract_profile" in capability_names
    assert "provider_contract_profile" in readiness["matched"]["capabilities"]
    assert "AgentCoreProviderContractProfile" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderAPIFormat" in readiness["matched"]["public_api"]
    assert "agent_core_provider_contract_profile" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderContractProfile" in stability["present_stable_api"]
    assert "AgentCoreProviderAPIFormat" in stability["present_stable_api"]
    assert "agent_core_provider_contract_profile" in stability["present_stable_api"]
