"""Provider adapter contract profile for the pure SDK boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentCoreProviderAPIFormat:
    """One provider API format shape the SDK can host through adapters."""

    name: str
    status: str
    adapter_boundary: str
    request_contracts: tuple[str, ...] = ()
    response_contracts: tuple[str, ...] = ()
    stream_contracts: tuple[str, ...] = ()
    supported_features: tuple[str, ...] = ()
    runtime_responsibilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-provider-api-format/v1",
            "name": self.name,
            "status": self.status,
            "adapter_boundary": self.adapter_boundary,
            "request_contracts": list(self.request_contracts),
            "response_contracts": list(self.response_contracts),
            "stream_contracts": list(self.stream_contracts),
            "supported_features": list(self.supported_features),
            "runtime_responsibilities": list(self.runtime_responsibilities),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreProviderContractProfile:
    """Provider-neutral profile describing supported adapter shapes."""

    formats: tuple[AgentCoreProviderAPIFormat, ...]
    required_core_contracts: tuple[str, ...]
    forbidden_core_dependencies: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-provider-contract-profile/v1",
            "format_count": len(self.formats),
            "formats": [format_.manifest() for format_ in self.formats],
            "format_names": [format_.name for format_ in self.formats],
            "required_core_contracts": list(self.required_core_contracts),
            "forbidden_core_dependencies": list(self.forbidden_core_dependencies),
            "metadata": dict(self.metadata),
        }


def agent_core_provider_contract_profile(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreProviderContractProfile:
    """Return provider API formats supported by the SDK boundary."""

    return AgentCoreProviderContractProfile(
        formats=(
            AgentCoreProviderAPIFormat(
                name="llm_provider_port",
                status="stable_contract",
                adapter_boundary="implement LLMProviderPort.complete and optionally stream",
                request_contracts=("LLMRequest", "LLMMessage", "LLMContentPart"),
                response_contracts=("LLMResponse", "LLMToolCall", "UsageInfo"),
                stream_contracts=("LLMStreamEvent", "LLMStreamAccumulator"),
                supported_features=(
                    "text",
                    "streaming",
                    "json_mode",
                    "json_schema",
                    "native_tool_calls",
                    "multimodal_content_parts",
                    "usage_accounting",
                    "retry_hints",
                ),
                runtime_responsibilities=(
                    "HTTP client",
                    "credentials",
                    "vendor request mapping",
                    "rate limits",
                    "tenant routing",
                ),
            ),
            AgentCoreProviderAPIFormat(
                name="agent_core_transport",
                status="stable_contract",
                adapter_boundary="wrap a dict transport with TransportLLMProvider and DefaultLLMProviderCodec",
                request_contracts=("agent-core-llm-transport-request/v1",),
                response_contracts=("DefaultLLMProviderCodec.decode_response",),
                stream_contracts=("DefaultLLMProviderCodec.decode_stream_event",),
                supported_features=(
                    "text",
                    "streaming",
                    "json_mode",
                    "json_schema",
                    "native_tool_calls",
                    "multimodal_content_parts",
                    "usage_accounting",
                    "retry_hints",
                ),
                runtime_responsibilities=(
                    "transport implementation",
                    "network retries outside SDK retry policy",
                    "secret loading",
                ),
            ),
            AgentCoreProviderAPIFormat(
                name="chat_completions_compatible",
                status="mvp_contract",
                adapter_boundary="use ChatCompletionsLLMProviderCodec behind a runtime-owned HTTP transport",
                request_contracts=("ChatCompletionsLLMProviderCodec.encode_request",),
                response_contracts=("ChatCompletionsLLMProviderCodec.decode_response",),
                stream_contracts=("ChatCompletionsLLMProviderCodec.decode_stream_event",),
                supported_features=(
                    "chat_messages",
                    "streaming_delta",
                    "tool_calls",
                    "json_schema_response_format",
                    "image_url_content_parts",
                    "usage_accounting",
                    "retryable_error_events",
                ),
                runtime_responsibilities=(
                    "base URL",
                    "API key",
                    "organization/project headers",
                    "deployment/model aliases",
                    "vendor-specific HTTP errors",
                ),
            ),
            AgentCoreProviderAPIFormat(
                name="custom_codec_transport",
                status="extension_contract",
                adapter_boundary="implement LLMProviderCodecPort and pass it to TransportLLMProvider",
                request_contracts=("LLMProviderCodecPort.encode_request",),
                response_contracts=("LLMProviderCodecPort.decode_response",),
                stream_contracts=("LLMProviderCodecPort.decode_stream_event",),
                supported_features=(
                    "vendor_specific_payloads",
                    "custom_stream_events",
                    "custom_usage_mapping",
                    "custom_error_mapping",
                ),
                runtime_responsibilities=(
                    "codec implementation",
                    "golden payload tests",
                    "vendor-specific feature negotiation",
                ),
            ),
        ),
        required_core_contracts=(
            "LLMProviderPort",
            "LLMRequest",
            "LLMResponse",
            "LLMStreamEvent",
            "LLMProviderCenter",
            "LLMProviderCodecPort",
            "TransportLLMProvider",
            "DefaultLLMProviderCodec",
            "ChatCompletionsLLMProviderCodec",
            "AgentCoreProviderConformanceHarness",
        ),
        forbidden_core_dependencies=(
            "openai",
            "agents",
            "anthropic",
            "google.genai",
            "fastapi",
            "requests",
            "httpx",
        ),
        metadata=dict(metadata or {}),
    )
