"""LLM provider abstractions used by the agent core."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Literal, Protocol
from uuid import uuid4


MessageRole = Literal["system", "user", "assistant", "tool"]
LLMContentPartKind = Literal["text", "image", "audio", "file", "binary", "json"]
LLMToolChoiceMode = Literal["auto", "none", "required", "tool"]
LLMResponseFormatKind = Literal["text", "json", "json_schema"]


@dataclass(frozen=True)
class LLMContentPart:
    """Provider-neutral message content part.

    Adapter packages map these parts to vendor-specific message formats while
    manifests keep raw payloads out of traces.
    """

    kind: LLMContentPartKind
    text: str = ""
    uri: str = ""
    mime_type: str = ""
    data: str = ""
    data_encoding: str = ""
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"text", "image", "audio", "file", "binary", "json"}:
            raise ValueError(f"unsupported content part kind: {self.kind}")
        if self.kind == "text" and not self.text:
            raise ValueError("text content part requires text")
        if self.kind in {"image", "audio", "file", "binary"} and not (self.uri or self.data):
            raise ValueError(f"{self.kind} content part requires uri or data")
        if self.kind == "json" and not self.text:
            raise ValueError("json content part requires text")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def text_part(
        cls,
        text: str,
        *,
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "LLMContentPart":
        return cls(kind="text", text=text, name=name, metadata=dict(metadata or {}))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-content-part/v1",
            "kind": self.kind,
            "text_bytes": len(self.text.encode("utf-8")),
            "has_uri": bool(self.uri),
            "mime_type": self.mime_type,
            "data_bytes": len(self.data.encode("utf-8")),
            "data_encoding": self.data_encoding,
            "name": self.name,
            "metadata": dict(self.metadata),
        }

    def transport_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-content-part/v1",
            "kind": self.kind,
            "text": self.text,
            "uri": self.uri,
            "mime_type": self.mime_type,
            "data": self.data,
            "data_encoding": self.data_encoding,
            "name": self.name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str
    name: str = ""
    content_parts: tuple[LLMContentPart, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_parts", tuple(self.content_parts))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def manifest(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "content_bytes": len(self.content.encode("utf-8")),
            "content_part_count": len(self.content_parts),
            "content_parts": [part.manifest() for part in self.content_parts],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class UsageInfo:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RetryHint:
    retryable: bool = False
    after_seconds: float | None = None
    reason: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "retryable": self.retryable,
            "after_seconds": self.after_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LLMRetryPolicy:
    max_retries: int = 0
    fallback_enabled: bool = True
    retry_on_stream_errors: bool = True
    max_retry_after_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_retries", max(0, int(self.max_retries)))
        if self.max_retry_after_seconds is not None:
            object.__setattr__(
                self,
                "max_retry_after_seconds",
                max(0.0, float(self.max_retry_after_seconds)),
            )

    def allows(self, hint: RetryHint, *, streamed: bool = False) -> bool:
        if not hint.retryable:
            return False
        if streamed and not self.retry_on_stream_errors:
            return False
        if (
            self.max_retry_after_seconds is not None
            and hint.after_seconds is not None
            and hint.after_seconds > self.max_retry_after_seconds
        ):
            return False
        return True

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-retry-policy/v1",
            "max_retries": self.max_retries,
            "fallback_enabled": self.fallback_enabled,
            "retry_on_stream_errors": self.retry_on_stream_errors,
            "max_retry_after_seconds": self.max_retry_after_seconds,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMUsageLimits:
    max_cost_usd: float | None = None
    max_call_attempts: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-usage-limits/v1",
            "max_cost_usd": self.max_cost_usd,
            "max_call_attempts": self.max_call_attempts,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMToolContract:
    """Provider-neutral function/tool contract for native model tool calling."""

    name: str
    description: str = ""
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    strict: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool contract name is required")
        object.__setattr__(self, "parameters_schema", dict(self.parameters_schema))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_tool_spec(
        cls,
        spec: Any,
        *,
        strict: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "LLMToolContract":
        """Build a provider-native tool contract from a ToolSpec-like object."""

        return cls(
            name=str(getattr(spec, "name", "") or ""),
            description=str(getattr(spec, "description", "") or ""),
            parameters_schema=dict(getattr(spec, "parameters_schema", {}) or {}),
            strict=strict,
            metadata={
                "source": "tool_spec",
                **dict(getattr(spec, "metadata", {}) or {}),
                **dict(metadata or {}),
            },
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-tool-contract/v1",
            "name": self.name,
            "description": self.description,
            "parameters_schema": dict(self.parameters_schema),
            "strict": self.strict,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMToolChoice:
    """Provider-neutral native tool selection policy."""

    mode: LLMToolChoiceMode = "auto"
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "none", "required", "tool"}:
            raise ValueError(f"unsupported tool choice mode: {self.mode}")
        if self.mode == "tool" and not self.tool_name:
            raise ValueError("tool choice mode 'tool' requires tool_name")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-tool-choice/v1",
            "mode": self.mode,
            "tool_name": self.tool_name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMResponseFormat:
    """Provider-neutral response format contract."""

    kind: LLMResponseFormatKind = "text"
    name: str = ""
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    strict: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"text", "json", "json_schema"}:
            raise ValueError(f"unsupported response format kind: {self.kind}")
        if self.kind == "json_schema" and not self.schema:
            raise ValueError("json_schema response format requires schema")
        object.__setattr__(self, "schema", dict(self.schema))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-response-format/v1",
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "schema": dict(self.schema),
            "strict": self.strict,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMToolCall:
    """Provider-neutral tool call requested by a model response."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("tool call name is required")
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def manifest(self) -> dict[str, Any]:
        raw_arguments = json.dumps(self.arguments, sort_keys=True, default=str)
        return {
            "schema_version": "agent-core-llm-tool-call/v1",
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "argument_keys": sorted(str(key) for key in self.arguments),
            "arguments_sha256": hashlib.sha256(raw_arguments.encode("utf-8")).hexdigest()
            if self.arguments
            else "",
            "metadata": dict(self.metadata),
        }

    def transport_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-tool-call/v1",
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "arguments": dict(self.arguments),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMRequest:
    messages: list[LLMMessage]
    model: str = ""
    temperature: float | None = None
    max_output_tokens: int | None = None
    tools: tuple[LLMToolContract, ...] = ()
    tool_choice: LLMToolChoice | None = None
    response_format: LLMResponseFormat | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.tool_choice is not None and self.tool_choice.mode != "none" and not self.tools:
            raise ValueError("tool_choice requires at least one tool contract")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-request/v1",
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "message_count": len(self.messages),
            "messages": [message.manifest() for message in self.messages],
            "tool_count": len(self.tools),
            "tools": [tool.manifest() for tool in self.tools],
            "tool_choice": self.tool_choice.manifest() if self.tool_choice else None,
            "response_format": self.response_format.manifest() if self.response_format else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMResponse:
    content: str = ""
    action: dict[str, Any] | None = None
    tool_calls: tuple[LLMToolCall, ...] = ()
    usage: UsageInfo = field(default_factory=UsageInfo)
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-response/v1",
            "content_bytes": len(self.content.encode("utf-8")),
            "has_action": self.action is not None,
            "tool_call_count": len(self.tool_calls),
            "tool_calls": [tool_call.manifest() for tool_call in self.tool_calls],
            "usage": self.usage.manifest(),
            "finish_reason": self.finish_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMStreamEvent:
    type: Literal[
        "message_start",
        "delta",
        "action",
        "tool_call",
        "usage",
        "error",
        "message_end",
    ]
    delta: str = ""
    action: dict[str, Any] | None = None
    tool_call: LLMToolCall | None = None
    usage: UsageInfo | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-stream-event/v1",
            "type": self.type,
            "delta_bytes": len(self.delta.encode("utf-8")),
            "has_action": self.action is not None,
            "has_tool_call": self.tool_call is not None,
            "tool_call": self.tool_call.manifest() if self.tool_call else None,
            "usage": self.usage.manifest() if self.usage else None,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass
class LLMStreamAccumulator:
    """Collect stream events into a response and stream audit summaries."""

    events: list[LLMStreamEvent] = field(default_factory=list)
    content_parts: list[str] = field(default_factory=list)
    action: dict[str, Any] | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)
    finish_reason: str = ""
    error: str = ""

    def add(self, event: LLMStreamEvent) -> None:
        self.events.append(event)
        if event.delta:
            self.content_parts.append(event.delta)
        if event.action is not None:
            self.action = dict(event.action)
        if event.tool_call is not None:
            self.tool_calls.append(event.tool_call)
        if event.usage is not None:
            self.usage = event.usage
        if event.type == "error":
            self.error = event.error
            self.finish_reason = "error"
        elif event.type == "message_end" and not self.finish_reason:
            self.finish_reason = "stop"

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    def response(self, *, metadata: dict[str, Any] | None = None) -> LLMResponse:
        return LLMResponse(
            content=self.content,
            action=dict(self.action) if self.action is not None else None,
            tool_calls=tuple(self.tool_calls),
            usage=self.usage,
            finish_reason=self.finish_reason,
            metadata={
                "streamed": True,
                "stream": self.summary_manifest(),
                **dict(metadata or {}),
            },
        )

    def summary_manifest(self) -> dict[str, Any]:
        event_types = [event.type for event in self.events]
        return {
            "schema_version": "agent-core-llm-stream-summary/v1",
            "event_count": len(self.events),
            "event_types": event_types,
            "delta_bytes": sum(len(event.delta.encode("utf-8")) for event in self.events),
            "content_bytes": len(self.content.encode("utf-8")),
            "has_action": self.action is not None,
            "tool_call_count": len(self.tool_calls),
            "tool_calls": [tool_call.manifest() for tool_call in self.tool_calls],
            "has_usage": any(event.usage is not None for event in self.events),
            "finish_reason": self.finish_reason,
            "error": self.error,
            "usage": self.usage.manifest(),
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-stream-accumulator/v1",
            "summary": self.summary_manifest(),
            "events": [event.manifest() for event in self.events],
        }


class LLMProviderPort(Protocol):
    """Provider-neutral interface for model calls."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one complete model response."""

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Yield provider stream events."""
        raise NotImplementedError


class LLMProviderCodecPort(Protocol):
    """Convert provider-neutral requests/responses to transport payloads."""

    def encode_request(self, request: LLMRequest) -> dict[str, Any]:
        """Return a transport-safe request payload."""

    def decode_response(self, payload: dict[str, Any]) -> LLMResponse:
        """Return a provider-neutral response."""

    def decode_stream_event(self, payload: dict[str, Any]) -> LLMStreamEvent:
        """Return a provider-neutral stream event."""


class LLMTransportPort(Protocol):
    """Transport boundary implemented by runtime/provider adapter packages."""

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return one response payload."""

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Yield stream event payloads."""
        raise NotImplementedError


class DefaultLLMProviderCodec:
    """Dependency-free codec for simple dict-based provider transports."""

    def encode_request(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-transport-request/v1",
            "model": request.model,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "name": message.name,
                    "content_parts": [
                        part.transport_payload() for part in message.content_parts
                    ],
                    "metadata": dict(message.metadata),
                }
                for message in request.messages
            ],
            "tools": [tool.manifest() for tool in request.tools],
            "tool_choice": request.tool_choice.manifest() if request.tool_choice else None,
            "response_format": request.response_format.manifest()
            if request.response_format
            else None,
            "metadata": dict(request.metadata),
        }

    def decode_response(self, payload: dict[str, Any]) -> LLMResponse:
        error = _provider_error_from_payload(payload)
        if error is not None:
            raise error
        usage = _usage_from_payload(payload.get("usage"))
        action = payload.get("action")
        tool_calls = _tool_calls_from_payload(payload.get("tool_calls"))
        return LLMResponse(
            content=str(payload.get("content") or ""),
            action=dict(action) if isinstance(action, dict) else None,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=str(payload.get("finish_reason") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    def decode_stream_event(self, payload: dict[str, Any]) -> LLMStreamEvent:
        usage_payload = payload.get("usage")
        action = payload.get("action")
        tool_call = _tool_call_from_payload(payload.get("tool_call"))
        return LLMStreamEvent(
            type=_stream_event_type(str(payload.get("type") or "delta")),
            delta=str(payload.get("delta") or ""),
            action=dict(action) if isinstance(action, dict) else None,
            tool_call=tool_call,
            usage=_usage_from_payload(usage_payload) if usage_payload is not None else None,
            error=str(payload.get("error") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "agent-core-default-llm-provider-codec/v1"}


class TransportLLMProvider(LLMProviderPort):
    """LLM provider backed by a runtime-supplied transport and codec."""

    def __init__(
        self,
        transport: LLMTransportPort,
        *,
        codec: LLMProviderCodecPort | None = None,
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.transport = transport
        self.codec = codec or DefaultLLMProviderCodec()
        self.name = name
        self.metadata = dict(metadata or {})

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self.codec.encode_request(request)
        try:
            raw = await self.transport.complete(payload)
            response = self.codec.decode_response(raw)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(str(exc), retry_hint=_retry_hint_from_exception(exc)) from exc
        return replace(
            response,
            metadata={
                **response.metadata,
                "transport_provider": self.name,
            },
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        payload = self.codec.encode_request(request)
        try:
            async for raw in self.transport.stream(payload):
                event = self.codec.decode_stream_event(raw)
                if event.type == "error":
                    raise LLMProviderError(
                        event.error or "provider stream error",
                        retry_hint=_retry_hint_from_stream_event(event),
                    )
                yield replace(
                    event,
                    metadata={
                        **event.metadata,
                        "transport_provider": self.name,
                    },
                )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(str(exc), retry_hint=_retry_hint_from_exception(exc)) from exc

    def manifest(self) -> dict[str, Any]:
        codec_manifest = getattr(self.codec, "manifest", None)
        transport_manifest = getattr(self.transport, "manifest", None)
        return {
            "schema_version": "agent-core-transport-llm-provider/v1",
            "name": self.name,
            "codec": codec_manifest() if callable(codec_manifest) else {},
            "transport": transport_manifest() if callable(transport_manifest) else {},
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMModelCapabilities:
    context_window_tokens: int = 0
    max_output_tokens: int = 0
    supports_streaming: bool = True
    supports_tool_calls: bool = False
    supports_structured_output: bool = False
    supports_json_mode: bool = False
    modalities: tuple[str, ...] = ("text",)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_window_tokens",
            max(0, int(self.context_window_tokens)),
        )
        object.__setattr__(
            self,
            "max_output_tokens",
            max(0, int(self.max_output_tokens)),
        )
        object.__setattr__(self, "modalities", tuple(str(item) for item in self.modalities))

    def supports_request(self, request: LLMRequest, *, streamed: bool = False) -> bool:
        if streamed and not self.supports_streaming:
            return False
        if _metadata_bool(request.metadata, "requires_streaming") and not self.supports_streaming:
            return False
        requested_modalities = _request_modalities(request)
        if requested_modalities and not requested_modalities <= set(self.modalities):
            return False
        if request.tools and not self.supports_tool_calls:
            return False
        if request.response_format is not None:
            if request.response_format.kind == "json" and not self.supports_json_mode:
                return False
            if (
                request.response_format.kind == "json_schema"
                and not self.supports_structured_output
            ):
                return False
        if _metadata_bool(request.metadata, "requires_tool_calls") and not self.supports_tool_calls:
            return False
        if (
            _metadata_bool(request.metadata, "requires_structured_output")
            and not self.supports_structured_output
        ):
            return False
        if _metadata_bool(request.metadata, "requires_json_mode") and not self.supports_json_mode:
            return False
        required = _metadata_strings(request.metadata, "required_capabilities")
        available = set(self.capability_names())
        if required and not set(required) <= available:
            return False
        requested_output = request.max_output_tokens or _metadata_int(
            request.metadata,
            "estimated_output_tokens",
        )
        if self.max_output_tokens and requested_output > self.max_output_tokens:
            return False
        if self.context_window_tokens:
            estimated_total = _metadata_int(request.metadata, "estimated_total_tokens")
            if estimated_total <= 0:
                estimated_total = _metadata_int(
                    request.metadata,
                    "estimated_input_tokens",
                ) + max(0, requested_output)
            if estimated_total > self.context_window_tokens:
                return False
        return True

    def capability_names(self) -> tuple[str, ...]:
        names = []
        if self.supports_streaming:
            names.append("streaming")
        if self.supports_tool_calls:
            names.append("tool_calls")
        if self.supports_structured_output:
            names.append("structured_output")
        if self.supports_json_mode:
            names.append("json_mode")
        names.extend(f"modality:{item}" for item in self.modalities)
        return tuple(dict.fromkeys(names))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-model-capabilities/v1",
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "supports_streaming": self.supports_streaming,
            "supports_tool_calls": self.supports_tool_calls,
            "supports_structured_output": self.supports_structured_output,
            "supports_json_mode": self.supports_json_mode,
            "modalities": list(self.modalities),
            "capabilities": list(self.capability_names()),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMProviderSpec:
    name: str
    models: tuple[str, ...] = ()
    default_model: str = ""
    priority: int = 0
    tags: tuple[str, ...] = ()
    default_capabilities: LLMModelCapabilities | None = None
    model_capabilities: dict[str, LLMModelCapabilities] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, model: str) -> bool:
        return not model or not self.models or model in self.models

    def capabilities_for(self, model: str) -> LLMModelCapabilities | None:
        if model and model in self.model_capabilities:
            return self.model_capabilities[model]
        if self.default_model and self.default_model in self.model_capabilities:
            return self.model_capabilities[self.default_model]
        return self.default_capabilities

    def supports_route(
        self,
        model: str,
        request: LLMRequest,
        *,
        streamed: bool = False,
    ) -> bool:
        if not self.supports(model):
            return False
        capabilities = self.capabilities_for(model)
        if capabilities is None:
            return True
        return capabilities.supports_request(request, streamed=streamed)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "models": list(self.models),
            "default_model": self.default_model,
            "priority": self.priority,
            "tags": list(self.tags),
            "default_capabilities": self.default_capabilities.manifest()
            if self.default_capabilities
            else None,
            "model_capabilities": {
                model: capabilities.manifest()
                for model, capabilities in sorted(self.model_capabilities.items())
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMProviderRoute:
    provider_name: str
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-provider-route/v1",
            "provider_name": self.provider_name,
            "model": self.model,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMProviderRouteCandidate:
    provider_name: str
    model: str = ""
    priority: int = 0
    selected: bool = False
    fallback_candidate: bool = False
    supported: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-provider-route-candidate/v1",
            "provider_name": self.provider_name,
            "model": self.model,
            "priority": self.priority,
            "selected": self.selected,
            "fallback_candidate": self.fallback_candidate,
            "supported": self.supported,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMProviderRoutePlan:
    requested_provider: str = ""
    requested_model: str = ""
    streamed: bool = False
    fallback_enabled: bool = True
    selected_route: LLMProviderRoute | None = None
    candidates: tuple[LLMProviderRouteCandidate, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.selected_route is not None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-provider-route-plan/v1",
            "requested_provider": self.requested_provider,
            "requested_model": self.requested_model,
            "streamed": self.streamed,
            "fallback_enabled": self.fallback_enabled,
            "ready": self.ready,
            "selected_route": self.selected_route.manifest() if self.selected_route else None,
            "candidates": [candidate.manifest() for candidate in self.candidates],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMCallRecord:
    provider_name: str
    model: str
    attempt: int
    status: Literal["completed", "failed"]
    streamed: bool = False
    usage: UsageInfo = field(default_factory=UsageInfo)
    error: str = ""
    retryable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-call-record/v1",
            "provider_name": self.provider_name,
            "model": self.model,
            "attempt": self.attempt,
            "status": self.status,
            "streamed": self.streamed,
            "usage": self.usage.manifest(),
            "error": self.error,
            "retryable": self.retryable,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _ProviderEntry:
    spec: LLMProviderSpec
    provider: LLMProviderPort


class LLMProviderNotFoundError(KeyError):
    pass


class LLMProviderError(RuntimeError):
    def __init__(self, message: str, *, retry_hint: RetryHint | None = None) -> None:
        super().__init__(message)
        self.retry_hint = retry_hint or RetryHint()


class LLMBudgetExceededError(RuntimeError):
    pass


class LLMProviderCenter(LLMProviderPort):
    """Registry and router for provider-neutral model calls."""

    def __init__(
        self,
        *,
        default_provider: str = "",
        default_model: str = "",
        max_retries: int = 0,
        max_cost_usd: float | None = None,
        fallback_enabled: bool = True,
        retry_policy: LLMRetryPolicy | None = None,
        usage_limits: LLMUsageLimits | None = None,
    ) -> None:
        self.default_provider = default_provider
        self.default_model = default_model
        self.retry_policy = retry_policy or LLMRetryPolicy(
            max_retries=max_retries,
            fallback_enabled=fallback_enabled,
        )
        self.usage_limits = usage_limits or LLMUsageLimits(max_cost_usd=max_cost_usd)
        self.max_retries = self.retry_policy.max_retries
        self.max_cost_usd = self.usage_limits.max_cost_usd
        self.fallback_enabled = self.retry_policy.fallback_enabled
        self.usage = UsageInfo()
        self.failures: list[dict[str, Any]] = []
        self.calls: list[LLMCallRecord] = []
        self._providers: dict[str, _ProviderEntry] = {}

    def register(
        self,
        name: str,
        provider: LLMProviderPort,
        *,
        models: tuple[str, ...] = (),
        default_model: str = "",
        priority: int = 0,
        tags: tuple[str, ...] = (),
        default_capabilities: LLMModelCapabilities | None = None,
        model_capabilities: dict[str, LLMModelCapabilities] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("provider name is required")
        self._providers[name] = _ProviderEntry(
            spec=LLMProviderSpec(
                name=name,
                models=tuple(models),
                default_model=default_model,
                priority=priority,
                tags=tuple(tags),
                default_capabilities=default_capabilities,
                model_capabilities=dict(model_capabilities or {}),
                metadata=dict(metadata or {}),
            ),
            provider=provider,
        )

    def get(self, name: str) -> LLMProviderPort:
        try:
            return self._providers[name].provider
        except KeyError as exc:
            raise LLMProviderNotFoundError(name) from exc

    def specs(self) -> tuple[LLMProviderSpec, ...]:
        return tuple(entry.spec for entry in self._ordered_entries())

    def search(self, query: str = "", *, tag: str = "", model: str = "") -> tuple[LLMProviderSpec, ...]:
        terms = tuple(part.casefold() for part in query.split() if part)
        matches: list[LLMProviderSpec] = []
        for entry in self._ordered_entries():
            spec = entry.spec
            if tag and tag not in spec.tags:
                continue
            if model and not spec.supports(model):
                continue
            haystack = " ".join((spec.name, *spec.models, *spec.tags)).casefold()
            if terms and not all(term in haystack for term in terms):
                continue
            matches.append(spec)
        return tuple(matches)

    def manifest(self) -> dict[str, Any]:
        return {
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "fallback_enabled": self.fallback_enabled,
            "max_retries": self.max_retries,
            "max_cost_usd": self.max_cost_usd,
            "retry_policy": self.retry_policy.manifest(),
            "usage_limits": self.usage_limits.manifest(),
            "usage": self.usage.manifest(),
            "call_count": len(self.calls),
            "failure_count": len(self.failures),
            "calls": [call.manifest() for call in self.calls],
            "failures": [dict(item) for item in self.failures],
            "providers": [spec.manifest() for spec in self.specs()],
        }

    def route_plan(
        self,
        request: LLMRequest,
        *,
        streamed: bool = False,
    ) -> LLMProviderRoutePlan:
        """Return an auditable provider route decision without calling a model."""

        requested_provider = str(request.metadata.get("provider") or "")
        requested_model = request.model
        selected_name = self._selected_provider_name(request, streamed=streamed)
        selected_route: LLMProviderRoute | None = None
        candidates: list[LLMProviderRouteCandidate] = []

        if requested_provider and requested_provider not in self._providers:
            candidates.append(
                LLMProviderRouteCandidate(
                    provider_name=requested_provider,
                    model=requested_model,
                    supported=False,
                    reason="provider_not_registered",
                )
            )

        for entry in self._ordered_entries():
            model = self._model_for_entry(request, entry)
            supports_model = entry.spec.supports(model)
            capabilities = entry.spec.capabilities_for(model)
            supports_capabilities = (
                capabilities.supports_request(request, streamed=streamed)
                if supports_model and capabilities is not None
                else supports_model
            )
            supported = supports_model and supports_capabilities
            selected = bool(selected_name and entry.spec.name == selected_name)
            fallback_candidate = (
                supported
                and not selected
                and not requested_provider
                and self.fallback_enabled
            )
            reason = self._route_candidate_reason(
                entry,
                requested_provider=requested_provider,
                supports_model=supports_model,
                supports_capabilities=supports_capabilities,
                selected=selected,
                fallback_candidate=fallback_candidate,
            )
            candidate = LLMProviderRouteCandidate(
                provider_name=entry.spec.name,
                model=model,
                priority=entry.spec.priority,
                selected=selected,
                fallback_candidate=fallback_candidate,
                supported=supported,
                reason=reason,
                metadata={
                    "model_capabilities": capabilities.manifest() if capabilities else None,
                    **entry.spec.metadata,
                },
            )
            candidates.append(candidate)
            if selected:
                selected_route = LLMProviderRoute(
                    provider_name=entry.spec.name,
                    model=model,
                    metadata={
                        "provider_priority": entry.spec.priority,
                        "model_capabilities": capabilities.manifest() if capabilities else None,
                        **entry.spec.metadata,
                    },
                )

        return LLMProviderRoutePlan(
            requested_provider=requested_provider,
            requested_model=requested_model,
            streamed=streamed,
            fallback_enabled=self.fallback_enabled,
            selected_route=selected_route,
            candidates=tuple(candidates),
            metadata={
                "default_provider": self.default_provider,
                "default_model": self.default_model,
            },
        )

    def select(self, request: LLMRequest) -> LLMProviderRoute:
        plan = self.route_plan(request)
        if plan.selected_route is None:
            raise LLMProviderNotFoundError(request.model or "<default>")
        return plan.selected_route

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._check_estimated_usage(request)
        last_error: Exception | None = None
        route_plan = self.route_plan(request)
        for entry in self._candidate_entries(request):
            routed = self._route_request(request, entry)
            attempts = self.max_retries + 1
            for attempt in range(attempts):
                self._check_call_attempt_limit()
                try:
                    response = await entry.provider.complete(routed)
                    self._record_usage(response.usage, provider=entry.spec.name, request=request)
                    self._record_call(
                        provider=entry.spec.name,
                        model=routed.model,
                        attempt=attempt + 1,
                        status="completed",
                        usage=response.usage,
                        metadata={
                            "route_plan": route_plan.manifest(),
                            "request": routed.manifest(),
                            "original_request": request.manifest(),
                            "response": response.manifest(),
                        },
                    )
                    return replace(
                        response,
                        metadata={
                            **response.metadata,
                            "provider": entry.spec.name,
                            "model": routed.model,
                            "attempt": attempt + 1,
                        },
                    )
                except LLMBudgetExceededError:
                    raise
                except Exception as exc:
                    last_error = exc
                    self._record_failure(
                        entry.spec.name,
                        routed.model,
                        attempt + 1,
                        exc,
                        metadata={
                            "route_plan": route_plan.manifest(),
                            "request": routed.manifest(),
                            "original_request": request.manifest(),
                        },
                    )
                    if not self._is_retryable(exc) or attempt + 1 >= attempts:
                        break
        if last_error is not None:
            raise last_error
        raise LLMProviderNotFoundError(request.model or "<default>")

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self._check_estimated_usage(request)
        last_error: Exception | None = None
        route_plan = self.route_plan(request, streamed=True)
        for entry in self._candidate_entries(request, streamed=True):
            routed = self._route_request(request, entry)
            attempts = self.max_retries + 1
            for attempt in range(attempts):
                self._check_call_attempt_limit()
                try:
                    events, usage, stream_summary = await self._collect_stream_attempt(
                        entry,
                        routed,
                        attempt=attempt + 1,
                    )
                    self._record_usage(usage, provider=entry.spec.name, request=request)
                    self._record_call(
                        provider=entry.spec.name,
                        model=routed.model,
                        attempt=attempt + 1,
                        status="completed",
                        streamed=True,
                        usage=usage,
                        metadata={
                            "route_plan": route_plan.manifest(),
                            "request": routed.manifest(),
                            "original_request": request.manifest(),
                            "stream_summary": stream_summary,
                        },
                    )
                    for event in events:
                        yield event
                    return
                except LLMBudgetExceededError:
                    raise
                except Exception as exc:
                    last_error = exc
                    self._record_failure(
                        entry.spec.name,
                        routed.model,
                        attempt + 1,
                        exc,
                        streamed=True,
                        metadata={
                            "route_plan": route_plan.manifest(),
                            "request": routed.manifest(),
                            "original_request": request.manifest(),
                        },
                    )
                    if not self._is_retryable(exc, streamed=True) or attempt + 1 >= attempts:
                        break
        if last_error is not None:
            raise last_error
        raise LLMProviderNotFoundError(request.model or "<default>")

    async def _collect_stream_attempt(
        self,
        entry: _ProviderEntry,
        request: LLMRequest,
        *,
        attempt: int,
    ) -> tuple[tuple[LLMStreamEvent, ...], UsageInfo, dict[str, Any]]:
        events: list[LLMStreamEvent] = []
        accumulator = LLMStreamAccumulator()
        async for event in entry.provider.stream(request):
            if event.type == "error":
                raise LLMProviderError(
                    event.error or "provider stream error",
                    retry_hint=_retry_hint_from_stream_event(event),
                )
            routed_event = LLMStreamEvent(
                type=event.type,
                delta=event.delta,
                action=event.action,
                tool_call=event.tool_call,
                usage=event.usage,
                error=event.error,
                metadata={
                    **event.metadata,
                    "provider": entry.spec.name,
                    "model": request.model,
                    "attempt": attempt,
                },
            )
            accumulator.add(routed_event)
            events.append(routed_event)
        return tuple(events), accumulator.usage, accumulator.summary_manifest()

    def _selected_provider_name(self, request: LLMRequest, *, streamed: bool = False) -> str:
        requested_provider = str(request.metadata.get("provider") or "")
        if requested_provider:
            entry = self._providers.get(requested_provider)
            if entry is None:
                return ""
            model = self._model_for_entry(request, entry)
            if entry.spec.supports_route(model, request, streamed=streamed):
                return entry.spec.name
            return ""

        if self.default_provider:
            entry = self._providers.get(self.default_provider)
            if entry and entry.spec.supports_route(
                self._model_for_entry(request, entry),
                request,
                streamed=streamed,
            ):
                return entry.spec.name

        for entry in self._ordered_entries():
            if entry.spec.supports_route(
                self._model_for_entry(request, entry),
                request,
                streamed=streamed,
            ):
                return entry.spec.name
        return ""

    @staticmethod
    def _route_candidate_reason(
        entry: _ProviderEntry,
        *,
        requested_provider: str,
        supports_model: bool,
        supports_capabilities: bool,
        selected: bool,
        fallback_candidate: bool,
    ) -> str:
        if requested_provider and entry.spec.name != requested_provider:
            return "provider_not_requested"
        if not supports_model:
            return "unsupported_model"
        if not supports_capabilities:
            return "unsupported_capabilities"
        if selected:
            return "selected"
        if fallback_candidate:
            return "fallback_candidate"
        return "eligible_not_selected"

    def _select_entry(self, request: LLMRequest, *, streamed: bool = False) -> _ProviderEntry:
        requested_provider = str(request.metadata.get("provider") or "")
        if requested_provider:
            entry = self._providers.get(requested_provider)
            if entry is None:
                raise LLMProviderNotFoundError(requested_provider)
            model = self._model_for_entry(request, entry)
            if not entry.spec.supports_route(model, request, streamed=streamed):
                raise LLMProviderNotFoundError(f"{requested_provider}:{request.model}")
            return entry

        if self.default_provider:
            entry = self._providers.get(self.default_provider)
            if entry and entry.spec.supports_route(
                self._model_for_entry(request, entry),
                request,
                streamed=streamed,
            ):
                return entry

        for entry in self._ordered_entries():
            if entry.spec.supports_route(
                self._model_for_entry(request, entry),
                request,
                streamed=streamed,
            ):
                return entry

        raise LLMProviderNotFoundError(request.model or "<default>")

    def _candidate_entries(
        self,
        request: LLMRequest,
        *,
        streamed: bool = False,
    ) -> tuple[_ProviderEntry, ...]:
        requested_provider = str(request.metadata.get("provider") or "")
        if requested_provider:
            return (self._select_entry(request, streamed=streamed),)

        selected = self._select_entry(request, streamed=streamed)
        if not self.fallback_enabled:
            return (selected,)
        entries = [selected]
        for entry in self._ordered_entries():
            if entry.spec.name == selected.spec.name:
                continue
            if entry.spec.supports_route(
                self._model_for_entry(request, entry),
                request,
                streamed=streamed,
            ):
                entries.append(entry)
        return tuple(entries)

    def _ordered_entries(self) -> tuple[_ProviderEntry, ...]:
        return tuple(
            sorted(
                self._providers.values(),
                key=lambda entry: (-entry.spec.priority, entry.spec.name),
            )
        )

    def _route_request(self, request: LLMRequest, entry: _ProviderEntry) -> LLMRequest:
        model = self._model_for_entry(request, entry)
        capabilities = entry.spec.capabilities_for(model)
        metadata = {
            **request.metadata,
            "provider": entry.spec.name,
            "provider_priority": entry.spec.priority,
            "model_capabilities": capabilities.manifest() if capabilities else None,
        }
        return replace(request, model=model, metadata=metadata)

    def _model_for_entry(self, request: LLMRequest, entry: _ProviderEntry) -> str:
        return request.model or entry.spec.default_model or self.default_model

    def _check_estimated_usage(self, request: LLMRequest) -> None:
        self._check_call_attempt_limit()
        max_cost = self._max_cost(request)
        estimated = float(request.metadata.get("estimated_cost_usd") or 0.0)
        if max_cost is not None and self.usage.cost_usd + estimated > max_cost:
            raise LLMBudgetExceededError(
                f"llm budget exceeded: spent={self.usage.cost_usd:.6f}, "
                f"estimated={estimated:.6f}, limit={max_cost:.6f}"
            )
        self._check_token_limit(
            "input",
            current=self.usage.input_tokens,
            incoming=_metadata_int(request.metadata, "estimated_input_tokens"),
            limit=self._limit_value(request, "max_input_tokens", self.usage_limits.max_input_tokens),
        )
        self._check_token_limit(
            "output",
            current=self.usage.output_tokens,
            incoming=_metadata_int(request.metadata, "estimated_output_tokens"),
            limit=self._limit_value(request, "max_output_tokens", self.usage_limits.max_output_tokens),
        )
        self._check_token_limit(
            "total",
            current=self.usage.total_tokens,
            incoming=_metadata_int(request.metadata, "estimated_total_tokens"),
            limit=self._limit_value(request, "max_total_tokens", self.usage_limits.max_total_tokens),
        )

    def _record_usage(self, usage: UsageInfo, *, provider: str, request: LLMRequest) -> None:
        if usage.cost_usd:
            max_cost = self._max_cost(request)
            if max_cost is not None and self.usage.cost_usd + usage.cost_usd > max_cost:
                raise LLMBudgetExceededError(
                    f"llm budget exceeded: spent={self.usage.cost_usd:.6f}, "
                    f"actual={usage.cost_usd:.6f}, limit={max_cost:.6f}"
                )
        self._check_token_limit(
            "input",
            current=self.usage.input_tokens,
            incoming=usage.input_tokens,
            limit=self._limit_value(request, "max_input_tokens", self.usage_limits.max_input_tokens),
        )
        self._check_token_limit(
            "output",
            current=self.usage.output_tokens,
            incoming=usage.output_tokens,
            limit=self._limit_value(request, "max_output_tokens", self.usage_limits.max_output_tokens),
        )
        self._check_token_limit(
            "total",
            current=self.usage.total_tokens,
            incoming=usage.total_tokens,
            limit=self._limit_value(request, "max_total_tokens", self.usage_limits.max_total_tokens),
        )
        self.usage = UsageInfo(
            input_tokens=self.usage.input_tokens + usage.input_tokens,
            output_tokens=self.usage.output_tokens + usage.output_tokens,
            total_tokens=self.usage.total_tokens + usage.total_tokens,
            cost_usd=self.usage.cost_usd + usage.cost_usd,
            metadata={"last_provider": provider},
        )

    def _max_cost(self, request: LLMRequest) -> float | None:
        value = self._limit_value(request, "max_cost_usd", self.usage_limits.max_cost_usd)
        return float(value) if value is not None else None

    def _check_call_attempt_limit(self) -> None:
        limit = self.usage_limits.max_call_attempts
        if limit is not None and len(self.calls) + 1 > int(limit):
            raise LLMBudgetExceededError(
                f"llm call attempt budget exceeded: attempts={len(self.calls)}, limit={int(limit)}"
            )

    @staticmethod
    def _limit_value(request: LLMRequest, key: str, default: int | float | None) -> Any:
        value = request.metadata.get(key)
        if value is not None:
            return value
        return default

    @staticmethod
    def _check_token_limit(
        kind: str,
        *,
        current: int,
        incoming: int,
        limit: int | None,
    ) -> None:
        if limit is None or incoming <= 0:
            return
        if current + incoming > int(limit):
            raise LLMBudgetExceededError(
                f"llm {kind} token budget exceeded: spent={current}, "
                f"incoming={incoming}, limit={int(limit)}"
            )

    def _record_call(
        self,
        *,
        provider: str,
        model: str,
        attempt: int,
        status: Literal["completed", "failed"],
        streamed: bool = False,
        usage: UsageInfo | None = None,
        error: str = "",
        retryable: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.calls.append(
            LLMCallRecord(
                provider_name=provider,
                model=model,
                attempt=attempt,
                status=status,
                streamed=streamed,
                usage=usage or UsageInfo(),
                error=error,
                retryable=retryable,
                metadata=dict(metadata or {}),
            )
        )

    def _record_failure(
        self,
        provider: str,
        model: str,
        attempt: int,
        exc: Exception,
        *,
        streamed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        retryable = self._is_retryable(exc, streamed=streamed)
        self.failures.append(
            {
                "provider": provider,
                "model": model,
                "attempt": attempt,
                "error": str(exc),
                "retryable": retryable,
                "streamed": streamed,
            }
        )
        self._record_call(
            provider=provider,
            model=model,
            attempt=attempt,
            status="failed",
            streamed=streamed,
            error=str(exc),
            retryable=retryable,
            metadata=metadata,
        )

    def _is_retryable(self, exc: Exception, *, streamed: bool = False) -> bool:
        if isinstance(exc, LLMProviderError):
            return self.retry_policy.allows(exc.retry_hint, streamed=streamed)
        return False


def _retry_hint_from_stream_event(event: LLMStreamEvent) -> RetryHint:
    return RetryHint(
        retryable=bool(event.metadata.get("retryable", False)),
        after_seconds=_optional_float(event.metadata.get("after_seconds")),
        reason=str(event.metadata.get("reason") or event.error or ""),
    )


def _retry_hint_from_exception(exc: Exception) -> RetryHint:
    retry_hint = getattr(exc, "retry_hint", None)
    if isinstance(retry_hint, RetryHint):
        return retry_hint
    retryable = bool(getattr(exc, "retryable", False))
    after_seconds = _optional_float(getattr(exc, "after_seconds", None))
    return RetryHint(retryable=retryable, after_seconds=after_seconds, reason=str(exc))


def _provider_error_from_payload(payload: dict[str, Any]) -> LLMProviderError | None:
    error = payload.get("error")
    if not error:
        return None
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("error") or "provider error")
        retryable = bool(error.get("retryable", False))
        after_seconds = _optional_float(error.get("after_seconds"))
        reason = str(error.get("reason") or message)
    else:
        message = str(error)
        retryable = bool(payload.get("retryable", False))
        after_seconds = _optional_float(payload.get("after_seconds"))
        reason = str(payload.get("reason") or message)
    return LLMProviderError(
        message,
        retry_hint=RetryHint(
            retryable=retryable,
            after_seconds=after_seconds,
            reason=reason,
        ),
    )


def _usage_from_payload(payload: Any) -> UsageInfo:
    if not isinstance(payload, dict):
        return UsageInfo()
    return UsageInfo(
        input_tokens=_metadata_int(payload, "input_tokens"),
        output_tokens=_metadata_int(payload, "output_tokens"),
        total_tokens=_metadata_int(payload, "total_tokens"),
        cost_usd=float(payload.get("cost_usd") or 0.0),
        metadata=dict(payload.get("metadata") or {}),
    )


def _tool_calls_from_payload(payload: Any) -> tuple[LLMToolCall, ...]:
    if payload is None:
        return ()
    if isinstance(payload, dict):
        call = _tool_call_from_payload(payload)
        return (call,) if call is not None else ()
    if isinstance(payload, (list, tuple)):
        calls = []
        for item in payload:
            call = _tool_call_from_payload(item)
            if call is not None:
                calls.append(call)
        return tuple(calls)
    return ()


def _tool_call_from_payload(payload: Any) -> LLMToolCall | None:
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("tool_name") or payload.get("name") or "").strip()
    if not name:
        return None
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("args")
    if not isinstance(arguments, dict):
        arguments = {}
    call_id = str(payload.get("call_id") or payload.get("id") or "").strip() or uuid4().hex
    return LLMToolCall(
        tool_name=name,
        arguments=dict(arguments),
        call_id=call_id,
        metadata=dict(payload.get("metadata") or {}),
    )


def _request_modalities(request: LLMRequest) -> set[str]:
    modalities = set(_metadata_strings(request.metadata, "required_modalities"))
    for message in request.messages:
        if message.content:
            modalities.add("text")
        for part in message.content_parts:
            modalities.add(_content_part_modality(part.kind))
    return {item for item in modalities if item}


def _content_part_modality(kind: str) -> str:
    if kind in {"text", "json"}:
        return "text"
    return kind


def _stream_event_type(
    value: str,
) -> Literal["message_start", "delta", "action", "tool_call", "usage", "error", "message_end"]:
    if value not in {
        "message_start",
        "delta",
        "action",
        "tool_call",
        "usage",
        "error",
        "message_end",
    }:
        raise LLMProviderError(f"invalid stream event type: {value}")
    return value  # type: ignore[return-value]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _metadata_bool(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def _metadata_strings(metadata: dict[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)

