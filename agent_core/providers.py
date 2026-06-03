"""LLM provider abstractions used by the agent core."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Literal, Protocol


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "content_bytes": len(self.content.encode("utf-8")),
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
class LLMRequest:
    messages: list[LLMMessage]
    model: str = ""
    temperature: float | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-request/v1",
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "message_count": len(self.messages),
            "messages": [message.manifest() for message in self.messages],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMResponse:
    content: str = ""
    action: dict[str, Any] | None = None
    usage: UsageInfo = field(default_factory=UsageInfo)
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-response/v1",
            "content_bytes": len(self.content.encode("utf-8")),
            "has_action": self.action is not None,
            "usage": self.usage.manifest(),
            "finish_reason": self.finish_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMStreamEvent:
    type: Literal["message_start", "delta", "action", "usage", "error", "message_end"]
    delta: str = ""
    action: dict[str, Any] | None = None
    usage: UsageInfo | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-llm-stream-event/v1",
            "type": self.type,
            "delta_bytes": len(self.delta.encode("utf-8")),
            "has_action": self.action is not None,
            "usage": self.usage.manifest() if self.usage else None,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class LLMProviderPort(Protocol):
    """Provider-neutral interface for model calls."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one complete model response."""

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Yield provider stream events."""
        raise NotImplementedError


@dataclass(frozen=True)
class LLMProviderSpec:
    name: str
    models: tuple[str, ...] = ()
    default_model: str = ""
    priority: int = 0
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, model: str) -> bool:
        return not model or not self.models or model in self.models

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "models": list(self.models),
            "default_model": self.default_model,
            "priority": self.priority,
            "tags": list(self.tags),
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
    ) -> None:
        self.default_provider = default_provider
        self.default_model = default_model
        self.max_retries = max(0, max_retries)
        self.max_cost_usd = max_cost_usd
        self.fallback_enabled = fallback_enabled
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
            "usage": self.usage.manifest(),
            "call_count": len(self.calls),
            "failure_count": len(self.failures),
            "calls": [call.manifest() for call in self.calls],
            "failures": [dict(item) for item in self.failures],
            "providers": [spec.manifest() for spec in self.specs()],
        }

    def select(self, request: LLMRequest) -> LLMProviderRoute:
        entry = self._select_entry(request)
        model = request.model or entry.spec.default_model or self.default_model
        return LLMProviderRoute(
            provider_name=entry.spec.name,
            model=model,
            metadata={"provider_priority": entry.spec.priority, **entry.spec.metadata},
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._check_estimated_budget(request)
        last_error: Exception | None = None
        for entry in self._candidate_entries(request):
            routed = self._route_request(request, entry)
            attempts = self.max_retries + 1
            for attempt in range(attempts):
                try:
                    response = await entry.provider.complete(routed)
                    self._record_usage(response.usage, provider=entry.spec.name, request=request)
                    self._record_call(
                        provider=entry.spec.name,
                        model=routed.model,
                        attempt=attempt + 1,
                        status="completed",
                        usage=response.usage,
                        metadata={"request": request.manifest()},
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
                        metadata={"request": request.manifest()},
                    )
                    if not self._is_retryable(exc) or attempt + 1 >= attempts:
                        break
        if last_error is not None:
            raise last_error
        raise LLMProviderNotFoundError(request.model or "<default>")

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self._check_estimated_budget(request)
        last_error: Exception | None = None
        for entry in self._candidate_entries(request):
            routed = self._route_request(request, entry)
            attempts = self.max_retries + 1
            for attempt in range(attempts):
                usage = UsageInfo()
                try:
                    async for event in entry.provider.stream(routed):
                        if event.usage is not None:
                            usage = event.usage
                        yield LLMStreamEvent(
                            type=event.type,
                            delta=event.delta,
                            action=event.action,
                            usage=event.usage,
                            error=event.error,
                            metadata={
                                **event.metadata,
                                "provider": entry.spec.name,
                                "model": routed.model,
                                "attempt": attempt + 1,
                            },
                        )
                    self._record_usage(usage, provider=entry.spec.name, request=request)
                    self._record_call(
                        provider=entry.spec.name,
                        model=routed.model,
                        attempt=attempt + 1,
                        status="completed",
                        streamed=True,
                        usage=usage,
                        metadata={"request": request.manifest()},
                    )
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
                        metadata={"request": request.manifest()},
                    )
                    if not self._is_retryable(exc) or attempt + 1 >= attempts:
                        break
        if last_error is not None:
            raise last_error
        raise LLMProviderNotFoundError(request.model or "<default>")

    def _select_entry(self, request: LLMRequest) -> _ProviderEntry:
        requested_provider = str(request.metadata.get("provider") or "")
        if requested_provider:
            entry = self._providers.get(requested_provider)
            if entry is None:
                raise LLMProviderNotFoundError(requested_provider)
            if not entry.spec.supports(request.model):
                raise LLMProviderNotFoundError(f"{requested_provider}:{request.model}")
            return entry

        if self.default_provider:
            entry = self._providers.get(self.default_provider)
            if entry and entry.spec.supports(request.model):
                return entry

        for entry in self._ordered_entries():
            if entry.spec.supports(request.model):
                return entry

        raise LLMProviderNotFoundError(request.model or "<default>")

    def _candidate_entries(self, request: LLMRequest) -> tuple[_ProviderEntry, ...]:
        requested_provider = str(request.metadata.get("provider") or "")
        if requested_provider:
            return (self._select_entry(request),)

        selected = self._select_entry(request)
        if not self.fallback_enabled:
            return (selected,)
        entries = [selected]
        for entry in self._ordered_entries():
            if entry.spec.name == selected.spec.name:
                continue
            if entry.spec.supports(request.model):
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
        model = request.model or entry.spec.default_model or self.default_model
        metadata = {
            **request.metadata,
            "provider": entry.spec.name,
            "provider_priority": entry.spec.priority,
        }
        return replace(request, model=model, metadata=metadata)

    def _check_estimated_budget(self, request: LLMRequest) -> None:
        max_cost = self._max_cost(request)
        if max_cost is None:
            return
        estimated = float(request.metadata.get("estimated_cost_usd") or 0.0)
        if self.usage.cost_usd + estimated > max_cost:
            raise LLMBudgetExceededError(
                f"llm budget exceeded: spent={self.usage.cost_usd:.6f}, "
                f"estimated={estimated:.6f}, limit={max_cost:.6f}"
            )

    def _record_usage(self, usage: UsageInfo, *, provider: str, request: LLMRequest) -> None:
        if usage.cost_usd:
            max_cost = self._max_cost(request)
            if max_cost is not None and self.usage.cost_usd + usage.cost_usd > max_cost:
                raise LLMBudgetExceededError(
                    f"llm budget exceeded: spent={self.usage.cost_usd:.6f}, "
                    f"actual={usage.cost_usd:.6f}, limit={max_cost:.6f}"
                )
        self.usage = UsageInfo(
            input_tokens=self.usage.input_tokens + usage.input_tokens,
            output_tokens=self.usage.output_tokens + usage.output_tokens,
            total_tokens=self.usage.total_tokens + usage.total_tokens,
            cost_usd=self.usage.cost_usd + usage.cost_usd,
            metadata={"last_provider": provider},
        )

    def _max_cost(self, request: LLMRequest) -> float | None:
        value = request.metadata.get("max_cost_usd")
        if value is not None:
            return float(value)
        return self.max_cost_usd

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
        retryable = self._is_retryable(exc)
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

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, LLMProviderError):
            return exc.retry_hint.retryable
        return False

