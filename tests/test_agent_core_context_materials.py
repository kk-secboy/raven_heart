from __future__ import annotations

import pytest

from agent_core.context import (
    ContextMaterial,
    ContextMaterialCenter,
    ContextMaterialQuery,
    ExternalContextMaterialStore,
    InMemoryContextMaterialStore,
)


class _FailingContextMaterialStore:
    async def search(self, query: ContextMaterialQuery) -> tuple[ContextMaterial, ...]:
        raise RuntimeError("context search failed")

    async def write(self, material: ContextMaterial) -> None:
        raise RuntimeError("context write failed")


@pytest.mark.asyncio
async def test_context_material_center_routes_and_manifests_external_backends() -> None:
    workspace = InMemoryContextMaterialStore(
        (
            ContextMaterial(
                name="auth_trace",
                content="auth callback csrf validation failed",
                role="timeline",
                priority=3,
                metadata={"tags": ("auth",), "namespace": "tenant-a"},
            ),
            ContextMaterial(
                name="dns_note",
                content="dns propagation completed",
                role="memory",
                priority=1,
                metadata={"tags": ("dns",), "namespace": "tenant-a"},
            ),
        )
    )
    pg_adapter = InMemoryContextMaterialStore(
        (
            ContextMaterial(
                name="risk_schema",
                content='{"required":["risk"]}',
                role="schema",
                priority=5,
                metadata={"tags": ("auth",), "namespace": "tenant-a"},
            ),
        )
    )
    external = ExternalContextMaterialStore(
        pg_adapter,
        name="tenant-pg",
        backend_kind="postgres",
        priority=20,
        namespace="tenant-a",
        supports_vector=True,
        tags=("auth", "durable"),
        location="postgres://context",
    )
    center = ContextMaterialCenter(default_store="workspace")
    center.register(
        "workspace",
        workspace,
        backend_kind="markdown",
        priority=5,
        namespace="tenant-a",
        tags=("auth", "local"),
    )
    center.register_spec(external.spec, external)

    await center.write(
        ContextMaterial(
            name="operator_hint",
            content="auth callback needs exploitability summary",
            role="runtime",
            priority=4,
            metadata={"store": "workspace", "tags": ("auth",), "namespace": "tenant-a"},
        )
    )
    results = await center.search(
        ContextMaterialQuery(
            query="auth risk callback",
            limit=3,
            namespace="tenant-a",
            filters={"tags": ("auth",)},
        )
    )
    manifest = center.manifest()

    assert [material.name for material in results] == [
        "risk_schema",
        "operator_hint",
        "auth_trace",
    ]
    assert results[0].metadata["store"] == "tenant-pg"
    assert manifest["stores"][0]["backend_kind"] == "postgres"
    assert manifest["stores"][0]["backend"]["kind"] == "postgres"
    assert "vector" in manifest["stores"][0]["backend"]["capabilities"]
    assert manifest["call_count"] == 3
    assert [call["operation"] for call in manifest["calls"]] == ["write", "search", "search"]
    assert manifest["calls"][1]["query"]["query_sha256"]
    assert "auth risk callback" not in str(manifest)
    assert external.manifest()["call_count"] == 1


@pytest.mark.asyncio
async def test_external_context_material_store_records_failed_runtime_calls() -> None:
    external = ExternalContextMaterialStore(
        _FailingContextMaterialStore(),
        name="tenant-pg",
        backend_kind="postgres",
    )

    with pytest.raises(RuntimeError):
        await external.search(ContextMaterialQuery(query="private target"))
    with pytest.raises(RuntimeError):
        await external.write(ContextMaterial(name="secret", content="sensitive context"))

    manifest = external.manifest()
    assert manifest["call_count"] == 2
    assert [call["status"] for call in manifest["calls"]] == ["failed", "failed"]
    assert manifest["calls"][0]["query"]["query_sha256"]
    assert "private target" not in str(manifest)
    assert manifest["calls"][1]["material"]["sha256"]
    assert "sensitive context" not in str(manifest)
