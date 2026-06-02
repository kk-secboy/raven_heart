"""Provider-neutral skill registry and prompt rendering."""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.prompt import estimate_tokens
from agent_core.search import SearchDocument, rank_documents


AVAILABLE_SKILLS_HEADER = "== Available Skills (use load_skill action to load) =="
LOADED_SKILLS_HEADER = "== Currently Loaded Skills =="
DEFAULT_SKILL_REGISTRY_TOKEN_BUDGET = 1200
VIEW_WINDOW_MAX_BYTES = 32 * 1024


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str = ""
    prompt: str = ""
    tags: tuple[str, ...] = ()
    priority: int = 0
    license: str = ""
    compatibility: str = ""
    disable_model_invocation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, task: str, tags: tuple[str, ...] = ()) -> bool:
        haystack = task.lower()
        own_tags = {item.lower() for item in self.tags}
        if tags and any(tag.lower() in own_tags for tag in tags):
            return True
        return any(tag.lower() in haystack for tag in self.tags)

    def brief(self) -> str:
        line = f"  - {self.name}: {self.description}".rstrip()
        if self.tags:
            line += f" tags={','.join(self.tags)}"
        return line

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "priority": self.priority,
            "license": self.license,
            "compatibility": self.compatibility,
            "disable_model_invocation": self.disable_model_invocation,
            "metadata": dict(self.metadata),
        }


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ValueError("skill name is required")
        if name in self._skills:
            raise ValueError(f"skill already registered: {name}")
        self._skills[name] = spec

    def get(self, name: str) -> SkillSpec | None:
        return self._skills.get(name)

    def list(self) -> tuple[SkillSpec, ...]:
        return tuple(sorted(self._skills.values(), key=lambda item: (-item.priority, item.name)))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-skill-registry/v1",
            "skills": [skill.manifest() for skill in self.list()],
        }

    def select(
        self,
        task: str,
        *,
        tags: tuple[str, ...] = (),
        limit: int = 5,
    ) -> tuple[SkillSpec, ...]:
        matched = [
            skill
            for skill in self.list()
            if skill.matches(task, tags)
            and (tags or not skill.disable_model_invocation)
        ]
        if not matched and not tags:
            matched = [
                skill
                for skill in self.list()
                if skill.priority > 0 and not skill.disable_model_invocation
            ]
        return tuple(matched[:limit])

    def render_prompt(self, skills: tuple[SkillSpec, ...]) -> str:
        parts = []
        for skill in skills:
            body = skill.prompt.strip() or skill.description.strip()
            if body:
                parts.append(f"[skill:{skill.name}]\n{transform_includes_to_resource_hints(body, skill.name)}")
        return "\n\n".join(parts)

    def load_markdown_file(self, path: str | Path) -> SkillSpec:
        source = Path(path).resolve()
        spec = skill_from_markdown(source.read_text(encoding="utf-8"))
        spec = _with_metadata(
            spec,
            {
                "source_path": str(source),
                "root_path": str(source.parent),
                "content_hash": compute_skill_hash(source.parent),
            },
        )
        self.register(spec)
        return spec

    def discover_markdown_skills(self, root: str | Path) -> tuple[SkillSpec, ...]:
        root_path = Path(root).resolve()
        loaded: list[SkillSpec] = []
        for skill_file in sorted(root_path.rglob("SKILL.md")):
            try:
                loaded.append(self.load_markdown_file(skill_file))
            except (OSError, ValueError):
                continue
        return tuple(loaded)

    def discover_archive_skills(
        self,
        archive_path: str | Path,
        *,
        extract_root: str | Path,
    ) -> tuple[SkillSpec, ...]:
        archive = Path(archive_path).resolve()
        destination = (Path(extract_root).resolve() / archive.stem).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        if archive.suffix.lower() != ".zip":
            raise ValueError("only .zip skill archives are supported")
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                member_path = _safe_archive_member_path(destination, member.filename)
                if member.is_dir():
                    member_path.mkdir(parents=True, exist_ok=True)
                    continue
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source:
                    member_path.write_bytes(source.read())
        loaded = self.discover_markdown_skills(destination)
        archived: list[SkillSpec] = []
        for skill in loaded:
            enriched = _with_metadata(
                skill,
                {
                    "archive_path": str(archive),
                    "archive_extract_root": str(destination),
                },
            )
            self._skills[skill.name] = enriched
            archived.append(enriched)
        return tuple(archived)

    def load_resource(self, ref: str, *, offset: int = 1, max_bytes: int = VIEW_WINDOW_MAX_BYTES) -> SkillViewWindow:
        skill_name, rel_path = parse_skill_resource_ref(ref)
        skill = self.get(skill_name)
        if skill is None:
            raise KeyError(skill_name)
        root = Path(str(skill.metadata.get("root_path") or "")).resolve()
        if not root:
            raise ValueError(f"skill has no root_path metadata: {skill_name}")
        target = (root / rel_path).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError("skill resource path escapes skill root")
        content = target.read_text(encoding="utf-8")
        window = SkillViewWindow(
            skill_name=skill_name,
            file_path=rel_path,
            content=content,
            offset=offset,
            max_bytes=max_bytes,
        )
        window.set_offset(offset)
        return window

    def search(self, query: str, *, limit: int = 8) -> tuple[SkillSpec, ...]:
        documents = tuple(
            SearchDocument(
                item=skill,
                text=" ".join(
                    (
                        skill.name,
                        skill.description,
                        " ".join(skill.tags),
                        skill.license,
                        skill.compatibility,
                    )
                ),
                priority=skill.priority,
                name=skill.name,
            )
            for skill in self.list()
        )
        return rank_documents(query, documents, limit=limit)

    def render_available(self, *, max_tokens: int = DEFAULT_SKILL_REGISTRY_TOKEN_BUDGET) -> str:
        listed, omitted = select_skills_by_token_budget(
            self.list(),
            max_tokens=max_tokens,
            section_header=AVAILABLE_SKILLS_HEADER + "\n",
        )
        lines = [AVAILABLE_SKILLS_HEADER]
        lines.extend(skill.brief() for skill in listed)
        if omitted:
            lines.append(f"  ... and {omitted} more skills. Use search_skills to find specific skills.")
        return "\n".join(lines)


class SkillsContext:
    """Tracks loaded skills and renders a stable prompt context."""

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry
        self._loaded: dict[str, SkillSpec] = {}
        self._views: dict[str, SkillViewWindow] = {}

    def load(self, name: str) -> SkillSpec:
        skill = self.registry.get(name)
        if skill is None:
            raise KeyError(name)
        self._loaded[skill.name] = skill
        return skill

    def unload(self, name: str) -> bool:
        return self._loaded.pop(name, None) is not None

    def loaded(self) -> tuple[SkillSpec, ...]:
        return tuple(sorted(self._loaded.values(), key=lambda item: (-item.priority, item.name)))

    def views(self) -> tuple["SkillViewWindow", ...]:
        return tuple(sorted(self._views.values(), key=lambda item: item.view_id))

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-skills-context/v1",
            "loaded_skills": [skill.manifest() for skill in self.loaded()],
            "available_skills_count": len(self.registry.list()),
            "views": [view.manifest() for view in self.views()],
        }

    def search(self, query: str, *, limit: int = 8) -> tuple[SkillSpec, ...]:
        return self.registry.search(query, limit=limit)

    def load_resource(self, ref: str, *, offset: int = 1, max_bytes: int = VIEW_WINDOW_MAX_BYTES) -> SkillViewWindow:
        view = self.registry.load_resource(ref, offset=offset, max_bytes=max_bytes)
        self._views[view.view_id] = view
        return view

    def change_view_offset(self, view_id: str, offset: int) -> SkillViewWindow:
        view = self._views.get(view_id)
        if view is None:
            raise KeyError(view_id)
        view.set_offset(offset)
        return view

    def render_stable(self, *, available_token_budget: int = DEFAULT_SKILL_REGISTRY_TOKEN_BUDGET) -> str:
        lines = [LOADED_SKILLS_HEADER]
        loaded = self.loaded()
        if loaded:
            lines.append(self.registry.render_prompt(loaded))
        else:
            lines.append("  (none)")
        lines.append("")
        lines.append(self.registry.render_available(max_tokens=available_token_budget))
        return "\n".join(lines).strip()


def select_skills_by_token_budget(
    skills: tuple[SkillSpec, ...],
    *,
    max_tokens: int,
    section_header: str = "",
) -> tuple[tuple[SkillSpec, ...], int]:
    current = estimate_tokens(section_header)
    listed: list[SkillSpec] = []
    for skill in tuple(sorted(skills, key=lambda item: item.name)):
        line = skill.brief() + "\n"
        line_tokens = estimate_tokens(line)
        if listed and current + line_tokens > max_tokens:
            break
        if not listed and current + line_tokens > max_tokens:
            break
        listed.append(skill)
        current += line_tokens
    omitted = max(0, len(skills) - len(listed))
    return tuple(listed), omitted


def skill_from_markdown(text: str) -> SkillSpec:
    raw = str(text or "")
    metadata: dict[str, Any] = {}
    body = raw.strip()
    if raw.lstrip().startswith("---"):
        stripped = raw.lstrip()
        end = stripped.find("\n---", 3)
        if end != -1:
            header = stripped[3:end].strip()
            body = stripped[end + len("\n---") :].strip()
            metadata = _parse_simple_frontmatter(header)

    title = _first_markdown_heading(body)
    name = str(metadata.pop("name", "") or title or "skill").strip()
    description = str(metadata.pop("description", "") or "").strip()
    tags_raw = metadata.pop("tags", ())
    tags = _coerce_tags(tags_raw)
    license_name = str(metadata.pop("license", "") or "").strip()
    compatibility = str(metadata.pop("compatibility", "") or "").strip()
    disable_model_invocation = _coerce_bool(metadata.pop("disable-model-invocation", False))
    priority_raw = metadata.pop("priority", 0)
    try:
        priority = int(priority_raw)
    except (TypeError, ValueError):
        priority = 0
    return SkillSpec(
        name=name,
        description=description,
        prompt=body,
        tags=tags,
        priority=priority,
        license=license_name,
        compatibility=compatibility,
        disable_model_invocation=disable_model_invocation,
        metadata=metadata,
    )


def parse_skill_resource_ref(ref: str) -> tuple[str, str]:
    text = str(ref or "").strip()
    if text.startswith("@"):
        text = text[1:]
    if "/" not in text:
        raise ValueError("skill resource reference must be @skill/path")
    skill_name, rel_path = text.split("/", 1)
    skill_name = skill_name.strip()
    rel_path = rel_path.strip().replace("\\", "/")
    if not skill_name or not rel_path:
        raise ValueError("skill resource reference must include skill and path")
    return skill_name, rel_path


def compute_skill_hash(root: str | Path, *, max_file_bytes: int = 10 * 1024) -> str:
    root_path = Path(root).resolve()
    digests: list[str] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            continue
    return hashlib.sha256("".join(digests).encode("utf-8")).hexdigest()


_INCLUDE_DIRECTIVE_RE = re.compile(r"<!--\s*include:\s*(.+?)\s*-->")


def transform_includes_to_resource_hints(content: str, skill_name: str) -> str:
    def replace(match: re.Match[str]) -> str:
        file_path = match.group(1).strip()
        if not file_path:
            return match.group(0)
        return (
            f"[Included file: {file_path} - use load_skill_resource "
            f"with @{skill_name}/{file_path} to read this content]"
        )

    return _INCLUDE_DIRECTIVE_RE.sub(replace, content)


def _with_metadata(spec: SkillSpec, metadata: dict[str, Any]) -> SkillSpec:
    return SkillSpec(
        name=spec.name,
        description=spec.description,
        prompt=spec.prompt,
        tags=spec.tags,
        priority=spec.priority,
        license=spec.license,
        compatibility=spec.compatibility,
        disable_model_invocation=spec.disable_model_invocation,
        metadata={**spec.metadata, **metadata},
    )


def _safe_archive_member_path(destination: Path, member_name: str) -> Path:
    normalized = str(member_name or "").replace("\\", "/")
    if not normalized.strip():
        raise ValueError("archive member path is empty")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"archive member path escapes destination: {member_name}")
    target = (destination / normalized).resolve()
    if not str(target).startswith(str(destination)):
        raise ValueError(f"archive member path escapes destination: {member_name}")
    return target


@dataclass
class SkillViewWindow:
    skill_name: str
    file_path: str
    content: str
    offset: int = 1
    nonce: str = ""
    max_bytes: int = VIEW_WINDOW_MAX_BYTES

    def __post_init__(self) -> None:
        if not self.nonce:
            raw = f"{self.skill_name}:{self.file_path}"
            self.nonce = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]

    @property
    def view_id(self) -> str:
        return f"{self.skill_name}:{self.file_path}:{self.nonce}"

    @property
    def lines(self) -> list[str]:
        return self.content.splitlines()

    def set_offset(self, offset: int) -> None:
        total = max(1, len(self.lines))
        self.offset = min(max(1, offset), total)

    def render(self) -> tuple[str, bool]:
        lines = self.lines
        if not lines:
            return "", False
        header = f"<|VIEW_WINDOW_{self.nonce}|>"
        footer = f"<|VIEW_WINDOW_END_{self.nonce}|>"
        if self.offset == 1:
            plain = "\n".join([header, *lines, footer])
            if len(plain.encode("utf-8")) <= self.max_bytes:
                return plain, False

        rendered = [header]
        if self.offset > 1:
            rendered.append("...")
        truncated = False
        for line_number, line in enumerate(lines[self.offset - 1 :], start=self.offset):
            candidate = "\n".join([*rendered, f"{line_number} | {line}", "...", footer])
            if len(candidate.encode("utf-8")) > self.max_bytes:
                truncated = True
                break
            rendered.append(f"{line_number} | {line}")
        if self.offset - 1 + len(rendered) - 1 < len(lines):
            rendered.append("...")
            truncated = True
        rendered.append(footer)
        return "\n".join(rendered), truncated

    def manifest(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "skill_name": self.skill_name,
            "file_path": self.file_path,
            "offset": self.offset,
            "total_lines": len(self.lines),
            "nonce": self.nonce,
            "max_bytes": self.max_bytes,
        }


def _parse_simple_frontmatter(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    current_map_key = ""
    for line in text.splitlines():
        if current_map_key and line.startswith((" ", "\t")) and ":" in line:
            nested_key, nested_value = line.split(":", 1)
            current = metadata.setdefault(current_map_key, {})
            if isinstance(current, dict):
                current[nested_key.strip()] = nested_value.strip().strip("'\"")
            continue
        current_map_key = ""
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value == "":
            metadata[key] = {}
            current_map_key = key
        elif value.startswith("[") and value.endswith("]"):
            metadata[key] = tuple(
                item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()
            )
        else:
            metadata[key] = value.strip("'\"")
    return metadata


def _coerce_tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (tuple, list)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _first_markdown_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""

