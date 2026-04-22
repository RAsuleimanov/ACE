"""
==============================================================================
playbook.py
==============================================================================

Utilities for parsing and manipulating ACE playbooks.
"""

import json
import re
from typing import Any

try:
    from .utils import get_section_slug
except ImportError:  # pragma: no cover - legacy standalone ACE entrypoints
    from utils import get_section_slug

DEFAULT_BULLET_STATUS = "active"
ACTIVE_BULLET_STATUSES = {"active", "candidate"}
DEFAULT_SECTION_CAP = 12
DEFAULT_WARMUP_WINDOW = 5
DEFAULT_MIN_OBSERVATIONS = 3

INT_METADATA_FIELDS = (
    "helpful",
    "harmful",
    "neutral",
    "created_step",
    "last_considered_step",
    "last_used_step",
    "times_considered_not_used",
)
METADATA_PATTERN = re.compile(r"([a-z_]+)=([^\s]+)")
ACE_BULLET_PREFIX_PATTERN = re.compile(
    r"^\[[^\]]+\]\s*helpful=\d+\s*harmful=\d+(?:\s+[a-z_]+=[^\s]+)*\s*::\s*"
)
BARE_BULLET_PREFIX_PATTERN = re.compile(r"^\[[^\]]+\]\s*::\s*")
BULLET_ID_LITERAL_PATTERN = re.compile(r"^[\w]+-\d{5}$", re.UNICODE)
BULLET_ID_FRAGMENT_PATTERN = re.compile(r"[\w]+-\d{5}", re.UNICODE)
BRACKETED_SEGMENT_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def normalize_section_name(section_raw: str) -> str:
    """Normalize a section header or operation section name."""
    return section_raw.lower().replace(" ", "_").replace("&", "and")


def parse_playbook_line(line: str) -> dict[str, Any] | None:
    """Parse a single playbook bullet line, supporting legacy and extended metadata."""
    line = line.strip()
    if not line or line.startswith("##"):
        return None

    match = re.match(r"\[([^\]]+)\]\s*(.*?)\s*::\s*(.*)", line)
    if not match:
        return None

    bullet_id, metadata_segment, content = match.groups()
    parsed: dict[str, Any] = {
        "id": bullet_id,
        "helpful": 0,
        "harmful": 0,
        "neutral": 0,
        "created_step": 0,
        "last_considered_step": 0,
        "last_used_step": 0,
        "times_considered_not_used": 0,
        "status": DEFAULT_BULLET_STATUS,
        "content": content.strip(),
        "raw_line": line,
    }

    for key, raw_value in METADATA_PATTERN.findall(metadata_segment):
        if key in INT_METADATA_FIELDS:
            try:
                parsed[key] = int(raw_value)
            except ValueError:
                continue
        elif key == "status":
            parsed["status"] = raw_value

    return parsed


def count_playbook_bullets(playbook_text: str) -> int:
    """Count legacy and lifecycle-enriched ACE bullets in a playbook."""
    return sum(1 for line in playbook_text.splitlines() if parse_playbook_line(line))


def strip_ace_bullet_prefix(text: str) -> str:
    """Remove an ACE bullet prefix from content while preserving the body text."""
    return ACE_BULLET_PREFIX_PATTERN.sub("", text, count=1)


def sanitize_curator_bullet_content(text: Any) -> str:
    """Strip leaked playbook-line prefixes from curator-provided bullet content."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    while cleaned:
        next_cleaned = strip_ace_bullet_prefix(cleaned)
        next_cleaned = BARE_BULLET_PREFIX_PATTERN.sub("", next_cleaned, count=1).lstrip()
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned

    return cleaned


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return unique values while preserving first-seen order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def extract_bullet_id_references(text: Any) -> tuple[list[str], list[str]]:
    """Extract exact and malformed bullet-ID references from content."""
    if not isinstance(text, str):
        return [], []

    refs: list[str] = []
    malformed: list[str] = []
    for inner in BRACKETED_SEGMENT_PATTERN.findall(text):
        candidate = inner.strip()
        if BULLET_ID_LITERAL_PATTERN.fullmatch(candidate):
            refs.append(candidate)
        elif BULLET_ID_FRAGMENT_PATTERN.search(candidate):
            malformed.append(candidate)

    return _dedupe_preserve_order(refs), _dedupe_preserve_order(malformed)


def cleanup_reference_artifacts(text: Any) -> str:
    """Normalize punctuation after deterministic reference stripping."""
    if not isinstance(text, str):
        return ""

    cleaned = text
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r",\s+(и|или|and|or)\b", r" \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"([(\[{])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]}])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def strip_bullet_id_references(
    text: Any,
    *,
    remove_ids: set[str] | None = None,
    remove_all: bool = False,
) -> str:
    """Strip selected or all bracketed bullet-ID references from content."""
    if not isinstance(text, str):
        return ""

    removable_ids = remove_ids or set()

    def replace(match: re.Match[str]) -> str:
        candidate = match.group(1).strip()
        if BULLET_ID_LITERAL_PATTERN.fullmatch(candidate):
            if remove_all or candidate in removable_ids:
                return ""
            return match.group(0)
        if BULLET_ID_FRAGMENT_PATTERN.search(candidate):
            return ""
        return match.group(0)

    cleaned = BRACKETED_SEGMENT_PATTERN.sub(replace, text)
    return cleanup_reference_artifacts(cleaned)


def _collect_bullets_by_id(playbook_text: str) -> dict[str, dict[str, Any]]:
    """Parse a playbook and return bullets indexed by ID."""
    items, _, _ = _parse_playbook_items(playbook_text)
    bullets: dict[str, dict[str, Any]] = {}
    for item in items:
        if item["kind"] == "bullet":
            bullets[item["bullet"]["id"]] = item["bullet"]
    return bullets


def validate_curator_reference_integrity(
    current_playbook: str,
    provisional_playbook: str,
    operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate new curator ops without blocking unrelated legacy reference debt."""
    current_bullets = _collect_bullets_by_id(current_playbook)
    provisional_bullets = _collect_bullets_by_id(provisional_playbook)
    current_ids = set(current_bullets)
    provisional_ids = set(provisional_bullets)

    changed_ids = provisional_ids - current_ids
    removed_ids: set[str] = set()

    for op in operations:
        if not isinstance(op, dict):
            continue
        op_type = op.get("type")
        if op_type == "UPDATE":
            bullet_id = op.get("bullet_id", "")
            if bullet_id in provisional_bullets:
                changed_ids.add(bullet_id)
        elif op_type == "ARCHIVE":
            bullet_id = op.get("bullet_id", "")
            if bullet_id in current_ids:
                removed_ids.add(bullet_id)
        elif op_type == "MERGE":
            removed_ids.update(
                source_id for source_id in op.get("source_ids", []) if source_id in current_ids
            )

    errors: list[dict[str, Any]] = []

    for bullet_id in sorted(changed_ids):
        bullet = provisional_bullets.get(bullet_id)
        if not bullet:
            continue
        if bullet.get("status", DEFAULT_BULLET_STATUS) not in ACTIVE_BULLET_STATUSES:
            continue

        refs, malformed = extract_bullet_id_references(bullet.get("content", ""))
        if malformed:
            errors.append(
                {
                    "type": "malformed_reference",
                    "bullet_id": bullet_id,
                    "refs": malformed,
                    "message": "Changed bullet contains malformed bullet-ID references.",
                }
            )

        seen_refs = [ref for ref in refs if ref in current_ids or ref in provisional_ids]
        unseen_refs = [ref for ref in refs if ref not in current_ids and ref not in provisional_ids]
        if seen_refs:
            errors.append(
                {
                    "type": "internal_reference",
                    "bullet_id": bullet_id,
                    "refs": seen_refs,
                    "message": "Changed bullet is not self-contained and still references playbook bullet IDs.",
                }
            )
        if unseen_refs:
            errors.append(
                {
                    "type": "unknown_reference",
                    "bullet_id": bullet_id,
                    "refs": unseen_refs,
                    "message": "Changed bullet references unseen bullet IDs.",
                }
            )

    for bullet_id, bullet in provisional_bullets.items():
        if bullet.get("status", DEFAULT_BULLET_STATUS) not in ACTIVE_BULLET_STATUSES:
            continue

        refs, _ = extract_bullet_id_references(bullet.get("content", ""))
        dangling_refs = sorted({ref for ref in refs if ref in removed_ids})
        if dangling_refs:
            errors.append(
                {
                    "type": "dangling_reference_after_archive",
                    "bullet_id": bullet_id,
                    "refs": dangling_refs,
                    "message": "Surviving active bullet still depends on ARCHIVE/MERGE source IDs.",
                }
            )

    return errors


def format_reference_validation_errors(
    errors: list[dict[str, Any]],
    *,
    max_entries: int = 12,
) -> str:
    """Format reference-validation failures for logs and repair prompts."""
    if not errors:
        return "No reference validation errors."

    lines: list[str] = []
    for error in errors[:max_entries]:
        refs = ", ".join(error.get("refs", [])) or "n/a"
        lines.append(f"- {error.get('bullet_id', 'unknown')}: {error.get('message', 'error')} Refs: {refs}")
    remaining = len(errors) - max_entries
    if remaining > 0:
        lines.append(f"- ... and {remaining} more validation errors")
    return "\n".join(lines)


def cleanup_playbook_references(
    playbook_text: str,
    *,
    strip_all_active_refs: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministically strip malformed/missing/dangling bullet-ID refs from active bullets."""
    items, _, _ = _parse_playbook_items(playbook_text)
    all_ids = {
        item["bullet"]["id"]
        for item in items
        if item["kind"] == "bullet"
    }
    archived_ids = {
        item["bullet"]["id"]
        for item in items
        if item["kind"] == "bullet" and item["bullet"].get("status") == "archived"
    }

    cleaned_bullets: list[dict[str, Any]] = []
    for item in items:
        if item["kind"] != "bullet":
            continue

        bullet = item["bullet"]
        if bullet.get("status", DEFAULT_BULLET_STATUS) not in ACTIVE_BULLET_STATUSES:
            continue

        content = bullet.get("content", "")
        refs, malformed = extract_bullet_id_references(content)
        remove_ids = {
            ref
            for ref in refs
            if strip_all_active_refs or ref not in all_ids or ref in archived_ids
        }
        if not remove_ids and not malformed:
            continue

        cleaned_content = strip_bullet_id_references(
            content,
            remove_ids=remove_ids,
            remove_all=strip_all_active_refs,
        )
        if cleaned_content == content:
            continue

        bullet["content"] = cleaned_content
        cleaned_bullets.append(
            {
                "bullet_id": bullet["id"],
                "removed_refs": sorted(remove_ids),
                "malformed_refs": malformed,
            }
        )

    return _render_playbook_items(items), cleaned_bullets


def get_next_global_id(playbook_text: str) -> int:
    """Extract the highest global ID and return the next available number."""
    max_id = 0
    lines = playbook_text.strip().split("\n")

    for line in lines:
        parsed = parse_playbook_line(line)
        if not parsed:
            continue

        id_match = re.search(r"-(\d+)$", parsed["id"])
        if id_match:
            max_id = max(max_id, int(id_match.group(1)))

    return max_id + 1


def format_playbook_line(
    bullet_id: str,
    helpful: int,
    harmful: int,
    content: str,
    *,
    neutral: int = 0,
    created_step: int = 0,
    last_considered_step: int = 0,
    last_used_step: int = 0,
    times_considered_not_used: int = 0,
    status: str = DEFAULT_BULLET_STATUS,
) -> str:
    """Format a bullet into the extended playbook line format."""
    return (
        f"[{bullet_id}] "
        f"helpful={helpful} "
        f"harmful={harmful} "
        f"neutral={neutral} "
        f"created_step={created_step} "
        f"last_considered_step={last_considered_step} "
        f"last_used_step={last_used_step} "
        f"times_considered_not_used={times_considered_not_used} "
        f"status={status} :: {content}"
    )


def format_parsed_playbook_line(parsed: dict[str, Any]) -> str:
    """Serialize a parsed bullet back into the canonical playbook line format."""
    return format_playbook_line(
        parsed["id"],
        parsed.get("helpful", 0),
        parsed.get("harmful", 0),
        parsed.get("content", ""),
        neutral=parsed.get("neutral", 0),
        created_step=parsed.get("created_step", 0),
        last_considered_step=parsed.get("last_considered_step", 0),
        last_used_step=parsed.get("last_used_step", 0),
        times_considered_not_used=parsed.get("times_considered_not_used", 0),
        status=parsed.get("status", DEFAULT_BULLET_STATUS),
    )


def _parse_playbook_items(
    playbook_text: str,
) -> tuple[list[dict[str, Any]], dict[str, int], set[str]]:
    """Parse the playbook into ordered items and an ID index."""
    items: list[dict[str, Any]] = []
    bullet_index: dict[str, int] = {}
    sections: set[str] = {"general"}
    current_section = "general"

    for line in playbook_text.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("##"):
            current_section = normalize_section_name(stripped[2:].strip())
            sections.add(current_section)
            items.append({"kind": "header", "section": current_section, "raw": line})
            continue

        parsed = parse_playbook_line(line)
        if parsed:
            parsed["section"] = current_section
            bullet_index[parsed["id"]] = len(items)
            items.append({"kind": "bullet", "section": current_section, "bullet": parsed})
            continue

        items.append({"kind": "text", "section": current_section, "raw": line})

    return items, bullet_index, sections


def _resolve_section_name(section_raw: str, sections: set[str]) -> str:
    """Resolve an operation section to an existing normalized section name."""
    section = normalize_section_name(section_raw or "general")
    if section in sections:
        return section

    available_sections = sorted(section for section in sections if section != "general")
    if available_sections:
        fallback = available_sections[0]
        print(f"Warning: Section '{section_raw}' not found, adding to '{fallback}'")
        return fallback

    print(f"Warning: Section '{section_raw}' not found, adding to OTHERS")
    return "others"


def _build_new_bullet(
    bullet_id: str,
    section: str,
    content: str,
    current_step: int,
    *,
    helpful: int = 0,
    harmful: int = 0,
    neutral: int = 0,
    last_considered_step: int = 0,
    last_used_step: int = 0,
    times_considered_not_used: int = 0,
    status: str = "candidate",
) -> dict[str, Any]:
    """Create a new bullet dictionary with lifecycle metadata."""
    return {
        "id": bullet_id,
        "section": section,
        "helpful": helpful,
        "harmful": harmful,
        "neutral": neutral,
        "created_step": current_step,
        "last_considered_step": last_considered_step,
        "last_used_step": last_used_step,
        "times_considered_not_used": times_considered_not_used,
        "status": status,
        "content": content,
    }


def _render_playbook_items(
    items: list[dict[str, Any]], additions_by_section: dict[str, list[dict[str, Any]]] | None = None
) -> str:
    """Render playbook items back to text, inserting pending additions after each section."""
    pending = {section: bullets[:] for section, bullets in (additions_by_section or {}).items()}
    final_lines: list[str] = []
    current_section: str | None = None

    for item in items:
        if item["kind"] == "header":
            if current_section and pending.get(current_section):
                final_lines.extend(
                    format_parsed_playbook_line(bullet) for bullet in pending[current_section]
                )
                pending[current_section] = []
            current_section = item["section"]
            final_lines.append(item["raw"])
        elif item["kind"] == "bullet":
            final_lines.append(format_parsed_playbook_line(item["bullet"]))
        else:
            final_lines.append(item["raw"])

    if current_section and pending.get(current_section):
        final_lines.extend(
            format_parsed_playbook_line(bullet) for bullet in pending[current_section]
        )
        pending[current_section] = []

    leftovers = [bullet for bullets in pending.values() for bullet in bullets]
    if leftovers:
        inserted = False
        for index, line in enumerate(final_lines):
            if line.strip() == "## OTHERS":
                final_lines[index + 1 : index + 1] = [
                    format_parsed_playbook_line(bullet) for bullet in leftovers
                ]
                inserted = True
                break
        if not inserted:
            final_lines.extend(format_parsed_playbook_line(bullet) for bullet in leftovers)

    return "\n".join(final_lines)


def get_bullet_observations(parsed: dict[str, Any]) -> int:
    """Return the number of scored observations for a bullet."""
    return parsed.get("helpful", 0) + parsed.get("harmful", 0) + parsed.get("neutral", 0)


def bullet_score(parsed: dict[str, Any]) -> float:
    """Rank bullets for prompt retention and section pruning."""
    active_bonus = 0.5 if parsed.get("status") == "active" else 0.0
    return (
        (parsed.get("helpful", 0) * 2.0)
        - (parsed.get("harmful", 0) * 3.0)
        - parsed.get("neutral", 0)
        - parsed.get("times_considered_not_used", 0)
        + active_bonus
        + (parsed.get("last_used_step", 0) * 0.01)
        + (parsed.get("last_considered_step", 0) * 0.001)
    )


def render_active_playbook(playbook_text: str) -> str:
    """Render only prompt-eligible bullets for the generator (full metadata)."""
    rendered_lines: list[str] = []

    for line in playbook_text.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("##") or not stripped:
            rendered_lines.append(line)
            continue

        parsed = parse_playbook_line(line)
        if not parsed:
            rendered_lines.append(line)
            continue

        if parsed.get("status", DEFAULT_BULLET_STATUS) in ACTIVE_BULLET_STATUSES:
            rendered_lines.append(format_parsed_playbook_line(parsed))

    return "\n".join(rendered_lines)


def render_minimal_playbook(playbook_text: str) -> str:
    """Render active bullets with only [id] + content. No lifecycle metadata.

    Used for Generator prompts where counters and recency just waste attention —
    generator only needs the ID (for citation in considered/used lists) and the
    actual bullet content. Reflector/Curator/Analyzer continue receiving full
    metadata because they need counts for tag decisions and prune scoring.
    """
    rendered_lines: list[str] = []

    for line in playbook_text.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("##") or not stripped:
            rendered_lines.append(line)
            continue

        parsed = parse_playbook_line(line)
        if not parsed:
            rendered_lines.append(line)
            continue

        if parsed.get("status", DEFAULT_BULLET_STATUS) in ACTIVE_BULLET_STATUSES:
            rendered_lines.append(f"[{parsed['id']}] :: {parsed.get('content', '')}")

    return "\n".join(rendered_lines)


def update_bullet_counts(
    playbook_text: str,
    bullet_tags: list[dict[str, str]],
    considered_bullet_ids: list[str] | None = None,
    used_bullet_ids: list[str] | None = None,
    current_step: int | None = None,
) -> str:
    """Update bullet evidence and lifecycle metadata based on considered/used bullets."""
    lines = playbook_text.strip().split("\n")
    updated_lines: list[str] = []

    tag_map: dict[str, str] = {}
    for tag in bullet_tags or []:
        if not isinstance(tag, dict):
            continue
        bullet_id = tag.get("id") or tag.get("bullet") or ""
        tag_value = tag.get("tag", "neutral")
        if bullet_id:
            tag_map[bullet_id] = tag_value

    considered_set = set(considered_bullet_ids or tag_map.keys())
    used_set = set(used_bullet_ids or [])

    if not tag_map and not considered_set and not used_set:
        print("Warning: No valid bullet evidence found to update counts")
        return playbook_text

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            updated_lines.append(line)
            continue

        parsed = parse_playbook_line(line)
        if not parsed:
            updated_lines.append(line)
            continue

        bullet_id = parsed["id"]
        was_considered = bullet_id in considered_set
        was_used = bullet_id in used_set

        # Guard against step-counter regressions: in multi-epoch runs, step resets
        # to 1 at each epoch boundary, so a bullet accumulated at step=274 in epoch 1
        # would have its timestamp overwritten by a smaller value in epoch 2. Use
        # max() to preserve the most recent absolute-time mark. Combined with the
        # epoch-offset fix in ace.py, this is belt-and-suspenders.
        if was_considered and current_step is not None:
            parsed["last_considered_step"] = max(parsed.get("last_considered_step", 0), current_step)
        if was_used and current_step is not None:
            parsed["last_used_step"] = max(parsed.get("last_used_step", 0), current_step)
        if was_considered and not was_used:
            parsed["times_considered_not_used"] += 1

        tag = tag_map.get(bullet_id)
        if tag == "helpful":
            parsed["helpful"] += 1
        elif tag == "harmful":
            parsed["harmful"] += 1
        elif tag == "neutral":
            parsed["neutral"] += 1

        if parsed.get("status") == "candidate" and (was_used or tag == "helpful"):
            parsed["status"] = "active"

        updated_lines.append(format_parsed_playbook_line(parsed))

    return "\n".join(updated_lines)


def apply_curator_operations(
    playbook_text: str,
    operations: list[dict[str, Any]],
    next_id: int,
    current_step: int = 0,
) -> tuple[str, int, dict[int, str]]:
    """Apply curator lifecycle operations to the playbook.

    Returns:
        (rendered_playbook, next_id, manifest) where manifest maps
        op_index → created/updated bullet_id for ADD, UPDATE, and MERGE ops.
    """
    items, bullet_index, sections = _parse_playbook_items(playbook_text)
    additions_by_section: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[int, str] = {}

    for op_idx, op in enumerate(operations):
        if not isinstance(op, dict):
            continue

        op_type = op.get("type")
        sanitized_content = sanitize_curator_bullet_content(op.get("content", ""))

        if op_type == "ADD":
            section = _resolve_section_name(op.get("section", "general"), sections)
            slug = get_section_slug(section)
            bullet_id = f"{slug}-{next_id:05d}"
            next_id += 1

            additions_by_section.setdefault(section, []).append(
                _build_new_bullet(bullet_id, section, sanitized_content, current_step)
            )
            manifest[op_idx] = bullet_id
            print(f"  Added bullet {bullet_id} to section {section}")
            continue

        if op_type == "UPDATE":
            bullet_id = op.get("bullet_id", "")
            item_index = bullet_index.get(bullet_id)
            if item_index is None:
                print(f"Warning: UPDATE skipped because bullet '{bullet_id}' was not found")
                continue

            bullet = items[item_index]["bullet"]
            bullet["content"] = sanitized_content or bullet["content"]
            bullet["status"] = "candidate"
            manifest[op_idx] = bullet_id
            print(f"  Updated bullet {bullet_id}")
            continue

        if op_type == "ARCHIVE":
            bullet_id = op.get("bullet_id", "")
            item_index = bullet_index.get(bullet_id)
            if item_index is None:
                print(f"Warning: ARCHIVE skipped because bullet '{bullet_id}' was not found")
                continue

            items[item_index]["bullet"]["status"] = "archived"
            print(f"  Archived bullet {bullet_id}")
            continue

        if op_type == "MERGE":
            source_ids = [
                source_id for source_id in op.get("source_ids", []) if source_id in bullet_index
            ]
            if len(source_ids) < 2:
                print("Warning: MERGE skipped because fewer than two source bullets were found")
                continue

            source_bullets = [items[bullet_index[source_id]]["bullet"] for source_id in source_ids]
            for source_bullet in source_bullets:
                source_bullet["status"] = "archived"

            section = _resolve_section_name(
                op.get("section", source_bullets[0]["section"]),
                sections | {source_bullets[0]["section"]},
            )
            slug = get_section_slug(section)
            merged_id = f"{slug}-{next_id:05d}"
            next_id += 1

            additions_by_section.setdefault(section, []).append(
                _build_new_bullet(
                    merged_id,
                    section,
                    sanitized_content,
                    current_step,
                    helpful=sum(bullet.get("helpful", 0) for bullet in source_bullets),
                    harmful=sum(bullet.get("harmful", 0) for bullet in source_bullets),
                    neutral=sum(bullet.get("neutral", 0) for bullet in source_bullets),
                    last_considered_step=max(
                        bullet.get("last_considered_step", 0) for bullet in source_bullets
                    ),
                    last_used_step=max(
                        bullet.get("last_used_step", 0) for bullet in source_bullets
                    ),
                    times_considered_not_used=sum(
                        bullet.get("times_considered_not_used", 0) for bullet in source_bullets
                    ),
                )
            )
            manifest[op_idx] = merged_id
            print(f"  Merged bullets {source_ids} into {merged_id}")
            continue

        if op_type == "CREATE_META":
            print("Warning: CREATE_META is ignored by the lifecycle executor")
            continue

        print(f"Warning: Unsupported curator operation '{op_type}'")

    return _render_playbook_items(items, additions_by_section), next_id, manifest


def prune_playbook(
    playbook_text: str,
    current_step: int,
    *,
    max_active_bullets_per_section: int = DEFAULT_SECTION_CAP,
    warmup_window: int = DEFAULT_WARMUP_WINDOW,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> tuple[str, list[str]]:
    """Apply deterministic archive rules to keep the active playbook bounded."""
    items, _, _ = _parse_playbook_items(playbook_text)
    section_bullets: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        if item["kind"] != "bullet":
            continue

        bullet = item["bullet"]
        if bullet.get("status", DEFAULT_BULLET_STATUS) not in ACTIVE_BULLET_STATUSES:
            continue

        section_bullets.setdefault(item["section"], []).append(bullet)

    archived_ids: list[str] = []
    archived_id_set: set[str] = set()

    for bullets in section_bullets.values():
        retained: list[dict[str, Any]] = []

        for bullet in bullets:
            created_step = bullet.get("created_step", 0)
            age = current_step - created_step if created_step > 0 else 0
            observations = get_bullet_observations(bullet)
            never_used = bullet.get("last_used_step", 0) == 0
            harmful_dominates = (
                observations >= min_observations
                and bullet.get("harmful", 0) > bullet.get("helpful", 0)
                and bullet.get("harmful", 0) >= max(1, bullet.get("neutral", 0))
            )

            if never_used and age >= warmup_window:
                archived_id_set.add(bullet["id"])
                archived_ids.append(bullet["id"])
                continue

            if harmful_dominates:
                archived_id_set.add(bullet["id"])
                archived_ids.append(bullet["id"])
                continue

            retained.append(bullet)

        ranked = sorted(retained, key=bullet_score, reverse=True)
        for bullet in ranked[max_active_bullets_per_section:]:
            if bullet["id"] not in archived_id_set:
                archived_id_set.add(bullet["id"])
                archived_ids.append(bullet["id"])

    if not archived_id_set:
        return playbook_text, []

    for item in items:
        if item["kind"] == "bullet" and item["bullet"]["id"] in archived_id_set:
            item["bullet"]["status"] = "archived"

    return _render_playbook_items(items), archived_ids


def get_playbook_stats(playbook_text: str) -> dict[str, Any]:
    """Generate statistics about the playbook, including lifecycle state counts."""
    lines = playbook_text.strip().split("\n")
    stats: dict[str, Any] = {
        "total_bullets": 0,
        "active_bullets": 0,
        "candidate_bullets": 0,
        "archived_bullets": 0,
        "high_performing": 0,
        "problematic": 0,
        "unused": 0,
        "by_section": {},
    }

    current_section = "general"

    for line in lines:
        if line.strip().startswith("##"):
            current_section = line.strip()[2:].strip()
            continue

        parsed = parse_playbook_line(line)
        if not parsed:
            continue

        stats["total_bullets"] += 1
        observations = get_bullet_observations(parsed)
        status = parsed.get("status", DEFAULT_BULLET_STATUS)

        if status == "active":
            stats["active_bullets"] += 1
        elif status == "candidate":
            stats["candidate_bullets"] += 1
        elif status == "archived":
            stats["archived_bullets"] += 1

        if parsed["helpful"] > 5 and parsed["harmful"] < 2:
            stats["high_performing"] += 1
        elif parsed["harmful"] > parsed["helpful"] and parsed["harmful"] > 0:
            stats["problematic"] += 1
        elif observations == 0:
            stats["unused"] += 1

        section_stats = stats["by_section"].setdefault(
            current_section,
            {
                "count": 0,
                "active": 0,
                "candidate": 0,
                "archived": 0,
                "helpful": 0,
                "harmful": 0,
                "neutral": 0,
            },
        )
        section_stats["count"] += 1
        section_stats[status] = section_stats.get(status, 0) + 1
        section_stats["helpful"] += parsed["helpful"]
        section_stats["harmful"] += parsed["harmful"]
        section_stats["neutral"] += parsed["neutral"]

    return stats


def extract_json_from_text(text, json_key=None):
    """Extract JSON object from text, handling various formats."""
    try:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        def find_json_objects(raw_text):
            """Find JSON objects using balanced brace counting."""
            json_objects = []
            i = 0
            while i < len(raw_text):
                if raw_text[i] == "{":
                    brace_count = 1
                    start = i
                    i += 1

                    while i < len(raw_text) and brace_count > 0:
                        if raw_text[i] == "{":
                            brace_count += 1
                        elif raw_text[i] == "}":
                            brace_count -= 1
                        elif raw_text[i] == '"':
                            i += 1
                            while i < len(raw_text) and raw_text[i] != '"':
                                if raw_text[i] == "\\":
                                    i += 1
                                i += 1
                        i += 1

                    if brace_count == 0:
                        json_objects.append(raw_text[start:i])
                else:
                    i += 1

            return json_objects

        for json_str in find_json_objects(text):
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    except Exception as exc:
        print(f"Failed to extract JSON: {exc}")
        preview = text[:500] + "..." if len(text) > 500 else text
        print(f"Raw content preview:\n{preview}")

    return None


def extract_playbook_bullets(playbook_text: str, bullet_ids: list[str]) -> str:
    """Extract specific bullets from the playbook for reflector input."""
    if not bullet_ids:
        return "(No bullets considered by generator)"

    found_bullets: list[str] = []
    bullet_id_set = set(bullet_ids)

    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in bullet_id_set:
            found_bullets.append(format_parsed_playbook_line(parsed))

    if not found_bullets:
        return "(Generator referenced bullet IDs but none were found in playbook)"

    return "\n".join(found_bullets)
