"""
Curator agent for ACE system.
Manages playbook operations (ADD, UPDATE, MERGE, ARCHIVE).
"""

import json
from pathlib import Path
from typing import Any, Optional

from llm import timed_llm_call
from logger import log_curator_failure, log_curator_operation_diff
from playbook_utils import (
    _collect_bullets_by_id,
    apply_curator_operations,
    cleanup_reference_artifacts,
    extract_bullet_id_references,
    extract_json_from_text,
    format_reference_validation_errors,
    strip_bullet_id_references,
    validate_curator_reference_integrity,
)

from ..prompts.curator import CURATOR_PROMPT, CURATOR_PROMPT_NO_GT

TARGETED_REPHRASE_PROMPT = """Перефразируй bullet так, чтобы он был самодостаточным.
Не ссылайся на другие ID — инлайнь суть.

Bullet: "{content}"

{ref_context}

Верни ТОЛЬКО текст нового bullet, без ID, метаданных и кавычек."""

REPAIR_SUMMARY_JSONL = "curator_repair_summary.jsonl"
REPAIR_COUNTERS_JSON = "curator_repair_counters.json"


class Curator:
    """
    Curator agent that manages the playbook by adding, updating,
    merging, and archiving bullets based on reflection feedback.
    """

    def __init__(
        self,
        api_client,
        api_provider,
        model: str,
        max_tokens: int = 4096,
        reasoning_effort: Optional[str] = None,
        reasoning: Optional[dict] = None,
    ):
        """
        Initialize the Curator agent.

        Args:
            api_client: OpenAI client for LLM calls
            api_provider: API provider for LLM calls
            model: Model name to use for curation
            max_tokens: Maximum tokens for curation
            reasoning_effort: Reasoning effort level for GPT-5.x models (none, low, medium, high)
        """
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning = reasoning

    def curate(
        self,
        current_playbook: str,
        recent_reflection: str,
        question_context: str,
        current_step: int,
        total_samples: int,
        token_budget: int,
        playbook_stats: dict[str, Any],
        use_ground_truth: bool = True,
        use_json_mode: bool = False,
        call_id: str = "curate",
        log_dir: Optional[str] = None,
        next_global_id: int = 1,
    ) -> tuple[str, int, list[dict[str, Any]], dict[str, Any]]:
        """
        Curate the playbook based on reflection feedback.

        Args:
            current_playbook: Current playbook content
            recent_reflection: Recent reflection from reflector
            question_context: Context for the current question
            current_step: Current training step
            total_samples: Total number of training samples
            token_budget: Total token budget for playbook
            playbook_stats: Statistics about current playbook
            use_ground_truth: Whether ground truth is available
            use_json_mode: Whether to use JSON mode
            call_id: Unique identifier for this call
            log_dir: Directory for logging
            next_global_id: Next available global ID for bullets

        Returns:
            Tuple of (updated_playbook, next_global_id, operations, call_info)
        """
        # Format playbook stats as JSON string (ensure_ascii=False for Russian readability)
        stats_str = json.dumps(playbook_stats, indent=2, ensure_ascii=False)

        # Select the appropriate prompt
        if use_ground_truth:
            prompt = CURATOR_PROMPT.format(
                current_step=current_step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats_str,
                recent_reflection=recent_reflection,
                current_playbook=current_playbook,
                question_context=question_context,
            )
        else:
            prompt = CURATOR_PROMPT_NO_GT.format(
                current_step=current_step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats_str,
                recent_reflection=recent_reflection,
                current_playbook=current_playbook,
                question_context=question_context,
            )

        # Make the LLM call
        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="curator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            reasoning_effort=self.reasoning_effort,
            reasoning=self.reasoning,
        )

        # Check for empty response error
        if response.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
            print("⏭️  Skipping curator operation due to empty response")
            log_curator_failure(log_dir, current_step, "empty_response", response[:200], 0)
            return current_playbook, next_global_id, [], call_info

        # Extract and validate operations
        try:
            operations_info = self._extract_and_validate_operations(response)

            operations = operations_info["operations"]
            print(f"✅ Curator JSON schema validated successfully: {len(operations)} operations")

            original_next_global_id = next_global_id
            candidate_playbook, candidate_next_global_id, validation_errors, manifest = (
                self._apply_validated_operations(
                    current_playbook,
                    operations,
                    next_global_id,
                    current_step,
                )
            )
            final_call_id = call_id
            repair_triggered = False
            repair_succeeded = False
            repair_failed = False

            if validation_errors:
                repair_triggered = True
                validation_summary = format_reference_validation_errors(validation_errors)
                print(f"❌ Curator reference validation failed:\n{validation_summary}")
                log_curator_failure(
                    log_dir,
                    current_step,
                    "reference_validation_error",
                    response,
                    0,
                    validation_summary,
                )

                repaired_ops, repair_call_info = self._targeted_repair(
                    current_playbook,
                    operations,
                    manifest,
                    validation_errors,
                    call_id=call_id,
                    log_dir=log_dir,
                )

                if repaired_ops is None:
                    print("⏭️  Skipping curator operation due to unresolvable reference errors")
                    repair_failed = True
                    self._log_repair_summary(
                        log_dir,
                        current_step,
                        call_id,
                        repair_triggered=repair_triggered,
                        repair_succeeded=repair_succeeded,
                        repair_failed=repair_failed,
                    )
                    return current_playbook, original_next_global_id, [], {
                        "initial": call_info,
                        "repair": repair_call_info,
                    }

                repaired_playbook, repaired_next_id, repair_errors, _ = (
                    self._apply_validated_operations(
                        current_playbook,
                        repaired_ops,
                        original_next_global_id,
                        current_step,
                    )
                )

                if repair_errors:
                    repair_summary = format_reference_validation_errors(repair_errors)
                    print(f"❌ Targeted repair still has errors:\n{repair_summary}")
                    log_curator_failure(
                        log_dir,
                        current_step,
                        "targeted_repair_still_invalid",
                        str(repaired_ops),
                        0,
                        repair_summary,
                    )
                    print("⏭️  Skipping curator operation after failed targeted repair")
                    repair_failed = True
                    self._log_repair_summary(
                        log_dir,
                        current_step,
                        call_id,
                        repair_triggered=repair_triggered,
                        repair_succeeded=repair_succeeded,
                        repair_failed=repair_failed,
                    )
                    return current_playbook, original_next_global_id, [], {
                        "initial": call_info,
                        "repair": repair_call_info,
                    }

                operations = repaired_ops
                candidate_playbook = repaired_playbook
                next_global_id = repaired_next_id
                call_info = {"initial": call_info, "repair": repair_call_info}
                final_call_id = f"{call_id}_repair"
                repair_succeeded = True
            else:
                next_global_id = candidate_next_global_id

            updated_playbook = candidate_playbook
            self._log_repair_summary(
                log_dir,
                current_step,
                call_id,
                repair_triggered=repair_triggered,
                repair_succeeded=repair_succeeded,
                repair_failed=repair_failed,
            )

            # Log detailed diff for each final operation before applying
            if log_dir is not None:
                for op in operations:
                    try:
                        log_curator_operation_diff(
                            Path(log_dir).parent, op, current_playbook, final_call_id
                        )
                    except Exception as e:
                        print(f"Warning: Failed to log curator operation diff: {e}")

            # Log operations
            for op in operations:
                try:
                    op_type = op.get("type", "UNKNOWN") if isinstance(op, dict) else "INVALID"
                    op_reason = (
                        op.get("reason", "No reason given")
                        if isinstance(op, dict)
                        else "Invalid operation format"
                    )
                    print(f"  - {op_type}: {op_reason}")
                except Exception as e:
                    print(f"  - UNKNOWN: Error logging operation: {e}")

            return updated_playbook, next_global_id, operations, call_info

        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            print(f"❌ Curator JSON parsing failed: {e}")
            print(f"📄 Raw curator response preview: {response[:300]}...")

            log_curator_failure(log_dir, current_step, "json_parse_error", response, 0, str(e))

            print("⏭️  Skipping curator operation due to invalid JSON format")
            return current_playbook, next_global_id, [], call_info

        except Exception as e:
            print(f"❌ Curator operation failed: {e}")
            print(f"📄 Raw curator response preview: {response[:300]}...")

            log_curator_failure(log_dir, current_step, "operation_error", response, 0, str(e))

            print("⏭️  Skipping curator operation and continuing training")
            return current_playbook, next_global_id, [], call_info

    def _apply_validated_operations(
        self,
        current_playbook: str,
        operations: list[dict[str, Any]],
        next_global_id: int,
        current_step: int,
    ) -> tuple[str, int, list[dict[str, Any]], dict[int, str]]:
        """Apply curator ops to a provisional playbook and validate reference integrity."""
        provisional_playbook, provisional_next_id, manifest = apply_curator_operations(
            current_playbook,
            operations,
            next_global_id,
            current_step=current_step,
        )
        validation_errors = validate_curator_reference_integrity(
            current_playbook,
            provisional_playbook,
            operations,
        )
        return provisional_playbook, provisional_next_id, validation_errors, manifest

    def _log_repair_summary(
        self,
        log_dir: Optional[str],
        current_step: int,
        call_id: str,
        *,
        repair_triggered: bool,
        repair_succeeded: bool,
        repair_failed: bool,
    ) -> None:
        """Append per-call repair stats and maintain aggregate counters."""
        if not log_dir:
            return

        run_dir = Path(log_dir).parent
        summary_path = run_dir / REPAIR_SUMMARY_JSONL
        counters_path = run_dir / REPAIR_COUNTERS_JSON

        entry = {
            "call_id": call_id,
            "step": current_step,
            "repair_triggered": repair_triggered,
            "repair_succeeded": repair_succeeded,
            "repair_failed": repair_failed,
        }

        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            counters = {
                "repair_triggered": 0,
                "repair_succeeded": 0,
                "repair_failed": 0,
            }
            if counters_path.exists():
                with open(counters_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for key in counters:
                    counters[key] = int(loaded.get(key, 0))

            counters["repair_triggered"] += int(repair_triggered)
            counters["repair_succeeded"] += int(repair_succeeded)
            counters["repair_failed"] += int(repair_failed)

            with open(counters_path, "w", encoding="utf-8") as f:
                json.dump(counters, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to write curator repair summary: {e}")

    def _targeted_repair(
        self,
        current_playbook: str,
        operations: list[dict[str, Any]],
        manifest: dict[int, str],
        validation_errors: list[dict[str, Any]],
        call_id: str = "repair",
        log_dir: Optional[str] = None,
    ) -> tuple[Optional[list[dict[str, Any]]], dict[str, Any]]:
        """Fix reference errors with targeted per-bullet rewrites instead of full re-prompt.

        Strategy per error type:
        - unknown_reference / malformed_reference: deterministic strip
        - internal_reference: targeted LLM rephrase (1-3 bullets individually, 4+ batched)
        - dangling_reference_after_archive: reject the ARCHIVE that caused it

        Returns:
            (repaired_operations or None if unresolvable, repair_call_info)
        """
        playbook_bullets = _collect_bullets_by_id(current_playbook)
        repair_call_info: dict[str, Any] = {}

        bullet_to_op_index: dict[str, int] = {}
        for op_idx, bullet_id in manifest.items():
            bullet_to_op_index[bullet_id] = op_idx
        for op_idx, op in enumerate(operations):
            if isinstance(op, dict) and op.get("type") == "UPDATE":
                bullet_to_op_index[op.get("bullet_id", "")] = op_idx

        repaired_ops = [dict(op) if isinstance(op, dict) else op for op in operations]

        dangling_archive_ids: set[str] = set()
        deterministic_bullets: set[str] = set()
        llm_rephrase_bullets: dict[str, list[str]] = {}

        for err in validation_errors:
            err_type = err.get("type", "")
            bullet_id = err.get("bullet_id", "")
            refs = err.get("refs", [])

            if err_type == "dangling_reference_after_archive":
                for ref in refs:
                    dangling_archive_ids.add(ref)

            elif err_type in ("unknown_reference", "malformed_reference"):
                deterministic_bullets.add(bullet_id)

            elif err_type == "internal_reference":
                llm_rephrase_bullets[bullet_id] = refs

        for op_idx, op in enumerate(repaired_ops):
            if not isinstance(op, dict):
                continue
            if op.get("type") == "ARCHIVE" and op.get("bullet_id") in dangling_archive_ids:
                print(f"  Repair: rejecting ARCHIVE of {op['bullet_id']} (would create dangling refs)")
                repaired_ops[op_idx] = {"type": "_REJECTED", "original": op}
            elif op.get("type") == "MERGE":
                source_ids = set(op.get("source_ids", []))
                rejected_sources = source_ids & dangling_archive_ids
                if rejected_sources:
                    print(f"  Repair: rejecting MERGE (sources {rejected_sources} would create dangling refs)")
                    repaired_ops[op_idx] = {"type": "_REJECTED", "original": op}

        for bullet_id in deterministic_bullets:
            op_idx = bullet_to_op_index.get(bullet_id)
            if op_idx is None or op_idx >= len(repaired_ops):
                continue
            op = repaired_ops[op_idx]
            if not isinstance(op, dict) or "content" not in op:
                continue
            original = op["content"]
            stripped = strip_bullet_id_references(original, remove_all=True)
            stripped = cleanup_reference_artifacts(stripped)
            if stripped and stripped != original:
                op["content"] = stripped
                print(f"  Repair: deterministic strip on {bullet_id}")

        if llm_rephrase_bullets:
            llm_rephrase_bullets = {
                bid: refs for bid, refs in llm_rephrase_bullets.items()
                if bid not in deterministic_bullets
            }

        if llm_rephrase_bullets:
            repair_items: list[dict[str, Any]] = []
            for bullet_id, refs in llm_rephrase_bullets.items():
                op_idx = bullet_to_op_index.get(bullet_id)
                if op_idx is None or op_idx >= len(repaired_ops):
                    continue
                op = repaired_ops[op_idx]
                if not isinstance(op, dict) or "content" not in op:
                    continue
                ref_lines = []
                for ref_id in refs:
                    ref_bullet = playbook_bullets.get(ref_id)
                    if ref_bullet:
                        ref_lines.append(f"[{ref_id}] означает: \"{ref_bullet.get('content', '')}\"")
                repair_items.append({
                    "op_idx": op_idx,
                    "bullet_id": bullet_id,
                    "content": op["content"],
                    "ref_context": "\n".join(ref_lines),
                })

            if repair_items:
                try:
                    repaired_contents = self._llm_rephrase_bullets(
                        repair_items, call_id=call_id, log_dir=log_dir,
                    )
                    repair_call_info = repaired_contents.get("_call_info", {})
                    for item in repair_items:
                        new_content = repaired_contents.get(item["bullet_id"])
                        if new_content:
                            repaired_ops[item["op_idx"]]["content"] = new_content
                            print(f"  Repair: LLM rephrase on {item['bullet_id']}")
                except Exception as e:
                    print(f"  Repair: LLM rephrase failed ({e}), falling back to deterministic strip")
                    for item in repair_items:
                        op = repaired_ops[item["op_idx"]]
                        stripped = strip_bullet_id_references(op["content"], remove_all=True)
                        stripped = cleanup_reference_artifacts(stripped)
                        if stripped:
                            op["content"] = stripped

        final_ops = [op for op in repaired_ops if not (isinstance(op, dict) and op.get("type") == "_REJECTED")]

        if not final_ops:
            print("  Repair: all operations rejected")
            return None, repair_call_info

        return final_ops, repair_call_info

    def _llm_rephrase_bullets(
        self,
        repair_items: list[dict[str, Any]],
        call_id: str = "repair",
        log_dir: Optional[str] = None,
    ) -> dict[str, str]:
        """Rephrase 1+ bullets via a single small LLM call.

        For 1-3 items: one bullet per call in the prompt.
        For 4+: batched JSON format.

        Returns dict of bullet_id → rephrased content, plus '_call_info' key.
        """
        if len(repair_items) <= 3:
            prompt_parts = []
            for item in repair_items:
                prompt_parts.append(
                    TARGETED_REPHRASE_PROMPT.format(
                        content=item["content"],
                        ref_context=item["ref_context"],
                    )
                )
            prompt = "\n---\n".join(prompt_parts)
            if len(repair_items) > 1:
                prompt += (
                    f"\n\nВерни ровно {len(repair_items)} перефразированных bullet, "
                    "каждый на отдельной строке, в том же порядке."
                )
        else:
            batch_items = []
            for item in repair_items:
                batch_items.append({
                    "id": item["bullet_id"],
                    "content": item["content"],
                    "ref_context": item["ref_context"],
                })
            prompt = (
                "Перефразируй каждый bullet так, чтобы он был самодостаточным. "
                "Не ссылайся на другие ID — инлайнь суть.\n\n"
                + json.dumps(batch_items, ensure_ascii=False, indent=2)
                + '\n\nВерни JSON массив: [{"id": "...", "content": "..."}]'
            )

        response, llm_call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="curator",
            call_id=f"{call_id}_targeted_repair",
            max_tokens=2048,
            log_dir=log_dir,
            reasoning_effort=self.reasoning_effort,
            reasoning=self.reasoning,
        )

        result: dict[str, str] = {"_call_info": llm_call_info}

        if response.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
            return result

        if len(repair_items) <= 3:
            lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
            for i, item in enumerate(repair_items):
                if i < len(lines):
                    cleaned = lines[i].strip().strip('"').strip("'")
                    if cleaned:
                        result[item["bullet_id"]] = cleaned
        else:
            try:
                parsed = extract_json_from_text(response, "content")
                if isinstance(parsed, list):
                    for entry in parsed:
                        if isinstance(entry, dict) and "id" in entry and "content" in entry:
                            result[entry["id"]] = entry["content"]
            except (ValueError, json.JSONDecodeError):
                lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
                for i, item in enumerate(repair_items):
                    if i < len(lines):
                        cleaned = lines[i].strip().strip('"').strip("'")
                        if cleaned:
                            result[item["bullet_id"]] = cleaned

        return result

    def _extract_and_validate_operations(self, response: str) -> dict[str, Any]:
        """
        Extract and validate operations from curator response.

        Args:
            response: The curator's response

        Returns:
            Dictionary with 'reasoning' and 'operations' keys

        Raises:
            ValueError: If JSON is invalid or missing required fields
        """
        # Extract operations info
        operations_info = extract_json_from_text(response, "operations")

        # Validate JSON structure is correct
        if not operations_info:
            raise ValueError("Failed to extract valid JSON from curator response")

        # Validate required fields
        if "reasoning" not in operations_info:
            raise ValueError("JSON missing required 'reasoning' field")

        if "operations" not in operations_info:
            raise ValueError("JSON missing required 'operations' field")

        # Validate field types
        if not isinstance(operations_info["reasoning"], str):
            raise ValueError("'reasoning' field must be a string")

        if not isinstance(operations_info["operations"], list):
            raise ValueError("'operations' field must be a list")

        # Validate operations structure
        for i, op in enumerate(operations_info["operations"]):
            if not isinstance(op, dict):
                raise ValueError(f"Operation {i} must be a dictionary")

            if "type" not in op:
                raise ValueError(f"Operation {i} missing required 'type' field")

            op_type = op["type"]

            if op_type not in ["ADD", "UPDATE", "MERGE", "ARCHIVE", "CREATE_META"]:
                raise ValueError(f"Unsupported operation type '{op_type}'")

            if op_type == "ADD":
                required_fields = {"type", "section", "content"}
                missing_fields = required_fields - set(op.keys())
                if missing_fields:
                    raise ValueError(f"ADD operation {i} missing fields: {list(missing_fields)}")
            elif op_type == "UPDATE":
                required_fields = {"type", "bullet_id", "content"}
                missing_fields = required_fields - set(op.keys())
                if missing_fields:
                    raise ValueError(f"UPDATE operation {i} missing fields: {list(missing_fields)}")
            elif op_type == "MERGE":
                required_fields = {"type", "source_ids", "section", "content"}
                missing_fields = required_fields - set(op.keys())
                if missing_fields:
                    raise ValueError(f"MERGE operation {i} missing fields: {list(missing_fields)}")
                if len(op.get("source_ids", [])) < 2:
                    raise ValueError(f"MERGE operation {i} must include at least two source_ids")
            elif op_type == "ARCHIVE":
                required_fields = {"type", "bullet_id", "reason"}
                missing_fields = required_fields - set(op.keys())
                if missing_fields:
                    raise ValueError(
                        f"ARCHIVE operation {i} missing fields: {list(missing_fields)}"
                    )

        return operations_info
