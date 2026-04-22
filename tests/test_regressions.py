"""Regression tests for evaluation, caching, and offline training contracts."""
import json
import os
import re
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/Users/jlocehok/ace_experiment/ace")

from ace import ACE
from ace.core.bulletpoint_analyzer import BulletpointAnalyzer
from ace.core.curator import Curator
from logger import log_curator_failure
import ace.core.curator as curator_module
from playbook_utils import (
    apply_curator_operations,
    cleanup_playbook_references,
    format_playbook_line,
    parse_playbook_line,
    update_bullet_counts,
    validate_curator_reference_integrity,
)
from utils import evaluate_test_set, EVAL_ERROR_SENTINEL, get_section_slug


def make_playbook() -> str:
    return "\n".join(
        [
            "## OTHERS",
            format_playbook_line("a-00001", 0, 0, "alpha"),
            format_playbook_line("b-00001", 0, 0, "beta"),
            format_playbook_line("c-00001", 0, 0, "gamma"),
        ]
    )


class ExactMatchProcessor:
    task_name = "dummy"

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        return predicted == ground_truth

    def evaluate_accuracy(self, predictions, ground_truths) -> float:
        if not ground_truths:
            return 0.0
        correct = sum(
            1 for predicted, truth in zip(predictions, ground_truths)
            if self.answer_is_correct(predicted, truth)
        )
        return correct / len(ground_truths)


class MixedGenerator:
    model = "mixed-generator"

    def generate(self, *, question, **kwargs):
        if question == "explode":
            raise RuntimeError("boom")
        return json.dumps({"final_answer": question}), [], [], {}


class RoundTripGenerator:
    model = "roundtrip-generator"

    def generate(self, *, call_id, **kwargs):
        if call_id.endswith("_gen_initial"):
            return json.dumps({"final_answer": "wrong"}), ["a-00001"], ["a-00001"], {}
        if "_post_reflect_round_0" in call_id:
            return json.dumps({"final_answer": "still-wrong"}), ["b-00001"], [], {}
        if "_post_reflect_round_1" in call_id:
            return json.dumps({"final_answer": "target"}), ["c-00001"], ["c-00001"], {}
        if call_id.endswith("_post_curate"):
            return json.dumps({"final_answer": "target"}), [], [], {}
        raise AssertionError(f"Unexpected generator call: {call_id}")


class RoundTripReflector:
    model = "roundtrip-reflector"

    def reflect(self, *, call_id, **kwargs):
        if "_round_0" in call_id:
            return "reflection 0", [{"id": "a-00001", "tag": "helpful"}], {}
        if "_round_1" in call_id:
            return "reflection 1", [{"id": "b-00001", "tag": "harmful"}], {}
        if call_id.endswith("_reflect_on_correct"):
            return "reflection correct", [], {}
        raise AssertionError(f"Unexpected reflector call: {call_id}")


def make_fake_ace(playbook: str) -> ACE:
    ace = ACE.__new__(ACE)
    ace.generator = RoundTripGenerator()
    ace.reflector = RoundTripReflector()
    ace.curator = SimpleNamespace(
        curate=lambda **kwargs: (_ for _ in ()).throw(AssertionError("curator not expected"))
    )
    ace.playbook = playbook
    ace.best_playbook = playbook
    ace.next_global_id = 1
    ace.max_tokens = 128
    ace.use_bulletpoint_analyzer = False
    ace.bulletpoint_analyzer = None
    ace.bulletpoint_analyzer_threshold = 0.9
    return ace


def test_evaluate_test_set_counts_failures() -> None:
    processor = ExactMatchProcessor()
    samples = [
        {"question": "ok", "context": "", "target": "ok"},
        {"question": "explode", "context": "", "target": "bad"},
    ]
    results, error_log = evaluate_test_set(
        processor,
        MixedGenerator(),
        "",
        samples,
        max_workers=1,
        use_json_mode=False,
    )
    assert results["total"] == 2
    assert results["correct"] == 1
    assert abs(results["accuracy"] - 0.5) < 1e-9
    assert len(error_log["sample_failures"]) == 1
    assert error_log["sample_failures"][0]["prediction"] == EVAL_ERROR_SENTINEL
    print("case 1 (evaluate_test_set mixed success/failure): OK")


def test_evaluate_test_set_all_failures_is_safe() -> None:
    processor = ExactMatchProcessor()
    samples = [{"question": "explode", "context": "", "target": "bad"}]
    results, error_log = evaluate_test_set(
        processor,
        MixedGenerator(),
        "",
        samples,
        max_workers=1,
        use_json_mode=False,
    )
    assert results["total"] == 1
    assert results["correct"] == 0
    assert results["accuracy"] == 0.0
    assert len(error_log["sample_failures"]) == 1
    print("case 2 (evaluate_test_set all failures): OK")


def test_cache_key_uses_dataset_fingerprint() -> None:
    ace = ACE.__new__(ACE)
    ace.generator = SimpleNamespace(model="cache-model")
    key_a, _ = ace._build_test_cache_key("pb", [{"question": "a"}], False)
    key_b, _ = ace._build_test_cache_key("pb", [{"question": "b"}], False)
    assert key_a != key_b
    print("case 3 (cache key uses dataset fingerprint): OK")


def test_log_curator_failure_accepts_none_path() -> None:
    log_curator_failure(None, 1, "type", "response", 0)
    print("case 4 (log_curator_failure tolerates None path): OK")


def test_api_analyzer_mode_runs_without_local_deps() -> None:
    analyzer = BulletpointAnalyzer(
        client=SimpleNamespace(),
        model="dummy",
        embedding_model_name="api:test-model",
    )
    called = {"compute": False}

    def fake_compute(bullets):
        called["compute"] = True
        return np.eye(len(bullets), dtype=np.float32)

    analyzer._compute_embeddings = fake_compute
    analyzer._find_similar_groups = lambda bullets, embeddings, threshold: []

    playbook = "\n".join(
        [
            "## OTHERS",
            format_playbook_line("misc-00001", 0, 0, "foo"),
            format_playbook_line("misc-00002", 0, 0, "bar"),
        ]
    )
    out = analyzer.analyze(playbook)
    assert called["compute"], "API embedding mode should not early-return before _compute_embeddings"
    assert out == playbook
    print("case 5 (api analyzer mode bypasses local dependency gate): OK")


def test_vis_banking_section_slugs_match_seed_prefixes() -> None:
    assert get_section_slug("Формат ответа") == "фмт"
    assert get_section_slug("Эскалация") == "эск"
    assert get_section_slug("Разграничения") == "разгр"
    print("case 6 (vis banking seed slugs match runtime): OK")


def test_batched_round_evidence_matches_sequential_updates() -> None:
    processor = ExactMatchProcessor()
    sample = {"question": "q", "context": "ctx", "target": "target"}
    config_params = {
        "max_num_rounds": 2,
        "curator_frequency": 99,
        "token_budget": 100,
        "use_json_mode": False,
        "no_ground_truth": False,
        "prune_max_active_bullets_per_section": 40,
        "prune_warmup_window": 50,
        "prune_min_observations": 3,
    }
    playbook = make_playbook()

    with tempfile.TemporaryDirectory() as tmpdir:
        usage_log_path = os.path.join(tmpdir, "usage.jsonl")
        sequential_ace = make_fake_ace(playbook)
        sequential_ace._train_single_sample(
            task_dict=sample,
            data_processor=processor,
            step_id="train_e_1_s_1",
            epoch=1,
            step=1,
            usage_log_path=usage_log_path,
            log_dir=tmpdir,
            config_params=config_params,
            total_samples=1,
        )

        batched_ace = make_fake_ace(playbook)
        phase1 = batched_ace._train_sample_phase1(
            sample,
            batched_ace.playbook,
            processor,
            "train_e_1_s_1",
            tmpdir,
            config_params,
            step=1,
        )
        for evidence in phase1["round_evidence"]:
            batched_ace.playbook = update_bullet_counts(
                batched_ace.playbook,
                evidence["bullet_tags"],
                considered_bullet_ids=evidence["considered_ids"],
                used_bullet_ids=evidence["used_ids"],
                current_step=1,
            )

        assert sequential_ace.playbook == batched_ace.playbook
        print("case 7 (batched round evidence matches sequential updates): OK")


def test_offline_train_without_val_samples() -> None:
    processor = ExactMatchProcessor()
    sample = {"question": "q", "context": "ctx", "target": "target"}
    playbook = make_playbook()

    with tempfile.TemporaryDirectory() as tmpdir:
        usage_log_path = os.path.join(tmpdir, "usage.jsonl")
        playbook_dir = os.path.join(tmpdir, "playbooks")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(playbook_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        ace = make_fake_ace(playbook)
        result = ace._offline_train(
            train_samples=[sample],
            val_samples=[],
            data_processor=processor,
            config={
                "num_epochs": 1,
                "eval_steps": 1,
                "save_steps": 10,
                "test_workers": 1,
                "json_mode": False,
                "curator_frequency": 99,
                "batch_size": 1,
                "task_name": "dummy",
                "save_dir": tmpdir,
                "max_num_rounds": 2,
                "playbook_token_budget": 100,
                "no_ground_truth": False,
            },
            save_path=tmpdir,
            usage_log_path=usage_log_path,
            playbook_dir=playbook_dir,
            log_dir=log_dir,
        )

        assert result["best_validation_accuracy"] == 0.0
        print("case 8 (offline training with empty val set): OK")


def test_curator_content_prefixes_are_sanitized() -> None:
    merged_prefix = (
        "[legacy-00999] helpful=1 harmful=0 neutral=0 created_step=0 "
        "last_considered_step=0 last_used_step=0 times_considered_not_used=0 "
        "status=candidate :: merged body with [c-00001] reference"
    )
    updated_playbook, next_id, _ = apply_curator_operations(
        make_playbook(),
        [
            {
                "type": "UPDATE",
                "bullet_id": "a-00001",
                "content": "[a-00001] :: rewritten body with [b-00001] reference",
            },
            {
                "type": "ADD",
                "section": "OTHERS",
                "content": "[ghost-12345] :: new body with [a-00001] reference",
            },
            {
                "type": "MERGE",
                "source_ids": ["b-00001", "c-00001"],
                "section": "OTHERS",
                "content": merged_prefix,
            },
        ],
        next_id=2,
        current_step=10,
    )

    parsed_bullets = []
    for line in updated_playbook.splitlines():
        parsed = parse_playbook_line(line)
        if parsed:
            parsed_bullets.append(parsed)

    contents_by_id = {bullet["id"]: bullet["content"] for bullet in parsed_bullets}
    assert contents_by_id["a-00001"] == "rewritten body with [b-00001] reference"
    assert "new body with [a-00001] reference" in contents_by_id.values()
    assert "merged body with [c-00001] reference" in contents_by_id.values()
    assert all(not re.match(r"^\[[^\]]+\]\s*::", content) for content in contents_by_id.values())
    assert next_id == 4
    print("case 9 (curator content prefixes are sanitized): OK")


def test_cleanup_playbook_references_strips_invalid_and_malformed_refs() -> None:
    playbook = "\n".join(
        [
            "## OTHERS",
            format_playbook_line(
                "a-00001",
                0,
                0,
                "Keep [c-00001], drop [z-99999], [b-00001], and [x-00001, y-00002].",
            ),
            format_playbook_line("b-00001", 0, 0, "archived source", status="archived"),
            format_playbook_line("c-00001", 0, 0, "active source"),
        ]
    )
    cleaned_playbook, cleaned_bullets = cleanup_playbook_references(playbook)
    parsed = {
        bullet["id"]: bullet["content"]
        for bullet in (parse_playbook_line(line) for line in cleaned_playbook.splitlines())
        if bullet
    }
    assert parsed["a-00001"] == "Keep [c-00001], drop and."
    assert cleaned_bullets == [
        {
            "bullet_id": "a-00001",
            "removed_refs": ["b-00001", "z-99999"],
            "malformed_refs": ["x-00001, y-00002"],
        }
    ]
    print("case 10 (cleanup strips invalid and malformed refs): OK")


def test_reference_validator_blocks_new_refs_and_archive_dependencies() -> None:
    current_playbook = "\n".join(
        [
            "## OTHERS",
            format_playbook_line("a-00001", 0, 0, "Use [b-00001] before answering."),
            format_playbook_line("b-00001", 0, 0, "Debt rule."),
            format_playbook_line("c-00001", 0, 0, "Fallback rule."),
        ]
    )
    provisional_playbook, _, _ = apply_curator_operations(
        current_playbook,
        [
            {
                "type": "UPDATE",
                "bullet_id": "c-00001",
                "content": "Still cite [b-00001] here.",
            },
            {
                "type": "ARCHIVE",
                "bullet_id": "b-00001",
                "reason": "stale",
            },
        ],
        next_id=2,
        current_step=5,
    )
    errors = validate_curator_reference_integrity(
        current_playbook,
        provisional_playbook,
        [
            {
                "type": "UPDATE",
                "bullet_id": "c-00001",
                "content": "Still cite [b-00001] here.",
            },
            {
                "type": "ARCHIVE",
                "bullet_id": "b-00001",
                "reason": "stale",
            },
        ],
    )
    error_types = {error["type"] for error in errors}
    assert "internal_reference" in error_types
    assert "dangling_reference_after_archive" in error_types
    print("case 11 (validator blocks refs and archive dependencies): OK")


def test_curator_repairs_reference_validation_failures_once() -> None:
    playbook = "\n".join(
        [
            "## OTHERS",
            format_playbook_line("a-00001", 0, 0, "Existing active rule."),
            format_playbook_line("b-00001", 0, 0, "Dependency rule."),
        ]
    )
    # First response: curator returns an UPDATE that references [b-00001] (invalid).
    # Second response: targeted rephrase returns clean plain-text bullet.
    curator_response = json.dumps(
        {
            "reasoning": "first try",
            "operations": [
                {
                    "type": "UPDATE",
                    "bullet_id": "a-00001",
                    "content": "Use [b-00001] when debt is mentioned.",
                }
            ],
        },
        ensure_ascii=False,
    )
    rephrase_response = "При упоминании долга сразу применяй правило задолженности напрямую."

    calls: list[dict] = []

    def fake_timed_llm_call(*args, **kwargs):
        calls.append({"prompt": args[3], "call_id": kwargs["call_id"]})
        if len(calls) == 1:
            return curator_response, {"call_id": kwargs["call_id"]}
        return rephrase_response, {"call_id": kwargs["call_id"]}

    original_timed_llm_call = curator_module.timed_llm_call
    curator_module.timed_llm_call = fake_timed_llm_call
    try:
        curator = Curator(
            api_client=None,
            api_provider="dummy",
            model="dummy-curator",
            max_tokens=512,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "detailed_llm_logs")
            os.makedirs(log_dir, exist_ok=True)
            updated_playbook, _, operations, call_info = curator.curate(
                current_playbook=playbook,
                recent_reflection="reflection",
                question_context="question",
                current_step=7,
                total_samples=10,
                token_budget=100,
                playbook_stats={"total_bullets": 2},
                use_ground_truth=True,
                use_json_mode=False,
                call_id="repair_test",
                log_dir=log_dir,
                next_global_id=3,
            )

            summary_path = os.path.join(tmpdir, "curator_repair_summary.jsonl")
            counters_path = os.path.join(tmpdir, "curator_repair_counters.json")
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_entries = [json.loads(line) for line in f]
            with open(counters_path, "r", encoding="utf-8") as f:
                counters = json.load(f)
    finally:
        curator_module.timed_llm_call = original_timed_llm_call

    parsed = {
        bullet["id"]: bullet["content"]
        for bullet in (parse_playbook_line(line) for line in updated_playbook.splitlines())
        if bullet
    }
    assert len(calls) == 2, f"Expected 2 LLM calls (curator + targeted rephrase), got {len(calls)}"
    assert "targeted_repair" in calls[1]["call_id"]
    assert "Перефразируй bullet" in calls[1]["prompt"]
    assert "b-00001" in calls[1]["prompt"], "Rephrase prompt must include referenced bullet content"
    assert parsed["a-00001"] == rephrase_response
    assert operations[0]["content"] == parsed["a-00001"]
    assert "repair" in call_info
    assert summary_entries == [
        {
            "call_id": "repair_test",
            "step": 7,
            "repair_triggered": True,
            "repair_succeeded": True,
            "repair_failed": False,
        }
    ]
    assert counters == {
        "repair_triggered": 1,
        "repair_succeeded": 1,
        "repair_failed": 0,
    }
    print("case 12 (curator targeted repair fixes refs): OK")


if __name__ == "__main__":
    test_evaluate_test_set_counts_failures()
    test_evaluate_test_set_all_failures_is_safe()
    test_cache_key_uses_dataset_fingerprint()
    test_log_curator_failure_accepts_none_path()
    test_api_analyzer_mode_runs_without_local_deps()
    test_vis_banking_section_slugs_match_seed_prefixes()
    test_batched_round_evidence_matches_sequential_updates()
    test_offline_train_without_val_samples()
    test_curator_content_prefixes_are_sanitized()
    test_cleanup_playbook_references_strips_invalid_and_malformed_refs()
    test_reference_validator_blocks_new_refs_and_archive_dependencies()
    test_curator_repairs_reference_validation_failures_once()
    print("\nALL REGRESSION TESTS PASS")
