"""Aggregate per-role token usage from a run's llm_calls.jsonl.

Each line in llm_calls.jsonl is a record produced by logger.log_llm_call
containing role + prompt_num_tokens + response_num_tokens (see llm.py:92-106).
Failed/empty calls have no token counts — they're counted by `error_calls`.
"""
import glob
import json
import os
from collections import defaultdict


def latest_run_dir(save_path: str, task_name: str, mode: str) -> str | None:
    pattern = os.path.join(save_path, f"ace_run_*_{task_name}_{mode}")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def summarize_tokens(run_dir: str) -> dict:
    log_path = os.path.join(run_dir, "detailed_llm_logs", "llm_calls.jsonl")
    if not os.path.exists(log_path):
        return {}

    by_role: dict = defaultdict(lambda: {
        "calls": 0, "error_calls": 0,
        "input_tokens": 0, "output_tokens": 0,
        "total_seconds": 0.0,
    })
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = rec.get("role", "unknown")
            r = by_role[role]
            r["calls"] += 1
            if rec.get("error"):
                r["error_calls"] += 1
            r["input_tokens"] += rec.get("prompt_num_tokens") or 0
            r["output_tokens"] += rec.get("response_num_tokens") or 0
            r["total_seconds"] += rec.get("total_time") or 0.0

    total = {k: 0 for k in ("calls", "error_calls", "input_tokens", "output_tokens")}
    total["total_seconds"] = 0.0
    for r in by_role.values():
        for k in total:
            total[k] += r[k]
    return {"by_role": dict(by_role), "total": total}


def write_summary(run_dir: str) -> dict:
    summary = summarize_tokens(run_dir)
    if not summary:
        return {}
    out_path = os.path.join(run_dir, "token_usage.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def format_summary(summary: dict) -> str:
    if not summary:
        return "(no llm_calls.jsonl found)"
    lines = []
    lines.append(f"{'role':<12} {'calls':>6} {'errs':>5} {'in_tok':>10} {'out_tok':>10} {'sec':>8}")
    lines.append("-" * 56)
    for role, r in sorted(summary["by_role"].items()):
        lines.append(
            f"{role:<12} {r['calls']:>6} {r['error_calls']:>5} "
            f"{r['input_tokens']:>10} {r['output_tokens']:>10} "
            f"{r['total_seconds']:>8.1f}"
        )
    t = summary["total"]
    lines.append("-" * 56)
    lines.append(
        f"{'TOTAL':<12} {t['calls']:>6} {t['error_calls']:>5} "
        f"{t['input_tokens']:>10} {t['output_tokens']:>10} "
        f"{t['total_seconds']:>8.1f}"
    )
    return "\n".join(lines)
