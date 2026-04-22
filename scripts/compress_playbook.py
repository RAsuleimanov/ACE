#!/usr/bin/env python3
"""One-shot playbook compression.

Rewrites over-sized bullets into compact form while preserving every rule,
ID mapping, and keyword reference. Keeps bullet IDs and metadata counters
intact (compress-in-place) so accrued helpful/harmful history survives.

Usage:
    python scripts/compress_playbook.py \\
        --input results/.../epoch_1_step_180_playbook.txt \\
        --output results/.../epoch_1_step_180_playbook_compressed.txt \\
        --threshold 1500 --target-chars 400
"""

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playbook_utils import parse_playbook_line, format_parsed_playbook_line
from utils import initialize_clients
from llm import timed_llm_call


COMPRESS_PROMPT = """You are compressing an over-sized playbook bullet for a Russian-language banking document routing task.

RULES:
1. Preserve EVERY rule, keyword mapping, and ID reference. Do not drop any bullet reference (e.g. [инст-00066], [пд-00011]) or document ID (e.g. ID 9, ID 13, ID 20).
2. Remove decoration: ⭐, ⛔, emoji emphasis, repeated "КРИТИЧЕСКОЕ"/"АБСОЛЮТНЫЙ"/"ОБЯЗАТЕЛЬНОЕ", ALL-CAPS headers. These add length without meaning.
3. Remove redundant re-phrasings (same rule stated twice in different words).
4. Target: ≤ {target_chars} characters. Plain Russian, compact sentences.
5. Preserve logical structure: "если X → Y" stays as "если X → Y".
6. Do NOT add new rules not present in the original.

Original bullet content:
---
{content}
---

Respond with ONLY the compressed content as plain text. No JSON, no markdown, no code blocks, no commentary."""


ID_REF_RE = re.compile(r"\[[\w\-]+-\d{5}\]", re.UNICODE)
DOC_ID_RE = re.compile(r"ID\s+\d+")


def extract_required_tokens(text):
    return ID_REF_RE.findall(text), DOC_ID_RE.findall(text)


def compress_bullet(parsed, target_chars, client, api_provider, model, reasoning, log_dir):
    content = parsed["content"]
    orig_len = len(content)
    prompt = COMPRESS_PROMPT.format(target_chars=target_chars, content=content)

    try:
        resp, _ = timed_llm_call(
            client, api_provider, model, prompt,
            role="compressor",
            call_id=f"compress_{parsed['id']}",
            max_tokens=2048,
            log_dir=log_dir,
            use_json_mode=False,
            reasoning=reasoning,
        )
    except Exception as e:
        return parsed, f"ERROR: {e}", orig_len, orig_len

    new_content = (resp or "").strip()
    new_content = re.sub(r"^```[a-z]*\s*\n?", "", new_content)
    new_content = re.sub(r"\n?```\s*$", "", new_content)
    new_content = new_content.strip()

    if not new_content:
        return parsed, "ERROR: empty response", orig_len, 0

    orig_refs, orig_doc_ids = extract_required_tokens(content)
    new_refs, new_doc_ids = extract_required_tokens(new_content)
    missing_refs = set(orig_refs) - set(new_refs)
    missing_docs = set(orig_doc_ids) - set(new_doc_ids)
    new_len = len(new_content)

    if missing_refs or missing_docs:
        missing = sorted(missing_refs | missing_docs)[:6]
        return parsed, f"REJECTED: dropped {missing}", orig_len, new_len

    if new_len >= orig_len:
        return parsed, f"SKIPPED: no shrinkage ({orig_len}→{new_len})", orig_len, new_len

    new_parsed = dict(parsed)
    new_parsed["content"] = new_content
    return new_parsed, f"OK: {orig_len} → {new_len} (-{100*(1-new_len/orig_len):.0f}%)", orig_len, new_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=int, default=1500,
                    help="Compress bullets with content > this many chars")
    ap.add_argument("--target-chars", type=int, default=400)
    ap.add_argument("--model", default="moonshotai/kimi-k2.5")
    ap.add_argument("--api-provider", default="openrouter")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and exit without calling LLM")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text()
    lines = text.split("\n")

    parsed_by_idx = {}
    targets = []
    for i, line in enumerate(lines):
        p = parse_playbook_line(line)
        if not p:
            continue
        parsed_by_idx[i] = p
        if p.get("status") == "archived":
            continue
        if len(p["content"]) > args.threshold:
            targets.append((i, p))

    print(f"Playbook: {input_path}")
    print(f"Parsed bullets: {len(parsed_by_idx)}")
    print(f"Active bullets above threshold ({args.threshold} chars): {len(targets)}")

    if not targets:
        print("Nothing to compress.")
        return

    sorted_targets = sorted(targets, key=lambda x: -len(x[1]["content"]))
    total_in = sum(len(p["content"]) for _, p in targets)
    print(f"Total chars to process: {total_in} (~{total_in // 2} tokens)")
    print("\nTop 10 offenders:")
    for _, p in sorted_targets[:10]:
        print(f"  {p['id']} helpful={p.get('helpful',0):>3} status={p.get('status','?'):<9} → {len(p['content'])} chars")

    if args.dry_run:
        print("\n--dry-run: exiting without LLM calls.")
        return

    reasoning = {"enabled": False}
    gen_client, _, _ = initialize_clients(args.api_provider)
    print(f"\nModel: {args.model}, reasoning={reasoning}, workers={args.workers}")
    print()

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_target = {
            ex.submit(compress_bullet, p, args.target_chars, gen_client,
                      args.api_provider, args.model, reasoning, args.log_dir): (idx, p)
            for idx, p in targets
        }
        for fut in as_completed(fut_to_target):
            idx, p = fut_to_target[fut]
            new_parsed, status, _, _ = fut.result()
            results[idx] = (new_parsed, status)
            print(f"  [{p['id']}] {status}")

    new_lines = list(lines)
    applied = 0
    for idx, (new_parsed, status) in results.items():
        if status.startswith("OK"):
            new_lines[idx] = format_parsed_playbook_line(new_parsed)
            applied += 1

    output_path.write_text("\n".join(new_lines))

    before_total = sum(len(p["content"]) for p in parsed_by_idx.values())
    after_total = 0
    for idx, p in parsed_by_idx.items():
        if idx in results and results[idx][1].startswith("OK"):
            after_total += len(results[idx][0]["content"])
        else:
            after_total += len(p["content"])

    print(f"\nWrote: {output_path}")
    print(f"Applied: {applied} / {len(targets)}")
    print(f"Active content: {before_total} → {after_total} chars "
          f"({100*(1-after_total/max(before_total,1)):.1f}% reduction)")


if __name__ == "__main__":
    main()
