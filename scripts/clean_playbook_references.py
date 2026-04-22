#!/usr/bin/env python3
"""Deterministically clean bullet-ID references from an ACE playbook."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playbook_utils import cleanup_playbook_references


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source playbook")
    parser.add_argument("--output", required=True, help="Path to cleaned playbook")
    parser.add_argument(
        "--strip-all-active-refs",
        action="store_true",
        help="Strip all active/candidate bullet-ID refs, not only missing/archived/malformed ones",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    playbook_text = input_path.read_text(encoding="utf-8")
    cleaned_text, cleaned_bullets = cleanup_playbook_references(
        playbook_text,
        strip_all_active_refs=args.strip_all_active_refs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned_text, encoding="utf-8")

    print(f"Wrote: {output_path}")
    print(f"Updated bullets: {len(cleaned_bullets)}")
    for entry in cleaned_bullets[:20]:
        removed = ", ".join(entry["removed_refs"]) or "-"
        malformed = ", ".join(entry["malformed_refs"]) or "-"
        print(
            f"  {entry['bullet_id']}: removed_refs={removed}; malformed_refs={malformed}"
        )
    remaining = len(cleaned_bullets) - 20
    if remaining > 0:
        print(f"  ... and {remaining} more")


if __name__ == "__main__":
    main()
