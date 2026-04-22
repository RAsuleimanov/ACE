#!/usr/bin/env python3
"""ACE entrypoint for the ViS banking-doc routing task.

Mirrors eval/finance/run.py; only data_processor + default config path change.
Run from the ace/ repo root, e.g.:

    uv run python -m eval.vis_banking.run \
        --task_name vis_banking \
        --mode eval_only \
        --api_provider openrouter \
        --generator_model google/gemma-4-31b-it \
        --reflector_model google/gemma-4-31b-it \
        --curator_model google/gemma-4-31b-it \
        --initial_playbook_path eval/vis_banking/data/seed_playbook.md \
        --save_path ../results/seed_baseline
"""
import os
import json
import argparse

import yaml

from .data_processor import DataProcessor, load_data
from .token_summary import latest_run_dir, write_summary, format_summary

from ace import ACE


def _load_run_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f) or {}
        return json.load(f)


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None,
                     help="YAML/JSON with run params (CLI flags override).")
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        description="ACE — ViS banking doc routing",
        parents=[pre],
    )
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--initial_playbook_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"])

    parser.add_argument("--api_provider", type=str, default="openrouter",
                        choices=["sambanova", "together", "openai", "commonstack", "openrouter"])
    parser.add_argument("--generator_model", type=str, default="google/gemma-4-31b-it")
    parser.add_argument("--reflector_model", type=str, default="google/gemma-4-31b-it")
    parser.add_argument("--curator_model", type=str, default="google/gemma-4-31b-it")

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=2)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--save_steps", type=int, default=50)

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=100000)
    parser.add_argument("--test_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Parallel sample batch in offline training (1 = sequential, paper-default).")

    # Deterministic lifecycle prune (fork §3.2 grow-and-refine)
    parser.add_argument("--prune_max_active_bullets_per_section", type=int, default=40,
                        help="Section cap. When active bullets in a section exceed this, "
                             "lowest-scoring are archived.")
    parser.add_argument("--prune_warmup_window", type=int, default=50,
                        help="Steps a fresh bullet gets to prove itself before eligible for 'never-used' archive.")
    parser.add_argument("--prune_min_observations", type=int, default=3,
                        help="Minimum (helpful+harmful+neutral) before 'harmful_dominates' archive can fire.")

    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")

    parser.add_argument("--use_bulletpoint_analyzer", action="store_true")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90)
    parser.add_argument("--bulletpoint_analyzer_bm25_threshold", type=float, default=0.0,
                        help="If >0, merge requires both dense cosine AND normalised BM25 "
                             "to clear their thresholds. Protects doc-ID-level distinctions.")
    parser.add_argument("--bulletpoint_analyzer_block_cross_section", type=lambda x: str(x).lower() != 'false',
                        default=True,
                        help="If true (default), bullets from different sections (e.g. инст vs то) never "
                             "merge — their semantic roles differ. Pass 'false' to disable.")
    parser.add_argument("--bulletpoint_analyzer_model", type=str, default='BAAI/bge-m3',
                        help="Sentence-transformers model for embedding. 'intfloat/multilingual-e5-small' "
                             "is ~5x lighter with comparable Russian quality.")

    # OpenRouter-style reasoning config for thinking models (e.g. Kimi K2.5).
    # Accept as JSON string from CLI; yaml can pass dict directly via set_defaults.
    parser.add_argument("--reflector_reasoning", type=json.loads, default=None,
                        help='JSON dict, e.g. \'{"enabled": false}\' to disable Kimi thinking.')
    parser.add_argument("--curator_reasoning", type=json.loads, default=None,
                        help='JSON dict, e.g. \'{"effort": "low"}\' or \'{"enabled": false}\'.')
    parser.add_argument("--analyzer_reasoning", type=json.loads, default=None,
                        help='Reasoning config for BulletpointAnalyzer merge LLM calls.')

    parser.add_argument("--config_path", type=str,
                        default="./eval/vis_banking/data/task_config.json")
    parser.add_argument("--save_path", type=str)

    parser.add_argument("--resume_from_playbook", type=str, default=None,
                        help="Path to intermediate_playbooks/epoch_E_step_N_playbook.txt to resume from. "
                             "Sets initial_playbook_path; pair with --skip_first_train_samples N matching "
                             "the step in the filename.")
    parser.add_argument("--skip_first_train_samples", type=int, default=0,
                        help="Skip first N train samples (use with --resume_from_playbook). "
                             "Step counter offsets so save/eval boundaries match the original run.")
    parser.add_argument("--resume_epoch", type=int, default=1,
                        help="Starting epoch number for naming + step-counter offset. "
                             "Use when resuming mid-training so filenames (epoch_N_step_M_playbook.txt) "
                             "reflect the absolute epoch, not the per-process loop counter.")

    if pre_args.config:
        cfg = _load_run_config(pre_args.config)
        unknown = set(cfg) - {a.dest for a in parser._actions}
        if unknown:
            parser.error(f"Unknown keys in {pre_args.config}: {sorted(unknown)}")
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    for required in ("task_name", "save_path"):
        if not getattr(args, required):
            parser.error(f"--{required} is required (set via CLI or --config)")

    if args.resume_from_playbook:
        # Resume always overrides any initial_playbook_path from config/CLI
        args.initial_playbook_path = args.resume_from_playbook
    if args.resume_from_playbook and args.skip_first_train_samples == 0:
        import re as _re
        m = _re.search(r"epoch_\d+_step_(\d+)_playbook", args.resume_from_playbook)
        if m:
            inferred = int(m.group(1))
            print(f"Resume: inferred --skip_first_train_samples={inferred} from filename")
            args.skip_first_train_samples = inferred
    # Auto-detect resume_epoch if not set explicitly.
    # - epoch_E_final_playbook     → epoch E fully done → resume into epoch E+1
    # - epoch_E_step_N_playbook    → mid-epoch E → continue epoch E (same number)
    if args.resume_from_playbook and args.resume_epoch == 1:
        import re as _re
        m_final = _re.search(r"epoch_(\d+)_final_playbook", args.resume_from_playbook)
        m_step = _re.search(r"epoch_(\d+)_step_\d+_playbook", args.resume_from_playbook)
        if m_final:
            prior = int(m_final.group(1))
            args.resume_epoch = prior + 1
            print(f"Resume: inferred --resume_epoch={args.resume_epoch} from filename (epoch {prior} final → next is {args.resume_epoch})")
        elif m_step:
            prior = int(m_step.group(1))
            args.resume_epoch = prior
            print(f"Resume: inferred --resume_epoch={prior} from filename (mid-epoch checkpoint → continue epoch {prior})")

    return args


def preprocess_data(task_name, config, mode):
    processor = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        if "test_data" not in config:
            raise ValueError(f"{mode} mode requires test_data in config.")
        test_samples = processor.process_task_data(load_data(config["test_data"]))
        print(f"{mode} mode: {len(test_samples)} examples")
    else:
        train_samples = processor.process_task_data(load_data(config["train_data"]))
        val_samples = processor.process_task_data(load_data(config["val_data"]))
        test_samples = (
            processor.process_task_data(load_data(config["test_data"]))
            if "test_data" in config else []
        )
        print(f"Offline mode: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def main():
    args = parse_args()

    print(f"\n{'='*60}\nACE — ViS banking\n{'='*60}")
    print(f"Task: {args.task_name}  Mode: {args.mode}")
    print(f"Provider: {args.api_provider}")
    print(f"Generator: {args.generator_model}")
    print(f"Reflector: {args.reflector_model}")
    print(f"Curator:   {args.curator_model}\n")

    with open(args.config_path, "r", encoding="utf-8") as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name, task_config[args.task_name], args.mode
    )

    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook\n")

    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
        bulletpoint_analyzer_bm25_threshold=args.bulletpoint_analyzer_bm25_threshold,
        bulletpoint_analyzer_block_cross_section=args.bulletpoint_analyzer_block_cross_section,
        bulletpoint_analyzer_model=args.bulletpoint_analyzer_model,
        reflector_reasoning=args.reflector_reasoning,
        curator_reasoning=args.curator_reasoning,
        analyzer_reasoning=args.analyzer_reasoning,
    )

    config = {
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "online_eval_frequency": args.online_eval_frequency,
        "save_steps": args.save_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "batch_size": args.batch_size,
        "skip_first_train_samples": args.skip_first_train_samples,
        "resume_epoch": args.resume_epoch,
        "prune_max_active_bullets_per_section": args.prune_max_active_bullets_per_section,
        "prune_warmup_window": args.prune_warmup_window,
        "prune_min_observations": args.prune_min_observations,
        "task_name": args.task_name,
        "mode": args.mode,
        "json_mode": args.json_mode,
        "no_ground_truth": args.no_ground_truth,
        "save_dir": args.save_path,
        "test_workers": args.test_workers,
        "initial_playbook_path": args.initial_playbook_path,
        "use_bulletpoint_analyzer": args.use_bulletpoint_analyzer,
        "bulletpoint_analyzer_threshold": args.bulletpoint_analyzer_threshold,
        "bulletpoint_analyzer_bm25_threshold": args.bulletpoint_analyzer_bm25_threshold,
        "bulletpoint_analyzer_block_cross_section": args.bulletpoint_analyzer_block_cross_section,
        "api_provider": args.api_provider,
    }

    try:
        ace_system.run(
            mode=args.mode,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    finally:
        # Always write token_usage.json — even on KeyboardInterrupt or crash —
        # since llm_calls.jsonl is line-buffered and flushed per call.
        run_dir = latest_run_dir(args.save_path, args.task_name, args.mode)
        if run_dir:
            summary = write_summary(run_dir)
            print(f"\n{'='*60}\nTOKEN USAGE\n{'='*60}")
            print(format_summary(summary))
            print(f"\nSaved to {os.path.join(run_dir, 'token_usage.json')}")


if __name__ == "__main__":
    main()
