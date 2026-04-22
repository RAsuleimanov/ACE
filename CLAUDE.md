# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Fork of `ace-agent/ace` (the Agentic Context Engineering paper implementation, arXiv:2510.04618) with a Level-2 lifecycle port from `DannyMac180/ace-platform` plus local patches for the ViS Russian banking-routing task.

Active branch: `port/lifecycle-level2` — do **not** merge to `main`, which tracks upstream. Stable checkpoint is `local-patches`.

Parent directory `/Users/jlocehok/ace_experiment/` contains the shared experiment spec (`CLAUDE.md` + `TASK.md` + shared `seed_playbook.md` used by both ACE and GEPA arms). Read that **before** touching anything that looks shared — see `feedback_scan_context_first` memory.

## Commands

Install + env:
```bash
uv sync                              # installs from pyproject.toml + uv.lock
cp .env.example .env                 # then edit to add OPENROUTER_API_KEY
```

Run offline training (ViS task):
```bash
PYTHONUNBUFFERED=1 python -m eval.vis_banking.run \
    --config eval/vis_banking/configs/gemma_kimi_offline.yaml \
    2>&1 | tee /Users/jlocehok/ace_experiment/results/ace_offline_clean.log
```

Resume mid-epoch (auto-detects `resume_epoch` + `skip_first_train_samples` from filename; pass `--skip_first_train_samples N` to force):
```bash
python -m eval.vis_banking.run \
    --config eval/vis_banking/configs/gemma_kimi_offline.yaml \
    --resume_from_playbook <path>/epoch_1_step_170_playbook.txt \
    --skip_first_train_samples 170
```

Tests (no LLM calls):
```bash
python tests/test_prune.py            # deterministic prune sanity
python tests/test_analyzer_live.py    # BGE-M3 + BM25 + B4 section gate
```

## Architecture (high-level)

### Three-role agentic loop

Each training sample goes through:
1. **Generator** (`ace/core/generator.py`) reads the playbook + question + context, emits `{reasoning, considered_bullet_ids, used_bullet_ids, final_answer}`. The considered/used split was added in the Level-2 port to fix attribution hallucination.
2. **Reflector** (`ace/core/reflector.py`) analyzes the generator's reasoning trace + ground truth, returns per-bullet tags (`helpful`/`harmful`/`neutral`).
3. **Curator** (`ace/core/curator.py`) reads playbook + aggregated reflections, emits lifecycle operations: `ADD`/`UPDATE`/`MERGE`/`ARCHIVE`.

### Critical orchestration detail (batched path)

`ace/ace.py::_offline_train_batched` runs a batch of N samples through Phase 1 (gen + reflect in parallel) → Phase 2a (apply bullet counters sequentially) → Phase 2b (**single curator call on aggregated reflections**, not N parallel curators — earlier version caused 48% duplicate UPDATEs) → Phase 3 (parallel post-curate gen).

### Playbook format + lifecycle

Every bullet line in a playbook `.txt` follows this exact format (extracted/formatted by `playbook_utils.py`):
```
[<prefix>-<5-digit-id>] helpful=N harmful=N neutral=N created_step=N last_considered_step=N last_used_step=N times_considered_not_used=N status=(active|candidate|archived) :: <content>
```
- Three statuses: `candidate` (new), `active` (promoted after use or helpful tag), `archived` (soft-delete; filtered from generator prompt but kept in file).
- Sections are discovered dynamically from `## Header` lines — **no built-in semantic ontology**. To teach curator where to place things, either seed representative bullets per section or describe section purposes in the curator prompt (not done by default in fork).

### Rendering

- `render_minimal_playbook(text)` → `[id] :: content` only (no metadata). **Used for generator prompts** to avoid attention pollution from counters.
- `render_active_playbook(text)` → full metadata, filters archived. Used for curator/reflector who need the counts.
- Raw `self.playbook` (with archived + metadata) is what gets saved to disk for audit.

### Prune (deterministic, in `playbook_utils.py::prune_playbook`)

Runs after every curator batch. Three rules:
- `never_used` — `last_used_step == 0 AND age >= warmup_window` → archive
- `harmful_dominates` — `observations >= min_observations AND harmful > helpful AND harmful >= max(1, neutral)` → archive
- `section_cap` — sort by `bullet_score`, archive anything below the top `max_active_bullets_per_section`

`bullet_score = helpful*2 - harmful*3 - neutral - times_considered_not_used + active_bonus(0.5) + last_used_step*0.01 + last_considered_step*0.001`

### Step counter is monotonic across epochs

`ace/ace.py` adds `epoch_offset = (epoch - 1) * original_total` to all `batch_steps` so `last_used_step` can't regress when the inner loop wraps. `update_bullet_counts` uses `max(existing, current_step)` as a belt-and-suspenders guard. `resume_epoch` auto-detects from filename: `epoch_E_final_playbook` → E+1; `epoch_E_step_N_playbook` → E.

### OpenRouter integration (`llm.py`)

- Model-aware provider routing in `timed_llm_call`: blacklists slow/expensive providers per model (Together/AkashML for Gemma; ModelRun/Venice/Novita for Kimi), sets `max_price` ceiling + `preferred_max_latency` p90.
- `reasoning` config (dict) is passed via `extra_body` — e.g. `{"enabled": false}` is the only reliable off-switch for Kimi K2.5 reasoning. `{"exclude": true}` still bills. `{"max_tokens": 0}` is silently ignored.
- Retry: 5× exp-backoff on empty responses, timeouts, 5xx, rate limits, JSON decode errors. Graceful degradation path in the embedding case (skips analyzer pass rather than killing the run).

### BulletpointAnalyzer

Deduplicates near-duplicate bullets via semantic similarity + BM25 AND-gate. Embedder is configurable via `bulletpoint_analyzer_model`:
- `BAAI/bge-m3` (local via sentence-transformers, ~1.5-2 GB MPS)
- `api:baai/bge-m3` (OpenRouter embeddings endpoint, ~$0.01/full run, zero local memory — preferred default)

Threshold `0.82` is empirically calibrated for BGE-M3 on this task's Russian bullets. Different embedders need different thresholds — re-measure against real bullets before changing the model.

### ViS banking task specifics (`eval/vis_banking/`)

- 29 classes: 26 doc-IDs + `FAQ` + `ОПЕРАТОР` + `CLIENT_CLARIFY`
- 275 train / 138 val
- Severe class imbalance: FAQ+ОПЕРАТОР = 46.5% of train; 10 classes have ≤2 samples
- Target matching is **regex-on-leading-number**: target `^1(?:\.|$)` matches output "1. Выписка по счёту" OR just "1". Doc name doesn't matter for grading.
- Russian content — do not translate prompts to English in evolving bullets.

## Config (`eval/vis_banking/configs/gemma_kimi_offline.yaml`)

Key knobs and why they matter:
- `num_epochs: 3`, `batch_size: 5`, `curator_frequency: 1` — with aggregated curator, curator_frequency now means "run curator every N batches", not samples
- `max_num_rounds: 3` — gen→reflect retry iterations for wrong answers
- `eval_steps: 110` — val eval triggers at step%110==0; set to divisor of 275 (55/137/275) if you want end-of-epoch eval
- `prune_warmup_window: 275` — one full epoch of grace before never_used can fire
- `*_reasoning: null | {"enabled": false} | {"effort": "low"}` per role — see the inline yaml comments; `enabled: false` is the only real off-switch for Kimi

## Things that are easy to break

- **`seed_playbook.md` was once a symlink** to the parent's shared seed. Now a regular file. Do not re-symlink or `Write` through it without `ls -la` check.
- **`status=candidate` is treated as active** for rendering/scoring (`ACTIVE_BULLET_STATUSES = {"active", "candidate"}`) — section_cap counts both.
- **`considered_bullet_ids` extractor** (`ace/core/generator.py`) must handle Cyrillic IDs (`инст-`, `пд-`, `то-`, `эск-`, `разгр-`, `фмт-`, `введ-`). The regex uses `\w{2,}` with `re.UNICODE`. `[a-z]` is wrong and silently produces empty used_bullet_ids (symptom: all `last_used_step=0`).
- **Empty sections in seed don't coax curator to use them.** Curator sees section names from `## Header`s but gets no ontology — it defaults to whichever section already has content (usually Инструкции). To activate a section, seed at least one canonical example and/or add section descriptions to `ace/prompts/curator.py`.
