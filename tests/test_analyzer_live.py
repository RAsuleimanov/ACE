"""Validate BGE-M3 + BM25 analyzer pipeline on real playbook bullets.

Runs _find_similar_groups under 3 configs and reports group diffs:
  1) new: BGE-M3 @ 0.93 + BM25 @ 0.1 (target prod config)
  2) dense-only: BGE-M3 @ 0.93
  3) old: all-mpnet-base-v2 @ 0.90 (previous prod baseline)

Does NOT call LLM merger — only checks which groups would be candidates.
"""
import sys
import os
sys.path.insert(0, '/Users/jlocehok/ace_experiment/ace')

import numpy as np
from ace.core.bulletpoint_analyzer import BulletpointAnalyzer

PB_PATH = "/Users/jlocehok/ace_experiment/results/ace_offline_gemma/ace_run_20260420_001747_vis_banking_offline/intermediate_playbooks/epoch_4_step_30_playbook.txt"

class FakeClient: pass


def load_bullets(path):
    from playbook_utils import parse_playbook_line
    bullets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().split("\n"):
            p = parse_playbook_line(line)
            if p:
                bullets.append(p)
    return bullets


def run_with(embedding_name, dense_threshold, bm25_threshold):
    a = BulletpointAnalyzer(
        FakeClient(), "dummy", max_tokens=1024,
        embedding_model_name=embedding_name,
        bm25_threshold=bm25_threshold,
    )
    a._load_embedding_model()
    embeddings = a._compute_embeddings(bullets)
    groups = a._find_similar_groups(bullets, embeddings, threshold=dense_threshold)
    return groups


bullets = load_bullets(PB_PATH)
print(f"bullets loaded: {len(bullets)}")
print(f"sections: {set(b['id'].split('-')[0] for b in bullets)}")

# Config 1: NEW (BGE-M3 dense@0.93 + BM25@0.1)
print("\n" + "="*60)
print("CONFIG 1 (NEW): BGE-M3 + BM25 AND-gate")
print("="*60)
groups_new = run_with('BAAI/bge-m3', 0.93, 0.1)
print(f"groups found: {len(groups_new)}")
for i, g in enumerate(groups_new):
    ids = [b['id'] for b in g['bullets']]
    contents = [b['content'][:80] for b in g['bullets']]
    print(f"  [{i+1}] {ids}")
    for c in contents:
        print(f"      • {c!r}")

# Config 2: BGE-M3 dense-only
print("\n" + "="*60)
print("CONFIG 2: BGE-M3 dense only @ 0.93 (no BM25)")
print("="*60)
groups_dense = run_with('BAAI/bge-m3', 0.93, 0.0)
print(f"groups found: {len(groups_dense)}")
for i, g in enumerate(groups_dense):
    ids = [b['id'] for b in g['bullets']]
    print(f"  [{i+1}] {ids}")

# Config 3: OLD baseline (all-mpnet-base-v2 @ 0.90, no BM25)
print("\n" + "="*60)
print("CONFIG 3 (OLD): all-mpnet-base-v2 @ 0.90")
print("="*60)
groups_old = run_with('all-mpnet-base-v2', 0.90, 0.0)
print(f"groups found: {len(groups_old)}")
for i, g in enumerate(groups_old):
    ids = [b['id'] for b in g['bullets']]
    print(f"  [{i+1}] {ids}")

# Diff analysis
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"OLD (mpnet @ 0.90):      {len(groups_old)} groups, {sum(len(g['bullets']) for g in groups_old)} bullets in groups")
print(f"Dense (BGE-M3 @ 0.93):   {len(groups_dense)} groups, {sum(len(g['bullets']) for g in groups_dense)} bullets in groups")
print(f"Dense+BM25 (NEW):        {len(groups_new)} groups, {sum(len(g['bullets']) for g in groups_new)} bullets in groups")

# Show groups blocked by BM25 (in dense-only but NOT in new)
dense_groups_set = {tuple(sorted(b['id'] for b in g['bullets'])) for g in groups_dense}
new_groups_set = {tuple(sorted(b['id'] for b in g['bullets'])) for g in groups_new}
blocked = dense_groups_set - new_groups_set
print(f"\nGroups BLOCKED by BM25 (dense says merge, lexical disagrees): {len(blocked)}")
for ids in list(blocked)[:10]:
    contents = [next(b['content'][:90] for b in bullets if b['id'] == bid) for bid in ids]
    print(f"  {list(ids)}")
    for c in contents:
        print(f"    • {c!r}")
