"""Sanity tests for deterministic prune_playbook (no LLM)."""
import sys
sys.path.insert(0, '/Users/jlocehok/ace_experiment/ace')
from playbook_utils import prune_playbook, format_playbook_line


def bullet(id_, section, *, helpful=0, harmful=0, neutral=0, created_step=0,
           last_used_step=0, status='active', content='x'):
    """Format a bullet under given section with lifecycle metadata."""
    return format_playbook_line(
        id_, helpful, harmful, content,
        neutral=neutral, created_step=created_step,
        last_used_step=last_used_step, status=status,
    )


# ============ Case 1: never-used + warmup expired → archived ============
# NOTE: fork's prune treats created_step=0 as "no age info, skip archive".
# Must set created_step > 0 for never-used eligibility to fire.
pb = "## Test\n" + "\n".join([
    bullet('a-1', 'test', created_step=10, last_used_step=0),   # never-used, age=90 > warmup → archive
    bullet('a-2', 'test', helpful=1, created_step=10, last_used_step=50),  # used → protect
])
pruned, archived = prune_playbook(pb, current_step=100,
                                   max_active_bullets_per_section=100,
                                   warmup_window=10,
                                   min_observations=3)
assert 'a-1' in archived, f"a-1 never-used should be archived; got {archived}"
assert 'a-2' not in archived, f"a-2 was used, keep; got {archived}"
print(f"case 1 (never-used archive): {archived}  OK")


# ============ Case 2: warmup still active → NOT archived ============
pb = "## Test\n" + bullet('b-1', 'test', created_step=95, last_used_step=0)
pruned, archived = prune_playbook(pb, current_step=100,
                                   max_active_bullets_per_section=100,
                                   warmup_window=50,   # only 5 steps old, warmup=50 → protect
                                   min_observations=3)
assert archived == [], f"fresh bullet should not be archived during warmup; got {archived}"
print("case 2 (warmup respects age): OK")


# ============ Case 3: harmful_dominates → archive ============
pb = "## Test\n" + "\n".join([
    bullet('c-1', 'test', helpful=0, harmful=5, last_used_step=80),  # harmful dominates
    bullet('c-2', 'test', helpful=5, harmful=0, last_used_step=80),  # clearly helpful
])
pruned, archived = prune_playbook(pb, current_step=100,
                                   max_active_bullets_per_section=100,
                                   warmup_window=10,
                                   min_observations=3)
assert 'c-1' in archived, f"c-1 harmful-dominates should archive; got {archived}"
assert 'c-2' not in archived, f"c-2 clearly helpful should survive; got {archived}"
print(f"case 3 (harmful_dominates): {archived}  OK")


# ============ Case 4: section_cap trims lowest-score ============
# 5 bullets in same section, cap=3 → 2 lowest-score archived
pb = "## Test\n" + "\n".join([
    bullet('d-1', 'test', helpful=10, harmful=0, last_used_step=90),
    bullet('d-2', 'test', helpful=5, harmful=0, last_used_step=80),
    bullet('d-3', 'test', helpful=1, harmful=0, last_used_step=50),
    bullet('d-4', 'test', helpful=0, harmful=0, last_used_step=0, created_step=0),
    bullet('d-5', 'test', helpful=0, harmful=1, last_used_step=20),
])
pruned, archived = prune_playbook(pb, current_step=100,
                                   max_active_bullets_per_section=3,
                                   warmup_window=10,
                                   min_observations=3)
# Top 3 by bullet_score should survive: d-1, d-2, d-3
assert 'd-1' not in archived and 'd-2' not in archived and 'd-3' not in archived, \
    f"top-3 should survive cap; got {archived}"
print(f"case 4 (section_cap): {archived}  OK (expected d-4, d-5 and any never-used dropped)")


print("\nALL TESTS PASS")
