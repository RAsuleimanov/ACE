"""ViS scoring — byte-faithful to GEPA's vis_adapter (adapter.py:113-121).

Empty `pattern` → True (GEPA: "no check required").
Invalid regex → False (GEPA: catches re.error and marks failed).
"""
import re

_GEPA_FLAGS = re.S | re.I


def matches_pattern(prediction: str, pattern: str) -> bool:
    if not pattern:
        return True
    if not prediction:
        return False
    try:
        return bool(re.search(pattern, prediction, _GEPA_FLAGS))
    except re.error:
        return False
