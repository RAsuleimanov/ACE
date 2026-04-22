"""ViS banking-document routing DataProcessor for ACE.

Scoring is byte-faithful to GEPA's vis_adapter (adapter.py:113):
    re.search(pattern, response, re.S | re.I)
where `pattern` is the per-example `additional_check` regex from the source
ScenarioData. Our JSONL stores that pattern as the `target` field
(see regenerate_jsonl.py).
"""
import os
import json
from typing import List, Dict, Any

from .label_parser import matches_pattern


def load_data(data_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


class DataProcessor:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        processed = []
        for item in raw_data:
            processed.append({
                "context": item.get("context", ""),
                "question": item.get("question", ""),
                "target": item.get("target", ""),
                "others": {
                    "label": item.get("label"),  # not used for matching — analysis only
                    "scenario_id": item.get("scenario_id"),
                    "scenario_description": item.get("scenario_description"),
                    "task": self.task_name,
                },
            })
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        return matches_pattern(predicted, ground_truth)

    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        if not ground_truths:
            return 0.0
        correct = sum(
            1 for p, g in zip(predictions, ground_truths) if self.answer_is_correct(p, g)
        )
        return correct / len(ground_truths)
