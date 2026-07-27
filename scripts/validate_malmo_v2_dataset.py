from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "malmo_v2"
FILES = {
    "train": DATA_DIR / "malmo_v2_train.jsonl",
    "validation": DATA_DIR / "malmo_v2_validation.jsonl",
    "test": DATA_DIR / "malmo_v2_test.jsonl",
}
EXPECTED_ROLES = ["system", "user", "assistant"]
NEGATIVE_PATTERNS = [
    r"\binte\b",
    r"\bingen\b",
    r"\binget\b",
    r"\binga\b",
    r"\baldrig\b",
    r"\bvägr(?:ar|ade|at)\b",
    r"\bslutat fungera\b",
    r"\bhar lagt av\b",
    r"\blagt av\b",
    r"\bglöm(?:mer|de|t)\b",
]


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item.get("messages")
        metadata = item.get("metadata")

        if not isinstance(messages, list):
            raise TypeError(f"{path}:{line_number}: messages must be a list")
        if [m.get("role") for m in messages] != EXPECTED_ROLES:
            raise ValueError(f"{path}:{line_number}: wrong role sequence")
        if not isinstance(metadata, dict):
            raise TypeError(f"{path}:{line_number}: metadata must be an object")

        for message in messages:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{path}:{line_number}: empty message content")

        rows.append(item)
    return rows


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def prompt_key(item: dict[str, Any]) -> tuple[str, str]:
    messages = item["messages"]
    return normalized(messages[0]["content"]), normalized(messages[1]["content"])


def has_negative_meaning(value: str) -> bool:
    normalized_value = value.casefold()
    return any(
        re.search(pattern, normalized_value)
        for pattern in NEGATIVE_PATTERNS
    )


def numbers(value: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", value)


def main() -> None:
    datasets = {name: load(path) for name, path in FILES.items()}

    keys = {name: {prompt_key(item) for item in rows} for name, rows in datasets.items()}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = keys[left] & keys[right]
        if overlap:
            raise RuntimeError(f"{left}/{right}: {len(overlap)} overlapping prompts")

    families = {
        name: {item["metadata"]["family"] for item in rows}
        for name, rows in datasets.items()
    }
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = families[left] & families[right]
        if overlap:
            raise RuntimeError(f"{left}/{right}: family leakage: {sorted(overlap)}")

    for name, rows in datasets.items():
        serialised = [
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in rows
        ]
        if len(serialised) != len(set(serialised)):
            raise RuntimeError(f"{name}: duplicate rows")

        for index, item in enumerate(rows, 1):
            user = item["messages"][1]["content"]
            assistant = item["messages"][2]["content"]
            task = item["metadata"]["task"]

            if task.startswith("rewrite") or task == "neutralize":
                user_is_negative = has_negative_meaning(user)
                assistant_is_negative = has_negative_meaning(assistant)
                if user_is_negative != assistant_is_negative:
                    raise RuntimeError(
                        f"{name}:{index}: semantic negation mismatch "
                        f"{user_is_negative} != {assistant_is_negative}"
                    )
                if numbers(user) != numbers(assistant):
                    raise RuntimeError(f"{name}:{index}: number mismatch")

    all_rows = [item for rows in datasets.values() for item in rows]
    all_outputs = [item["messages"][2]["content"].casefold() for item in all_rows]
    assa_count = sum("asså" in output for output in all_outputs)
    assa_ratio = assa_count / len(all_outputs)
    if assa_ratio > 0.08:
        raise RuntimeError(f"'asså' ratio too high: {assa_ratio:.2%}")

    task_counts = Counter(
        item["metadata"]["task"]
        for item in all_rows
    )

    print("MALMO V2 DATASET VALIDATION: PASSED")
    for name, rows in datasets.items():
        print(f"{name}: {len(rows)} examples")
        print(f"{name} families: {sorted(families[name])}")
    print("Cross-split prompt overlap: 0")
    print("Cross-split family overlap: 0")
    print("Duplicate rows: 0")
    print(f"'asså' output ratio: {assa_ratio:.2%}")
    print("Task counts:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
