from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "train": ROOT / "data/train/scanian_malmo_train.jsonl",
    "validation": ROOT / "data/validation/scanian_malmo_validation.jsonl",
    "test": ROOT / "data/test/scanian_malmo_test.jsonl",
}
EXPECTED_ROLES = ["system", "user", "assistant"]


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        row = json.loads(line)
        messages = row.get("messages")

        if not isinstance(messages, list):
            raise TypeError(f"{path}:{line_number}: messages must be a list")

        roles = [message.get("role") for message in messages]
        if roles != EXPECTED_ROLES:
            raise ValueError(
                f"{path}:{line_number}: expected roles {EXPECTED_ROLES}, got {roles}"
            )

        for message in messages:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{path}:{line_number}: empty message content")

        rows.append(row)

    return rows


def key(row: dict[str, Any]) -> tuple[str, str]:
    messages = row["messages"]
    return (
        messages[0]["content"].strip().casefold(),
        messages[1]["content"].strip().casefold(),
    )


def main() -> None:
    datasets = {name: load(path) for name, path in FILES.items()}

    for name, rows in datasets.items():
        serialised = [
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
        ]
        if len(serialised) != len(set(serialised)):
            raise RuntimeError(f"{name}: duplicate rows found")

    keys = {name: {key(row) for row in rows} for name, rows in datasets.items()}

    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left, right in pairs:
        overlap = keys[left] & keys[right]
        if overlap:
            raise RuntimeError(f"{left}/{right}: {len(overlap)} overlapping prompts")

    print("DATASET VALIDATION: PASSED")
    for name, rows in datasets.items():
        print(f"{name}: {len(rows)} examples")
    print("Duplicate rows: 0")
    print("Cross-split prompt overlap: 0")


if __name__ == "__main__":
    main()
