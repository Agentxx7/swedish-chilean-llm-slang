from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/examples/swedish_slang_train_seed.jsonl"
VALIDATION_PATH = ROOT / "data/examples/swedish_slang_validation_seed.jsonl"

EXPECTED_ROLES = ["system", "user", "assistant"]


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
            continue

        if not isinstance(item, dict):
            errors.append(f"{path.name}:{line_number}: row must be an object")
            continue

        messages = item.get("messages")

        if not isinstance(messages, list):
            errors.append(f"{path.name}:{line_number}: messages must be a list")
            continue

        roles: list[str] = []
        row_has_error = False

        for message_number, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                errors.append(
                    f"{path.name}:{line_number}: "
                    f"message {message_number} must be an object"
                )
                row_has_error = True
                continue

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str):
                errors.append(
                    f"{path.name}:{line_number}: "
                    f"message {message_number} has invalid role"
                )
                row_has_error = True
            else:
                roles.append(role)

            if not isinstance(content, str) or not content.strip():
                errors.append(
                    f"{path.name}:{line_number}: "
                    f"message {message_number} has empty content"
                )
                row_has_error = True

        if roles != EXPECTED_ROLES:
            errors.append(
                f"{path.name}:{line_number}: "
                f"roles must be {EXPECTED_ROLES}, found {roles}"
            )
            row_has_error = True

        if not row_has_error:
            rows.append(item)

    return rows, errors


def prompt_key(item: dict[str, Any]) -> tuple[str, str]:
    messages = item["messages"]

    system_text = messages[0]["content"].strip().lower()
    user_text = messages[1]["content"].strip().lower()

    return system_text, user_text


def duplicate_errors(
    name: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    seen: dict[str, int] = {}

    for index, item in enumerate(rows, start=1):
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)

        if key in seen:
            errors.append(
                f"{name}: duplicate rows {seen[key]} and {index}"
            )
        else:
            seen[key] = index

    return errors


def main() -> None:
    train_rows, train_errors = load_jsonl(TRAIN_PATH)
    validation_rows, validation_errors = load_jsonl(VALIDATION_PATH)

    errors = train_errors + validation_errors
    errors.extend(duplicate_errors("training", train_rows))
    errors.extend(duplicate_errors("validation", validation_rows))

    train_prompts = {prompt_key(item) for item in train_rows}
    validation_prompts = {prompt_key(item) for item in validation_rows}

    overlap = train_prompts & validation_prompts

    if overlap:
        errors.append(
            f"training and validation contain {len(overlap)} overlapping prompts"
        )

    if errors:
        print("DATASET VALIDATION: FAILED")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("DATASET VALIDATION: PASSED")
    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(validation_rows)}")
    print("Prompt overlap: 0")
    print("Duplicate rows: 0")


if __name__ == "__main__":
    main()
