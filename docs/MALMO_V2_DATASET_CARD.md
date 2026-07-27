# Malmö LLM dataset v2 — synthetic reviewed-seed candidate

## Goal

Train a controllable local chatbot that can:

- answer briefly in modern Malmö-oriented colloquial Swedish
- rewrite neutral Swedish into mild or medium Malmö-oriented register
- return slang or regional text to neutral standard Swedish
- preserve meaning, negation, numbers and named facts

This is an adapter dataset, not a complete language model by itself.

## Design changes from v1

- topic families are exclusive to train, validation and test
- chat-answer tasks are included, not only rewrites
- `asså` is deliberately rare
- mild and medium Malmö registers are separate
- neutralisation examples teach the model to turn the style off
- metadata records family, task and review state
- validator checks negations, numbers, duplicates and family leakage

## Size

- train: 336 examples
- validation: 84 examples
- test: 56 examples

The test set must remain untouched until model selection is complete.

## Limitation

Every row is synthetic and marked `synthetic_needs_review`.
The CSV review sheet must be reviewed by a Malmö/Skåne speaker before this is
treated as a gold dataset.

## Files

- `data/malmo_v2/malmo_v2_train.jsonl`
- `data/malmo_v2/malmo_v2_validation.jsonl`
- `data/malmo_v2/malmo_v2_test.jsonl`
- `data/malmo_v2/malmo_v2_review.csv`
- `scripts/validate_malmo_v2_dataset.py`

## Validation

```bash
uv run python scripts/validate_malmo_v2_dataset.py
```

## Review workflow

Open:

```text
data/malmo_v2/malmo_v2_review.csv
```

Set `approved` to `yes` or `no` and explain corrections in `review_note`.
Do not train a production adapter from unreviewed rows.
