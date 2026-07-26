# Scanian and Malmö Swedish synthetic starter dataset

## Purpose

This pack is a supervised fine-tuning starter dataset for four controllable
text modes:

1. Swedish everyday slang
2. Modern written Scanian-flavoured Swedish
3. Modern colloquial Malmö Swedish
4. Neutral standard Swedish

## Size

- Training: 300 examples
- Validation: 60 examples
- Final test: 60 examples

The final test file must not be used to select epochs, learning rates or
checkpoints.

## Important limitation

This is synthetic training material, not a linguistic gold-standard corpus.
Scanian contains substantial geographic variation, and traditional dialect
vocabulary is not automatically representative of modern Malmö speech.

The dataset therefore:

- avoids phonetic spellings intended to imitate accent
- uses regional vocabulary sparingly
- separates modern Scanian and Malmö-oriented styles
- includes standard-Swedish control examples
- should be reviewed by speakers before a larger production training run

Text fine-tuning can influence vocabulary, syntax and conversational register.
It cannot teach an audible Scanian accent; that requires an appropriate TTS or
speech model.

## Review requirement

Before training, inspect all examples for:

- naturalness
- unchanged semantic meaning
- regional authenticity
- accidental stereotypes
- overuse of words such as `mög`, `klyddigt`, `rälig`, `hialös` or `hutta`
- unwanted personal data or copyrighted dialogue

## Files

- `data/train/scanian_malmo_train.jsonl`
- `data/validation/scanian_malmo_validation.jsonl`
- `data/test/scanian_malmo_test.jsonl`
- `scripts/validate_scanian_malmo_dataset.py`

## Validation

Run from the repository root:

```bash
uv run python scripts/validate_scanian_malmo_dataset.py
```

Expected result:

```text
DATASET VALIDATION: PASSED
train: 300 examples
validation: 60 examples
test: 60 examples
Duplicate rows: 0
Cross-split prompt overlap: 0
```

## Reference basis

The design was informed by public material from the Institute for Language and
Folklore (ISOF) on Scanian and South Swedish dialects, including regional words
such as `klyddigt`, `mög`, `rälig`, `hialös` and `hutta`, and by the City of
Malmö's overview of Malmö speech. These references do not certify every
synthetic sentence as authentic contemporary usage.
