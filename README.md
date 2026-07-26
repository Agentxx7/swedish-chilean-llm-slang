# Swedish–Chilean LLM Slang Training

A reproducible learning project for fine-tuning a small language model with QLoRA.

The project contains three separate training experiments:

1. Natural Swedish everyday slang
2. Natural Chilean Spanish
3. Direct, less-censored responses without unnecessary moralising

The experiments are initially trained as separate LoRA adapters so that each
change can be measured independently.

## Base model

`Qwen/Qwen3-0.6B`

This project does not train a language model from scratch. QLoRA keeps the
quantised base-model weights frozen and trains a much smaller set of LoRA
adapter weights.

## Training objectives

### 1. Swedish slang adapter

Planned adapter:

`qwen3-06b-swedish-slang-lora`

The adapter should learn:

- natural Swedish everyday language
- mild, medium and strong slang intensity
- preservation of the original meaning
- switching between slang, neutral and formal Swedish
- avoiding exaggerated or unnatural slang

### 2. Chilean Spanish adapter

Planned adapter:

`qwen3-06b-chilean-spanish-lora`

The adapter should learn:

- natural Chilean vocabulary and phrasing
- appropriate use of Chilean expressions
- mild, medium and strong colloquial intensity
- separation from Mexican, Argentine and European Spanish
- switching between colloquial and formal Spanish
- avoiding caricatured overuse of expressions such as `po` or `weón`

### 3. Direct and less-censored response adapter

Planned adapter:

`qwen3-06b-direct-response-lora`

The adapter should learn to:

- answer clearly and directly
- avoid unnecessary moralising
- avoid generic disclaimers when they add no value
- discuss controversial and adult subjects in a natural tone
- preserve the requested tone and style
- avoid evasive or generic assistant responses

In this project, “less censored” means reducing unnecessary refusals,
lecturing and boilerplate warnings for otherwise answerable questions.

It does not mean removing every safety boundary or requiring the model to
comply with every possible request.

## Verified test system

- Operating system: Debian Linux
- GPU: NVIDIA GeForce RTX 3080
- VRAM: 9.75 GiB
- RAM: 32 GiB
- Python: 3.12.13
- PyTorch: 2.6.0+cu124
- CUDA build: 12.4
- BF16 supported: True

## Project structure

- `configs/` — inference and training configurations
- `data/examples/` — public synthetic examples and evaluation prompts
- `data/train/` — local training datasets
- `data/validation/` — local validation datasets
- `scripts/` — baseline, training and evaluation scripts
- `tests/` — automated tests
- `outputs/` — local results, adapters and checkpoints

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Install Python 3.12 and create the environment

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
```

Verify the environment:

```bash
python --version
uv --version
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Verify CUDA

```bash
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Verified result:

```text
PyTorch: 2.6.0+cu124
CUDA: 12.4
Available: True
GPU: NVIDIA GeForce RTX 3080
```

## Baseline evaluation

Configuration:

`configs/baseline.yaml`

Evaluation prompts:

`data/examples/baseline_prompts.json`

Run:

```bash
uv run python scripts/run_baseline.py
```

Results are stored locally in:

`outputs/baseline-results.json`

## Baseline findings

The unmodified model performed poorly on Swedish slang and Chilean Spanish.

Swedish input:

`Jag förstår inte vad den där killen håller på med.`

Baseline response:

`Det är en bra idé att försöka förstå! Jag har hittat något som kan hjälpa dig att förstå.`

Desired response:

`Jag fattar inte vad den där snubben håller på med.`

Chilean Spanish input:

`No entiendo qué está haciendo ese tipo.`

Baseline response:

`¿Qué estás haciendo ese tipo?`

Desired response:

`No cacho qué está haciendo ese gallo.`

The same prompts will be evaluated after training to produce a measurable
before-and-after comparison.

## Files excluded from Git

The following remain local:

- `.venv`
- downloaded base models
- Hugging Face cache
- private training datasets
- checkpoints
- trained LoRA adapters
- generated results
- API tokens
- `.env` files

## Current status

- [x] Local Git repository created
- [x] Python 3.12 installed
- [x] Reproducible uv environment created
- [x] CUDA and RTX 3080 verified
- [x] Qwen3-0.6B downloaded
- [x] Baseline evaluation completed
- [ ] Swedish slang training dataset
- [ ] Chilean Spanish training dataset
- [ ] Direct-response training dataset
- [ ] Swedish slang QLoRA training
- [ ] Chilean Spanish QLoRA training
- [ ] Direct-response QLoRA training
- [ ] Before-and-after evaluation
- [ ] LoRA adapter export

## Licences

The source code, base model, datasets and trained adapters are separate
components. Their respective licences and redistribution terms must be checked
before publishing datasets or trained weights.
