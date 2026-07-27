from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "malmo_v2_train.yaml"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError("Training configuration must be a YAML object.")

    return config


def project_path(value: str) -> Path:
    return ROOT / value


def maximum_token_length(
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> int:
    lengths: list[int] = []

    for row in rows:
        rendered = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        token_ids = tokenizer(
            rendered,
            add_special_tokens=False,
        )["input_ids"]

        lengths.append(len(token_ids))

    return max(lengths)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training has been stopped.")

    config = load_config(CONFIG_PATH)

    model_config = config["model"]
    data_config = config["data"]
    quantization_config = config["quantization"]
    lora_config = config["lora"]
    training_config = config["training"]
    output_config = config["output"]

    model_id = str(model_config["id"])
    max_length = int(model_config["max_length"])

    train_path = project_path(str(data_config["train_file"]))
    validation_path = project_path(str(data_config["validation_file"]))

    run_directory = project_path(str(output_config["run_directory"]))
    adapter_directory = project_path(str(output_config["adapter_directory"]))
    metrics_path = project_path(str(output_config["metrics_file"]))

    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    if run_directory.exists() and any(run_directory.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {run_directory}\n"
            "Remove it before starting a fresh experiment."
        )

    run_directory.mkdir(parents=True, exist_ok=True)

    seed = int(training_config["seed"])
    set_seed(seed)

    print("MALMO CHATBOT V2 TRAINING")
    print(f"Model: {model_id}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GiB",
    )
    print(f"Seed: {seed}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )

    train_rows = list(dataset["train"])
    validation_rows = list(dataset["validation"])

    longest_train = maximum_token_length(train_rows, tokenizer)
    longest_validation = maximum_token_length(validation_rows, tokenizer)
    longest_example = max(longest_train, longest_validation)

    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(validation_rows)}")
    print(f"Longest example: {longest_example} tokens")
    print(f"Maximum allowed: {max_length} tokens")

    if longest_example > max_length:
        raise RuntimeError(
            f"An example contains {longest_example} tokens, "
            f"which exceeds max_length={max_length}."
        )

    print()
    print("Loading the 4-bit base model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(
            quantization_config["quant_type"]
        ),
        bnb_4bit_use_double_quant=bool(
            quantization_config["double_quant"]
        ),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    peft_config = LoraConfig(
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=str(lora_config["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(run_directory),
        num_train_epochs=float(training_config["epochs"]),
        learning_rate=float(training_config["learning_rate"]),
        per_device_train_batch_size=int(
            training_config["train_batch_size"]
        ),
        per_device_eval_batch_size=int(
            training_config["eval_batch_size"]
        ),
        gradient_accumulation_steps=int(
            training_config["gradient_accumulation_steps"]
        ),
        warmup_ratio=float(training_config["warmup_ratio"]),
        max_length=max_length,
        assistant_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print()
    print("Trainable parameters:")
    trainer.model.print_trainable_parameters()

    print()
    print("Validation before training:")
    initial_evaluation = trainer.evaluate()
    print(initial_evaluation)

    print()
    print("Starting training...")
    training_result = trainer.train()

    print()
    print("Final validation:")
    final_evaluation = trainer.evaluate()
    print(final_evaluation)

    adapter_directory.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_directory))
    tokenizer.save_pretrained(str(adapter_directory))

    summary = {
        "model_id": model_id,
        "seed": seed,
        "training_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "epochs": float(training_config["epochs"]),
        "longest_example_tokens": longest_example,
        "initial_evaluation": initial_evaluation,
        "training_metrics": training_result.metrics,
        "final_evaluation": final_evaluation,
        "adapter_directory": str(adapter_directory),
        "log_history": trainer.state.log_history,
    }

    metrics_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("TRAINING COMPLETED")
    print(f"Adapter: {adapter_directory}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
