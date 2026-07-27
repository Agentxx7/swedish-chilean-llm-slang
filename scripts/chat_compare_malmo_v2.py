from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = ROOT / "outputs" / "malmo-v2-run" / "final-adapter"
OUTPUT_DIR = ROOT / "outputs" / "malmo-v2-evaluation"
OUTPUT_FILE = OUTPUT_DIR / "comparisons.jsonl"

SYSTEM_PROMPT = (
    "Du är en hjälpsam svensk assistent. "
    "Svara kort, naturligt och korrekt. "
    "Bevara betydelse, namn, siffror och negationer. "
    "Lägg inte till information som användaren inte har gett dig."
)

MAX_NEW_TOKENS = 128


def load_model() -> tuple[AutoTokenizer, PeftModel]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA hittades inte. Kontrollera NVIDIA-drivrutin och PyTorch.")

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"LoRA-adaptern saknas: {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        is_trainable=False,
    )

    model.eval()
    return tokenizer, model


def build_inputs(
    tokenizer: AutoTokenizer,
    model: PeftModel,
    user_prompt: str,
) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in inputs.items()}


def generate_answer(
    tokenizer: AutoTokenizer,
    model: PeftModel,
    user_prompt: str,
    *,
    adapter_enabled: bool,
) -> str:
    inputs = build_inputs(tokenizer, model, user_prompt)
    input_length = inputs["input_ids"].shape[1]

    adapter_context = (
        nullcontext()
        if adapter_enabled
        else model.disable_adapter()
    )

    with adapter_context, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0, input_length:]
    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def save_comparison(
    user_prompt: str,
    base_response: str,
    lora_response: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_model": MODEL_ID,
        "adapter": str(ADAPTER_DIR),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "base_response": base_response,
        "lora_response": lora_response,
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "repetition_penalty": 1.05,
        },
    }

    with OUTPUT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    print("MALMÖ V2 INTERAKTIV JÄMFÖRELSE")
    print(f"Basmodell: {MODEL_ID}")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'saknas'}")
    print("Skriv exit eller quit för att avsluta.")
    print()

    tokenizer, model = load_model()

    while True:
        try:
            user_prompt = input("Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAvslutar.")
            break

        if not user_prompt:
            continue

        if user_prompt.lower() in {"exit", "quit"}:
            print("Avslutar.")
            break

        base_response = generate_answer(
            tokenizer,
            model,
            user_prompt,
            adapter_enabled=False,
        )

        lora_response = generate_answer(
            tokenizer,
            model,
            user_prompt,
            adapter_enabled=True,
        )

        print("\n=== BASE MODEL ===")
        print(base_response)

        print("\n=== MALMÖ V2 LORA ===")
        print(lora_response)
        print()

        save_comparison(
            user_prompt,
            base_response,
            lora_response,
        )


if __name__ == "__main__":
    main()
