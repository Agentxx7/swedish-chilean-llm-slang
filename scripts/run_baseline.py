from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "baseline.yaml"
PROMPTS_PATH = ROOT / "data" / "examples" / "baseline_prompts.json"

DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Konfiguration saknas: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"Ogiltig YAML-konfiguration: {path}")

    return data


def load_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Testfrågor saknas: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError("Baseline-filen måste innehålla minst ett testfall.")

    return data


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA är inte tillgängligt. Baseline-körningen stoppas.")

    config = load_yaml(CONFIG_PATH)
    prompts = load_prompts(PROMPTS_PATH)

    model_config = config["model"]
    generation_config = config["generation"]

    model_id = str(model_config["id"])
    dtype_name = str(model_config["dtype"])
    max_new_tokens = int(model_config["max_new_tokens"])

    if dtype_name not in DTYPES:
        raise ValueError(f"Okänd dtype: {dtype_name}")

    output_path = ROOT / str(config["output"]["file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Laddar modell: {model_id}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Datatyp: {dtype_name}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=DTYPES[dtype_name],
        device_map="auto",
    )
    model.eval()

    results: list[dict[str, Any]] = []

    for index, item in enumerate(prompts, start=1):
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
        ]

        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = tokenizer(
            [rendered_prompt],
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=bool(generation_config["do_sample"]),
                repetition_penalty=float(
                    generation_config["repetition_penalty"]
                ),
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[0][model_inputs.input_ids.shape[1] :]
        response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        result = {
            **item,
            "response": response,
        }
        results.append(result)

        print(f"[{index}/{len(prompts)}] {item['id']}")
        print(response)
        print("-" * 70)

    document = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "dtype": dtype_name,
        "device": torch.cuda.get_device_name(0),
        "thinking_enabled": False,
        "results": results,
    }

    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Baseline sparad: {output_path}")
    print(f"Antal resultat: {len(results)}")


if __name__ == "__main__":
    main()
