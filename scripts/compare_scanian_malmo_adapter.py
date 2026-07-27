from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = ROOT / "outputs" / "scanian-malmo-run" / "final-adapter"
OUTPUT_JSON = ROOT / "outputs" / "scanian-malmo-comparison.json"
OUTPUT_MD = ROOT / "outputs" / "scanian-malmo-comparison.md"
MAX_NEW_TOKENS = 80
SEED = 42

PROMPTS: list[dict[str, str]] = [
    {
        "id": "slang-01",
        "mode": "svensk slang",
        "system": (
            "Skriv om meningen på naturlig svensk vardagsslang. "
            "Intensitet: medium. Behåll betydelsen och lägg inte till ny information."
        ),
        "user": "Jag förstår inte varför bussen aldrig kommer.",
    },
    {
        "id": "slang-02",
        "mode": "svensk slang",
        "system": (
            "Skriv om meningen på naturlig svensk vardagsslang. "
            "Intensitet: medium. Behåll betydelsen och lägg inte till ny information."
        ),
        "user": "Det här programmet fungerar väldigt dåligt efter uppdateringen.",
    },
    {
        "id": "slang-03",
        "mode": "svensk slang",
        "system": (
            "Skriv om meningen på naturlig svensk vardagsslang. "
            "Intensitet: medium. Behåll betydelsen och lägg inte till ny information."
        ),
        "user": "Jag orkar inte fortsätta vänta på att de ska bestämma sig.",
    },
    {
        "id": "scanian-01",
        "mode": "modern skånska",
        "system": (
            "Skriv om meningen på modern skånsk vardagssvenska. "
            "Använd regionala ord sparsamt, skriv inte fonetiskt och undvik karikatyr. "
            "Behåll betydelsen."
        ),
        "user": "Det är svårt att få ordning på alla konton och behörigheter.",
    },
    {
        "id": "scanian-02",
        "mode": "modern skånska",
        "system": (
            "Skriv om meningen på modern skånsk vardagssvenska. "
            "Använd regionala ord sparsamt, skriv inte fonetiskt och undvik karikatyr. "
            "Behåll betydelsen."
        ),
        "user": "Jag är mycket otålig när folk aldrig bestämmer sig.",
    },
    {
        "id": "scanian-03",
        "mode": "modern skånska",
        "system": (
            "Skriv om meningen på modern skånsk vardagssvenska. "
            "Använd regionala ord sparsamt, skriv inte fonetiskt och undvik karikatyr. "
            "Behåll betydelsen."
        ),
        "user": "Släng den gamla kartongen innan vi går.",
    },
    {
        "id": "malmo-01",
        "mode": "modern malmöitiska",
        "system": (
            "Skriv om meningen på modern malmöitisk vardagssvenska. "
            "Tonen ska vara urban och avslappnad men inte stereotyp. "
            "Behåll betydelsen och lägg inte till ny information."
        ),
        "user": "Jag tänker inte stå här och vänta hela kvällen.",
    },
    {
        "id": "malmo-02",
        "mode": "modern malmöitiska",
        "system": (
            "Skriv om meningen på modern malmöitisk vardagssvenska. "
            "Tonen ska vara urban och avslappnad men inte stereotyp. "
            "Behåll betydelsen och lägg inte till ny information."
        ),
        "user": "Han förstår inte varför alla är så irriterade.",
    },
    {
        "id": "malmo-03",
        "mode": "modern malmöitiska",
        "system": (
            "Skriv om meningen på modern malmöitisk vardagssvenska. "
            "Tonen ska vara urban och avslappnad men inte stereotyp. "
            "Behåll betydelsen och lägg inte till ny information."
        ),
        "user": "Det tog väldigt lång tid att få ordning på inställningarna.",
    },
    {
        "id": "standard-01",
        "mode": "neutral standardsvenska",
        "system": (
            "Skriv om meningen på neutral standardsvenska utan slang eller regionala "
            "uttryck. Behåll betydelsen."
        ),
        "user": "Det där är bara klyddigt och skitstörigt, asså.",
    },
    {
        "id": "standard-02",
        "mode": "neutral standardsvenska",
        "system": (
            "Skriv om meningen på neutral standardsvenska utan slang eller regionala "
            "uttryck. Behåll betydelsen."
        ),
        "user": "Hutta den där påsen, den är rälig.",
    },
    {
        "id": "standard-03",
        "mode": "neutral standardsvenska",
        "system": (
            "Skriv om meningen på neutral standardsvenska utan slang eller regionala "
            "uttryck. Behåll betydelsen."
        ),
        "user": "Jag pallar inte med det här möget längre, asså.",
    },
]


def load_base_model() -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )
    model.to("cuda")
    model.eval()
    return model


def generate_answer(model: Any, tokenizer: Any, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = encoded["input_ids"].shape[1]
    answer_tokens = generated[0, prompt_length:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def run_model(model: Any, tokenizer: Any, label: str) -> dict[str, str]:
    answers: dict[str, str] = {}

    print()
    print(f"GENERATING WITH {label.upper()}")

    for index, prompt in enumerate(PROMPTS, start=1):
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            system=prompt["system"],
            user=prompt["user"],
        )
        answers[prompt["id"]] = answer
        print(f"[{index:02d}/{len(PROMPTS)}] {prompt['id']}: {answer}")

    return answers


def release_model(model: Any) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def markdown_safe(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>")


def write_outputs(
    base_answers: dict[str, str],
    adapter_answers: dict[str, str],
) -> None:
    rows: list[dict[str, str]] = []

    for prompt in PROMPTS:
        rows.append(
            {
                **prompt,
                "base_answer": base_answers[prompt["id"]],
                "adapter_answer": adapter_answers[prompt["id"]],
            }
        )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "adapter_directory": str(ADAPTER_DIR),
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "enable_thinking": False,
                    "seed": SEED,
                },
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Base model vs Scanian/Malmö LoRA",
        "",
        "Deterministic generation: `do_sample=False`, thinking disabled.",
        "",
        "| ID | Läge | Inmatning | Basmodell | LoRA-adapter |",
        "|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_safe(row["id"]),
                    markdown_safe(row["mode"]),
                    markdown_safe(row["user"]),
                    markdown_safe(row["base_answer"]),
                    markdown_safe(row["adapter_answer"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Manuell bedömning",
            "",
            "Bedöm varje LoRA-svar på:",
            "",
            "- betydelsen bevarad",
            "- rätt stil",
            "- naturligt språk",
            "- ingen överdriven dialekt",
            "- ingen tillagd information",
            "",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Adapter not found: {ADAPTER_DIR}")

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("SCANIAN/MALMO BASE VS LORA COMPARISON")
    print(f"Base model: {MODEL_ID}")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Prompts: {len(PROMPTS)}")

    base_model = load_base_model()
    base_answers = run_model(base_model, tokenizer, "base model")
    release_model(base_model)

    adapter_base_model = load_base_model()
    adapter_model = PeftModel.from_pretrained(
        adapter_base_model,
        ADAPTER_DIR,
        is_trainable=False,
    )
    adapter_model.eval()
    adapter_answers = run_model(adapter_model, tokenizer, "LoRA adapter")
    release_model(adapter_model)

    write_outputs(base_answers, adapter_answers)

    print()
    print("COMPARISON COMPLETED")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
