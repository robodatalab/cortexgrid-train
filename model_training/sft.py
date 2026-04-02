"""Supervised fine-tuning with LoRA."""

from __future__ import annotations

from typing import Any

import peft
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_gateway import deploy_model, Message, Tool, ToolSpec
from model_gateway.utils import normalize_tools


def tokenize_masking_non_assistant_messages(
    messages: list[Message],
    tokenizer: Any,
    tool_specs: list[ToolSpec],
) -> dict[str, torch.Tensor]:
    """Tokenize a conversation, masking non-assistant turns with -100 labels."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tool_specs,
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    labels = torch.full_like(input_ids, -100)

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            if i == 0:
                first_token_idx = 0
            else:
                prompt_before = tokenizer.apply_chat_template(
                    messages[:i],
                    tools=tool_specs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                first_token_idx = tokenizer(prompt_before, return_tensors="pt")[
                    "input_ids"
                ].shape[-1]

            prompt_after = tokenizer.apply_chat_template(
                messages[: i + 1],
                tools=tool_specs,
                tokenize=False,
                add_generation_prompt=False,
            )
            last_token_idx = tokenizer(prompt_after, return_tensors="pt")[
                "input_ids"
            ].shape[-1]

            labels[0, first_token_idx:last_token_idx] = input_ids[
                0, first_token_idx:last_token_idx
            ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_sft(model_id: str, tools: list[Tool], train_ds: Dataset) -> None:
    """Run supervised fine-tuning with LoRA on a deployed model."""
    model = deploy_model(model_id=model_id)

    lora_config = peft.LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=8,
        lora_alpha=16,
    )
    lora_model = peft.get_peft_model(model=model.model, peft_config=lora_config)
    optimizer = torch.optim.AdamW(params=lora_model.parameters(), lr=2e-4)

    tool_specs = normalize_tools(tools) or []

    train_dl = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=lambda x: x[0]
    )
    for sample in tqdm(train_dl, desc="training"):
        optimizer.zero_grad()

        messages = sample["messages"]
        inputs = tokenize_masking_non_assistant_messages(
            messages=messages, tokenizer=model.tokenizer, tool_specs=tool_specs
        )

        outputs = lora_model(
            input_ids=inputs["input_ids"].to(device=model.device),
            labels=inputs["labels"].to(device=model.device),
            attention_mask=inputs["attention_mask"].to(device=model.device),
        )

        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
        optimizer.step()

    lora_model.save_pretrained(f"./models/{model_id}-FT")
