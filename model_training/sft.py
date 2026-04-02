"""Supervised fine-tuning with LoRA.

Supports only the Huggingface model API.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

import peft
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from model_gateway import Message, ToolSpec
from model_gateway.providers.huggingface import HuggingFaceModel

log = logging.getLogger(__name__)


class LossFn(Protocol):
    """Custom loss function: (model_outputs, batch) -> scalar tensor."""

    def __call__(
        self, outputs: Any, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor: ...


def _default_loss(outputs: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Standard causal LM cross-entropy (already computed by HuggingFace)."""
    return outputs.loss


def train_sft(
    deployed: HuggingFaceModel,
    dataset: Dataset,
    loss_fn: LossFn | Callable | None = None,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-5,
    lora_rank: int = 8,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
) -> peft.PeftModel | peft.PeftMixedModel:
    """Run LoRA-based supervised fine-tuning on a deployed model.

    Args:
        deployed: A DeployedModel with .model, .tokenizer, .device attributes.
        dataset: A torch Dataset. Each item should be a dict of tensors.
            Must include "input_ids" and "attention_mask".
            If no custom loss_fn is provided, must also include "labels".
        loss_fn: Custom loss function (outputs, batch) -> scalar tensor.
            Defaults to outputs.loss (standard causal LM cross-entropy).
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        lr: Learning rate for AdamW.
        lora_rank: LoRA rank (r).
        lora_alpha: LoRA alpha. Defaults to 2 * lora_rank.
        lora_dropout: LoRA dropout.
        max_grad_norm: Gradient clipping norm.
        warmup_ratio: Fraction of total steps used for LR warmup.

    Returns:
        The trained PeftModel (caller is responsible for saving).
    """
    loss_fn = loss_fn or _default_loss
    lora_alpha = lora_alpha if lora_alpha is not None else lora_rank * 2
    device = deployed.device

    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = peft.get_peft_model(deployed.model, lora_config)
    model.print_trainable_parameters()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * warmup_ratio)),
        num_training_steps=total_steps,
    )

    log.info("Starting training: %d epochs, %d steps/epoch", epochs, len(loader))
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch.get("labels", batch["input_ids"].clone())
            if "labels" not in batch:
                labels[batch["attention_mask"] == 0] = -100

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )

            loss = loss_fn(outputs, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1

        avg = epoch_loss / max(epoch_steps, 1)
        log.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg)

    return model
