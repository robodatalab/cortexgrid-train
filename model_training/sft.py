"""Supervised fine-tuning with LoRA.

Supports only the Huggingface model API.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

import mlflow
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from model_training.util import mlflow_active


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
    model: torch.nn.Module,
    dataset: Dataset,
    loss_fn: LossFn | Callable | None = None,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-5,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
) -> torch.nn.Module:
    """Run supervised fine-tuning on a model.

    Args:
        model: Model to be tuned.
        dataset: A torch Dataset. Each item should be a dict of tensors.
            Must include "input_ids" and "attention_mask".
            If no custom loss_fn is provided, must also include "labels".
        loss_fn: Custom loss function (outputs, batch) -> scalar tensor.
            Defaults to outputs.loss (standard causal LM cross-entropy).
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        lr: Learning rate for AdamW.
        max_grad_norm: Gradient clipping norm.
        warmup_ratio: Fraction of total steps used for LR warmup.

    Returns:
        The trained PeftModel (caller is responsible for saving).
    """
    loss_fn = loss_fn or _default_loss
    device = model.device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * warmup_ratio)),
        num_training_steps=total_steps,
    )

    log.info("Starting training: %d epochs, %d steps/epoch", epochs, len(loader))

    if mlflow_active():
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_grad_norm": max_grad_norm,
                "warmup_ratio": warmup_ratio,
                "total_steps": total_steps,
            }
        )

    model.train()
    global_step = 0

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

            step_loss = loss.item()
            epoch_loss += step_loss
            epoch_steps += 1
            global_step += 1

            if mlflow_active():
                mlflow.log_metrics(
                    {
                        "train/loss": step_loss,
                        "train/lr": float(scheduler.get_last_lr()[0]),
                    },
                    step=global_step,
                )

        avg = epoch_loss / max(epoch_steps, 1)
        log.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg)
        if mlflow_active():
            mlflow.log_metric("train/epoch_loss", avg, step=epoch + 1)

    return model
