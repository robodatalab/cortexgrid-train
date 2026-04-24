"""Tests for cortexflow tracking integration in train_sft."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from model_training import train_sft


class _FakeDataset(Dataset):
    def __init__(self, n=1):
        self.samples = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, 2, 3]),
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class _MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    @property
    def device(self):
        return self.weight.device

    def forward(self, input_ids, attention_mask, labels):
        return SimpleNamespace(loss=self.weight.sum())


@patch("model_training.sft.cortexflow")
class TestSftCortexflowTracking(unittest.TestCase):
    def test_logs_step_metrics(self, mock_cortexflow):
        mock_cortexflow.resume.return_value = None
        train_sft(
            model=_MockModel(),
            dataset=_FakeDataset(2),
            epochs=1,
            batch_size=1,
        )
        self.assertEqual(mock_cortexflow.log_metrics.call_count, 2)
        for call in mock_cortexflow.log_metrics.call_args_list:
            metrics = call[0][0]
            self.assertIn("train/loss", metrics)
            self.assertIn("train/lr", metrics)

    def test_logs_epoch_loss(self, mock_cortexflow):
        mock_cortexflow.resume.return_value = None
        train_sft(
            model=_MockModel(),
            dataset=_FakeDataset(1),
            epochs=2,
            batch_size=1,
        )
        epoch_loss_calls = [
            c
            for c in mock_cortexflow.log_metric.call_args_list
            if c[0][0] == "train/epoch_loss"
        ]
        self.assertEqual(len(epoch_loss_calls), 2)

    def test_logs_heartbeat_per_epoch(self, mock_cortexflow):
        mock_cortexflow.resume.return_value = None
        train_sft(
            model=_MockModel(),
            dataset=_FakeDataset(1),
            epochs=3,
            batch_size=1,
        )
        heartbeat_calls = [
            c
            for c in mock_cortexflow.log_metric.call_args_list
            if c[0][0] == "heartbeat"
        ]
        self.assertEqual(len(heartbeat_calls), 3)


if __name__ == "__main__":
    unittest.main()
