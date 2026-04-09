"""Tests for MLflow tracking integration in train_sft."""

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


class TestSftMlflowTracking(unittest.TestCase):
    @patch("model_training.sft.mlflow")
    @patch("model_training.sft.mlflow_active", return_value=True)
    def test_logs_params_when_active(self, _active, mock_mlflow):
        train_sft(
            model=_MockModel(),
            dataset=_FakeDataset(1),
            epochs=1,
            batch_size=1,
        )
        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        self.assertEqual(params["epochs"], 1)
        self.assertEqual(params["batch_size"], 1)

    @patch("model_training.sft.mlflow")
    @patch("model_training.sft.mlflow_active", return_value=True)
    def test_logs_step_metrics_when_active(self, _active, mock_mlflow):
        train_sft(
            model=_MockModel(),
            dataset=_FakeDataset(2),
            epochs=1,
            batch_size=1,
        )
        # 2 steps -> 2 log_metrics calls
        self.assertEqual(mock_mlflow.log_metrics.call_count, 2)
        for call in mock_mlflow.log_metrics.call_args_list:
            metrics = call[0][0]
            self.assertIn("train/loss", metrics)
            self.assertIn("train/lr", metrics)

    @patch("model_training.sft.mlflow")
    @patch("model_training.sft.mlflow_active", return_value=True)
    def test_logs_epoch_loss_when_active(self, _active, mock_mlflow):
        train_sft(
            model=_MockModel(),
            dataset=_FakeDataset(1),
            epochs=2,
            batch_size=1,
        )
        # 2 epochs -> 2 log_metric calls for epoch_loss
        self.assertEqual(mock_mlflow.log_metric.call_count, 2)
        for call in mock_mlflow.log_metric.call_args_list:
            self.assertEqual(call[0][0], "train/epoch_loss")

    @patch("model_training.sft.mlflow")
    @patch("model_training.sft.mlflow_active", return_value=False)
    def test_no_mlflow_calls_when_inactive(self, _active, mock_mlflow):
        train_sft(
            model=_MockModel(),
            dataset=_FakeDataset(2),
            epochs=2,
            batch_size=1,
        )
        mock_mlflow.log_params.assert_not_called()
        mock_mlflow.log_metrics.assert_not_called()
        mock_mlflow.log_metric.assert_not_called()


class TestSftMlflowMidTrainingActivation(unittest.TestCase):
    """Verify that tracking responds to mlflow_active() changing mid-training."""

    @patch("model_training.sft.mlflow")
    @patch("model_training.sft.mlflow_active")
    def test_starts_inactive_then_becomes_active(self, mock_active, mock_mlflow):
        # First call (log_params check) -> inactive
        # Subsequent calls (step metrics, epoch loss) -> active
        mock_active.side_effect = [False, True, True]
        train_sft(
            model=_MockModel(),
            dataset=_FakeDataset(1),
            epochs=1,
            batch_size=1,
        )
        mock_mlflow.log_params.assert_not_called()
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.log_metric.assert_called_once()


if __name__ == "__main__":
    unittest.main()
