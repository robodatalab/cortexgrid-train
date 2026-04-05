"""Unit tests for model_training public API."""

import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from model_training import train_sft


class _FakeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class _MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.call_count = 0

    @property
    def device(self):
        return self.weight.device

    def forward(self, input_ids, attention_mask, labels):
        self.call_count += 1
        return SimpleNamespace(loss=self.weight.sum())


class TestTrainSft(unittest.TestCase):
    @staticmethod
    def _make_dataset(n=1):
        return _FakeDataset(
            [
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "attention_mask": torch.tensor([1, 1, 1]),
                    "labels": torch.tensor([-100, 2, 3]),
                }
                for _ in range(n)
            ]
        )

    def test_forward_pass_per_batch(self):
        model = _MockModel()
        n, epochs = 3, 1
        train_sft(
            model=model,
            dataset=self._make_dataset(n),
            epochs=epochs,
            batch_size=2,
        )
        # ceil(3/2) = 2 batches per epoch, 1 epoch
        self.assertEqual(model.call_count, 2)

    def test_trains_for_multiple_epochs(self):
        model = _MockModel()
        train_sft(
            model=model,
            dataset=self._make_dataset(1),
            epochs=3,
            batch_size=1,
        )
        # 1 batch per epoch * 3 epochs
        self.assertEqual(model.call_count, 3)


if __name__ == "__main__":
    unittest.main()
