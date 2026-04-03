"""Unit tests for model_training public API."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import Dataset

from model_training import train_sft_with_lora


class _FakeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@patch("model_training.sft.peft")
class TestTrainSft(unittest.TestCase):

    @staticmethod
    def _make_dataset(n=1):
        return _FakeDataset([
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, 2, 3]),
            }
            for _ in range(n)
        ])

    @staticmethod
    def _make_deployed():
        deployed = MagicMock()
        deployed.device = "cpu"
        return deployed

    @staticmethod
    def _setup_peft(mock_peft):
        param = torch.nn.Parameter(torch.tensor([1.0]))
        lora_model = MagicMock()
        lora_model.parameters.return_value = [param]

        def make_output(*args, **kwargs):
            output = MagicMock()
            output.loss = torch.tensor(1.0, requires_grad=True)
            return output

        lora_model.side_effect = make_output
        mock_peft.get_peft_model.return_value = lora_model
        return lora_model

    def test_wraps_deployed_model_with_peft(self, mock_peft):
        self._setup_peft(mock_peft)
        deployed = self._make_deployed()
        train_sft_with_lora(deployed=deployed, dataset=self._make_dataset(), epochs=1)
        mock_peft.get_peft_model.assert_called_once_with(
            deployed.model, mock_peft.LoraConfig.return_value,
        )

    def test_creates_lora_with_expected_config(self, mock_peft):
        self._setup_peft(mock_peft)
        train_sft_with_lora(deployed=self._make_deployed(), dataset=self._make_dataset(), epochs=1)
        mock_peft.LoraConfig.assert_called_once()
        kw = mock_peft.LoraConfig.call_args.kwargs
        self.assertEqual(kw["target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(kw["r"], 8)
        self.assertEqual(kw["lora_alpha"], 16)

    def test_custom_lora_rank(self, mock_peft):
        self._setup_peft(mock_peft)
        train_sft_with_lora(
            deployed=self._make_deployed(),
            dataset=self._make_dataset(),
            epochs=1,
            lora_rank=16,
            lora_alpha=32,
        )
        kw = mock_peft.LoraConfig.call_args.kwargs
        self.assertEqual(kw["r"], 16)
        self.assertEqual(kw["lora_alpha"], 32)

    def test_returns_peft_model(self, mock_peft):
        lora_model = self._setup_peft(mock_peft)
        result = train_sft_with_lora(
            deployed=self._make_deployed(), dataset=self._make_dataset(), epochs=1,
        )
        self.assertIs(result, lora_model)

    def test_forward_pass_per_batch(self, mock_peft):
        lora_model = self._setup_peft(mock_peft)
        n, epochs = 3, 1
        train_sft_with_lora(
            deployed=self._make_deployed(),
            dataset=self._make_dataset(n),
            epochs=epochs,
            batch_size=2,
        )
        # ceil(3/2) = 2 batches per epoch, 1 epoch
        self.assertEqual(lora_model.call_count, 2)

    def test_trains_for_multiple_epochs(self, mock_peft):
        lora_model = self._setup_peft(mock_peft)
        train_sft_with_lora(
            deployed=self._make_deployed(),
            dataset=self._make_dataset(1),
            epochs=3,
            batch_size=1,
        )
        # 1 batch per epoch * 3 epochs
        self.assertEqual(lora_model.call_count, 3)


if __name__ == "__main__":
    unittest.main()
