"""Unit tests for model_training public API."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import Dataset

from model_training import tokenize_masking_non_assistant_messages, train_sft


class _CharTokenizer:
    """Character-level tokenizer with a simple chat template for testing."""

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
    ) -> str:
        parts = [
            f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages
        ]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, *, return_tensors=None):
        ids = [ord(c) for c in text]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# tokenize_masking_non_assistant_messages
# ---------------------------------------------------------------------------


class TestTokenizeMaskingNonAssistantMessages(unittest.TestCase):

    def setUp(self):
        self.tokenizer = _CharTokenizer()
        self.tool_specs: list[dict] = []

    def _token_len(self, text: str) -> int:
        """Number of tokens for *text* under the char-level tokenizer."""
        return len(text)

    # -- structure ----------------------------------------------------------

    def test_returns_required_keys(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = tokenize_masking_non_assistant_messages(
            messages=messages,
            tokenizer=self.tokenizer,
            tool_specs=self.tool_specs,
        )
        self.assertEqual(
            set(result.keys()), {"input_ids", "attention_mask", "labels"}
        )

    def test_output_shapes_match(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = tokenize_masking_non_assistant_messages(
            messages=messages,
            tokenizer=self.tokenizer,
            tool_specs=self.tool_specs,
        )
        self.assertEqual(result["input_ids"].shape, result["labels"].shape)
        self.assertEqual(result["input_ids"].shape, result["attention_mask"].shape)

    # -- masking behaviour --------------------------------------------------

    def test_non_assistant_tokens_masked(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = tokenize_masking_non_assistant_messages(
            messages=messages,
            tokenizer=self.tokenizer,
            tool_specs=self.tool_specs,
        )
        prefix = self.tokenizer.apply_chat_template(
            messages[:1],
            tools=self.tool_specs,
            tokenize=False,
            add_generation_prompt=True,
        )
        first_idx = self._token_len(prefix)
        labels = result["labels"][0]
        self.assertTrue(torch.all(labels[:first_idx] == -100).item())

    def test_assistant_tokens_unmasked(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = tokenize_masking_non_assistant_messages(
            messages=messages,
            tokenizer=self.tokenizer,
            tool_specs=self.tool_specs,
        )
        prefix = self.tokenizer.apply_chat_template(
            messages[:1],
            tools=self.tool_specs,
            tokenize=False,
            add_generation_prompt=True,
        )
        full = self.tokenizer.apply_chat_template(
            messages[:2],
            tools=self.tool_specs,
            tokenize=False,
            add_generation_prompt=False,
        )
        start, end = self._token_len(prefix), self._token_len(full)
        labels = result["labels"][0]
        input_ids = result["input_ids"][0]
        self.assertTrue(
            torch.all(labels[start:end] == input_ids[start:end]).item()
        )

    def test_no_assistant_turns_all_masked(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "you are helpful"},
        ]
        result = tokenize_masking_non_assistant_messages(
            messages=messages,
            tokenizer=self.tokenizer,
            tool_specs=self.tool_specs,
        )
        self.assertTrue(torch.all(result["labels"] == -100).item())

    def test_multiple_assistant_turns(self):
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        result = tokenize_masking_non_assistant_messages(
            messages=messages,
            tokenizer=self.tokenizer,
            tool_specs=self.tool_specs,
        )
        labels = result["labels"][0]
        input_ids = result["input_ids"][0]

        # First assistant span
        p1 = self.tokenizer.apply_chat_template(
            messages[:1], tools=self.tool_specs, tokenize=False, add_generation_prompt=True,
        )
        f1 = self.tokenizer.apply_chat_template(
            messages[:2], tools=self.tool_specs, tokenize=False, add_generation_prompt=False,
        )
        s1, e1 = self._token_len(p1), self._token_len(f1)

        # Second assistant span
        p2 = self.tokenizer.apply_chat_template(
            messages[:3], tools=self.tool_specs, tokenize=False, add_generation_prompt=True,
        )
        f2 = self.tokenizer.apply_chat_template(
            messages[:4], tools=self.tool_specs, tokenize=False, add_generation_prompt=False,
        )
        s2, e2 = self._token_len(p2), self._token_len(f2)

        self.assertTrue(torch.all(labels[s1:e1] == input_ids[s1:e1]).item())
        self.assertTrue(torch.all(labels[s2:e2] == input_ids[s2:e2]).item())
        # Gap between the two assistant turns must be masked
        self.assertTrue(torch.all(labels[e1:s2] == -100).item())

    def test_assistant_as_first_message(self):
        messages = [{"role": "assistant", "content": "hello"}]
        result = tokenize_masking_non_assistant_messages(
            messages=messages,
            tokenizer=self.tokenizer,
            tool_specs=self.tool_specs,
        )
        full = self.tokenizer.apply_chat_template(
            messages, tools=self.tool_specs, tokenize=False, add_generation_prompt=False,
        )
        end = self._token_len(full)
        labels = result["labels"][0]
        input_ids = result["input_ids"][0]
        # i==0 branch: first_token_idx = 0, so entire message is unmasked
        self.assertTrue(torch.all(labels[:end] == input_ids[:end]).item())

    def test_tool_specs_forwarded_to_template(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "prompt"
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        tokenizer.return_value = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }
        tool_specs = [{"type": "function", "function": {"name": "my_tool"}}]
        messages = [{"role": "user", "content": "hi"}]

        tokenize_masking_non_assistant_messages(
            messages=messages, tokenizer=tokenizer, tool_specs=tool_specs,
        )

        for c in tokenizer.apply_chat_template.call_args_list:
            self.assertEqual(c.kwargs["tools"], tool_specs)


# ---------------------------------------------------------------------------
# train_sft
# ---------------------------------------------------------------------------


@patch("model_training.sft.tokenize_masking_non_assistant_messages")
@patch("model_training.sft.peft")
@patch("model_training.sft.deploy_model")
@patch("model_training.sft.normalize_tools")
class TestTrainSft(unittest.TestCase):

    @staticmethod
    def _make_dataset(n=1):
        return _FakeDataset([
            {"messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]}
            for i in range(n)
        ])

    @staticmethod
    def _setup(mock_normalize, mock_deploy, mock_peft, mock_tokenize):
        deployed = MagicMock()
        deployed.device = "cpu"
        mock_deploy.return_value = deployed

        param = torch.nn.Parameter(torch.tensor([1.0]))
        lora_model = MagicMock()
        lora_model.parameters.return_value = [param]
        mock_peft.get_peft_model.return_value = lora_model

        output = MagicMock()
        output.loss = torch.tensor(1.0, requires_grad=True)
        lora_model.return_value = output

        mock_normalize.return_value = []
        mock_tokenize.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, 2, 3]]),
        }
        return deployed, lora_model

    def test_deploys_model_with_given_id(self, mock_norm, mock_deploy, mock_peft, mock_tok):
        self._setup(mock_norm, mock_deploy, mock_peft, mock_tok)
        train_sft(model_id="my-model", tools=[], train_ds=self._make_dataset())
        mock_deploy.assert_called_once_with(model_id="my-model")

    def test_creates_lora_with_expected_config(self, mock_norm, mock_deploy, mock_peft, mock_tok):
        self._setup(mock_norm, mock_deploy, mock_peft, mock_tok)
        train_sft(model_id="m", tools=[], train_ds=self._make_dataset())
        mock_peft.LoraConfig.assert_called_once()
        kw = mock_peft.LoraConfig.call_args.kwargs
        self.assertEqual(kw["target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(kw["r"], 8)
        self.assertEqual(kw["lora_alpha"], 16)

    def test_normalizes_tools(self, mock_norm, mock_deploy, mock_peft, mock_tok):
        self._setup(mock_norm, mock_deploy, mock_peft, mock_tok)
        tools = [lambda: None]
        train_sft(model_id="m", tools=tools, train_ds=self._make_dataset())
        mock_norm.assert_called_once_with(tools)

    def test_saves_model_to_expected_path(self, mock_norm, mock_deploy, mock_peft, mock_tok):
        _, lora_model = self._setup(mock_norm, mock_deploy, mock_peft, mock_tok)
        train_sft(model_id="my-model", tools=[], train_ds=self._make_dataset())
        lora_model.save_pretrained.assert_called_once_with("./models/my-model-FT")

    def test_forward_pass_per_sample(self, mock_norm, mock_deploy, mock_peft, mock_tok):
        _, lora_model = self._setup(mock_norm, mock_deploy, mock_peft, mock_tok)
        n = 3
        train_sft(model_id="m", tools=[], train_ds=self._make_dataset(n))
        self.assertEqual(lora_model.call_count, n)

    def test_tokenizes_each_sample(self, mock_norm, mock_deploy, mock_peft, mock_tok):
        self._setup(mock_norm, mock_deploy, mock_peft, mock_tok)
        n = 2
        train_sft(model_id="m", tools=[], train_ds=self._make_dataset(n))
        self.assertEqual(mock_tok.call_count, n)


if __name__ == "__main__":
    unittest.main()
