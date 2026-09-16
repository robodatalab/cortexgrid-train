# cortexgrid-train

Training loops for models you train on the
[cortexgrid](https://github.com/robodatalab/cortexgrid) infrastructure.

## What this library is

cortexgrid-train is an add-on to cortexgrid. It does not offer better training
algorithms or custom optimizations. The loops are standard: `train_sft` is AdamW with a
cosine warmup schedule and gradient clipping, the same loop you would write by hand.

It exists so you don't have to hand-write the same cortexgrid instrumentation into
each project's training loop. Every loop in this library:

- **logs metrics** to the cortexgrid experiment run (MLflow): per-step loss and
  learning rate, and per-epoch loss;
- **logs a heartbeat** at the start of every epoch, so a stalled run is visible;
- **checkpoints** model, optimizer, and scheduler state through `cortexgrid.checkpoint()`
  after every epoch;
- **resumes** from the latest checkpoint through `cortexgrid.resume()`, so a retried
  job continues after its last completed epoch instead of starting over.

**It is not limited to LLMs.** The library covers training any PyTorch model on
cortexgrid: language models, diffusion models, vision models, and so on. The algorithm
it ships today, `train_sft`, is supervised fine-tuning for HuggingFace causal LMs.
Loops for other model types and objectives belong here as well (see
[Adding a training algorithm](#adding-a-training-algorithm)).

## Install

```bash
pip install cortexgrid-train
```

## Using it in your project

### 1. Get a model to train

The training loops take any `torch.nn.Module`, so the model can come from anywhere:
the HuggingFace Hub (`from_pretrained("org/name")`), a module you define, or weights
already stored in the cortexgrid model registry.

To train on the cluster, stage the weights in the registry with
[cortexgrid-infer](https://github.com/robodatalab/cortexgrid-infer). The download runs
on the cluster, so the weights never pass through your machine:

```python
import time

import cortexgrid
import cortexgrid_infer

cortexgrid.Experiment.init("qwen-sft")

MODEL_ID = "hf:Qwen/Qwen2.5-0.5B-Instruct"
cortexgrid_infer.upload_model(MODEL_ID)

status = cortexgrid_infer.deployment_status(MODEL_ID)
while status is None or status.phase != "ready":
    time.sleep(10)
    status = cortexgrid_infer.deployment_status(MODEL_ID)
```

cortexgrid-infer registers `hf:<org>/<name>` in the current run. The family is the
part of `<name>` before its last `-`, and the suffix is the part after it. For example,
`Qwen2.5-0.5B-Instruct` becomes family `Qwen2.5-0.5B` and suffix `Instruct`. A job
submitted from the same run gets the weights as a local directory with
`cortexgrid.load_model(family, suffix, run_name)`.

### 2. Write the training job

A training job is a plain function. It loads the model, builds a dataset, calls a
training loop, and saves the result. Saving is up to you: the loop only returns the
trained model.

```python
import cortexgrid
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from cortexgrid_train import train_sft


def train_job():
    run_name = cortexgrid.Experiment.get_instance().run_name()
    weights_dir = cortexgrid.load_model("Qwen2.5-0.5B", "Instruct", run_name)

    model = AutoModelForCausalLM.from_pretrained(weights_dir).to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(r=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )

    dataset = build_dataset(weights_dir)  # your code, see below
    model = train_sft(model, dataset, epochs=3, batch_size=4, lr=2e-4)

    model.save_pretrained("adapter")
    cortexgrid.upload_dir("adapter", f"adapters/{run_name}")
```

`train_sft` expects a `torch.utils.data.Dataset` whose items are dicts of tensors with
`input_ids` and `attention_mask`, plus optionally `labels`. When `labels` is missing,
the loop copies `input_ids` and masks out padding positions. To use something other
than the model's built-in causal-LM loss, pass `loss_fn=lambda outputs, batch: ...`.

### 3. Run it on the cluster

Submit the job from the same experiment run that holds the weights:

```python
job_id = cortexgrid.remote(train_job, num_gpus=1, retry=True)
```

Metrics appear in the run while the job trains. If the job is retried, the loop picks
up from the last checkpoint. See the
[cortexgrid docs](https://github.com/robodatalab/cortexgrid) for job status, retries,
and logs.

## Adding a training algorithm

To add a new objective or model type (DPO, reward modeling, diffusion fine-tuning, a
vision classifier, ...), add it as a new loop next to SFT:

1. Create a module, e.g. `cortexgrid_train/dpo.py`, with a `train_dpo(...)` function,
   and export it from `cortexgrid_train/__init__.py`.
2. Write the algorithm however it needs to work: its own dataset format, loss, optimizer,
   and schedule. Nothing here assumes a language model.
3. Keep the cortexgrid contract. It is what makes the loop belong in this library.
   [`cortexgrid_train/sft.py`](cortexgrid_train/sft.py) is the reference
   implementation.
   - Before training, call `cortexgrid.resume()`. If it returns a checkpoint, restore
     the training state and skip the epochs that already completed.
   - At the start of each epoch, call
     `cortexgrid.log_metric("heartbeat", time.time(), step=global_step)`.
   - After each step, call `cortexgrid.log_metrics({"train/...": ...}, step=global_step)`.
   - After each epoch, open `with cortexgrid.checkpoint() as ckpt:`, set `ckpt.epoch`
     and `ckpt.global_step`, and call `ckpt.save_training_state(model, optimizer, scheduler)`.
     To store any other state the algorithm needs to resume, assign it as an
     attribute on `ckpt`.
   - Return the trained model and let the caller save it.
4. Add unit tests that patch `cortexgrid_train.<module>.cortexgrid` and check the metric
   and checkpoint calls, as in [`tests/test_sft_mlflow.py`](tests/test_sft_mlflow.py).
   Run them with `uv run python -m unittest discover -s tests -v`.

## License

Apache-2.0
