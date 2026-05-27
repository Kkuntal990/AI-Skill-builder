<!--
Provenance: this task is a port of MLAgentBench's `llama-inference` benchmark.
Upstream: https://github.com/snap-stanford/MLAgentBench/tree/main/MLAgentBench/benchmarks/llama-inference
Upstream license: MIT.

The "Task" section below is the verbatim 289-byte research_problem.txt from
upstream — nothing added, nothing paraphrased, nothing edited. The "Framework
integration notes" section is OURS — strictly the minimum needed to make the
task runnable in this harness (metric-parse line, in-pod path to the starter,
HF mirror substitution because the upstream model repo was removed). It is
labeled so it shows up in the audit trail.
-->

## Task

Given a inference script inference.py, execute it to see the current generation speed per token and then try to improve it with accelerate library. The script is run on a single A100 GPU. Before you give the final answer, please ask yourself if there is any other way to improve the speed.

---

## Framework integration notes (not part of the upstream task)

- The starter `inference.py` is at `./input/inference.py` (the agent's
  working directory layout is `./input/` for read-only task files,
  `./working/` for scratch). You should copy or modify it as your solution.
- Your final solution code must run the inference and print TWO lines as the
  very last output:
    ``Average per token generation time:  <float>``
    ``Final Validation Score: <float>``
  Both should hold the **same** number (per-token generation time in
  seconds; **lower is better**). The first line is the upstream MLAgentBench
  metric format; the second is what this harness parses as the trajectory's
  validation score. The starter already prints the first line; you just
  need to also print the second.
- This is a **latency benchmark**, NOT a classification task. There is no
  train/val split, no submission.csv, no test set. The single dataset
  (`wikitext-103-v1` test split, 100 batches of 4) is your benchmark
  workload; running it and measuring per-token latency *is* the evaluation.
- The starter already uses `huggyllama/llama-7b` (a Meta-license-compliant
  mirror with identical weights to the now-removed
  `decapoda-research/llama-7b-hf`). The unused
  `prepare_model_for_int8_training` import has also been removed from the
  starter since `peft` renamed it. The rest of the upstream task is
  verbatim.
- GPU: this run uses an **NVIDIA A10 (24 GB)**, not the A100 mentioned in
  the upstream task. Llama-7B fp16 is ~14 GB → fits with batch_size=4, but
  do NOT raise the batch size assuming A100 memory.

---

## Starter script (verbatim from upstream)

```python
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, get_peft_config
from peft import PeftModel, PeftConfig
import torch
from accelerate import Accelerator
from datasets import load_from_disk
import numpy as np
import datasets
from torch.utils.data import DataLoader
from transformers import default_data_collator
import argparse
from transformers import LlamaForCausalLM
import time
from datasets import load_dataset


#### DO NOT EDIT ######

generation_length = 1
context_length = 128

tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b").to("cuda")
eval_dataset = load_dataset("wikitext", 'wikitext-103-v1', split="test")

# tokenize the dataset and filter out examples that are shorter than the context length
def tokenize_and_filter_function(examples):
    tokenized_examples = tokenizer(examples["text"], truncation=True, max_length=context_length)
    # only keep the examples where the context is not too long
    result = {
        "input_ids": [],
        "attention_mask": [],
    }
    for i, input_ids in enumerate(tokenized_examples["input_ids"]):
        if len(input_ids) == context_length:
            result["input_ids"].append(input_ids)
            result["attention_mask"].append(tokenized_examples["attention_mask"][i])
    return result

eval_dataset = eval_dataset.map(tokenize_and_filter_function, batched=True, num_proc=4, remove_columns=["text"])


#################

batch_size = 4

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size)

with torch.no_grad():
    model.eval()
    # record average step time
    total_time = 0
    for idx, batch in enumerate(eval_dataloader):
        if idx == 100:
            break
        input_ids = batch["input_ids"].to("cuda")
        start_time = time.time()
        outputs = model.generate(
            input_ids=input_ids,
            max_length= generation_length + context_length,
        )
        total_time += time.time() - start_time
    print("Average per token generation time: ", total_time / 100)
```
