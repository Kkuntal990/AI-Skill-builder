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

- The starter `inference.py` is at `$MLEVAL_TASK_DATA_DIR/inference.py`.
- Your final modified script must end with a single print of the form
  ``Average per token generation time:  <float>`` (this is what the starter
  already prints — preserving that line is sufficient). The judge parses
  this line as the trajectory's metric; lower is better.
- The starter already uses `huggyllama/llama-7b` (a Meta-license-compliant
  mirror with identical weights to the now-removed
  `decapoda-research/llama-7b-hf`). The unused
  `prepare_model_for_int8_training` import has also been removed from the
  starter since `peft` renamed it. The rest of the upstream task is
  verbatim.
- No CSV submission is required for this task; only the printed metric.

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
