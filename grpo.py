"""
adapted from will brown's grpo demo: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
"""

import re
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig
from custom_trainer import CustomGRPOTrainer, extract_xml_answer
import wandb

from transformers.trainer_callback import ProgressCallback
def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)
ProgressCallback.on_log = on_log

# Load dataset using the new system
dataset_dict = load_from_disk("arithmetic_dataset")
train_dataset = dataset_dict['train']
val_dataset = dataset_dict['validation']

# reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"\nQuestion:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def numeric_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    def is_numeric(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    return [0.5 if is_numeric(r) else 0.0 for r in extracted_responses]

def count_xml(text) -> float:
    count = 0.0
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# Create reward function to penalize longer completions
def length_penalty_reward_func(completions, **kwargs) -> list[float]:
    """
    Penalizes longer completions by returning a negative reward proportional to the length.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [-len(c) * 0.001 for c in contents]  # Adjust the factor as needed

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "unsloth/Qwen3-0.6B"

output_dir="outputs/arithmetic-grpo"

wandb.init(
    project="apart-physics",
    entity="grpo-llc",
)

# Training configuration
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
num_generations = 4

# Calculate eval_steps based on dataset size and training configuration
# This ensures evaluation happens at ~10% intervals throughout an epoch
train_dataset_size = len(train_dataset)
effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
steps_per_epoch = (train_dataset_size * num_generations) // effective_batch_size
eval_steps = max(1, steps_per_epoch // 5)  # Evaluate at ~20% intervals

print(f"Training dataset size: {train_dataset_size}")
print(f"Effective batch size: {effective_batch_size}")  
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Eval steps (20% intervals): {eval_steps}")
    
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-6,
    weight_decay = 0.1,
    warmup_ratio = 0.05,
    lr_scheduler_type='cosine',
    logging_steps=1,
    optim="adamw_8bit",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_generations=num_generations,
    max_prompt_length=128,
    loss_type="dr_grpo",
    max_completion_length=512,
    num_train_epochs=1,
    report_to="wandb",
    log_on_each_node=False,
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps, # so this requires around 12 GB of memory
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",    
)
        
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

trainer = CustomGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        correctness_reward_func,
        xmlcount_reward_func,
        numeric_reward_func,
        length_penalty_reward_func
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# trainer.evaluate()
trainer.train()
