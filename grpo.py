"""
adapted from will brown's grpo demo: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
"""

import re
import os
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig
from custom_trainer import CustomGRPOTrainer, extract_xml_answer
import wandb

# Set required environment variables for single-node vLLM execution
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

from transformers.trainer_callback import ProgressCallback
def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)
ProgressCallback.on_log = on_log

# Load dataset using the new system
dataset_dict = load_from_disk("puzzle_dataset")
train_dataset = dataset_dict['train']
val_dataset = dataset_dict['validation']

# reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"\nQuestion:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def right_string_length_reward_func(completions, answer, **kwargs) -> list[float]:
    """
    Rewards if the extracted answer has the same length as the target answer.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if len(r) == len(a) else 0.0 for r, a in zip(extracted_responses, answer)]

def longest_common_substring(s1, s2):
    """
    Calculate the length of the longest common substring between two strings.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0
    
    return max_length

def lcs_reward_func(completions, answer, **kwargs) -> list[float]:
    """
    Rewards based on the longest common substring similarity between predicted and target answer.
    This is the most important reward function for the puzzle dataset.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    rewards = []
    for r, a in zip(extracted_responses, answer):
        if len(r) == 0 or len(a) == 0:
            rewards.append(0.0)
        else:
            lcstr_length = longest_common_substring(r, a)
            # Normalize by the maximum possible substring length
            max_length = max(len(r), len(a))
            similarity = lcstr_length / max_length
            # Scale the reward (1.0 max for perfect substring match)
            rewards.append(similarity * 1.0)
    
    return rewards

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

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
#model_name = "unsloth/Qwen3-0.6B"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

output_dir="../../workspace/outputs/puzzle-grpo"

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
eval_steps = max(1, steps_per_epoch // 10)  # Evaluate at ~10% intervals

print(f"Training dataset size: {train_dataset_size}")
print(f"Effective batch size: {effective_batch_size}")  
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Eval steps (10% intervals): {eval_steps}")
    
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=8e-6,
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
    num_train_epochs=10,
    report_to="wandb",
    log_on_each_node=False,
    max_grad_norm=10.0,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps, # so this requires around 12 GB of memory
    use_vllm=True,
    vllm_mode="colocate",
    # Best thinking mode parameters (do not use greedy as it can lead to repetition!)
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
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
        right_string_length_reward_func,
        lcs_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# trainer.evaluate()
trainer.train()
