"""
adapted from will brown's grpo demo: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
"""

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import random

# Load and prep dataset

SYSTEM_PROMPT = """
Respond ONLY with the answer to the question in the following format:
<answer>
...
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# custom dataset generation with tunable parameters
def generate_arithmetic_problems(
        max_digits: int = 4,
        num_terms: int = 2,
        num_problems: int = 10,
) -> Dataset:
    problems = {"problem": [], "answer": []}
    for _ in range(num_problems):
        terms = [random.randint(1, 10**max_digits) for _ in range(num_terms)]

        problem = f"{terms[0]}"

        for i in range(num_terms-1):
            operation = random.choice(['+', '-', '*', '/'])
            # avoid division by zero
            if operation == '/' and terms[i + 1] == 0:
                terms[i + 1] = 1
            problem += f" {operation} {terms[i + 1]} "
        problem += f" = "
        problems["problem"].append(problem)
        
        # Calculate answer and convert to string for consistent comparison
        numeric_answer = eval(problem.replace('=', '').strip())
        # Round division results to avoid floating point precision issues
        if '/' in problem:
            if numeric_answer == int(numeric_answer):
                answer_str = str(int(numeric_answer))
            else:
                answer_str = f"{numeric_answer:.2f}".rstrip('0').rstrip('.')
        else:
            answer_str = str(int(numeric_answer))
        
        problems["answer"].append(answer_str)

    data = Dataset.from_dict(problems)
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        "answer": x["answer"]
    })
    return data

dataset = generate_arithmetic_problems(
    max_digits=3,
    num_terms=2,
    num_problems=1000,
)

# reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
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

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

output_dir="outputs/arithmetic-grpo"
run_name="trial-1-arithmetic"
    
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
)
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")
        
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        correctness_reward_func,
        xmlcount_reward_func,
        int_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
    #peft_config=peft_config
)
trainer.train()
