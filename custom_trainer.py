import re
from collections import defaultdict, deque
import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler
from datasets import Dataset
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import RepeatSampler
from typing import Optional, List, Dict

# Imports for LLC calculation
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim.sgmcmc import SGMCMC
from devinterp.utils import default_nbeta
import wandb

def extract_xml_answer(text: str) -> str:
    """
    Extracts the content from within the first <answer> tag.
    Handles missing or incomplete tags gracefully.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback for incomplete tags or different formatting
    if "<answer>" in text:
        answer = text.split("<answer>")[-1]
        if "</answer>" in answer:
            answer = answer.split("</answer>")[0]
        return answer.strip()
        
    return "" # Return empty string if no answer found

# This evaluate function is specifically for devinterp's LLC calculation.
# It computes the standard cross-entropy loss for a causal language model.
def evaluate_llc(model, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """
    Computes the loss for the given model and batch.
    The batch is expected to be a dictionary containing 'input_ids', 'attention_mask', and 'labels'.
    """
    outputs = model(
        input_ids=batch['input_ids'], 
        attention_mask=batch['attention_mask'], 
        labels=batch['labels']
    )
    return outputs.loss, {"logits": outputs.logits}

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, fast_eval: bool = True, **kwargs):
        self.fast_eval = fast_eval
        self.eval_dataset_for_metrics = kwargs.get('eval_dataset')
        super().__init__(*args, **kwargs)

    def llc_data_collator(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Data collator to prepare batches for LLC calculation.
        It tokenizes prompts and answers, combines them, and creates labels
        where the loss is only computed on the answer part.
        """
        prompts = [d['prompt'] for d in data]
        answers = [d['answer'] for d in data]
        
        prompt_strs = [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        
        # Get prompt lengths before padding
        prompt_lengths = [len(self.tokenizer.encode(s)) for s in prompt_strs]
        
        # Combine prompt and answer
        texts = [p + a + self.tokenizer.eos_token for p, a in zip(prompt_strs, answers)]
        
        # Tokenize with padding
        tokenized = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_completion_length
        )
        
        # Create labels and mask prompt part
        labels = tokenized.input_ids.clone()
        
        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100
        
        return {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask, "labels": labels}

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        num_samples = len(eval_dataset)

        # The log container is a deque with maxlen=generation_batch_size.
        # We must resize it to hold all evaluation samples.
        log_attr_name = '_textual_logs' if hasattr(self, '_textual_logs') else '_logs'
        log_container = getattr(self, log_attr_name)
        
        # Re-initialize the deques with a new, larger maxlen
        for key in log_container:
            if isinstance(log_container[key], defaultdict):
                log_container[key] = defaultdict(lambda: deque(maxlen=num_samples))
            else:
                log_container[key] = deque(maxlen=num_samples)
        
        print(f"--- Resized log container to hold {num_samples} evaluation samples. ---")

        original_num_generations = self.num_generations
        eval_generations = original_num_generations
        
        if self.fast_eval:
            self.num_generations = 1
            eval_generations = 1
        
        try:
            output_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            output_metrics["eval_generations_used"] = eval_generations
        except Exception as e:
            print(f"Error during evaluation: {e}")
                
        finally:
            self.num_generations = original_num_generations
            
        return output_metrics
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Overrides the dataloader creation to guarantee a sequential, non-shuffled
        order, which is critical for calculating metrics correctly.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        # Determine k and choose the sampler.
        if self.fast_eval:
            k = 1
            # For fast eval, a standard SequentialSampler is perfect. It is, by
            # definition, ordered and not shuffled.
            sampler = SequentialSampler(eval_dataset)
            batch_size = self.args.per_device_eval_batch_size
        else:
            k = self.args.num_generations
            # For full eval, we use the RepeatSampler but crucially pass `shuffle=False`.
            sampler = RepeatSampler(
                data_source=eval_dataset,
                mini_repeat_count=k,
                batch_size=self.args.per_device_eval_batch_size,
                repeat_count=1,
                shuffle=False
            )
            batch_size = self.args.per_device_eval_batch_size * k
            
        return DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """
        Overrides the log method to add custom accuracy metrics.
        This method is called by the Trainer at the end of each logging/evaluation step.
        """
        # This part handles the accuracy logging you already had.
        # It's kept separate from the new LLC logic.
        is_eval = not self.model.training
        log_attr_name = '_textual_logs' if hasattr(self, '_textual_logs') else '_logs'

        # Check if we are in an evaluation context and have the necessary data to compute accuracy
        if is_eval and self.eval_dataset_for_metrics is not None and hasattr(self, log_attr_name):
            log_container = getattr(self, log_attr_name)
            if "completion" in log_container and len(log_container["completion"]) > 0:
                k = self.num_generations if not self.fast_eval else 1
                completions_text = list(log_container["completion"])
                
                per_condition_scores = defaultdict(list)
                num_prompts_to_score = len(completions_text) // k
                
                extracted_answers = [extract_xml_answer(c) for c in completions_text]

                for i in range(num_prompts_to_score):
                    eval_example = self.eval_dataset_for_metrics[i]
                    true_answer = str(eval_example['answer'])
                    op = eval_example['operation']
                    digits = eval_example['num_digits']
                    terms = eval_example['num_terms']
                    condition_key = f"{op}_{digits}d_{terms}t"

                    prompt_answers = extracted_answers[i*k : (i+1)*k]
                    
                    is_correct = 1.0 if true_answer in prompt_answers else 0.0
                    per_condition_scores[condition_key].append(is_correct)
                
                overall_scores = []
                metric_prefix = f"accuracy_pass@{k}"
                for key, scores in per_condition_scores.items():
                    if scores:
                        logs[f"eval_{metric_prefix}/{key}"] = sum(scores) / len(scores)
                        overall_scores.extend(scores)
                if overall_scores:
                    logs[f"eval_{metric_prefix}/overall"] = sum(overall_scores) / len(overall_scores)

        super().log(logs)