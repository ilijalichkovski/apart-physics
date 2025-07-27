import re
from collections import defaultdict

import torch
from trl import GRPOTrainer
from typing import Optional, Any, Union, Dict, List
from torch.utils.data import DataLoader
from datasets import Dataset
from collections import deque

from trl.trainer.grpo_trainer import RepeatSampler
from torch.utils.data import SequentialSampler


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

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, fast_eval: bool = True, **kwargs):
        self.fast_eval = fast_eval
        self.eval_dataset_for_metrics = kwargs.get('eval_dataset')
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        num_samples = len(eval_dataset)

        # The log container is a deque with maxlen=generation_batch_size (16).
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

        # --- The rest of the state management logic from before ---
        original_num_generations = self.num_generations
        eval_generations = original_num_generations
        
        if self.fast_eval:
            self.num_generations = 1
            eval_generations = 1
        
        try:
            output_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            output_metrics["eval_generations_used"] = eval_generations
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
        is_eval = not self.model.training

        # Use the correct variable name for your installed version of trl.
        log_attr_name = '_textual_logs' if hasattr(self, '_textual_logs') else '_logs'
        k = 1

        if is_eval and self.eval_dataset_for_metrics is not None:
            log_container = getattr(self, log_attr_name)
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