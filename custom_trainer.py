import re
from collections import defaultdict

import torch
from trl import GRPOTrainer
from typing import Optional, Any, Union, Dict, List
from torch.utils.data import DataLoader
from datasets import Dataset


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
    """
    A custom GRPOTrainer that provides an option for fast single-generation
    evaluation by correctly orchestrating the trainer's internal state.
    
    Args:
        fast_eval (bool): If True, evaluation will use only 1 generation for speed.
                          If False, it will use the training `num_generations`.
    """
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
        """
        Overrides the main evaluation method to temporarily set num_generations to 1
        if `fast_eval` is True. This ensures that all internal logic, from the
        sampler to the reward reshaping, is in sync.
        """
        # --- Store original state and temporarily modify it for evaluation ---
        original_num_generations = self.num_generations
        # This variable will hold the number of generations used in this specific run
        eval_generations = original_num_generations
        
        if self.fast_eval:
            self.num_generations = 1
            eval_generations = 1
            print(f"--- Starting fast evaluation with num_generations = {self.num_generations} ---")
        else:
            print(f"--- Starting full evaluation with num_generations = {self.num_generations} ---")

        # We use a try...finally block to ensure the original value is always restored.
        try:
            # Call the parent's evaluate method. It will now use the modified
            # self.num_generations (if fast_eval=True) for ALL its internal
            # operations, including creating the sampler and reshaping rewards.
            output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            
            # Pass the number of generations used to the logs for our custom metric calculation
            output["eval_generations_used"] = eval_generations

        finally:
            # Restore the original value for subsequent training steps
            if self.fast_eval:
                print(f"--- Restoring num_generations to {original_num_generations} for training ---")
                self.num_generations = original_num_generations
            
        return output

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