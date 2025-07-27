import typing
from typing import Type, List, Dict
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from devinterp.optim.sgmcmc import SGMCMC
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.slt.llc import OnlineLLCEstimator
from devinterp.vis_utils import EpsilonBetaAnalyzer
from devinterp.utils import plot_trace, default_nbeta

# --- Configuration Flags ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DO_TUNING = False

# --- Data Preparation and Evaluation Functions ---

class LLCDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [d['prompt'] for d in data]
        answers = [d['answer'] for d in data]
        
        prompt_strs = [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        prompt_lengths = [len(self.tokenizer.encode(s, add_special_tokens=False)) for s in prompt_strs]
        texts = [p + a + self.tokenizer.eos_token for p, a in zip(prompt_strs, answers)]
        
        tokenized = self.tokenizer(
            texts, 
            padding='longest', 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
        
        labels = tokenized.input_ids.clone()
        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100
        
        return {"input_ids": tokenized.input_ids.to(DEVICE), "attention_mask": tokenized.attention_mask.to(DEVICE), "labels": labels.to(DEVICE)}

def evaluate_loss(model, batch: Dict[str, torch.Tensor]) -> typing.Tuple[torch.Tensor, dict]:
    outputs = model(
        input_ids=batch['input_ids'], 
        attention_mask=batch['attention_mask'], 
        labels=batch['labels']
    )
    return outputs.loss, {"logits": outputs.logits}

# --- Main Analysis Functions ---

def estimate_llc_given_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    evaluate: typing.Callable,
    epsilon: float,
    beta: float,
    sampling_method: Type[torch.optim.Optimizer] = SGMCMC.sgld,
    localization: float = 100.0,
    num_chains: int = 2,
    num_draws: int = 250,
    num_burnin_steps: int = 50,
    num_steps_bw_draws: int = 1,
    device: str = DEVICE,
    online: bool = True,
    verbose: bool = False,
):
    optimizer_kwargs = dict(lr=epsilon, localization=localization, nbeta=beta)
    
    sweep_stats = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        device=device,
        online=online,
        verbose=verbose,
    )

    sweep_stats["llc/trace"] = np.array(sweep_stats["llc/trace"])
    return sweep_stats

def tune_llc(model, loader):
    print("--- Starting Interactive Hyperparameter Tuning for LLC ---")
    while True:
        try:
            lr = float(input("Enter learning rate (e.g., 1e-5): "))
            gamma = float(input("Enter localization gamma (e.g., 100.0): "))
            nbeta = float(input(f"Enter temperature nbeta (recommended: {default_nbeta(loader):.2f}): "))
            num_draws = int(input("Enter number of draws (e.g., 200): "))
            num_burnin_steps = int(input("Enter number of burn-in steps (e.g., 200): "))
            num_chains = int(input("Enter number of chains (e.g., 2): "))
        except ValueError:
            print("Invalid input. Please enter numbers.")
            continue

        print("Running SGLD sampler to observe convergence...")
        llc_stats = estimate_learning_coeff_with_summary(
            model,
            loader=loader,
            evaluate=evaluate_loss,
            sampling_method=SGMCMC.sgld,
            optimizer_kwargs=dict(lr=lr, nbeta=nbeta, localization=gamma),
            num_chains=num_chains,
            num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            device=DEVICE,
            online=True,
            verbose=True,
        )
        llc_trace = llc_stats["llc/trace"]
        
        plot_trace(
            llc_trace,
            "LLC",
            x_axis="Draw",
            title=f"LLC Trace (lr={lr}, gamma={gamma}, nbeta={nbeta})\nFinal Mean LLC: {llc_stats['llc/means'][-1]:.3f}",
            plot_mean=True,
            plot_std=True,
            fig_size=(12, 6),
        )
        plt.show()

        if input("Are you happy with these results? (y/n): ").lower() == "y":
            break
    
    print("--- Hyperparameter tuning complete ---")
    return lr, gamma, nbeta, num_draws, num_burnin_steps, num_chains

def main():
    output_dir = "outputs/arithmetic-grpo"
    # CORRECTED: Set batch size to 4 for the analysis script
    batch_size = 4
    
    try:
        checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        checkpoint_dirs.sort(key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)))
        checkpoint_paths = [os.path.join(output_dir, d) for d in checkpoint_dirs]
        print(f"Found {len(checkpoint_paths)} checkpoints.")
    except (FileNotFoundError, AttributeError, TypeError):
        print(f"Error: Could not find valid checkpoint directories in {output_dir}")
        return

    if not checkpoint_paths:
        print("No checkpoints found. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_paths[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_dict = load_from_disk("arithmetic_dataset")
    val_dataset = dataset_dict['validation']

    data_collator = LLCDataCollator(tokenizer)
    loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)

    if DO_TUNING:
        print("--- Performing automated hyperparameter sweep on the final checkpoint ---")
        last_checkpoint_model = AutoModelForCausalLM.from_pretrained(checkpoint_paths[-1]).to(DEVICE)
        analyzer = EpsilonBetaAnalyzer()
        analyzer.configure_sweep(
            llc_estimator=estimate_llc_given_model,
            llc_estimator_kwargs=dict(
                model=last_checkpoint_model,
                evaluate=evaluate_loss,
                device=DEVICE,
                loader=loader,
                verbose=True,
            ),
            min_epsilon=1e-6,
            max_epsilon=1e-4,
            epsilon_samples=5,
            beta_range=[1.0, 5.0, 10.0],
        )
        analyzer.sweep()
        analyzer.plot().show()
        del last_checkpoint_model
        torch.cuda.empty_cache()
        
        print("\nLoading the final checkpoint for interactive tuning...")
        final_model = AutoModelForCausalLM.from_pretrained(checkpoint_paths[-1]).to(DEVICE)
        lr, gamma, nbeta, num_draws, num_burnin_steps, num_chains = tune_llc(final_model, loader)
        del final_model
        torch.cuda.empty_cache()
    else:
        print("Skipping tuning. Using default hyperparameters.")
        lr = 1e-5
        gamma = 100.0
        nbeta = default_nbeta(loader)
        num_draws = 200
        num_burnin_steps = 200
        num_chains = 2
        print(f"Defaults: lr={lr}, gamma={gamma}, nbeta={nbeta:.2f}, num_draws={num_draws}, burn-in={num_burnin_steps}, num_chains={num_chains}")

    print("\n--- Starting full LLC estimation across all checkpoints ---")
    all_results = []
    for checkpoint_path in checkpoint_paths:
        step = int(re.search(r"checkpoint-(\d+)", checkpoint_path).group(1))
        print(f"\nProcessing checkpoint at step {step}...")
        
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(DEVICE)
        
        checkpoint_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        
        llc_stats = estimate_learning_coeff_with_summary(
            model,
            loader=checkpoint_loader,
            evaluate=evaluate_loss,
            sampling_method=SGMCMC.sgld,
            optimizer_kwargs=dict(lr=lr, nbeta=nbeta, localization=gamma),
            num_chains=num_chains,
            num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            device=DEVICE,
            online=False,
            verbose=True,
        )
        
        all_results.append({
            "step": step,
            "llc_mean": llc_stats["llc/mean"],
            "llc_std": llc_stats["llc/std"],
        })
        
        del model
        torch.cuda.empty_cache()

    steps = [r["step"] for r in all_results]
    llc_means = [r["llc_mean"] for r in all_results]
    llc_stds = [r["llc_std"] for r in all_results]

    plt.figure(figsize=(12, 7))
    plt.errorbar(steps, llc_means, yerr=llc_stds, fmt='-o', capsize=5, label='LLC (Mean ± Std)')
    plt.xlabel("Training Steps")
    plt.ylabel("Local Learning Coefficient (LLC)")
    plt.title("LLC Evolution During GRPO Training")
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig("llc_evolution.png")
    print("\nSaved final plot to llc_evolution.png")
    plt.show()

if __name__ == "__main__":
    main()