import typing
from typing import Type
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import os

from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce
from devinterp.vis_utils import EpsilonBetaAnalyzer
from devinterp.utils import plot_trace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def estimate_llc_given_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    evaluate: typing.Callable,
    epsilon: float,
    beta: float,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    localization: float = 5.0,
    num_chains: int = 2,
    num_draws: int = 500,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    device: torch.device = DEVICE,
    online: bool = True,
    verbose: bool = False,
):
    sweep_stats = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        optimizer_kwargs=dict(lr=epsilon, localization=localization, nbeta=beta),
        num_chains=num_chains,  # How many independent chains to run
        num_draws=num_draws,  # How many samples to draw per chain
        num_burnin_steps=num_burnin_steps,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=num_steps_bw_draws,  # How many steps to take between each sample
        device=device,
        online=online,
        verbose=verbose,
    )

    sweep_stats["llc/trace"] = np.array(sweep_stats["llc/trace"])
    return sweep_stats

def tune_llc(all_checkpointed_models, train_dataset):
    """
    Choosing the LLC hyperparameters for the given model checkpoints and dataset.
    """
    
    print("Tuning LLC hyperparameters...")
    while True:

        lr = float(input("Enter the learning rate: "))
        gamma = int(input("Enter the gamma: "))
        nbeta = float(input("Enter the nbeta: "))
        num_draws = int(input("Enter the number of draws: "))
        num_chains = int(input("Enter the number of chains: "))

        learning_coeff_stats = estimate_learning_coeff_with_summary(
            all_checkpointed_models[-1],
            loader=DataLoader(train_dataset, batch_size=256, shuffle=True),
            evaluate=evaluate_ce,
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=lr, nbeta=nbeta, localization=gamma),
            num_chains=num_chains,
            num_draws=num_draws,
            device=DEVICE,
            online=True,
        )
        trace = learning_coeff_stats["loss/trace"]

        plot_trace(
            trace,
            "Loss",
            x_axis="Step",
            title=f"Loss Trace, avg LLC = {sum(learning_coeff_stats['llc/means']) / len(learning_coeff_stats['llc/means']):.2f}",
            plot_mean=False,
            plot_std=False,
            fig_size=(12, 9),
            true_lc=None,
        )

        if input("Are you happy with the results? (y/n): ") == "y":
            break
    return lr, gamma, nbeta, num_draws, num_chains

def main():
    
    output_dir = "outputs/arithmetic-grpo"
    models_list = os.listdir(output_dir)
    all_checkpointed_models = [torch.load(f"{output_dir}/{model}") for model in models_list]

    dataset_dict = load_from_disk("arithmetic_dataset")
    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['validation']

    loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
    analyzer = EpsilonBetaAnalyzer()
    analyzer.configure_sweep(
        llc_estimator=estimate_llc_given_model,
        llc_estimator_kwargs=dict(
            model=all_checkpointed_models[-1],
            evaluate=evaluate_ce,
            device=DEVICE,
            loader=loader,
        ),
        min_epsilon=3e-5,
        max_epsilon=3e-1,
        epsilon_samples=5,
        min_beta=None,
        max_beta=None,
        beta_samples=5,
        dataloader=loader,
    )
    analyzer.sweep()

    analyzer.plot()

    lr, gamma, nbeta, num_draws, num_chains = tune_llc(all_checkpointed_models, train_dataset)

    print("Hyperparameter tuning complete; starting full LLC estimation")

    llcs = [
        estimate_learning_coeff_with_summary(
            model_checkpoint,
            loader=DataLoader(train_dataset, batch_size=256, shuffle=True),
            evaluate=evaluate_ce,
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=lr, nbeta=nbeta, localization=gamma),
            num_chains=1,
            num_draws=num_draws,
            device=DEVICE,
            online=False,
        )
        for model_checkpoint in all_checkpointed_models
    ]

    llc_means = [llc["llc/mean"] for llc in llcs]
    llc_stds = [llc["llc/std"] for llc in llcs]

    plt.plot(llc_means)
    plt.plot(llc_stds)
    plt.show()

if __name__ == "__main__":
    main()