# apart-physics

**What do localized learning coefficients tell us during GRPO training?**

A research project investigating the relationship between Localized Learning Coefficients (LLCs) and training dynamics during Gradient-based Reward Policy Optimization (GRPO). This work explores how LLCs can reveal phase transitions and learning patterns in reinforcement learning from human feedback (RLHF) training.

## 🔬 Research Overview

This project analyzes the behavior of neural networks during GRPO training by computing Localized Learning Coefficients - a measure that captures how much each parameter contributes to the model's learning at different stages of training. By correlating LLC dynamics with reward signals and accuracy metrics, we aim to understand:

- **Phase transitions** during RLHF training
- **Learning dynamics** and their relationship to model performance  
- **Mechanistic insights** into how models adapt during policy optimization

## 📊 Key Findings

Our analysis reveals distinct phase transitions during training, highlighted by:
- **Suspected phase transition regions**: Steps 44-66, 88-110, and 187-209
- **Correlated changes** in LLCs, reward functions, and accuracy metrics
- **Visual evidence** of learning dynamics through comprehensive plotting analysis

## 🚀 Quick Start

### Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ilijalichkovski/apart-physics.git
cd apart-physics

# Install dependencies
uv sync
```

## 🛠️ Usage

### 1. Generate Training Dataset

Create the arithmetic dataset for training:

```bash
uv run python generate_dataset.py
```

This generates:
- `arithmetic_dataset_train.jsonl` - Training examples
- `arithmetic_dataset_validation.jsonl` - Validation examples
- `arithmetic_dataset/` - Processed dataset directory

### 2. Train the Model with GRPO

Run the GRPO training process:

```bash
uv run python grpo.py
```

This will:
- Load the generated dataset
- Train the model using Gradient-based Reward Policy Optimization
- Save model checkpoints to `outputs/`
- Log training metrics to `wandb/`

### 3. Compute Localized Learning Coefficients

Calculate LLCs from the saved checkpoints:

```bash
uv run python llc.py
```

This analyzes the checkpoints and generates:
- `llc_results.csv` - LLC metrics over training steps
- Detailed analysis of parameter-wise learning dynamics

### 4. Generate Analysis Plots

Create comprehensive visualizations:

```bash
# Open the Jupyter notebook
uv run jupyter lab results/create_plots.ipynb
```

Or run the plotting script directly to generate high-DPI figures:
- `reward_plots_error_bars.png` - Stacked LLC and reward analysis
- `reward_plots_combined.png` - Combined reward comparison
- `accuracy_plots_grid.png` - Accuracy performance grid

## 📁 Project Structure

```
apart-physics/
├── generate_dataset.py      # Dataset generation script
├── grpo.py                  # GRPO training implementation  
├── llc.py                   # LLC computation and analysis
├── custom_trainer.py        # Custom training utilities
├── results/
│   ├── create_plots.ipynb   # Comprehensive plotting notebook
│   ├── wandb_results.csv    # Training metrics
│   ├── wandb_accuracies.csv # Accuracy data
│   └── llc_results.csv      # LLC analysis results
├── arithmetic_dataset/      # Generated training data
├── outputs/                 # Model checkpoints and artifacts
└── wandb/                   # Weights & Biases logs
```