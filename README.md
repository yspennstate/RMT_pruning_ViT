# Random Matrix Theory pruning of Vision Transformers 

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

This repository contains the code for the paper "Efficient Pruning of Vision Transformers using Random Matrix Theory". The code provides implementations for pruning Vision Transformers (ViT) using Random Matrix Theory (RMT) techniques.

## Purpose

The main goal of this project is to efficiently prune Vision Transformers (ViT) to reduce their size and computational requirements while maintaining their performance. This is achieved using Random Matrix Theory (RMT) to guide the pruning process.

## Repository Structure

- `validation.py`: Contains functions for evaluating the model.
- `utils.py`: Utility functions used across the project.
- `training.py`: Functions for training and fine-tuning the model.
- `SplittableLayers.py`: Custom layers that support splitting and pruning.
- `RMT.py`: Implementation of Random Matrix Theory functions.
- `pruning.py`: Functions for pruning the model.
- `prune.py`: Script for pruning and fine-tuning the model.
- `fine_tune.py`: Script for fine-tuning the pruned model.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Fine-Tuning

To fine-tune the Vision Transformer model, use the `fine_tune.py` script. This script fine-tunes a pre-trained ViT model with specified hyperparameters.

#### Command

```bash
python fine_tune.py --device <device> --weights_path <path_to_weights> --save_path <path_to_save_model>
```

#### Arguments

- `--device`: Device to use for computation (default: `cuda:0`).
- `--weights_path`: Path to the model weights to fine-tune.
- `--save_path`: Path to save the fine-tuned model.

### Pruning

To prune the Vision Transformer model, use the `prune.py` script. This script prunes the model using RMT techniques and fine-tunes it after each pruning cycle.

#### Command

```bash
python prune.py --device <device> --save_path <path_to_save_model> --plot <True/False>
```

#### Arguments

- `--device`: Device to use for computation (default: `cuda:0`).
- `--save_path`: Path to save the pruned model.
- `--plot`: Whether to plot the results (default: `True`).

## Citation

If you use this code in your research, please consider citing the following paper:

```
@article{rmt_pruning_vit,
  title={Efficient Pruning of Vision Transformers using Random Matrix Theory},
  author={Authors},
  journal={Journal},
  year={2022}
}
```