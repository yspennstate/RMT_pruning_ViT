# RMT Pruning ViT

This repository contains the code for the paper "Efficient Pruning of Vision Transformers using Random Matrix Theory" (link will be updated later). The code provides implementations for pruning Vision Transformers (ViT) using Random Matrix Theory (RMT) techniques.

## Repository Structure

- `validation.py`: Contains functions for evaluating the model.
- `utils.py`: Utility functions used across the project.
- `training.py`: Functions for training and fine-tuning the model.
- `SplittableLayers.py`: Custom layers that support splitting and pruning.
- `RMT.py`: Implementation of Random Matrix Theory functions.
- `pruning.py`: Functions for pruning the model.
- `pruning_script.py`: Script for running the pruning process.
- `prune.py`: Script for pruning and fine-tuning the model.
- `flops.py`: Functions for calculating FLOPs (Floating Point Operations).
- `fine_tune.py`: Script for fine-tuning the pruned model.

## Usage

1. **Training and Fine-Tuning**: Use `training.py` and `fine_tune.py` to train and fine-tune the Vision Transformer model.
2. **Pruning**: Use `pruning.py` and `prune.py` to prune the model using RMT techniques.
3. **Evaluation**: Use `validation.py` to evaluate the performance of the model.
4. **FLOPs Calculation**: Use `flops.py` to calculate the FLOPs of the model.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

# Citation

If you use this code in your research, please consider citing the following paper:

```
@article{rmt_pruning_vit,
  title={Efficient Pruning of Vision Transformers using Random Matrix Theory},
  author={Authors},
  journal={Journal},
  year={2022}
}
```
