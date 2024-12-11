import argparse
import matplotlib.pyplot as plt
import math
import os

import torch
from torchvision.models import ViT_B_16_Weights
import torch.optim as optim
import timm

from training import fine_tune_model
from src.utils import save_state
from src.validation import evaluate, get_val_dataset
from src.pruning import (
    count_nonzero_params,
    count_total_params,
    prune_model,
    replace_layers,
)

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Pruning script for ViT model")
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to use for computation"
)
parser.add_argument("--save_path", type=str, default=".", help="Path to save the model")
parser.add_argument(
    "--plot", type=bool, default=True, help="Whether to plot the results"
)
args = parser.parse_args()


# Set device for computation
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
preprocess = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
val_loader = get_val_dataset(preprocess=preprocess)

# Define hyperparameters of pruning
alpha = 0.25
beta = 0.8
n_prune_cycles = 35  # Number of pruning cycles
fine_tune_epochs = 5000  # Number of epochs for fine-tuning after each pruning
learning_rate = 0.05e-6  # Learning rate for fine-tuning
l1_lambda = 0.0000005  # Adjust as needed for L1 regularization
weight_decay = 0.0000002  # Adjust as needed for L2 regularization
GoF = 1


# Revised training and pruning loop
accuracies1, accuracies5, total_num_para = [], [], []

# Set base path and create directories for saving models
base_path = args.save_path
project_name = "ViTL16_pruning"
model_save_path = os.path.join(base_path, project_name, "saved_models")
os.makedirs(model_save_path, exist_ok=True)

# Initialize model and optimizer
model = timm.create_model("vit_large_patch16_224", pretrained=True).to(device)  # type: ignore
replace_layers(model, alpha, beta, GoF, depth=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Save initial state
save_state(
    model=model,
    optimizer=optimizer,
    cycle_index=0,
    fine_tune_epochs=fine_tune_epochs,
    learning_rate=learning_rate,
    filename_prefix="initial",
    model_save_path=model_save_path,
    accuracies1=accuracies1,
    accuracies5=accuracies5,
    total_num_para=total_num_para,
)

# Evaluate unpruned model
top1_unpruned, top5_unpruned = evaluate(val_loader, model, device)
top1_unpruned = (
    top1_unpruned.cpu().item()
    if isinstance(top1_unpruned, torch.Tensor)
    else top1_unpruned
)
top5_unpruned = (
    top5_unpruned.cpu().item()
    if isinstance(top5_unpruned, torch.Tensor)
    else top5_unpruned
)

# Append to accuracies and parameter lists
accuracies1.append(top1_unpruned)
accuracies5.append(top5_unpruned)
total_num_para.append(model)


# Pruning loop
for i in range(1, n_prune_cycles):
    print(f"Starting pruning cycle {i}")

    model.to(device)
    layers = []
    for name, module in model.named_modules():
        try:
            module.goodnessOfFitCutoff
            layers.append((name, module))
        except:
            pass

    target_reduction = 0.06 * 1.02**i
    linf_errors = prune_model(model, target_reduction, i, n_prune_cycles, device)

    # Save state after pruning
    save_state(
        model=model,
        optimizer=optimizer,
        cycle_index=i,
        fine_tune_epochs=fine_tune_epochs,
        learning_rate=learning_rate,
        filename_prefix="pruned",
        accuracies1=accuracies1,
        accuracies5=accuracies5,
        total_num_para=total_num_para,
        model_save_path=model_save_path,
    )

    total_epochs = math.floor(fine_tune_epochs)

    # Calculate the number of epochs for each fine-tuning function
    epochs_for_fine_tune_2 = math.floor(total_epochs * 1)  # 1% of epochs
    epochs_for_fine_tune_1 = math.floor(
        total_epochs - epochs_for_fine_tune_2
    )  # Remaining 90% of epochs

    model, optimizer = fine_tune_model(
        model,
        i,
        epochs_for_fine_tune_2,
        learning_rate,
        l1_lambda,
        weight_decay,
        linf_errors,
    )

    # Save state after fine-tuning
    save_state(
        model=model,
        optimizer=optimizer,
        cycle_index=i,
        fine_tune_epochs=fine_tune_epochs,
        learning_rate=learning_rate,
        filename_prefix="pruned",
        accuracies1=accuracies1,
        accuracies5=accuracies5,
        total_num_para=total_num_para,
        model_save_path=model_save_path,
    )

    fine_tune_epochs += 1000

    num_nonzero = count_nonzero_params(model)
    print(f"Number of non zero parameters: {num_nonzero}")
    top1, top5 = evaluate(val_loader, model, device)
    accuracies1.append(top1)
    accuracies5.append(top5)
    total_num_para.append(num_nonzero)

    # Save state after evaluation
    save_state(
        model=model,
        optimizer=optimizer,
        cycle_index=i,
        fine_tune_epochs=fine_tune_epochs,
        learning_rate=learning_rate,
        filename_prefix="pruned",
        accuracies1=accuracies1,
        accuracies5=accuracies5,
        total_num_para=total_num_para,
        model_save_path=model_save_path,
    )

# Evaluate unpruned model
unpruned_resnet = timm.create_model("vit_large_patch16_224", pretrained=True).to(device)  # type: ignore
num_params_unpruned = count_total_params(unpruned_resnet)
top1_unpruned, top5_unpruned = evaluate(val_loader, unpruned_resnet, device)
top1_unpruned = (
    top1_unpruned.cpu().item()
    if isinstance(top1_unpruned, torch.Tensor)
    else top1_unpruned
)
top5_unpruned = (
    top5_unpruned.cpu().item()
    if isinstance(top5_unpruned, torch.Tensor)
    else top5_unpruned
)

# Append to accuracies and parameter lists
accuracies1.append(top1_unpruned)
accuracies5.append(top5_unpruned)
total_num_para.append(num_params_unpruned)
params_kept_percentages = [
    100 * total / num_params_unpruned for total in total_num_para
]

# Convert elements to CPU numpy arrays or Python scalars if they are tensors
accuracies1 = [
    acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accuracies1
]
accuracies5 = [
    acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accuracies5
]
total_num_para = [
    param.cpu().item() if isinstance(param, torch.Tensor) else param
    for param in total_num_para
]

# Plot the results if PLOT is True
if args.plot:
    plt.figure(figsize=(10, 8))
    plt.plot(total_num_para, accuracies1, label="Top 1 Accuracy")
    plt.plot(total_num_para, accuracies5, label="Top 5 Accuracy")

    # Add a horizontal line for the unpruned model accuracy
    plt.axhline(
        y=top1_unpruned,
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Unpruned Top 1",
    )
    plt.axhline(
        y=top5_unpruned,
        color="purple",
        linestyle="-",
        linewidth=2,
        label="Unpruned Top 5",
    )

    plt.xlabel("Number of Non-zero Parameters")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Non-zero Parameters")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(range(n_prune_cycles + 1), accuracies1, label="Top 1 Accuracy")
    plt.plot(range(n_prune_cycles + 1), accuracies5, label="Top 5 Accuracy")

    # Add a horizontal line for the unpruned model accuracy
    plt.axhline(
        y=top1_unpruned,
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Unpruned Top 1",
    )
    plt.axhline(
        y=top5_unpruned,
        color="purple",
        linestyle="-",
        linewidth=2,
        label="Unpruned Top 5",
    )

    plt.xlabel("Pruning Cycles")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy over Pruning Cycles")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Convert accuracies and parameters to NumPy arrays for easier handling
    acc1_np = torch.Tensor(accuracies1).detach().cpu().numpy()
    acc5_np = torch.Tensor(accuracies5).detach().cpu().numpy()
    total_num_para_np = torch.Tensor(total_num_para).detach().cpu().numpy()

    plt.figure(figsize=(10, 8))

    # Plot Top 1 Accuracy
    for x, y in zip(total_num_para_np, acc1_np):
        plt.scatter(x, y, marker="o", s=100)  # s is the size of the dot
        plt.annotate(
            f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    # Add unpruned DNN data point and horizontal line
    plt.scatter(
        num_params_unpruned,
        top1_unpruned,
        color="purple",
        marker="o",
        s=100,
        label="Unpruned",
    )
    plt.axhline(
        y=top1_unpruned,
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Unpruned Baseline",
    )

    plt.xlabel("Number of Non-zero Parameters")
    plt.ylabel("Top 1 Accuracy")
    plt.title("Top 1 Accuracy vs Number of Non-zero Parameters")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 8))

    # Plot Top 5 Accuracy
    for x, y in zip(total_num_para_np, acc5_np):
        plt.scatter(x, y, marker="o", s=100)
        plt.annotate(
            f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    # Add unpruned DNN data point and horizontal line
    plt.scatter(
        num_params_unpruned,
        top5_unpruned,
        color="purple",
        marker="o",
        s=100,
        label="Unpruned",
    )
    plt.axhline(
        y=top5_unpruned,
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Unpruned Baseline",
    )

    plt.xlabel("Number of Non-zero Parameters")
    plt.ylabel("Top 5 Accuracy")
    plt.title("Top 5 Accuracy vs Number of Non-zero Parameters")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 8))

    # Plot Top 1 Accuracy over Pruning Cycles
    for i, acc in enumerate(accuracies1):
        plt.scatter(i, acc, marker="o", s=100)  # s is the size of the dot
        plt.annotate(
            f"{acc:.1f}%",
            (i, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Add a horizontal line for the unpruned model Top 1 accuracy
    plt.axhline(
        y=top1_unpruned,
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Unpruned Top 1",
    )

    # Plot Top 5 Accuracy over Pruning Cycles
    for i, acc in enumerate(accuracies5):
        plt.scatter(i, acc, marker="o", s=100)
        plt.annotate(
            f"{acc:.1f}%",
            (i, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Add a horizontal line for the unpruned model Top 5 accuracy
    plt.axhline(
        y=top5_unpruned,
        color="purple",
        linestyle="-",
        linewidth=2,
        label="Unpruned Top 5",
    )

    plt.xlabel("Pruning Cycles")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy over Pruning Cycles")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Top 1 Accuracy over Pruning Cycles
    plt.figure(figsize=(10, 8))
    for x, y in zip(n_prune_cycles, acc1_np):  # type: ignore
        plt.scatter(x, y, marker="o", s=100)
        plt.annotate(
            f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    plt.axhline(
        y=top1_unpruned,
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Unpruned Top 1",
    )

    plt.xlabel("Pruning Cycles")
    plt.ylabel("Top 1 Accuracy")
    plt.title("Top 1 Accuracy over Pruning Cycles")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Top 5 Accuracy over Pruning Cycles
    plt.figure(figsize=(10, 8))
    for x, y in zip(n_prune_cycles, acc5_np):  # type: ignore
        plt.scatter(x, y, marker="o", s=100)
        plt.annotate(
            f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    plt.axhline(
        y=top5_unpruned,
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Unpruned Top 5",
    )

    plt.xlabel("Pruning Cycles")
    plt.ylabel("Top 5 Accuracy")
    plt.title("Top 5 Accuracy over Pruning Cycles")
    plt.legend()
    plt.grid(True)
    plt.show()
