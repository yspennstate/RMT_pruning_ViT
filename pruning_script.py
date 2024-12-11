# %%


# %%
import json
import matplotlib.pyplot as plt
import random
from random import shuffle
import math
import numpy as np
import pandas as pd
from PIL import Image
import glob
import os
from tqdm.notebook import tqdm as tqdm
import seaborn as sns
from sklearn.model_selection import ParameterGrid  # type: ignore

import torch
from torchvision import datasets
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import timm  # type: ignore


from SplittableLayers import (
    SplittableConv,
    SplittableLinear,
)
from training import fine_tune_model_2  # type: ignore
from utils import load_checkpoint, save_state
from flops import calculate_flops, calculate_vit_flops
from validation import evaluate, get_val_dataset
from pruning import (
    naive_prune,
    count_nonzero_params,
    count_total_params,
    prune_model,
    replace_layers,
    perform_splitting,
)

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
preprocess = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()

# %%


def test_parameters(alpha, beta, goodnessOfFitCutoff, split):
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
    vit.eval()
    vit.to(device)
    replace_layers(vit, alpha, beta, goodnessOfFitCutoff)
    perform_splitting(vit, split)
    vit.to(device)
    val = get_val_dataset(preprocess=preprocess)
    return *evaluate(val, vit, device), sum(p.numel() for p in vit.parameters())


# Load the pre-trained model
model = timm.create_model("vit_large_patch16_224", pretrained=True).to(device)

# Resolve data configuration from timm
data_config = timm.data.resolve_model_data_config(model)
preprocess = timm.data.create_transform(**data_config, is_training=False)

# Load the dataset with preprocessing
dataset = datasets.ImageFolder("imagenet/val", transform=preprocess)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Shuffle indices for the entire dataset
indices = list(range(len(dataset)))
random.shuffle(indices)

# Calculate sizes for training and validation subsets
training_size = int(len(dataset) * 0.0) + 1
validation_size = len(dataset) - training_size

# Split indices for training and validation
train_indices = indices[:training_size]
val_indices = indices[training_size : training_size + validation_size]

# Create subsets for training and validation
train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

# Create DataLoaders for the subsets
train_loader = DataLoader(train_subset, batch_size=50, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)

# Print sizes to verify the splits
print(f"Total dataset size: {len(dataset)}")
print(f"Validation set size: {len(val_subset)}")
print(
    f"Training set size: {len(train_subset)}"
)  # Should be about 1% of the validation set size


# num_params = 1 gof= 0.1 0.05 0.02 0.01
alpha = 0.25
beta = 0.8
n_prune_cycles = 35  # Number of pruning cycles
fine_tune_epochs = 500  # Number of epochs for fine-tuning after each pruning
learning_rate = 0.05e-6  # Learning rate for fine-tuning
l1_lambda = 0.0000005  # Adjust as needed for L1 regularization
weight_decay = 0.0000002  # Adjust as needed for L2 regularization
GoF = 1


# Revised training and pruning loop
accuracies1, accuracies5, total_num_para = [], [], []


# unpruned_resnet = unpruned_resnet.to(device)


base_path = "/content/drive/My Drive/ModelExperiments"
project_name = "ViTL16_pruning_1"  # Ensure there's no space or '.ipynb' unless it's actually part of the folder name
model_save_path = os.path.join(base_path, project_name, "saved_models")
os.makedirs(model_save_path, exist_ok=True)


# Load a specific checkpoint


model = timm.create_model("vit_large_patch16_224", pretrained=True).to(device)
replace_layers(model, alpha, beta, GoF, depth=0)


optimizer = optim.Adam(model.parameters(), lr=0.05e-6)  # Adjust lr as needed


## Existing model save path
existing_model_save_path = (
    "/content/drive/My Drive/ModelExperiments/ViTL16_pruning_1/saved_models"
)


# Initialize model and optimizer
model = timm.create_model("vit_large_patch16_224", pretrained=True).to(device)
replace_layers(model, alpha, beta, GoF, depth=0)
new_learning_rate = 0.05e-6  # Define the new learning rate here
optimizer = optim.Adam(model.parameters(), lr=new_learning_rate)

replace_layers(model, alpha, beta, 1, depth=0)
# Load the checkpoint for cycle 6 from the existing experiment
checkpoint_filename = "pruned_cycle_0.pth.tar"
checkpoint = load_checkpoint(model, checkpoint_filename, optimizer)


if checkpoint:
    # Print out some details from the loaded checkpoint
    last_completed_cycle = checkpoint["cycle_index"]
    fine_tune_epochs = 300
    learning_rate = 0.05e-6
    accuracies1 = checkpoint.get("accuracies1", [])
    accuracies5 = checkpoint.get("accuracies5", [])
    total_num_para = checkpoint.get("total_num_para", [])
    train_indices = checkpoint.get("train_indices", [])
    val_indices = checkpoint.get("val_indices", [])

    # Continue processing or resume training
    print(f"Resuming from cycle: {last_completed_cycle+1}")
else:
    print("Failed to load the specified checkpoint.")


# Assuming you save the state after creating these subsets
save_state(
    model=model,
    optimizer=optimizer,
    cycle_index=0,
    fine_tune_epochs=fine_tune_epochs,
    learning_rate=learning_rate,
    filename_prefix="initial",
    train_indices=train_indices,
    val_indices=val_indices,
    model_save_path=model_save_path,
    accuracies1=accuracies1,
    accuracies5=accuracies5,
    total_num_para=total_num_para,
)


# num_params_unpruned = count_total_params(unpruned_resnet)
# top1_unpruned, top5_unpruned = evaluate(val_loader, unpruned_resnet)


top1_unpruned, top5_unpruned = evaluate(val_loader, model, device)


# Convert to Python scalars if necessarytop
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
# params_kept_percentages = [100 * total / num_params_unpruned for total in total_num_para]


# num_params_unpruned = count_total_params(unpruned_resnet)
# top1_unpruned, top5_unpruned = evaluate(val_loader, unpruned_resnet)


fine_tune_epochs = 5000

# last_completed_cycle = 0

last_completed_cycle = 0
print(f"Starting pruning cycle {last_completed_cycle}")

for i in range(last_completed_cycle + 1, n_prune_cycles):
    print(f"Starting pruning cycle {i}")

    # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    # model.to(device)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    GOF = 1

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

    # top1,top5 = evaluate(val_loader,model)

    # Save state after pruning

    print(f"Type of model before save_state: {type(model)}")
    print(f"Type of optimizer before save_state: {type(optimizer)}")

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

    model, optimizer = fine_tune_model_2(
        model,
        i,
        train_loader,
        epochs_for_fine_tune_2,
        learning_rate,
        l1_lambda,
        weight_decay,
        linf_errors,
    )

    # model, optimizer = fine_tune_model_1(model, i, train_loader, epochs_for_fine_tune_1, learning_rate, l1_lambda, weight_decay,linf_errors)

    # Note: Assume fine_tune_model_1 has similar parameters; adjust if different

    # model = fine_tune_model_2(model, train_loader, math.floor(fine_tune_epochs),learning_rate, l1_lambda, weight_decay)

    # model, optimizer = fine_tune_model_2(model,i, train_loader, math.floor(fine_tune_epochs),learning_rate, l1_lambda, weight_decay,linf_errors)

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

    fine_tune_epochs = fine_tune_epochs + 1000

    # fine_tune_epochs = min(fine_tune_epochs + 5,70)

    # l1_lambda = l1_lambda*1.05
    # weight_decay = weight_decay*1.05

    num_nonzero = count_nonzero_params(model)
    print(num_nonzero)
    top1, top5 = evaluate(val_loader, model, device)
    # top1_new,top5_new=evaluate(val_loader_2,model)
    accuracies1.append(top1)
    accuracies5.append(top5)
    total_num_para.append(num_nonzero)
    print(top1)
    print(top5)
    print(num_nonzero)
    # Save the model
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unpruned_resnet = timm.create_model("vit_large_patch16_224", pretrained=True).to(device)

# unpruned_resnet = timm.create_model('deit_tiny_patch16_224', pretrained=True)

unpruned_resnet = unpruned_resnet.to(device)


# unpruned_resnet = unpruned_resnet.to(device)

num_params_unpruned = count_total_params(unpruned_resnet)
top1_unpruned, top5_unpruned = evaluate(val_loader, unpruned_resnet, device)

# Convert to Python scalars if necessary
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

plt.figure(figsize=(10, 8))
plt.plot(total_num_para, accuracies1, label="Top 1 Accuracy")
plt.plot(total_num_para, accuracies5, label="Top 5 Accuracy")

# Add a horizontal line for the unpruned model accuracy
plt.axhline(
    y=top1_unpruned, color="purple", linestyle="--", linewidth=2, label="Unpruned Top 1"
)
plt.axhline(
    y=top5_unpruned, color="purple", linestyle="-", linewidth=2, label="Unpruned Top 5"
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
    y=top1_unpruned, color="purple", linestyle="--", linewidth=2, label="Unpruned Top 1"
)
plt.axhline(
    y=top5_unpruned, color="purple", linestyle="-", linewidth=2, label="Unpruned Top 5"
)

plt.xlabel("Pruning Cycles")
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Pruning Cycles")
plt.legend()
plt.grid(True)
plt.show()


# Assuming all necessary imports and data preparation is done.

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
        f"{acc:.1f}%", (i, acc), textcoords="offset points", xytext=(0, 10), ha="center"
    )

# Add a horizontal line for the unpruned model Top 1 accuracy
plt.axhline(
    y=top1_unpruned, color="purple", linestyle="--", linewidth=2, label="Unpruned Top 1"
)

# Plot Top 5 Accuracy over Pruning Cycles
for i, acc in enumerate(accuracies5):
    plt.scatter(i, acc, marker="o", s=100)
    plt.annotate(
        f"{acc:.1f}%", (i, acc), textcoords="offset points", xytext=(0, 10), ha="center"
    )

# Add a horizontal line for the unpruned model Top 5 accuracy
plt.axhline(
    y=top5_unpruned, color="purple", linestyle="-", linewidth=2, label="Unpruned Top 5"
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
    y=top1_unpruned, color="purple", linestyle="--", linewidth=2, label="Unpruned Top 1"
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
    y=top5_unpruned, color="purple", linestyle="--", linewidth=2, label="Unpruned Top 5"
)

plt.xlabel("Pruning Cycles")
plt.ylabel("Top 5 Accuracy")
plt.title("Top 5 Accuracy over Pruning Cycles")
plt.legend()
plt.grid(True)
plt.show()


class TestDataset(Dataset):

    def __init__(self, path, transform=None):

        self.image_paths = glob.glob(path + "*.JPEG")

        self.transform = transform

    def __getitem__(self, index):

        path = self.image_paths[index]

        x = Image.open(path).convert("RGB")

        if self.transform is not None:

            x = self.transform(x)

        return path.split("/")[-1], x

    def __len__(self):

        return len(self.image_paths)


# here preprocess is the same as the one used in train and val datasets

test_dataset = TestDataset("./imagenet/test/", transform=preprocess)

testloader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# Then the evaluation code is:


def get_test_submission(model, dataloader, device):

    model.eval()

    # prepare a df for submission

    submission = pd.DataFrame(columns=[f"choice {i}" for i in range(1, 6)])

    steps = 0

    with torch.no_grad():

        for paths, images in dataloader:

            images = images.to(device)

            outputs = model(images)

            # get top 5 predictions

            _, predicted = torch.topk(outputs, 5)

            submission_batch = pd.DataFrame(
                data=predicted.cpu().numpy(), columns=submission.columns, index=paths
            )

            submission = pd.concat([submission, submission_batch])

            steps += 1

            if steps == 20:

                break

    # sort submission by index

    submission = submission.sort_index()

    return submission


# You can get the submission by running:

submission = get_test_submission(model, testloader, device)

submission.to_csv("submission.csv", index=False, header=False)


def select_models(
    accuracies1,
    accuracies5,
    total_num_para,
    num_params_unpruned,
    target_reductions=[25, 40],
):
    best_top1_model = np.argmax(accuracies1)
    best_top5_model = np.argmax(accuracies5)

    closest_25_model = min(
        range(len(total_num_para)),
        key=lambda i: abs(total_num_para[i] - num_params_unpruned * 0.75),
    )
    closest_40_model = min(
        range(len(total_num_para)),
        key=lambda i: abs(total_num_para[i] - num_params_unpruned * 0.60),
    )

    return best_top1_model, best_top5_model, closest_25_model, closest_40_model


best_top1, best_top5, closest_25, closest_40 = select_models(
    accuracies1, accuracies5, total_num_para, num_params_unpruned
)


selected_models = [best_top1, best_top5, closest_25, closest_40]
unique_selected_models = set(selected_models)  # Remove duplicates

for model_idx in unique_selected_models:
    model_path = os.path.join(model_save_path, f"model_pruned_{model_idx}.pth")
    model.load_state_dict(torch.load(model_path))

    submission = get_test_submission(model, testloader, device)
    submission.to_csv(f"submission_model_{model_idx}.csv", index=False, header=False)
    print(f"Evaluation complete for model {model_idx}")


# %%


# %%


# %%


# %%


# %%


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
# Set img_size to 288 for the training model
model_train = timm.create_model(
    "vit_base_patch16_224", pretrained=True, img_size=288
).to(device)

# Load the pre-trained model for validation
model_val = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)

# Resolve data configuration from timm for the validation set
val_data_config = timm.data.resolve_model_data_config(model_val)
val_preprocess = timm.data.create_transform(**val_data_config, is_training=False)

# Set up preprocessing with new resolution for the training set
train_data_config = val_data_config.copy()
train_data_config["input_size"] = (3, 288, 288)  # Change resolution to 288x288
train_preprocess = timm.data.create_transform(**train_data_config, is_training=True)

# Load the validation dataset with default preprocessing
val_dataset = datasets.ImageFolder("imagenet/val", transform=val_preprocess)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Shuffle indices for the entire dataset
indices = list(range(len(val_dataset)))
random.shuffle(indices)

# Calculate sizes for validation and training subsets
training_size = int(len(val_dataset) * 0.01)
validation_size = len(val_dataset) - training_size

# Split indices for training and validation
train_indices = indices[:training_size]
val_indices = indices[training_size : training_size + validation_size]

# Create subsets for training and validation
train_subset = Subset(val_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

# Apply the training preprocessing transform
train_subset.dataset.transform = train_preprocess  # type: ignore

# Create DataLoaders for the subsets
train_loader = DataLoader(train_subset, batch_size=50, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)

# Print sizes to verify the splits
print(f"Total dataset size: {len(val_dataset)}")
print(f"Validation set size: {len(val_subset)}")
print(
    f"Training set size: {len(train_subset)}"
)  # Should be about 1% of the validation set size


def count_approx_zero_params(layer, threshold=1e-6):
    total_nonzero = 0
    for param in layer.parameters():
        if param.requires_grad:
            # Use a threshold to count weights that are nearly zero as zero
            total_nonzero += torch.sum(torch.abs(param.data) > threshold).item()
    return total_nonzero


def calculate_lasso_strength(linf_error):
    # Define the logic for calculating lasso strength
    return linf_error


def freeze_pruned_weights(model):
    # Define the logic for freezing pruned weights
    pass


# Fine-tune the model with 10% data training and 90% without data
def fine_tune_model_2(
    model, i, train_dataloader, epochs, lr, l1_lambda, weight_decay, linf_errors
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine if we should use data (10% chance)
            if random.random() < 0.10:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            else:
                loss = 0  # No data, loss starts at 0

            for name, module in model.named_modules():
                if name in linf_errors:
                    lasso_strength = calculate_lasso_strength(linf_errors[name])
                    for param in module.parameters():
                        group_lasso = lasso_strength / 100 * param.pow(2).sum().sqrt()
                        loss += group_lasso

            # L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

            # Calculate L2 loss
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += weight_decay * l2_norm

            # Scale gradients based on LinfError
            for name, module in model.named_modules():
                if name in linf_errors:
                    scale_factor = calculate_lasso_strength(linf_errors[name])
                    for param in module.parameters():
                        if param.grad is not None:
                            param.grad *= scale_factor

            if loss != 0:
                loss.backward()
                freeze_pruned_weights(model)
                optimizer.step()
            epoch_loss += loss.item()  # type: ignore

        scheduler.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs} completed.")
            print(f"Total Loss: {epoch_loss / len(train_dataloader)}")
            print(f"L1 Loss: {l1_lambda * l1_norm.item()}")  # type: ignore
            print(f"L2 Loss: {weight_decay * l2_norm.item()}")  # type: ignore

    return model, optimizer


# Example usage of the fine-tune function
linf_errors = {}  # Replace with actual linf_errors if available
fine_tune_epochs = 50000
learning_rate = 0.05e-6
l1_lambda = 0.0005
weight_decay = 0.0001

# Fine-tune with training model (288x288)
model_train, optimizer = fine_tune_model_2(
    model_train,
    0,
    train_loader,
    fine_tune_epochs,
    learning_rate,
    l1_lambda,
    weight_decay,
    linf_errors,
)

# Evaluation on the validation set with validation model (224x224)
top1, top5 = evaluate(val_loader, model_val, device)
print(f"Top-1 Accuracy: {top1}")
print(f"Top-5 Accuracy: {top5}")


z_start = 0.015  # Starting value of z
z_end = 0.005  # Ending value of z

m = 20
GOF = 1
# num_params = 1 gof= 0.1 0.05 0.02 0.01
alpha = 0.17
beta = 0.8
# Linear decay of z
z = [z_start - (z_start - z_end) * k / (m - 1) for k in range(m)]
ratio = [1 + 0.0 * k / (m - 1) for k in range(m)]
resnets = [timm.create_model("vit_base_patch16_224", pretrained=True) for k in range(m)]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unpruned_resnet = timm.create_model("vit_base_patch16_224", pretrained=True)


total_num_para = []
accuracies1 = []
accuracies5 = []
for j in range(m):
    threshold = z[j]
    resnets[j].to(device)
    replace_layers(resnets[j], alpha, beta, GOF, depth=0)
    layers = []
    for name, module in resnets[j].named_modules():
        try:
            module.goodnessOfFitCutoff
            layers.append((name, module))
        except:
            pass

    n_splits = 1  # Number of times each layer should be split

    total_LinfError = 0
    total_percentage_less_than_splus = 0
    num_layers = 0
    # Stage 1: Recursive Splitting

    for split_round in range(1, 1 + n_splits):
        # Calculate the slope
        # l = (0.2 - 0.85) / (n_splits - 1)

        # The y-intercept is the initial GOF
        # b = 0.85

        # Calculate GOF
        # l2 = l * (split_round - 1) + b
        # print(f"Splitting round {split_round}/{n_splits}, GOF: {GOF}, l2: {l2}")
        # print("GOF", GOF)

        # Collect layers that can be split
        splittable_layers = [
            (name, module)
            for name, module in resnets[j].named_modules()
            if isinstance(module, (SplittableConv, SplittableLinear))
        ]

        for name, layer in splittable_layers:

            # prune_channels(layer.layer1)

            result, splus, LinfError, percentage_less_than_splus = layer.split(
                ratio[j], 1.8 * z[j]
            )  # type: ignore

            print(result)

            # Iterate over all sub-modules of the layer
            for submodule in layer.modules():
                # Check if the submodule is a Conv2d or Linear layer
                if isinstance(submodule, torch.nn.Conv2d) or isinstance(
                    submodule, torch.nn.Linear
                ):
                    # Apply naive pruning to the submodule
                    naive_prune(
                        submodule,
                        1.5 * (1 - LinfError) * z[j] * percentage_less_than_splus / 100,
                    )
                    print("pruned sublayer")

            total_LinfError += LinfError
            total_percentage_less_than_splus += percentage_less_than_splus
            num_layers += 1

        # M_size = count_nonzero_params(layer)

        # thre=min(.002,  10*math.tanh(Splus) * (1 - LinfError) * (1 / math.sqrt(M_size)))

        # print(thre)

        # naive_prune2(layer,thre)
        # naive_prune2(layer.layer1,thre)
        # naive_prune2(layer.layer2,thre)

        # Counting the number of splittable layers after splitting
        splittable_layers_after_split = [
            (name, module)
            for name, module in resnets[j].named_modules()
            if isinstance(module, (SplittableConv, SplittableLinear))
        ]
        print(
            f"Total number of splittable layers after round {split_round + 1}: {len(splittable_layers_after_split)}"
        )

    average_LinfError = total_LinfError / (num_layers + 0.1)
    average_percentage_less_than_splus = total_percentage_less_than_splus / (
        num_layers + 0.1
    )

    print(f"Average LinfError across splittable layers: {average_LinfError}")
    print(
        f"Average percentage of singular values less than splus: {average_percentage_less_than_splus}%"
    )

    # Apply naive pruning to all layers after split
    for name, module in resnets[j].named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            naive_prune(module, 0.5 * z[j])
        # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        # naive_prune(module, threshold)
        # random_pruning(module, 2*threshold)

        # elif isinstance(module, SplittableLinear) or isinstance(module, SplittableConv):
        # prune_custom_layer1(module,threshold)

    # for name, module in resnets[j].named_modules():

    # if isinstance(module, torch.nn.Conv2d):
    # bn_name = name.replace('conv', 'bn')
    # if bn_name in resnets[j].named_modules():
    # bn_layer = dict(resnets[j].named_modules())[bn_name]
    # global_prune(module, bn_layer, 200*threshold)
    # prune_channels(module,200*threshold)
    # prune_filters(module,200*threshold)

    # elif  isinstance(module, SplittableConv):
    # prune_custom_cannel(module,150*threshold)
    # bn_name = name + '_bn'  # Assuming a naming convention
    # if bn_name in resnets[j].named_modules():
    # bn_layer = dict(resnets[j].named_modules())[bn_name]
    # prune_custom_layer(module, bn_layer, 200*threshold)

    for name, param in resnets[j].named_parameters():
        if param.device.type == "cpu":
            param.data = param.data.to(device)

    num_nonzero = count_nonzero_params(resnets[j])
    print(num_nonzero)
    top1, top5 = evaluate(val_loader, resnets[j], device)
    accuracies1.append(top1)
    accuracies5.append(top5)
    total_num_para.append(num_nonzero)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unpruned_resnet = unpruned_resnet.to(device)

num_params_unpruned = count_total_params(unpruned_resnet)
top1_unpruned, top5_unpruned = evaluate(val_loader, unpruned_resnet, device)

# Convert to Python scalars if necessary
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
total_num_para.append(
    num_params_unpruned
)  # Note: Check if this duplication is intentional
params_kept_percentages = [
    100 * total / num_params_unpruned for total in total_num_para
]


top1_filename = "Top1_Accuracy_vs_Params.png"
top5_filename = "Top5_Accuracy_vs_Params.png"

# Assuming accuracies1, accuracies5, total_num_para, and params_kept_percentages are defined previously

# Filter data based on accuracy thresholds
threshold_top1 = 70
threshold_top5 = 90

# Convert accuracies to NumPy arrays for easier handling
acc1_np = torch.Tensor(accuracies1).detach().cpu().numpy()
acc5_np = torch.Tensor(accuracies5).detach().cpu().numpy()

# Filter data for Top 1 Accuracy plot
filtered_top1_indices = [i for i, acc in enumerate(acc1_np) if acc >= threshold_top1]
filtered_top1_para = [total_num_para[i] for i in filtered_top1_indices]
filtered_top1_acc = [acc1_np[i] for i in filtered_top1_indices]
filtered_top1_pct = [params_kept_percentages[i] for i in filtered_top1_indices]

# Filter data for Top 5 Accuracy plot
filtered_top5_indices = [i for i, acc in enumerate(acc5_np) if acc >= threshold_top5]
filtered_top5_para = [total_num_para[i] for i in filtered_top5_indices]
filtered_top5_acc = [acc5_np[i] for i in filtered_top5_indices]
filtered_top5_pct = [params_kept_percentages[i] for i in filtered_top5_indices]

# Convert top1_unpruned to a CPU tensor if it's not already
if isinstance(top1_unpruned, torch.Tensor):
    top1_unpruned = (
        top1_unpruned.cpu().item()
    )  # This will also convert it to a Python scalar

# Now, the rest of your code should work fine
plt.figure(figsize=(10, 8))
for x, y, pct in zip(filtered_top1_para, filtered_top1_acc, filtered_top1_pct):
    plt.scatter(x, y, marker="o")
    plt.annotate(
        f"{pct:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )

if top1_unpruned >= threshold_top1:
    plt.scatter(
        num_params_unpruned,
        top1_unpruned,
        color="purple",
        marker="x",
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
plt.xlabel("Number of Parameters Kept")
plt.ylabel("Top 1 Accuracy")
plt.title(f"Number of Parameters Kept vs Top 1 Accuracy (Above {threshold_top1}%)")
plt.legend()
plt.grid(True)
plt.savefig("Top1_Accuracy_vs_Params.png")
plt.show()

# Convert top5_unpruned to a CPU tensor if it's a GPU tensor
if isinstance(top5_unpruned, torch.Tensor):
    top5_unpruned = (
        top5_unpruned.cpu().item()
    )  # Assuming top5_unpruned is a single value tensor

# Convert num_params_unpruned to a CPU tensor if it's a GPU tensor
if isinstance(num_params_unpruned, torch.Tensor):
    num_params_unpruned = (
        num_params_unpruned.cpu().item()
    )  # Assuming num_params_unpruned is a single value tensor


plt.figure(figsize=(10, 8))
for x, y, pct in zip(filtered_top5_para, filtered_top5_acc, filtered_top5_pct):
    plt.scatter(x, y, marker="o")
    plt.annotate(
        f"{pct:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )

if top5_unpruned >= threshold_top5:
    plt.scatter(
        num_params_unpruned,
        top5_unpruned,
        color="purple",
        marker="x",
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
plt.xlabel("Number of Parameters Kept")
plt.ylabel("Top 5 Accuracy")
plt.title(f"Number of Parameters Kept vs Top 5 Accuracy (Above {threshold_top5}%)")
plt.legend()
plt.grid(True)
plt.savefig("Top5_Accuracy_vs_Params.png")
plt.show()


# Plotting Top 5 Accuracy vs Number of Parameters Kept with the unpruned DNN included
plt.figure(figsize=(10, 8))
for x, y, pct in zip(filtered_top5_para, filtered_top5_acc, filtered_top5_pct):
    plt.scatter(x, y, marker="o")
    plt.annotate(
        f"{pct:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )
# Add a special marker and line for the unpruned model if it meets the accuracy threshold
if top5_unpruned >= threshold_top5:
    plt.scatter(
        num_params_unpruned,
        top5_unpruned,
        color="purple",
        marker="x",
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
plt.xlabel("Number of Parameters Kept")
plt.ylabel("Top 5 Accuracy")
plt.title(f"Number of Parameters Kept vs Top 5 Accuracy (Above {threshold_top5}%)")
plt.legend()
plt.grid(True)
plt.savefig("Top5_Accuracy_vs_Params.png")
plt.show()


top1_filename_flop = "Top1_Accuracy_vs_FLOPs.png"
top5_filename_flop = "Top5_Accuracy_vs_FLOPs.png"


# Calculate FLOPs for each model
flops_list = [
    calculate_flops(model, input_shape=(1, 3, 224, 224))
    for model in resnets + [unpruned_resnet]
]

# Calculate FLOPs
flops_list = []
for model in resnets + [unpruned_resnet]:
    model.to(device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    flops = calculate_vit_flops(model)
    flops_list.append(flops)

# Convert accuracy tensors to CPU if necessary
accuracies1_cpu = [
    acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accuracies1
]
accuracies5_cpu = [
    acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accuracies5
]

# Convert unpruned accuracies to CPU if necessary
top1_unpruned_cpu = (
    top1_unpruned.cpu().item()
    if isinstance(top1_unpruned, torch.Tensor)
    else top1_unpruned
)
top5_unpruned_cpu = (
    top5_unpruned.cpu().item()
    if isinstance(top5_unpruned, torch.Tensor)
    else top5_unpruned
)

# Calculate FLOPs reduction percentage
flops_unpruned = flops_list[-1]
flops_reduction_pct = [(1 - (flop / flops_unpruned)) * 100 for flop in flops_list]

# Plot and save Top 1 Accuracy vs FLOPs
plt.figure(figsize=(10, 8))
for flop, acc, reduction in zip(flops_list, accuracies1_cpu, flops_reduction_pct):
    plt.scatter(
        flop, acc, marker="o", label="Pruned Models" if flop == flops_list[0] else ""
    )
    plt.annotate(
        f"{reduction:.1f}%",
        (flop, acc),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# Unpruned model (red X and purple line)
plt.scatter(
    flops_unpruned, top1_unpruned_cpu, color="red", marker="x", label="Unpruned"
)
plt.axhline(
    y=top1_unpruned_cpu,
    color="purple",
    linestyle="--",
    linewidth=2,
    label="Unpruned Baseline",
)

plt.xlabel("FLOPs")
plt.ylabel("Top 1 Accuracy")
plt.title("Top 1 Accuracy vs FLOPs")
plt.legend()
plt.grid(True)
plt.savefig("Top1_Accuracy_vs_FLOPs.png")
plt.show()
# Calculate FLOPs reduction percentage
flops_unpruned = flops_list[-1]
flops_reduction_pct = [(1 - (flop / flops_unpruned)) * 100 for flop in flops_list]

# Plot and save Top 5 Accuracy vs FLOPs
plt.figure(figsize=(10, 8))
for flop, acc, reduction in zip(flops_list, accuracies5_cpu, flops_reduction_pct):
    plt.scatter(
        flop, acc, marker="o", label="Pruned Models" if flop == flops_list[0] else ""
    )
    plt.annotate(
        f"{reduction:.1f}%",
        (flop, acc),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# Unpruned model (red X and purple line)
plt.scatter(
    flops_unpruned, top5_unpruned_cpu, color="red", marker="x", label="Unpruned"
)
plt.axhline(
    y=top5_unpruned_cpu,
    color="purple",
    linestyle="--",
    linewidth=2,
    label="Unpruned Baseline",
)

plt.xlabel("FLOPs")
plt.ylabel("Top 5 Accuracy")
plt.title("Top 5 Accuracy vs FLOPs")
plt.legend()
plt.grid(True)
plt.savefig("Top5_Accuracy_vs_FLOPs.png")
plt.show()

# %%


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unpruned_resnet = timm.create_model("vit_base_patch16_224", pretrained=True)

unpruned_resnet = unpruned_resnet.to(device)


# unpruned_resnet = unpruned_resnet.to(device)

num_params_unpruned = count_total_params(unpruned_resnet)
top1_unpruned, top5_unpruned = evaluate(val_loader, unpruned_resnet, device)

# Convert to Python scalars if necessary
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


plt.figure(figsize=(10, 8))
plt.plot(total_num_para, accuracies1, label="Top 1 Accuracy")
plt.plot(total_num_para, accuracies5, label="Top 5 Accuracy")

# Add a scatter point for the unpruned model accuracy
plt.scatter(
    num_params_unpruned,
    top1_unpruned,
    color="purple",
    marker="x",
    s=100,
    label="Unpruned Top 1",
)
plt.scatter(
    num_params_unpruned,
    top5_unpruned,
    color="purple",
    marker="o",
    s=100,
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
    y=top1_unpruned, color="purple", linestyle="--", linewidth=2, label="Unpruned Top 1"
)
plt.axhline(
    y=top5_unpruned, color="purple", linestyle="-", linewidth=2, label="Unpruned Top 5"
)

plt.xlabel("Pruning Cycles")
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Pruning Cycles")
plt.legend()
plt.grid(True)
plt.show()

# %%
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

plt.figure(figsize=(10, 8))
plt.plot(total_num_para, accuracies1, label="Top 1 Accuracy")
plt.plot(total_num_para, accuracies5, label="Top 5 Accuracy")

# Add a horizontal line for the unpruned model accuracy
plt.axhline(
    y=top1_unpruned, color="purple", linestyle="--", linewidth=2, label="Unpruned Top 1"
)
plt.axhline(
    y=top5_unpruned, color="purple", linestyle="-", linewidth=2, label="Unpruned Top 5"
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
    y=top1_unpruned, color="purple", linestyle="--", linewidth=2, label="Unpruned Top 1"
)
plt.axhline(
    y=top5_unpruned, color="purple", linestyle="-", linewidth=2, label="Unpruned Top 5"
)

plt.xlabel("Pruning Cycles")
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Pruning Cycles")
plt.legend()
plt.grid(True)
plt.show()


# %%


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unpruned_resnet = unpruned_resnet.to(device)

num_params_unpruned = count_total_params(unpruned_resnet)
top1_unpruned, top5_unpruned = evaluate(val_loader, unpruned_resnet, device)

# Convert to Python scalars if necessary
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
total_num_para.append(
    num_params_unpruned
)  # Note: Check if this duplication is intentional
params_kept_percentages = [
    100 * total / num_params_unpruned for total in total_num_para
]


top1_filename = "Top1_Accuracy_vs_Params.png"
top5_filename = "Top5_Accuracy_vs_Params.png"

# Assuming accuracies1, accuracies5, total_num_para, and params_kept_percentages are defined previously

# Filter data based on accuracy thresholds
threshold_top1 = 70
threshold_top5 = 90

# Convert accuracies to NumPy arrays for easier handling
acc1_np = torch.Tensor(accuracies1).detach().cpu().numpy()
acc5_np = torch.Tensor(accuracies5).detach().cpu().numpy()

# Filter data for Top 1 Accuracy plot
filtered_top1_indices = [i for i, acc in enumerate(acc1_np) if acc >= threshold_top1]
filtered_top1_para = [total_num_para[i] for i in filtered_top1_indices]
filtered_top1_acc = [acc1_np[i] for i in filtered_top1_indices]
filtered_top1_pct = [params_kept_percentages[i] for i in filtered_top1_indices]

# Filter data for Top 5 Accuracy plot
filtered_top5_indices = [i for i, acc in enumerate(acc5_np) if acc >= threshold_top5]
filtered_top5_para = [total_num_para[i] for i in filtered_top5_indices]
filtered_top5_acc = [acc5_np[i] for i in filtered_top5_indices]
filtered_top5_pct = [params_kept_percentages[i] for i in filtered_top5_indices]

# Convert top1_unpruned to a CPU tensor if it's not already
if isinstance(top1_unpruned, torch.Tensor):
    top1_unpruned = (
        top1_unpruned.cpu().item()
    )  # This will also convert it to a Python scalar

# Now, the rest of your code should work fine
plt.figure(figsize=(10, 8))
for x, y, pct in zip(filtered_top1_para, filtered_top1_acc, filtered_top1_pct):
    plt.scatter(x, y, marker="o")
    plt.annotate(
        f"{pct:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )

if top1_unpruned >= threshold_top1:
    plt.scatter(
        num_params_unpruned,
        top1_unpruned,
        color="purple",
        marker="x",
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
plt.xlabel("Number of Parameters Kept")
plt.ylabel("Top 1 Accuracy")
plt.title(f"Number of Parameters Kept vs Top 1 Accuracy (Above {threshold_top1}%)")
plt.legend()
plt.grid(True)
plt.savefig("Top1_Accuracy_vs_Params.png")
plt.show()

# Convert top5_unpruned to a CPU tensor if it's a GPU tensor
if isinstance(top5_unpruned, torch.Tensor):
    top5_unpruned = (
        top5_unpruned.cpu().item()
    )  # Assuming top5_unpruned is a single value tensor

# Convert num_params_unpruned to a CPU tensor if it's a GPU tensor
if isinstance(num_params_unpruned, torch.Tensor):
    num_params_unpruned = (
        num_params_unpruned.cpu().item()
    )  # Assuming num_params_unpruned is a single value tensor


plt.figure(figsize=(10, 8))
for x, y, pct in zip(filtered_top5_para, filtered_top5_acc, filtered_top5_pct):
    plt.scatter(x, y, marker="o")
    plt.annotate(
        f"{pct:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )

if top5_unpruned >= threshold_top5:
    plt.scatter(
        num_params_unpruned,
        top5_unpruned,
        color="purple",
        marker="x",
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
plt.xlabel("Number of Parameters Kept")
plt.ylabel("Top 5 Accuracy")
plt.title(f"Number of Parameters Kept vs Top 5 Accuracy (Above {threshold_top5}%)")
plt.legend()
plt.grid(True)
plt.savefig("Top5_Accuracy_vs_Params.png")
plt.show()


# Plotting Top 5 Accuracy vs Number of Parameters Kept with the unpruned DNN included
plt.figure(figsize=(10, 8))
for x, y, pct in zip(filtered_top5_para, filtered_top5_acc, filtered_top5_pct):
    plt.scatter(x, y, marker="o")
    plt.annotate(
        f"{pct:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )
# Add a special marker and line for the unpruned model if it meets the accuracy threshold
if top5_unpruned >= threshold_top5:
    plt.scatter(
        num_params_unpruned,
        top5_unpruned,
        color="purple",
        marker="x",
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
plt.xlabel("Number of Parameters Kept")
plt.ylabel("Top 5 Accuracy")
plt.title(f"Number of Parameters Kept vs Top 5 Accuracy (Above {threshold_top5}%)")
plt.legend()
plt.grid(True)
plt.savefig("Top5_Accuracy_vs_Params.png")
plt.show()


top1_filename_flop = "Top1_Accuracy_vs_FLOPs.png"
top5_filename_flop = "Top5_Accuracy_vs_FLOPs.png"


# Calculate FLOPs for each model
flops_list = [
    calculate_flops(model, input_shape=(1, 3, 224, 224))
    for model in resnets + [unpruned_resnet]
]

# Calculate FLOPs
flops_list = []
for model in resnets + [unpruned_resnet]:
    model.to(device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    flops = calculate_vit_flops(model)
    flops_list.append(flops)

# Convert accuracy tensors to CPU if necessary
accuracies1_cpu = [
    acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accuracies1
]
accuracies5_cpu = [
    acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accuracies5
]

# Convert unpruned accuracies to CPU if necessary
top1_unpruned_cpu = (
    top1_unpruned.cpu().item()
    if isinstance(top1_unpruned, torch.Tensor)
    else top1_unpruned
)
top5_unpruned_cpu = (
    top5_unpruned.cpu().item()
    if isinstance(top5_unpruned, torch.Tensor)
    else top5_unpruned
)

# Calculate FLOPs reduction percentage
flops_unpruned = flops_list[-1]
flops_reduction_pct = [(1 - (flop / flops_unpruned)) * 100 for flop in flops_list]

# Plot and save Top 1 Accuracy vs FLOPs
plt.figure(figsize=(10, 8))
for flop, acc, reduction in zip(flops_list, accuracies1_cpu, flops_reduction_pct):
    plt.scatter(
        flop, acc, marker="o", label="Pruned Models" if flop == flops_list[0] else ""
    )
    plt.annotate(
        f"{reduction:.1f}%",
        (flop, acc),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# Unpruned model (red X and purple line)
plt.scatter(
    flops_unpruned, top1_unpruned_cpu, color="red", marker="x", label="Unpruned"
)
plt.axhline(
    y=top1_unpruned_cpu,
    color="purple",
    linestyle="--",
    linewidth=2,
    label="Unpruned Baseline",
)

plt.xlabel("FLOPs")
plt.ylabel("Top 1 Accuracy")
plt.title("Top 1 Accuracy vs FLOPs")
plt.legend()
plt.grid(True)
plt.savefig("Top1_Accuracy_vs_FLOPs.png")
plt.show()

# Calculate FLOPs reduction percentage
flops_unpruned = flops_list[-1]
flops_reduction_pct = [(1 - (flop / flops_unpruned)) * 100 for flop in flops_list]

# Plot and save Top 5 Accuracy vs FLOPs
plt.figure(figsize=(10, 8))
for flop, acc, reduction in zip(flops_list, accuracies5_cpu, flops_reduction_pct):
    plt.scatter(
        flop, acc, marker="o", label="Pruned Models" if flop == flops_list[0] else ""
    )
    plt.annotate(
        f"{reduction:.1f}%",
        (flop, acc),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# Unpruned model (red X and purple line)
plt.scatter(
    flops_unpruned, top5_unpruned_cpu, color="red", marker="x", label="Unpruned"
)
plt.axhline(
    y=top5_unpruned_cpu,
    color="purple",
    linestyle="--",
    linewidth=2,
    label="Unpruned Baseline",
)

plt.xlabel("FLOPs")
plt.ylabel("Top 5 Accuracy")
plt.title("Top 5 Accuracy vs FLOPs")
plt.legend()
plt.grid(True)
plt.savefig("Top5_Accuracy_vs_FLOPs.png")
plt.show()


# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unpruned_resnet = unpruned_resnet.to(device)

num_params_unpruned = count_total_params(unpruned_resnet)
top1_unpruned, top5_unpruned = evaluate(val_loader, unpruned_resnet, device)

# Convert to Python scalars if necessary
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
total_num_para.append(
    num_params_unpruned
)  # Note: Check if this duplication is intentional

# Rest of your code for plotting...


# Calculate percentages of parameters kept
params_kept_percentages = [
    100 * total / num_params_unpruned for total in total_num_para
]


top1_filename = "Top1_Accuracy_vs_Params.png"
top5_filename = "Top5_Accuracy_vs_Params.png"

# Assuming accuracies1, accuracies5, total_num_para, and params_kept_percentages are defined previously

# Filter data based on accuracy thresholds
threshold_top1 = 70
threshold_top5 = 90

# Convert accuracies to NumPy arrays for easier handling
acc1_np = torch.Tensor(accuracies1).detach().cpu().numpy()
acc5_np = torch.Tensor(accuracies5).detach().cpu().numpy()

# Filter data for Top 1 Accuracy plot
filtered_top1_indices = [i for i, acc in enumerate(acc1_np) if acc >= threshold_top1]
filtered_top1_para = [total_num_para[i] for i in filtered_top1_indices]
filtered_top1_acc = [acc1_np[i] for i in filtered_top1_indices]
filtered_top1_pct = [params_kept_percentages[i] for i in filtered_top1_indices]

# Filter data for Top 5 Accuracy plot
filtered_top5_indices = [i for i, acc in enumerate(acc5_np) if acc >= threshold_top5]
filtered_top5_para = [total_num_para[i] for i in filtered_top5_indices]
filtered_top5_acc = [acc5_np[i] for i in filtered_top5_indices]
filtered_top5_pct = [params_kept_percentages[i] for i in filtered_top5_indices]

# Convert top1_unpruned to a CPU tensor if it's not already
if isinstance(top1_unpruned, torch.Tensor):
    top1_unpruned = (
        top1_unpruned.cpu().item()
    )  # This will also convert it to a Python scalar

# Now, the rest of your code should work fine
plt.figure(figsize=(10, 8))
for x, y, pct in zip(filtered_top1_para, filtered_top1_acc, filtered_top1_pct):
    plt.scatter(x, y, marker="o")
    plt.annotate(
        f"{pct:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )

if top1_unpruned >= threshold_top1:
    plt.scatter(
        num_params_unpruned,
        top1_unpruned,
        color="purple",
        marker="x",
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
plt.xlabel("Number of Parameters Kept")
plt.ylabel("Top 1 Accuracy")
plt.title(f"Number of Parameters Kept vs Top 1 Accuracy (Above {threshold_top1}%)")
plt.legend()
plt.grid(True)
plt.savefig("Top1_Accuracy_vs_Params.png")
plt.show()

# Convert top5_unpruned to a CPU tensor if it's a GPU tensor
if isinstance(top5_unpruned, torch.Tensor):
    top5_unpruned = (
        top5_unpruned.cpu().item()
    )  # Assuming top5_unpruned is a single value tensor

# Convert num_params_unpruned to a CPU tensor if it's a GPU tensor
if isinstance(num_params_unpruned, torch.Tensor):
    num_params_unpruned = (
        num_params_unpruned.cpu().item()
    )  # Assuming num_params_unpruned is a single value tensor

# Now, the rest of your code should work fine
plt.figure(figsize=(10, 8))
for x, y, pct in zip(filtered_top5_para, filtered_top5_acc, filtered_top5_pct):
    plt.scatter(x, y, marker="o")
    plt.annotate(
        f"{pct:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )

if top5_unpruned >= threshold_top5:
    plt.scatter(
        num_params_unpruned,
        top5_unpruned,
        color="purple",
        marker="x",
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
plt.xlabel("Number of Parameters Kept")
plt.ylabel("Top 5 Accuracy")
plt.title(f"Number of Parameters Kept vs Top 5 Accuracy (Above {threshold_top5}%)")
plt.legend()
plt.grid(True)
plt.savefig("Top5_Accuracy_vs_Params.png")
plt.show()


# Plotting Top 5 Accuracy vs Number of Parameters Kept with the unpruned DNN included
plt.figure(figsize=(10, 8))
for x, y, pct in zip(filtered_top5_para, filtered_top5_acc, filtered_top5_pct):
    plt.scatter(x, y, marker="o")
    plt.annotate(
        f"{pct:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )
# Add a special marker and line for the unpruned model if it meets the accuracy threshold
if top5_unpruned >= threshold_top5:
    plt.scatter(
        num_params_unpruned,
        top5_unpruned,
        color="purple",
        marker="x",
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
plt.xlabel("Number of Parameters Kept")
plt.ylabel("Top 5 Accuracy")
plt.title(f"Number of Parameters Kept vs Top 5 Accuracy (Above {threshold_top5}%)")
plt.legend()
plt.grid(True)
plt.savefig("Top5_Accuracy_vs_Params.png")
plt.show()


# %%


# Calculate FLOPs for each model
flops_list = [
    calculate_flops(model, input_shape=(1, 3, 224, 224))
    for model in resnets + [unpruned_resnet]
]

# Calculate FLOPs
flops_list = []
for model in resnets + [unpruned_resnet]:
    model.to(device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    flops = calculate_flops(model)
    flops_list.append(flops)

# Convert accuracy tensors to CPU if necessary
accuracies1_cpu = [
    acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accuracies1
]
accuracies5_cpu = [
    acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in accuracies5
]

# Convert unpruned accuracies to CPU if necessary
top1_unpruned_cpu = (
    top1_unpruned.cpu().item()
    if isinstance(top1_unpruned, torch.Tensor)
    else top1_unpruned
)
top5_unpruned_cpu = (
    top5_unpruned.cpu().item()
    if isinstance(top5_unpruned, torch.Tensor)
    else top5_unpruned
)

# Plot and save Top 1 Accuracy vs FLOPs
plt.figure(figsize=(10, 8))
plt.scatter(flops_list, accuracies1_cpu, marker="o", label="Pruned Models")
plt.scatter(
    flops_list[-1], top1_unpruned_cpu, color="red", marker="x", label="Unpruned"
)
plt.xlabel("FLOPs")
plt.ylabel("Top 1 Accuracy")
plt.title("Top 1 Accuracy vs FLOPs")
plt.legend()
plt.grid(True)
plt.savefig("Top1_Accuracy_vs_FLOPs.png")
plt.show()

# Plot and save Top 5 Accuracy vs FLOPs
plt.figure(figsize=(10, 8))
plt.scatter(flops_list, accuracies5_cpu, marker="o", label="Pruned Models")
plt.scatter(
    flops_list[-1], top5_unpruned_cpu, color="red", marker="x", label="Unpruned"
)
plt.xlabel("FLOPs")
plt.ylabel("Top 5 Accuracy")
plt.title("Top 5 Accuracy vs FLOPs")
plt.legend()
plt.grid(True)
plt.savefig("Top5_Accuracy_vs_FLOPs.png")
plt.show()


# %%
k = 3
param_grid = {
    "alpha": [0.1, 0.3, 0.01],
    "beta": [0.99, 0.999],
    "goodnessOfFitCutoff": [0.135],  # np.linspace(0,1.0,k+1,endpoint=False)[1:],
    "split": np.linspace(0.3, 1.0, 10, endpoint=False),
}
for value in param_grid.values():
    shuffle(value)

grid = ParameterGrid(param_grid)
try:
    with open("take_gof_to_zero.txt", "r") as file:
        res = json.load(file)
except:
    res = {}
for params in tqdm(grid):
    to_tuple = str(
        (
            params["alpha"],
            params["beta"],
            params["goodnessOfFitCutoff"],
            params["split"],
        )
    )
    if to_tuple in res:
        continue

    top1, top5, params = test_parameters(**params)
    res[to_tuple] = (float(top1.cpu()), float(top5.cpu()), params)  # type: ignore

    with open("take_gof_to_zero.txt", "w") as file:
        file.write(json.dumps(res))


# %%


with open("take_gof_to_zero.txt", "r") as file:
    res = json.load(file)


df = pd.DataFrame.from_dict(
    res, orient="index", columns=["top1", "top5", "param number"]
)
# change the name of the index column of df
df.index.name = "(alpha,beta,goodnessOfFitCutoff,split)"


def unstring(string):
    """given a string of the form str((a,b,c,d)) return (a,b,c,d)
    example '(1,2,3,4)' -> (1,2,3,4)"""
    without_parentheses = string[1:-1]
    return tuple(float(x) for x in without_parentheses.split(","))


df["params"] = df.index.map(unstring)
df["alpha"] = df["params"].map(lambda x: x[0])
df["beta"] = df["params"].map(lambda x: x[1])
df["goodnessOfFitCutoff"] = df["params"].map(lambda x: x[2])
df["split"] = df["params"].map(lambda x: x[3])
df = df.drop(columns=["params"])
df.reset_index(drop=True, inplace=True)

# %%


# %%
# show df sorted by param number
# make a plot of top1 vs param number


sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(
    data=df,
    x="split",
    y="top1",
    hue="goodnessOfFitCutoff",
    size="param number",
    sizes=(20, 200),
)


# %%
small_split = df[(df["split"] == df.loc[1]["split"])]
# sort by goodnessOfFitCutoff
small_split = small_split.sort_values(by=["top1"])

# %%
small_split.sort_values(by=["top5"])

# %%
test_parameters(alpha=0.3, beta=0.6, goodnessOfFitCutoff=0.2, split=0.8333333333333334)
