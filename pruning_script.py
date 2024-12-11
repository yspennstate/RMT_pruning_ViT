# %%


# %%
import matplotlib.pyplot as plt
import math
import os
from tqdm.notebook import tqdm as tqdm

import torch
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
import torch.optim as optim
import timm  # type: ignore


from training import fine_tune_model_2  # type: ignore
from utils import load_checkpoint, save_state
from validation import evaluate, get_val_dataset
from pruning import (
    count_nonzero_params,
    count_total_params,
    prune_model,
    replace_layers,
    perform_splitting,
)

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
preprocess = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()

val_loader = get_val_dataset(preprocess=preprocess)

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


"""# Load the pre-trained model
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


# Create subsets for training and validation
train_subset = Subset(dataset, train_indices)


# Create DataLoaders for the subsets
train_loader = DataLoader(train_subset, batch_size=50, shuffle=True)
"""
# val_indices = indices[training_size : training_size + validation_size]
# val_subset = Subset(dataset, val_indices)
# val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)


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

"""
model = timm.create_model("vit_large_patch16_224", pretrained=True).to(device)
replace_layers(model, alpha, beta, GoF, depth=0)


optimizer = optim.Adam(model.parameters(), lr=0.05e-6)  # Adjust lr as needed
"""

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

# %%
