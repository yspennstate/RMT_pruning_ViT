from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Necessary for cropped images in Imagenet

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from src.pruning import freeze_pruned_weights, replace_layers
from src.validation import evaluate
import timm
import argparse
import os

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Fine tuning script for ViT model")
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to use for computation"
)
parser.add_argument(
    "--weights_path", type=str, help="Path to model weights to fine-tune"
)
parser.add_argument("--save_path", type=str, help="Path to save fine-tuned model")
args = parser.parse_args()

# Create save directory if it doesn't exist
os.makedirs(args.save_path, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.empty_cache()

# Set device for computation
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# Initialize and prepare the model
model = timm.create_model("vit_large_patch16_224", pretrained=True)  # type: ignore
replace_layers(model, 0.01, 0.5, 0, depth=0)
model.load_state_dict(torch.load(args.weights_path)["state_dict"])
model.to(device)

# Prepare data transforms and loaders
data_config = timm.data.resolve_model_data_config(model)  # type: ignore
preprocess = timm.data.create_transform(**data_config, is_training=False)  # type: ignore
train_dataset = datasets.ImageFolder("imagenet/train", transform=preprocess)
val_dataset = datasets.ImageFolder("imagenet/val", transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Hyperparameters for fine-tuning
fine_tune_epoch = 20
weight_decay = 1e-6
learning_rate = 3e-5
lambda_original = 1e-8

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Learning rate scheduler
scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, fine_tune_epoch)
scheduler_warmup = optim.lr_scheduler.ConstantLR(optimizer, factor=0.01, total_iters=1)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[scheduler_warmup, scheduler_cos], milestones=[2]
)

# Fine-tuning loop
for i in range(fine_tune_epoch):
    loop = tqdm(train_loader)
    total_loss = 0
    steps = 1
    model.train()
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss
        loss.backward()
        freeze_pruned_weights(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        loop.set_postfix(loss=total_loss.to("cpu").item() / steps)
        steps += 1

    # Evaluate the model on validation set
    accuracy, top5 = evaluate(model, val_loader, device)

    # Save the model state
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "accuracy": accuracy,
            "top5": top5,
        },
        f"{args.save_path}/epoch_{i}_lr_{scheduler.get_last_lr()}_accuracy_{accuracy:.4f}_top5_{top5:.4f}.pt",
    )
    scheduler.step()
