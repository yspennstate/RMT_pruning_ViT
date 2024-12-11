import torch
import os


def save_checkpoint(checkpoint, filename, model_save_path):
    full_path = os.path.join(model_save_path, filename)
    torch.save(checkpoint, full_path)
    print(f"Checkpoint saved to {full_path}")


def load_checkpoint(model, filename, model_save_path, optimizer=None):
    full_path = os.path.join(model_save_path, filename)
    if os.path.exists(full_path):
        print(f"Loading checkpoint from {full_path}")
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("Checkpoint loaded successfully.")
        return checkpoint
    else:
        print("No checkpoint found.")
        return None


def save_state(
    model,
    optimizer,
    cycle_index,
    fine_tune_epochs,
    learning_rate,
    filename_prefix,
    accuracies1,
    accuracies5,
    total_num_para,
    model_save_path,
    train_indices=None,
    val_indices=None,
):
    print(f"Inside save_state - Type of model: {type(model)}")
    print(f"Inside save_state - Type of optimizer: {type(optimizer)}")

    state = {
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cycle_index": cycle_index,
        "fine_tune_epochs": fine_tune_epochs,
        "learning_rate": learning_rate,
        "accuracies1": accuracies1,
        "accuracies5": accuracies5,
        "total_num_para": total_num_para,
        "train_indices": train_indices,  # Saving the train indices
        "val_indices": val_indices,  # Saving the validation indices
    }
    filename = f"{filename_prefix}_cycle_{cycle_index}.pth.tar"
    save_checkpoint(state, filename, model_save_path)
    save_checkpoint(
        state, "last_checkpoint.pth.tar", model_save_path
    )  # Keep updating the last checkpoint
