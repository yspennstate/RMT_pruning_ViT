import torch
import os


def save_checkpoint(checkpoint, filename, model_save_path):
    full_path = os.path.join(model_save_path, filename)
    torch.save(checkpoint, full_path)
    print(f"Checkpoint saved to {full_path}")


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
):

    state = {
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cycle_index": cycle_index,
        "fine_tune_epochs": fine_tune_epochs,
        "learning_rate": learning_rate,
        "accuracies1": accuracies1,
        "accuracies5": accuracies5,
        "total_num_para": total_num_para,
    }
    filename = f"{filename_prefix}_cycle_{cycle_index}.pth.tar"
    save_checkpoint(state, filename, model_save_path)
    save_checkpoint(
        state, "last_checkpoint.pth.tar", model_save_path
    )  # Keep updating the last checkpoint
