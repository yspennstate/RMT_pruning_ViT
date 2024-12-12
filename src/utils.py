import torch
import os


def save_checkpoint(checkpoint, filename, model_save_path):
    """
    Saves the checkpoint to the specified path.

    Args:
        checkpoint (dict): The checkpoint data to save.
        filename (str): The name of the file to save the checkpoint as.
        model_save_path (str): The directory path to save the checkpoint.
    """
    full_path = os.path.join(model_save_path, filename)
    torch.save(checkpoint, full_path)


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
    """
    Saves the state of the model and optimizer along with training metadata.

    Args:
        model (torch.nn.Module): The model whose state is to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state is to be saved.
        cycle_index (int): The current cycle index.
        fine_tune_epochs (int): Number of fine-tuning epochs.
        learning_rate (float): The learning rate used.
        filename_prefix (str): Prefix for the filename.
        accuracies1 (list): List of top-1 accuracies.
        accuracies5 (list): List of top-5 accuracies.
        total_num_para (int): Total number of parameters in the model.
        model_save_path (str): The directory path to save the state.
    """
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
    # Keep updating the last checkpoint
    save_checkpoint(state, "last_checkpoint.pth.tar", model_save_path)
