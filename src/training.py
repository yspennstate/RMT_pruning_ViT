from pruning import freeze_pruned_weights, get_base_name, calculate_lasso_strength
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def fine_tune_model(model, i, epochs, lr, l1_lambda, weight_decay, linf_errors):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss = 0

        # Iterate through named modules and apply regularization
        for name, module in model.named_modules():
            base_name = get_base_name(name)
            if base_name in linf_errors:
                lasso_strength = calculate_lasso_strength(linf_errors[base_name], i)
                for param in module.parameters():
                    group_lasso = lasso_strength / 1000000 * param.pow(2).sum().sqrt()
                    total_loss += group_lasso

        # L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        total_loss += l1_lambda * l1_norm

        # Calculate L2 loss
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        total_loss += weight_decay * l2_norm

        if total_loss != 0:
            total_loss.backward(retain_graph=True)
            # Scale gradients based on LinfError
            for name, module in model.named_modules():
                base_name = get_base_name(name)
                if base_name in linf_errors:
                    scale_factor = calculate_lasso_strength(linf_errors[base_name], i)
                    for param in module.parameters():
                        if param.grad is not None:
                            param.grad *= scale_factor

        freeze_pruned_weights(model)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs} completed.")
            print(f"Total Loss: {total_loss.item()}")  # type: ignore
            print(f"L1 Loss: {l1_lambda * l1_norm.item()}")  # type: ignore
            print(f"L2 Loss: {weight_decay * l2_norm.item()}")  # type: ignore

    return model, optimizer
