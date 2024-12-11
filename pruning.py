import torch
from SplittableLayers import (
    SplittableConv,
    SplittableLinear,
)


def naive_prune(layer, threshold):
    with torch.no_grad():
        weight_mask = torch.abs(layer.weight) > threshold
        layer.weight *= weight_mask.float()


def naive_prune2(layer, threshold):
    with torch.no_grad():
        if hasattr(layer, "weight") and isinstance(layer.weight, torch.nn.Parameter):
            # Prune standard layers
            weight_mask = torch.abs(layer.weight) > threshold
            layer.weight.data *= weight_mask.float()
            # print("pruned")


def count_nonzero_params(model):
    count = 0
    for param in model.parameters():
        count += torch.count_nonzero(param).item()
    return count


def count_total_params(models):
    if isinstance(models, list):
        return sum(p.numel() for model in models for p in model.parameters())
    else:
        return sum(p.numel() for p in models.parameters())


def count_approx_zero_params(layer, threshold=1e-6):
    total_nonzero = 0
    for param in layer.parameters():
        if param.requires_grad:
            # Use a threshold to count weights that are nearly zero as zero
            total_nonzero += torch.sum(torch.abs(param.data) > threshold).item()
    return total_nonzero


def replace_layers(m, alpha, beta, goodnessOfFitCutoff, depth=0):
    replacable_layers = {}
    for name, module in m.named_children():
        if name == "":
            continue
        try:
            module.goodnessOfFitCutoff
            continue
        except:
            pass

        if isinstance(module, torch.nn.MultiheadAttention):
            replacable_layers[name] = (module, "attention")
            continue

        if len(list(module.named_modules())) > 1:
            replace_layers(module, alpha, beta, goodnessOfFitCutoff, depth=depth + 1)
            continue

        if isinstance(module, torch.nn.Conv2d):
            replacable_layers[name] = (module, "conv")
            continue

        if isinstance(module, torch.nn.Linear):
            replacable_layers[name] = (module, "linear")
            continue
    for name, pair in replacable_layers.items():
        module, type_str = pair
        if type_str == "conv":
            setattr(
                m,
                name,
                SplittableConv.from_layer(
                    module,
                    alpha=alpha,
                    beta=beta,
                    goodnessOfFitCutoff=goodnessOfFitCutoff,
                ),
            )
        if type_str == "linear":
            setattr(
                m,
                name,
                SplittableLinear.from_layer(
                    module,
                    alpha=alpha,
                    beta=beta,
                    goodnessOfFitCutoff=goodnessOfFitCutoff,
                ),
            )


def perform_splitting(vit, split):
    k = 0
    splittable_layers = []
    for name, module in vit.named_modules():
        try:
            module.split
            splittable_layers.append((name, module))
        except:
            pass

    for n, splittable_layer in splittable_layers:
        splittable_layer.split(split, save_plot=None)
        k += 1


def prune_model(model, target_reduction, i, n_prune_cycles, device):

    linf_errors = {}

    splittable_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (SplittableConv, SplittableLinear))
    ]

    for name, layer in splittable_layers:

        current_reduction = 0
        num_params_unpruned = 0

        initial_pruning_factor = 0.00001 * i  # Starting pruning factor
        step_size = 0.00002  # Incremental step for adjusting pruning factor

        num_params_unpruned_now = count_nonzero_params(layer)

        # print(num_params_unpruned_now)

        pruning_factor = initial_pruning_factor

        W = layer.get_matrix()
        M, N = W.shape

        end_scale = 1

        scale = 1 + i * (end_scale - 1) / n_prune_cycles

        # Determine the scale based on the cycle number
        if i % 2 == 0:
            result, splus, LinfError, percentage_less_than_splus = layer.split(
                1, scale * 750 * 0.000000015 * target_reduction * N * M
            )  # type: ignore
            print(result)
            # print(splus)
            # print(LinfError)
            # print(percentage_less_than_splus
        else:
            result, splus, LinfError, percentage_less_than_splus = layer.split(
                1, 0
            )  # type: ignore # Pass zero scale for odd cycles
            print("skip")

        # result,splus,LinfError, percentage_less_than_splus=layer.split(1, scale*500*.000000015*target_reduction*N*M)

        # result,splus,LinfError, percentage_less_than_splus=layer.split(1, scale*50*.000000015* (1-LinfError)*percentage_less_than_splus/100*target_reduction*num_params_unpruned_now)

        # split_called_for_current_factor = True

        linf_errors[name] = LinfError

        # print(linf_errors)

        # print(scale)

        # print(splus)
        # print(LinfError)
        # print(percentage_less_than_splus)
        # Step 1: Get the matrix of the layer

        while (
            current_reduction
            < (1 - LinfError) ** (1.5 / (i))
            * (percentage_less_than_splus / 100) ** (1.5 / (i))
            * target_reduction
            * num_params_unpruned_now
        ):
            # Reset model to its original state if not the first iteration
            # result,splus,LinfError, percentage_less_than_splus=layer.split(1-200*pruning_factor, 0)
            # print(result)

            # print((1-LinfError)**1.4*percentage_less_than_splus/100*target_reduction)

            # layer.pruningRMT(name, .1*(1-LinfError) * pruning_factor*percentage_less_than_splus/100, splus, LinfError, percentage_less_than_splus, n_prune_cycles)

            # Iterate over all sub-modules of the layer
            for submodule in layer.modules():
                # Check if the submodule is a Conv2d or Linear layer
                if isinstance(submodule, torch.nn.Conv2d) or isinstance(
                    submodule, torch.nn.Linear
                ):
                    # Apply naive pruning to the submodule
                    naive_prune(
                        submodule,
                        4
                        * ((1 - LinfError) * percentage_less_than_splus / 100)
                        ** (1.5 / (i))
                        * pruning_factor,
                    )
                    naive_prune(submodule, 3 * pruning_factor)

            pruning_factor += step_size

            # print(pruning_factor)

            # print(target_reduction*num_params_unpruned,"number")
            # Call the pruning function

            # Count non-zero parameters
            num_nonzero_now = count_nonzero_params(layer)
            current_reduction = num_params_unpruned_now - num_nonzero_now
            # print(current_reduction)
            # print(num_nonzero)

        if current_reduction >= target_reduction * num_params_unpruned:
            num_nonzero_now = count_nonzero_params(layer)
            # print(num_nonzero_now)

    num_nonzero = count_nonzero_params(model)
    print(num_nonzero)

    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            param.data = param.data.to(device)

    return linf_errors


def freeze_pruned_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Set the gradients of the pruned weights to zero where the parameter data is zero
            param.grad.data.mul_(param.data.ne(0).float())


def get_weight_matrices(model):
    weights = []
    for name, param in model.named_parameters():
        if (
            "weight" in name and len(param.size()) > 1
        ):  # This condition ensures we're only grabbing weights of layers and not biases
            weights.append(param)
    return weights


def group_lasso_norm(weights):
    group_lasso = 0
    for weight in weights:
        group_lasso += weight.pow(2).sum().sqrt()  # Euclidean norm of the weight matrix
    return group_lasso


def calculate_lasso_strength(linf_error, i):
    # print('check')
    return 1 / (1 + 10000 * linf_error / (i))


def get_base_name(name):
    parts = name.split(".")
    if parts[-1] in ["layer_1", "layer_2"]:
        return ".".join(parts[:-1])
    return name
