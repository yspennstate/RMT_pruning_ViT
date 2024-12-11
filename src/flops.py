import torch
import torch.nn as nn


def calculate_flops(model, input_shape=(1, 3, 224, 224)):
    total_flops = 0

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # Count the non-zero weights
            non_zero_weights = torch.count_nonzero(layer.weight).item()

            # Calculate the output dimensions
            n, c, h, w = input_shape
            h_out = (
                (
                    h
                    + 2 * layer.padding[0]
                    - layer.dilation[0] * (layer.kernel_size[0] - 1)
                    - 1
                )
                // layer.stride[0]
            ) + 1
            w_out = (
                (
                    w
                    + 2 * layer.padding[1]
                    - layer.dilation[1] * (layer.kernel_size[1] - 1)
                    - 1
                )
                // layer.stride[1]
            ) + 1
            output_dims = (n, layer.out_channels, h_out, w_out)

            # FLOPs for a Conv2D layer
            flops_per_weight = layer.kernel_size[0] * layer.kernel_size[1] * c
            flops = flops_per_weight * non_zero_weights * h_out * w_out
            total_flops += flops

            # Update input_shape for next layers
            input_shape = output_dims

        elif isinstance(layer, nn.Linear):
            # Count the non-zero weights
            non_zero_weights = torch.count_nonzero(layer.weight).item()

            # FLOPs for a Linear layer
            flops = non_zero_weights * layer.out_features
            total_flops += flops

    return total_flops


def calculate_vit_flops(model, input_shape=(1, 3, 224, 224), sparsity=1.0):
    total_flops = 0
    seq_len = input_shape[1] * input_shape[2]  # Height * Width

    for layer in model.modules():
        if isinstance(layer, nn.MultiheadAttention):
            total_flops += calculate_attention_flops(
                layer, sparsity, seq_len, layer.num_heads
            )
        elif isinstance(layer, nn.Linear):
            total_flops += calculate_feedforward_flops(layer, sparsity)

    return total_flops


def calculate_attention_flops(module, sparsity, seq_len, num_heads):
    # Assuming sparsity is the fraction of non-zero attention scores
    dense_flops = 2 * seq_len**2  # Softmax and matrix multiply in attention
    sparse_flops = dense_flops * sparsity
    return sparse_flops * num_heads


def calculate_feedforward_flops(layer, sparsity):
    # Assuming sparsity is the fraction of non-zero weights
    input_features, output_features = layer.weight.shape
    dense_flops = 2 * input_features * output_features
    return dense_flops * sparsity
