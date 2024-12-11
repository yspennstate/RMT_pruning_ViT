import torch.nn as nn
import numpy as np
from RMT import bema_inside, error, MP_Density_Wrapper
import math
import matplotlib.pyplot as plt
from functools import partial
import torch


class Splittable(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def forward(self, x):
        return self.layer2(self.layer1(x))

    @property
    def goodnessOfFitCutoff(self):
        return self.goodnessOfFitCutoff_

    @property
    def param_numbers(self):
        if self.splitted:
            return (self.in_features + self.out_features) * self.layer1.out_features
        return self.in_features * self.out_features

    @staticmethod
    def shift(singular_values, gamma, sigma):
        ratios = singular_values / sigma
        new_ratios = np.sqrt(
            ratios**2 - gamma - 1 + np.sqrt((ratios**2 - gamma - 1) ** 2 - 4 * gamma)
        )
        min_mult = np.sqrt(2 * np.sqrt(gamma)) / (1 + np.sqrt(gamma))
        backup_ratios = ratios * min_mult
        new_ratios[np.isnan(new_ratios)] = backup_ratios[np.isnan(new_ratios)]
        return sigma * new_ratios

    def fit_MP(self, U, singular_values, V, save_plot):
        eigenvals = singular_values**2 / V.shape[0]
        eigenvals = np.sort(eigenvals)

        p = min(U.shape[0], V.shape[0])
        n = max(U.shape[0], V.shape[0])
        gamma = p / n
        sigma_sq, lamda_plus, l2 = bema_inside(p, n, eigenvals, self.alpha, self.beta)
        Splus = math.sqrt(V.shape[1] * lamda_plus)
        LinfError = error(eigenvals, self.alpha, p, gamma, sigma_sq)
        goodFit = LinfError < self.goodnessOfFitCutoff
        if True:  # save_plot is not None:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
            Z = eigenvals[eigenvals < lamda_plus]
            Y = MP_Density_Wrapper(gamma, sigma_sq, Z)
            # plt.plot(Z,Y, color = "orange", label = "Predicted Density")
            ax[0].axvline(
                x=lamda_plus, label=f"Lambda Plus={lamda_plus:.2e}", color="red"
            )
            ax[0].plot(
                np.sort(eigenvals),
                np.linspace(0, 1, len(eigenvals), endpoint=False),
                color="orange",
                label="Empirical CDF",
            )
            ax[0].set_xscale("log")
            ax[0].legend()
            ax[0].title.set_text("Empirical CDF")

            ax[1].hist(
                Z,
                bins=200,
                color="black",
                label="Truncated Empirical Density",
                density=True,
            )
            ax[1].plot(Z, Y, color="orange", label="Predicted Density")
            ax[1].legend()
            ax[1].title.set_text("Density Comparison Zoomed")

            fig.suptitle(f'{save_plot}, Decision : {"no"*int(not(goodFit))} goodFit')
            fig.savefig(f"{save_plot}.png")
            plt.close(fig)
            # plt.show()
        shifted_singular_values = Splittable.shift(singular_values, gamma, sigma_sq)

        return Splus, goodFit, shifted_singular_values, LinfError

    def pruningRMT(
        self,
        layer_name,
        threshold,
        splus,
        LinfError,
        percentage_less_than_splus,
        n_prune_cycles,
    ):
        # Step 1: Get the matrix of the layer
        W = self.get_matrix()
        M, N = W.shape

        # Step 2: Calculate sigma^2 from Lambda_+ (splus)
        Q = M / N
        lambda_plus = splus**2
        sigma_squared = lambda_plus / ((1 + Q) ** 2)

        # Construct Spike Model
        R = np.random.normal(0, np.sqrt(sigma_squared / N), size=W.shape)
        S = W - R  # Deterministic matrix S

        # Histogram of R
        plt.hist(R.flatten(), bins=50)
        plt.title(f"Histogram of R Parameters After Pruning - Layer: {layer_name}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        # plt.show()

        # Step 3: Prune elements of R
        R[np.abs(R) < threshold] = 0
        # R -= .000005

        # Step 4: Update Weight Layer
        W_new = R + S

        self.set_params(
            "layer1",
            torch.from_numpy(W_new).float(),
            bias=None,
            change_bias=False,
        )

    def split(
        self, ratio, thresh, shift=False, recombine_when_no_fit=False, save_plot=None
    ):
        matrix = self.get_matrix()
        U, S, V = np.linalg.svd(matrix)
        Splus, goodFit, shifted_singular_values, LinfError = self.fit_MP(
            U, S, V, save_plot
        )
        print(goodFit)
        # Keeping the original logic for singular values
        sparsify_mask = S < Splus
        percentage = np.sum(sparsify_mask) / len(S) * 100

        if not goodFit:
            return f"{self.name} no good fit, left as is", Splus, LinfError, percentage

        if shift:
            S = shifted_singular_values
        else:
            S = S

        significant_singulars = np.sum(S > Splus)
        inner_dim = max(
            int((S.shape[0] - significant_singulars) * ratio) + significant_singulars,
            self.min_dim,
        )

        original_params = matrix.shape[0] * matrix.shape[1]
        reduced_params = (matrix.shape[0] + matrix.shape[1]) * inner_dim

        if goodFit:
            # Keeping the original logic for singular values
            filtered_S = S[:inner_dim]
            sparsify_mask = S < Splus
            percentage = np.sum(sparsify_mask) / len(S) * 100
            # print(f"Percentage of singular values less than Splus: {percentage:.2f}%")

            filtered_U = U[:, :inner_dim]
            filtered_V = V[:inner_dim, :]
            dynamic_thresh = thresh * ((1 - S / Splus) ** 30)

            for i in range(inner_dim):
                if sparsify_mask[i]:
                    filtered_U[:, i] = np.where(
                        np.abs(filtered_U[:, i]) < dynamic_thresh[i],
                        0,
                        filtered_U[:, i],
                    )
                    filtered_V[i, :] = np.where(
                        np.abs(filtered_V[i, :]) < dynamic_thresh[i],
                        0,
                        filtered_V[i, :],
                    )

            # Apply pruning to all elements in U and V matrices
            for i in range(inner_dim):
                filtered_U[:, i] = np.where(
                    np.abs(filtered_U[:, i]) < thresh / 750, 0, filtered_U[:, i]
                )
                filtered_V[i, :] = np.where(
                    np.abs(filtered_V[i, :]) < thresh / 750, 0, filtered_V[i, :]
                )

            if self.splitted == True:
                return f"{self.name} Already splitted", Splus, LinfError, percentage

            elif self.param_numbers <= reduced_params:
                # Recompose the matrix if parameter reduction is not significant
                new_matrix = (filtered_U * filtered_S[None, :inner_dim]) @ filtered_V
                self.set_params(
                    "layer1",
                    torch.from_numpy(new_matrix).float(),
                    bias=None,
                    change_bias=False,
                )
                return (
                    f"{self.name} not enough param reduction, matrix recomposed",
                    Splus,
                    LinfError,
                    percentage,
                )

            else:
                # Proceed with the split
                new_weights1 = np.sqrt(filtered_S)[:inner_dim, None] * filtered_V
                new_weights2 = filtered_U * np.sqrt(filtered_S)[None, :inner_dim]

                # Clone bias
                original_bias = None
                try:
                    original_bias = nn.Parameter(self.layer1.bias.clone())
                except AttributeError:
                    pass

                self.layer1, self.layer2 = self.make_splitted_layers(inner_dim)
                self.set_params(
                    "layer1", torch.from_numpy(new_weights1).float(), bias=None
                )
                self.set_params(
                    "layer2", torch.from_numpy(new_weights2).float(), original_bias
                )
                self.splitted = True
                return (
                    f"{self.name} splitted, new dims {(self.in_features, inner_dim, self.out_features)}",
                    Splus,
                    LinfError,
                    percentage,
                )


class SplittableLinear(Splittable):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(
        self,
        in_features,
        out_features,
        alpha,
        beta,
        goodnessOfFitCutoff,
        name="splittable_linear",
        bias=True,
        min_dim=10,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer1 = nn.Linear(in_features, out_features, bias=bias)
        self.layer2 = nn.Identity()
        self.splitted = False
        self.alpha = alpha
        self.beta = beta
        self.goodnessOfFitCutoff_ = goodnessOfFitCutoff
        self.name = name
        self.min_dim = min_dim

    @property
    def param_numbers(self):
        if self.splitted:
            return (self.in_features + self.out_features) * self.layer1.out_features
        return self.in_features * self.out_features

    @staticmethod
    def from_layer(linear, alpha, beta, goodnessOfFitCutoff):
        bias = linear.bias != None
        splittable_lin = SplittableLinear(
            linear.in_features,
            linear.out_features,
            alpha,
            beta,
            goodnessOfFitCutoff,
            bias=bias,
        )
        splittable_lin.set_params("layer1", linear.weight, linear.bias)
        return splittable_lin

    def __str__(self):
        return (
            f"Linear(in_features={self.in_features}, out_features={self.out_features})"
        )

    def get_matrix(self):
        layerMatrix = torch.as_tensor(self.layer1.weight)
        layerMatrix = layerMatrix.cpu()
        layerMatrix = layerMatrix.detach().numpy()
        if not self.splitted:
            return layerMatrix
        layerMatrix2 = torch.as_tensor(self.layer2.weight)
        layerMatrix2 = layerMatrix2.cpu()
        layerMatrix2 = layerMatrix2.detach().numpy()
        return layerMatrix2 @ layerMatrix

    def set_params(self, which_layer, weight, bias, change_bias=True):
        assert which_layer in ["layer1", "layer2"]
        getattr(self, which_layer).weight = nn.Parameter(weight)
        if change_bias:
            if bias is None:
                getattr(self, which_layer).bias = None
            else:
                getattr(self, which_layer).bias = nn.Parameter(bias)

    def make_splitted_layers(self, inner_dim):
        layer1 = nn.Linear(self.in_features, inner_dim, bias=False)
        layer2 = nn.Linear(inner_dim, self.out_features, bias=False)
        return layer1, layer2


class SplittableConv(Splittable):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha,
        beta,
        goodnessOfFitCutoff,
        name="splittable_conv",
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        padding_mode="zeros",
        min_dim=10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = kernel_size * kernel_size * in_channels
        self.out_features = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.padding_mode = padding_mode

        self.layer1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.layer2 = nn.Identity()
        self.splitted = False
        self.alpha = alpha
        self.beta = beta
        self.goodnessOfFitCutoff_ = goodnessOfFitCutoff
        self.name = name
        self.min_dim = min_dim

    @property
    def param_numbers(self):
        if self.splitted:
            return np.prod(self.layer1.weight.shape) + np.prod(self.layer2.weight.shape)
        return self.in_features * self.out_features

    def __str__(self):
        return f"SplittableConv2d({self.in_channels},{self.out_channels},kernel_size=({self.kernel_size},{self.kernel_size}),stride=({self.stride},{self.stride}))"

    @staticmethod
    def from_layer(conv, alpha, beta, goodnessOfFitCutoff):
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        assert conv.kernel_size[0] == conv.kernel_size[1]
        kernel_size = conv.kernel_size[0]
        assert conv.stride[0] == conv.stride[1]  # type: ignore
        stride = conv.stride[0]  # type: ignore
        assert conv.padding[0] == conv.padding[1]  # type: ignore
        padding = conv.padding[0]  # type: ignore
        groups = conv.groups
        padding_mode = conv.padding_mode
        bias = conv.bias != None
        splittable_conv = SplittableConv(
            in_channels,
            out_channels,
            kernel_size,
            alpha,
            beta,
            goodnessOfFitCutoff,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        splittable_conv.set_params("layer1", conv.weight, conv.bias)
        return splittable_conv

    def get_matrix(self):
        layerMatrix = torch.as_tensor(self.layer1.weight)
        layerMatrix = layerMatrix.cpu()
        layerMatrix = layerMatrix.detach().numpy()
        layerMatrix = np.reshape(layerMatrix, (layerMatrix.shape[0], -1))
        if not self.splitted:
            return layerMatrix
        layerMatrix2 = torch.as_tensor(self.layer2.weight)
        layerMatrix2 = layerMatrix2.cpu()
        layerMatrix2 = layerMatrix2.detach().numpy()
        layerMatrix2 = np.reshape(layerMatrix2, (layerMatrix2.shape[0], -1))
        return layerMatrix2 @ layerMatrix

    def set_params(self, which_layer, weight, bias, change_bias=True):
        assert which_layer in ["layer1", "layer2"]
        if len(weight.shape) == 2:
            weight = torch.reshape(weight, getattr(self, which_layer).weight.shape)
        getattr(self, which_layer).weight = nn.Parameter(weight)
        if change_bias:
            if bias is None:
                getattr(self, which_layer).bias = None
            else:
                getattr(self, which_layer).bias = nn.Parameter(bias)

    def make_splitted_layers(self, inner_dim):
        layer1 = nn.Conv2d(
            self.in_channels,
            inner_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=False,
            padding_mode=self.padding_mode,
        )
        layer2 = nn.Conv2d(
            inner_dim,
            self.out_channels,
            1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
            padding_mode=self.padding_mode,
        )
        return layer1, layer2
