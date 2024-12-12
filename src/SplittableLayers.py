import torch.nn as nn
import numpy as np
from RMT import bema_inside, error, MP_Density_Wrapper
import math
import matplotlib.pyplot as plt
import torch


class Splittable(nn.Module):
    """
    Base class for splittable layers.
    """

    def forward(self, x):
        """
        Forward pass through the layer.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        return self.layer2(self.layer1(x))

    @property
    def goodnessOfFitCutoff(self):
        """
        Get the goodness of fit cutoff.

        Returns:
        float: Goodness of fit cutoff.
        """
        return self.goodnessOfFitCutoff_

    @property
    def param_numbers(self):
        """
        Get the number of parameters in the layer.

        Returns:
        int: Number of parameters.
        """
        if self.splitted:
            return (self.in_features + self.out_features) * self.layer1.out_features
        return self.in_features * self.out_features

    @staticmethod
    def shift(singular_values, gamma, sigma):
        """
        Shift singular values based on gamma and sigma.

        Parameters:
        singular_values (array): Singular values.
        gamma (float): Ratio of dimensions.
        sigma (float): Variance.

        Returns:
        array: Shifted singular values.
        """
        ratios = singular_values / sigma
        new_ratios = np.sqrt(
            ratios**2 - gamma - 1 + np.sqrt((ratios**2 - gamma - 1) ** 2 - 4 * gamma)
        )
        min_mult = np.sqrt(2 * np.sqrt(gamma)) / (1 + np.sqrt(gamma))
        backup_ratios = ratios * min_mult
        new_ratios[np.isnan(new_ratios)] = backup_ratios[np.isnan(new_ratios)]
        return sigma * new_ratios

    def fit_MP(self, U, singular_values, V, save_plot):
        """
        Fit the Marcenko-Pastur distribution to the singular values.

        Parameters:
        U (array): Left singular vectors.
        singular_values (array): Singular values.
        V (array): Right singular vectors.
        save_plot (str): Path to save the plot.

        Returns:
        tuple: Splus, goodFit, shifted_singular_values, LinfError
        """
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
        """
        Prune the layer using Random Matrix Theory.

        Parameters:
        layer_name (str): Name of the layer.
        threshold (float): Threshold for pruning.
        splus (float): Splus value.
        LinfError (float): Linf error.
        percentage_less_than_splus (float): Percentage of singular values less than Splus.
        n_prune_cycles (int): Number of pruning cycles.
        """
        W = self.get_matrix()
        M, N = W.shape

        Q = M / N
        lambda_plus = splus**2
        sigma_squared = lambda_plus / ((1 + Q) ** 2)

        R = np.random.normal(0, np.sqrt(sigma_squared / N), size=W.shape)
        S = W - R

        plt.hist(R.flatten(), bins=50)
        plt.title(f"Histogram of R Parameters After Pruning - Layer: {layer_name}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        R[np.abs(R) < threshold] = 0

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
        """
        Split the layer based on singular values.

        Parameters:
        ratio (float): Ratio for splitting.
        thresh (float): Threshold for pruning.
        shift (bool): If True, shift singular values, default is False.
        recombine_when_no_fit (bool): If True, recombine when no good fit, default is False.
        save_plot (str): Path to save the plot.

        Returns:
        tuple: Result message, Splus, LinfError, percentage
        """
        matrix = self.get_matrix()
        U, S, V = np.linalg.svd(matrix)
        Splus, goodFit, shifted_singular_values, LinfError = self.fit_MP(
            U, S, V, save_plot
        )
        sparsify_mask = S < Splus
        percentage = np.sum(sparsify_mask) / len(S) * 100

        if not goodFit:
            return f"{self.name} no good fit, left as is", Splus, LinfError, percentage

        if shift:
            S = shifted_singular_values

        significant_singulars = np.sum(S > Splus)
        inner_dim = max(
            int((S.shape[0] - significant_singulars) * ratio) + significant_singulars,
            self.min_dim,
        )

        original_params = matrix.shape[0] * matrix.shape[1]
        reduced_params = (matrix.shape[0] + matrix.shape[1]) * inner_dim

        if goodFit:
            filtered_S = S[:inner_dim]
            sparsify_mask = S < Splus
            percentage = np.sum(sparsify_mask) / len(S) * 100

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

            for i in range(inner_dim):
                filtered_U[:, i] = np.where(
                    np.abs(filtered_U[:, i]) < thresh / 750, 0, filtered_U[:, i]
                )
                filtered_V[i, :] = np.where(
                    np.abs(filtered_V[i, :]) < thresh / 750, 0, filtered_V[i, :]
                )

            if self.splitted:
                return f"{self.name} Already splitted", Splus, LinfError, percentage

            elif self.param_numbers <= reduced_params:
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
                new_weights1 = np.sqrt(filtered_S)[:inner_dim, None] * filtered_V
                new_weights2 = filtered_U * np.sqrt(filtered_S)[None, :inner_dim]

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
    """
    Custom Linear layer that mimics a standard linear layer but can be split.
    """

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
        """
        Initialize the SplittableLinear layer.

        Parameters:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        alpha (float): Alpha parameter.
        beta (float): Beta parameter.
        goodnessOfFitCutoff (float): Goodness of fit cutoff.
        name (str): Name of the layer, default is "splittable_linear".
        bias (bool): If True, include a bias term, default is True.
        min_dim (int): Minimum dimension for splitting, default is 10.
        """
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
        """
        Get the number of parameters in the layer.

        Returns:
        int: Number of parameters.
        """
        if self.splitted:
            return (self.in_features + self.out_features) * self.layer1.out_features
        return self.in_features * self.out_features

    @staticmethod
    def from_layer(linear, alpha, beta, goodnessOfFitCutoff):
        """
        Create a SplittableLinear layer from an existing Linear layer.

        Parameters:
        linear (torch.nn.Linear): The existing Linear layer.
        alpha (float): Alpha parameter.
        beta (float): Beta parameter.
        goodnessOfFitCutoff (float): Goodness of fit cutoff.

        Returns:
        SplittableLinear: The created SplittableLinear layer.
        """
        bias = linear.bias is not None
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
        """
        Get the weight matrix of the layer.

        Returns:
        array: Weight matrix.
        """
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
        """
        Set the parameters of the layer.

        Parameters:
        which_layer (str): Which layer to set parameters for ("layer1" or "layer2").
        weight (array): Weight matrix.
        bias (array): Bias vector.
        change_bias (bool): If True, change the bias, default is True.
        """
        assert which_layer in ["layer1", "layer2"]
        getattr(self, which_layer).weight = nn.Parameter(weight)
        if change_bias:
            if bias is None:
                getattr(self, which_layer).bias = None
            else:
                getattr(self, which_layer).bias = nn.Parameter(bias)

    def make_splitted_layers(self, inner_dim):
        """
        Create the splitted layers.

        Parameters:
        inner_dim (int): Inner dimension for the split.

        Returns:
        tuple: The created layers.
        """
        layer1 = nn.Linear(self.in_features, inner_dim, bias=False)
        layer2 = nn.Linear(inner_dim, self.out_features, bias=False)
        return layer1, layer2


class SplittableConv(Splittable):
    """Custom Linear layer but mimics a standard linear layer with the ability to split into two layers."""

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
        """
        Initialize the SplittableConv layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            alpha (float): Alpha parameter for splitting.
            beta (float): Beta parameter for splitting.
            goodnessOfFitCutoff (float): Cutoff value for goodness of fit.
            name (str): Name of the layer.
            stride (int): Stride of the convolution. Default is 1.
            padding (int): Padding added to all four sides of the input. Default is 0.
            groups (int): Number of blocked connections from input channels to output channels. Default is 1.
            bias (bool): If True, adds a learnable bias to the output. Default is True.
            padding_mode (str): Padding mode. Default is "zeros".
            min_dim (int): Minimum dimension for splitting. Default is 10.
        """
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
        """
        Calculate the number of parameters in the layer.

        Returns:
            int: Number of parameters.
        """
        if self.splitted:
            return np.prod(self.layer1.weight.shape) + np.prod(self.layer2.weight.shape)
        return self.in_features * self.out_features

    def __str__(self):
        """
        String representation of the layer.

        Returns:
            str: String representation.
        """
        return f"SplittableConv2d({self.in_channels},{self.out_channels},kernel_size=({self.kernel_size},{self.kernel_size}),stride=({self.stride},{self.stride}))"

    @staticmethod
    def from_layer(conv, alpha, beta, goodnessOfFitCutoff):
        """
        Create a SplittableConv layer from an existing Conv2d layer.

        Args:
            conv (nn.Conv2d): Existing Conv2d layer.
            alpha (float): Alpha parameter for splitting.
            beta (float): Beta parameter for splitting.
            goodnessOfFitCutoff (float): Cutoff value for goodness of fit.

        Returns:
            SplittableConv: New SplittableConv layer.
        """
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
        """
        Get the weight matrix of the layer.

        Returns:
            np.ndarray: Weight matrix.
        """
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
        """
        Set the parameters of a specified layer.

        Args:
            which_layer (str): The layer to set parameters for ("layer1" or "layer2").
            weight (torch.Tensor): The weight tensor.
            bias (torch.Tensor): The bias tensor.
            change_bias (bool): Whether to change the bias. Default is True.
        """
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
        """
        Create the splitted layers.

        Args:
            inner_dim (int): The inner dimension for the split.

        Returns:
            tuple: The two splitted layers.
        """
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
