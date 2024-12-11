### This code is copied from torchstat and modified


import queue
import time
from typing import OrderedDict
import torch
import torch.nn as nn


import numpy as np
import pandas as pd


def compute_flops(module, inp, out):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_flops(module, inp, out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_flops(module, inp, out)
    elif isinstance(
        module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.GELU)
    ):
        return compute_ReLU_flops(module, inp, out)
    elif isinstance(module, nn.Upsample):
        return compute_Upsample_flops(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp, out)
    elif (
        isinstance(module, nn.Dropout)
        or isinstance(module, nn.Identity)
        or isinstance(module, nn.LayerNorm)
    ):
        return 0
    else:
        print(f"[Flops]: {type(module).__name__} is not supported!")
        return 0
    pass


"""def compute_SplittableLinear_flops(module, inp, out):
    assert len(inp.size()) == 2 and len(out.size()) == 2
    if module.splitted:
        return compute_Linear_flops(module.layer1, inp, out) + compute_Linear_flops(
            module.layer2, inp, out
        )
    return compute_Linear_flops(module.layer1, inp, out)"""


def compute_Conv2d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_BatchNorm2d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    in_c, in_h, in_w = inp.size()[1:]
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops


def compute_ReLU_flops(module, inp, out):
    assert isinstance(
        module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.GELU)
    )
    batch_size = inp.size()[0]
    active_elements_count = batch_size

    for s in inp.size()[1:]:
        active_elements_count *= s

    return active_elements_count


def compute_Pool2d_flops(module, inp, out):
    assert isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    return np.prod(inp.shape)


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    if len(inp.size()) == 3:
        assert len(out.size()) == 3
        assert out.size()[0] == inp.size()[0]
        return out.size()[0] * compute_Linear_flops(module, inp[0], out[0])
    assert len(inp.size()) == 2 and len(out.size()) == 2
    batch_size = inp.size()[0]
    return batch_size * torch.count_nonzero(module.weight)
    return batch_size * inp.size()[1] * out.size()[1]


def compute_Upsample_flops(module, inp, out):
    assert isinstance(module, nn.Upsample)
    output_size = out[0]
    batch_size = inp.size()[0]
    output_elements_count = batch_size
    for s in output_size.shape[1:]:
        output_elements_count *= s

    return output_elements_count


def compute_Conv2d_madd(module, inp, out):
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    # ops per output element
    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
    kernel_add_group = kernel_add * out_h * out_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return total_mul + total_add


def compute_ConvTranspose2d_madd(module, inp, out):
    assert isinstance(module, nn.ConvTranspose2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c, in_h, in_w = inp.size()[1:]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * in_h * in_w * (out_c // groups)
    kernel_add_group = kernel_add * in_h * in_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return total_mul + total_add


def compute_BatchNorm2d_madd(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c, in_h, in_w = inp.size()[1:]

    # 1. sub mean
    # 2. div standard deviation
    # 3. mul alpha
    # 4. add beta
    return 4 * in_c * in_h * in_w


def compute_MaxPool2d_madd(module, inp, out):
    assert isinstance(module, nn.MaxPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out.size()[1:]

    return (k_h * k_w - 1) * out_h * out_w * out_c


def compute_AvgPool2d_madd(module, inp, out):
    assert isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out.size()[1:]

    kernel_add = k_h * k_w - 1
    kernel_avg = 1

    return (kernel_add + kernel_avg) * (out_h * out_w) * out_c


def compute_ReLU_madd(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU))

    count = 1
    for i in inp.size()[1:]:
        count *= i
    return count


def compute_Softmax_madd(module, inp, out):
    assert isinstance(module, nn.Softmax)
    assert len(inp.size()) > 1

    count = 1
    for s in inp.size()[1:]:
        count *= s
    exp = count
    add = count - 1
    div = count
    return exp + add + div


def compute_Linear_madd(module, inp, out):
    assert isinstance(module, nn.Linear)
    if len(inp.size()) == 3:
        assert len(out.size()) == 3
        assert out.size()[0] == inp.size()[0]
        return out.size()[0] * compute_Linear_madd(module, inp[0], out[0])
    assert len(inp.size()) == 2 and len(out.size()) == 2

    num_in_features = inp.size()[1]
    num_out_features = out.size()[1]

    mul = num_in_features
    add = num_in_features - 1
    return num_out_features * (mul + add)


def compute_Bilinear_madd(module, inp1, inp2, out):
    assert isinstance(module, nn.Bilinear)
    assert len(inp1.size()) == 2 and len(inp2.size()) == 2 and len(out.size()) == 2

    num_in_features_1 = inp1.size()[1]
    num_in_features_2 = inp2.size()[1]
    num_out_features = out.size()[1]

    mul = num_in_features_1 * num_in_features_2 + num_in_features_2
    add = num_in_features_1 * num_in_features_2 + num_in_features_2 - 1
    return num_out_features * (mul + add)


def compute_madd(module, inp, out):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_madd(module, inp, out)
    elif isinstance(module, nn.ConvTranspose2d):
        return compute_ConvTranspose2d_madd(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_madd(module, inp, out)
    elif isinstance(module, nn.MaxPool2d):
        return compute_MaxPool2d_madd(module, inp, out)
    elif isinstance(module, nn.AvgPool2d):
        return compute_AvgPool2d_madd(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU)):
        return compute_ReLU_madd(module, inp, out)
    elif isinstance(module, nn.Softmax):
        return compute_Softmax_madd(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_madd(module, inp, out)
    elif isinstance(module, nn.Bilinear):
        return compute_Bilinear_madd(module, inp[0], inp[1], out)
    elif (
        isinstance(module, nn.Dropout)
        or isinstance(module, nn.Identity)
        or isinstance(module, nn.LayerNorm)
    ):
        return 0
    else:
        print(f"[MAdd]: {type(module).__name__} is not supported!")
        return 0


def compute_memory(module, inp, out):
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU, nn.GELU)):
        return compute_ReLU_memory(module, inp, out)
    elif isinstance(module, nn.PReLU):
        return compute_PReLU_memory(module, inp, out)
    elif isinstance(module, nn.Conv2d):
        return compute_Conv2d_memory(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_memory(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_memory(module, inp, out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_memory(module, inp, out)
    elif (
        isinstance(module, nn.Dropout)
        or isinstance(module, nn.Identity)
        or isinstance(module, nn.LayerNorm)
    ):
        return (0, 0)
    else:
        print(f"[Memory]: {type(module).__name__} is not supported!")
        return (0, 0)
    pass


def num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_ReLU_memory(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU, nn.GELU))
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * inp.size()[1:].numel()

    return (mread, mwrite)


def compute_PReLU_memory(module, inp, out):
    assert isinstance(module, (nn.PReLU))
    batch_size = inp.size()[0]
    mread = batch_size * (inp.size()[1:].numel() + num_params(module))
    mwrite = batch_size * inp.size()[1:].numel()

    return (mread, mwrite)


def compute_Conv2d_memory(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    out_c, out_h, out_w = out.size()[1:]

    # This includes weighs with bias if the module contains it.
    mread = batch_size * (inp.size()[1:].numel() + num_params(module))
    mwrite = batch_size * out_c * out_h * out_w
    return (mread, mwrite)


def compute_BatchNorm2d_memory(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_size, in_c, in_h, in_w = inp.size()

    mread = batch_size * (inp.size()[1:].numel() + 2 * in_c)
    mwrite = inp.size().numel()
    return (mread, mwrite)


def compute_Linear_memory(module, inp, out):
    assert isinstance(module, nn.Linear)
    if len(inp.size()) == 3:
        assert len(out.size()) == 3
        assert out.size()[0] == inp.size()[0]
        return out.size()[0] * compute_Linear_memory(module, inp[0], out[0])

    assert len(inp.size()) == 2 and len(out.size()) == 2
    batch_size = inp.size()[0]
    mread = batch_size * (inp.size()[1:].numel() + num_params(module))
    mwrite = out.size().numel()

    return (mread, mwrite)


def compute_Pool2d_memory(module, inp, out):
    assert isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    return (mread, mwrite)


class ModelHook(object):
    def __init__(self, model, input_size):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (list, tuple))

        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook

        self._hook_model()
        x = torch.rand(1, *self._input_size)  # add module duration time
        self._model.eval()
        self._model(x)

    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        if len(list(module.children())) > 0:
            return

        module.register_buffer("input_shape", torch.zeros(3).int())
        module.register_buffer("output_shape", torch.zeros(3).int())
        module.register_buffer("parameter_quantity", torch.zeros(1).int())
        module.register_buffer("inference_memory", torch.zeros(1).long())
        module.register_buffer("MAdd", torch.zeros(1).long())
        module.register_buffer("duration", torch.zeros(1).float())
        module.register_buffer("Flops", torch.zeros(1).long())
        module.register_buffer("Memory", torch.zeros(2).long())

    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call

            # Itemsize for memory
            itemsize = input[0].detach().numpy().itemsize

            start = time.time()
            output = self._origin_call[module.__class__](module, *input, **kwargs)
            end = time.time()
            module.duration = torch.from_numpy(
                np.array([end - start], dtype=np.float32)
            )

            module.input_shape = torch.from_numpy(
                np.array(input[0].size()[1:], dtype=np.int32)
            )
            module.output_shape = torch.from_numpy(
                np.array(output.size()[1:], dtype=np.int32)
            )

            parameter_quantity = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                parameter_quantity += 0 if p is None else torch.numel(p.data)
            module.parameter_quantity = torch.from_numpy(
                np.array([parameter_quantity], dtype=np.longlong)
            )

            inference_memory = 1
            for s in output.size()[1:]:
                inference_memory *= s
            # memory += parameters_number  # exclude parameter memory
            inference_memory = inference_memory * 4 / (1024**2)  # shown as MB unit
            module.inference_memory = torch.from_numpy(
                np.array([inference_memory], dtype=np.float32)
            )

            if len(input) == 1:
                madd = compute_madd(module, input[0], output)
                flops = compute_flops(module, input[0], output)
                Memory = compute_memory(module, input[0], output)
            elif len(input) > 1:
                madd = compute_madd(module, input, output)
                flops = compute_flops(module, input, output)
                Memory = compute_memory(module, input, output)
            else:  # error
                madd = 0
                flops = 0
                Memory = (0, 0)
            module.MAdd = torch.from_numpy(np.array([madd], dtype=np.int64))
            module.Flops = torch.from_numpy(np.array([flops], dtype=np.int64))
            Memory = np.array(Memory, dtype=np.int32) * itemsize
            module.Memory = torch.from_numpy(Memory)

            return output

        for module in self._model.modules():
            if (
                len(list(module.children())) == 0
                and module.__class__ not in self._origin_call
            ):
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        self._sub_module_call_hook()

    @staticmethod
    def _retrieve_leaf_modules(model):
        leaf_modules = []
        for name, m in model.named_modules():
            if len(list(m.children())) == 0:
                leaf_modules.append((name, m))
        return leaf_modules

    def retrieve_leaf_modules(self):
        return OrderedDict(self._retrieve_leaf_modules(self._model))


pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 10000)


def round_value(value, binary=False):
    divisor = 1024.0 if binary else 1000.0

    if value // divisor**4 > 0:
        return str(round(value / divisor**4, 2)) + "T"
    elif value // divisor**3 > 0:
        return str(round(value / divisor**3, 2)) + "G"
    elif value // divisor**2 > 0:
        return str(round(value / divisor**2, 2)) + "M"
    elif value // divisor > 0:
        return str(round(value / divisor, 2)) + "K"
    return str(value)


def get_report_dataframe(collected_nodes):
    data = list()
    for node in collected_nodes:
        name = node.name
        input_shape = " ".join(["{:>3d}"] * len(node.input_shape)).format(
            *[e for e in node.input_shape]
        )
        output_shape = " ".join(["{:>3d}"] * len(node.output_shape)).format(
            *[e for e in node.output_shape]
        )
        parameter_quantity = node.parameter_quantity
        inference_memory = node.inference_memory
        MAdd = node.MAdd
        Flops = node.Flops
        mread, mwrite = [i for i in node.Memory]
        duration = node.duration
        data.append(
            [
                name,
                input_shape,
                output_shape,
                parameter_quantity,
                inference_memory,
                MAdd,
                duration,
                Flops,
                mread,
                mwrite,
            ]
        )
    df = pd.DataFrame(data)
    df.columns = [
        "module name",
        "input shape",
        "output shape",
        "params",
        "memory(MB)",
        "MAdd",
        "duration",
        "Flops",
        "MemRead(B)",
        "MemWrite(B)",
    ]
    df["duration[%]"] = df["duration"] / (df["duration"].sum() + 1e-7)
    df["MemR+W(B)"] = df["MemRead(B)"] + df["MemWrite(B)"]
    total_parameters_quantity = df["params"].sum()
    total_memory = df["memory(MB)"].sum()
    total_operation_quantity = df["MAdd"].sum()
    total_flops = df["Flops"].sum()
    total_duration = df["duration[%]"].sum()
    total_mread = df["MemRead(B)"].sum()
    total_mwrite = df["MemWrite(B)"].sum()
    total_memrw = df["MemR+W(B)"].sum()
    del df["duration"]

    """# Add Total row
    total_df = pd.Series(
        [
            total_parameters_quantity,
            total_memory,
            total_operation_quantity,
            total_flops,
            total_duration,
            mread,
            mwrite,
            total_memrw,
        ],
        index=[
            "params",
            "memory(MB)",
            "MAdd",
            "Flops",
            "duration[%]",
            "MemRead(B)",
            "MemWrite(B)",
            "MemR+W(B)",
        ],
        name="total",
    )
    df = pd.concat([df.T, total_df]).T"""
    df.loc["Total"] = df.sum(numeric_only=True)

    df = df.fillna(" ")
    df["memory(MB)"] = df["memory(MB)"].apply(lambda x: f"{x:.2f}")
    df["duration[%]"] = df["duration[%]"].apply(lambda x: f"{x:.2%}")
    df["MAdd"] = df["MAdd"].apply(lambda x: f"{x:,}")
    df["Flops"] = df["Flops"].apply(lambda x: f"{x:,}")

    return (
        df,
        total_parameters_quantity,
        total_memory,
        total_operation_quantity,
        total_flops,
        total_memrw,
    )


def report_format(
    df,
    total_parameters_quantity,
    total_memory,
    total_operation_quantity,
    total_flops,
    total_memrw,
):

    summary = str(df) + "\n"
    summary += "=" * len(str(df).split("\n")[0])
    summary += "\n"
    summary += "Total params: {:,}\n".format(total_parameters_quantity)

    summary += "-" * len(str(df).split("\n")[0])
    summary += "\n"
    summary += "Total memory: {:.2f}MB\n".format(total_memory)
    summary += "Total MAdd: {}MAdd\n".format(round_value(total_operation_quantity))
    summary += "Total Flops: {}Flops\n".format(round_value(total_flops))
    summary += "Total MemR+W: {}B\n".format(round_value(total_memrw, True))
    return summary


class StatTree(object):
    def __init__(self, root_node):
        assert isinstance(root_node, StatNode)

        self.root_node = root_node

    def get_same_level_max_node_depth(self, query_node):
        if query_node.name == self.root_node.name:
            return 0
        same_level_depth = max([child.depth for child in query_node.parent.children])
        return same_level_depth

    def update_stat_nodes_granularity(self):
        q = queue.Queue()
        q.put(self.root_node)
        while not q.empty():
            node = q.get()
            node.granularity = self.get_same_level_max_node_depth(node)
            for child in node.children:
                q.put(child)

    def get_collected_stat_nodes(self, query_granularity):
        self.update_stat_nodes_granularity()

        collected_nodes = []
        stack = list()
        stack.append(self.root_node)
        while len(stack) > 0:
            node = stack.pop()
            for child in reversed(node.children):
                stack.append(child)
            if node.depth == query_granularity:
                collected_nodes.append(node)
            if node.depth < query_granularity <= node.granularity:
                collected_nodes.append(node)
        return collected_nodes


class StatNode(object):
    def __init__(self, name=str(), parent=None):
        self._name = name
        self._input_shape = None
        self._output_shape = None
        self._parameter_quantity = 0
        self._inference_memory = 0
        self._MAdd = 0
        self._Memory = (0, 0)
        self._Flops = 0
        self._duration = 0
        self._duration_percent = 0

        self._granularity = 1
        self._depth = 1
        self.parent = parent
        self.children = list()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def granularity(self):
        return self._granularity

    @granularity.setter
    def granularity(self, g):
        self._granularity = g

    @property
    def depth(self):
        d = self._depth
        if len(self.children) > 0:
            d += max([child.depth for child in self.children])
        return d

    @property
    def input_shape(self):
        if len(self.children) == 0:  # leaf
            return self._input_shape
        else:
            return self.children[0].input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        self._input_shape = input_shape

    @property
    def output_shape(self):
        if len(self.children) == 0:  # leaf
            return self._output_shape
        else:
            return self.children[-1].output_shape

    @output_shape.setter
    def output_shape(self, output_shape):
        assert isinstance(output_shape, (list, tuple))
        self._output_shape = output_shape

    @property
    def parameter_quantity(self):
        # return self.parameters_quantity
        total_parameter_quantity = self._parameter_quantity
        for child in self.children:
            total_parameter_quantity += child.parameter_quantity
        return total_parameter_quantity

    @parameter_quantity.setter
    def parameter_quantity(self, parameter_quantity):
        assert parameter_quantity >= 0
        self._parameter_quantity = parameter_quantity

    @property
    def inference_memory(self):
        total_inference_memory = self._inference_memory
        for child in self.children:
            total_inference_memory += child.inference_memory
        return total_inference_memory

    @inference_memory.setter
    def inference_memory(self, inference_memory):
        self._inference_memory = inference_memory

    @property
    def MAdd(self):
        total_MAdd = self._MAdd
        for child in self.children:
            total_MAdd += child.MAdd
        return total_MAdd

    @MAdd.setter
    def MAdd(self, MAdd):
        self._MAdd = MAdd

    @property
    def Flops(self):
        total_Flops = self._Flops
        for child in self.children:
            total_Flops += child.Flops
        return total_Flops

    @Flops.setter
    def Flops(self, Flops):
        self._Flops = Flops

    @property
    def Memory(self):
        total_Memory = self._Memory
        for child in self.children:
            total_Memory[0] += child.Memory[0]  # type: ignore
            total_Memory[1] += child.Memory[1]  # type: ignore
            print(total_Memory)
        return total_Memory

    @Memory.setter
    def Memory(self, Memory):
        assert isinstance(Memory, (list, tuple))
        self._Memory = Memory

    @property
    def duration(self):
        total_duration = self._duration
        for child in self.children:
            total_duration += child.duration
        return total_duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    def find_child_index(self, child_name):
        assert isinstance(child_name, str)

        index = -1
        for i in range(len(self.children)):
            if child_name == self.children[i].name:
                index = i
        return index

    def add_child(self, node):
        assert isinstance(node, StatNode)

        if self.find_child_index(node.name) == -1:  # not exist
            self.children.append(node)


def get_parent_node(root_node, stat_node_name):
    assert isinstance(root_node, StatNode)

    node = root_node
    names = stat_node_name.split(".")
    for i in range(len(names) - 1):
        node_name = ".".join(names[0 : i + 1])
        child_index = node.find_child_index(node_name)
        assert child_index != -1
        node = node.children[child_index]
    return node


def convert_leaf_modules_to_stat_tree(leaf_modules):
    assert isinstance(leaf_modules, OrderedDict)

    create_index = 1
    root_node = StatNode(name="root", parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        names = leaf_module_name.split(".")
        for i in range(len(names)):
            create_index += 1
            stat_node_name = ".".join(names[0 : i + 1])
            parent_node = get_parent_node(root_node, stat_node_name)
            node = StatNode(name=stat_node_name, parent=parent_node)
            parent_node.add_child(node)
            if i == len(names) - 1:  # leaf module itself
                input_shape = leaf_module.input_shape.numpy().tolist()
                output_shape = leaf_module.output_shape.numpy().tolist()
                node.input_shape = input_shape
                node.output_shape = output_shape
                node.parameter_quantity = leaf_module.parameter_quantity.numpy()[0]
                node.inference_memory = leaf_module.inference_memory.numpy()[0]
                node.MAdd = leaf_module.MAdd.numpy()[0]
                node.Flops = leaf_module.Flops.numpy()[0]
                node.duration = leaf_module.duration.numpy()[0]
                node.Memory = leaf_module.Memory.numpy().tolist()
    return StatTree(root_node)


class ModelStat(object):
    def __init__(self, model, input_size, query_granularity=1):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (tuple, list)) and len(input_size) == 3
        self._model = model
        self._input_size = input_size
        self._query_granularity = query_granularity

    def _analyze_model(self):
        model_hook = ModelHook(self._model, self._input_size)
        leaf_modules = model_hook.retrieve_leaf_modules()
        stat_tree = convert_leaf_modules_to_stat_tree(leaf_modules)
        collected_nodes = stat_tree.get_collected_stat_nodes(self._query_granularity)
        return collected_nodes

    def make_report(self):
        collected_nodes = self._analyze_model()
        self.report_data = get_report_dataframe(collected_nodes)

    def get_FLOPs(self):
        try:
            return self.report_data[4]
        except:
            self.make_report()
            return self.report_data[4]

    def show_report(self):
        try:
            report = report_format(*self.report_data)
            print(report)
        except:
            self.make_report()
            report = report_format(*self.report_data)
            print(report)


def stat(model, input_size, query_granularity=1):
    ms = ModelStat(model, input_size, query_granularity)
    ms.show_report()


def statFLOPs(model, input_size, query_granularity=1):
    ms = ModelStat(model, input_size, query_granularity)
    return ms.get_FLOPs()


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
