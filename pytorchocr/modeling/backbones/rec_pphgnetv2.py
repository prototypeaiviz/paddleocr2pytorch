# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/PaddlePaddle/PaddleClas/blob/2f36cab604e439b59d1a854df34ece3b10d888e3/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py
"""

from __future__ import absolute_import, division, print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from paddle import ParamAttr
# from paddle.nn import Conv2D, BatchNorm, Linear, BatchNorm2D, MaxPool2D, AvgPool2D
# from paddle.nn.initializer import Uniform
# from paddle.regularizer import L2Decay

from typing import Tuple, List, Dict, Union, Callable, Any
#from ppocr.modeling.backbones.rec_donut_swin import DonutSwinModelOutput
from pytorchocr.modeling.backbones.rec_donut_swin import DonutSwinModelOutput


class IdentityBasedConv1x1(nn.Conv2d):
    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias_attr=False,
        )

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.Tensor(id_value)
        self.weight.set_value(torch.zeros_like(self.weight))

    def forward(self, input):
        kernel = self.weight + self.id_tensor
        result = F.conv2d(
            input,
            kernel,
            None,
            stride=1,
            padding=0,
            dilation=self._dilation,
            groups=self._groups,
        )
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor


class BNAndPad(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        epsilon=1e-5,
        momentum=0.1,
        last_conv_bias=None,
        bn=nn.BatchNorm2d,
    ):
        super().__init__()
        self.bn = bn(num_features, momentum=momentum, epsilon=epsilon)
        self.pad_pixels = pad_pixels
        self.last_conv_bias = last_conv_bias

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            bias = -self.bn._mean
            if self.last_conv_bias is not None:
                bias += self.last_conv_bias
            pad_values = self.bn.bias + self.bn.weight * (
                bias / torch.sqrt(self.bn._variance + self.bn._epsilon)
            )
            """ pad """
            # TODO: n,h,w,c format is not supported yet
            n, c, h, w = output.shape
            values = pad_values.reshape([1, -1, 1, 1])
            w_values = values.expand([n, -1, self.pad_pixels, w])
            x = torch.cat([w_values, output, w_values], dim=2)
            h = h + self.pad_pixels * 2
            h_values = values.expand([n, -1, h, self.pad_pixels])
            x = torch.cat([h_values, x, h_values], dim=3)
            output = x
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def _mean(self):
        return self.bn._mean

    @property
    def _variance(self):
        return self.bn._variance

    @property
    def _epsilon(self):
        return self.bn._epsilon


def conv_bn(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
):
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias_attr=False,
        padding_mode=padding_mode,
    )
    bn_layer = nn.BatchNorm2D(num_features=out_channels)
    se = nn.Sequential()
    se.add_sublayer("conv", conv_layer)
    se.add_sublayer("bn", bn_layer)
    return se


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn._variance + bn._epsilon).sqrt()
    return (
        kernel * ((gamma / std).reshape([-1, 1, 1, 1])),
        bn.bias - bn._mean * gamma / std,
    )


def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.transpose([1, 0, 2, 3]))
        b_hat = (k2 * b1.reshape([1, -1, 1, 1])).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.transpose([1, 0, 2, 3])
        k1_group_width = k1.shape[0] // groups
        k2_group_width = k2.shape[0] // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width : (g + 1) * k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width : (g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append(
                (
                    k2_slice
                    * b1[g * k1_group_width : (g + 1) * k1_group_width].reshape(
                        [1, -1, 1, 1]
                    )
                ).sum((1, 2, 3))
            )
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)


def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = (
        1.0 / kernel_size**2
    )
    return k


def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.shape[2]) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.shape[3]) // 2
    return F.pad(
        kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
    )


class DiverseBranchBlock(nn.Module):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
        is_repped=False,
        single_init=False,
        **kwargs,
    ):
        super().__init__()

        padding = (filter_size - 1) // 2
        dilation = 1

        in_channels = num_channels
        out_channels = num_filters
        kernel_size = filter_size
        internal_channels_1x1_3x3 = None
        nonlinear = act

        self.is_repped = is_repped

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nn.ReLU()

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if is_repped:
            self.dbb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.dbb_origin = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_sublayer(
                    "conv",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=groups,
                        bias=False,
                    ),
                )
                self.dbb_avg.add_sublayer(
                    "bn", BNAndPad(pad_pixels=padding, num_features=out_channels)
                )
                self.dbb_avg.add_sublayer(
                    "avg",
                    nn.AvgPool2D(kernel_size=kernel_size, stride=stride, padding=0),
                )
                self.dbb_1x1 = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                )
            else:
                self.dbb_avg.add_sublayer(
                    "avg",
                    nn.AvgPool2D(
                        kernel_size=kernel_size, stride=stride, padding=padding
                    ),
                )

            self.dbb_avg.add_sublayer("avgbn", nn.BatchNorm2D(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = (
                    in_channels if groups < out_channels else 2 * in_channels
                )  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_sublayer(
                    "idconv1", IdentityBasedConv1x1(channels=in_channels, groups=groups)
                )
            else:
                self.dbb_1x1_kxk.add_sublayer(
                    "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=internal_channels_1x1_3x3,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=groups,
                        bias=False,
                    ),
                )
            self.dbb_1x1_kxk.add_sublayer(
                "bn1",
                BNAndPad(pad_pixels=padding, num_features=internal_channels_1x1_3x3),
            )
            self.dbb_1x1_kxk.add_sublayer(
                "conv2",
                nn.Conv2d(
                    in_channels=internal_channels_1x1_3x3,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias=False,
                ),
            )
            self.dbb_1x1_kxk.add_sublayer("bn2", nn.BatchNorm2D(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def forward(self, inputs):
        if self.is_repped:
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, "dbb_1x1"):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(
            self.dbb_origin.conv.weight, self.dbb_origin.bn
        )

        if hasattr(self, "dbb_1x1"):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, "idconv1"):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(
            k_1x1_kxk_first, self.dbb_1x1_kxk.bn1
        )
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(
            self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2
        )
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(
            k_1x1_kxk_first,
            b_1x1_kxk_first,
            k_1x1_kxk_second,
            b_1x1_kxk_second,
            groups=self.groups,
        )

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg, self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, "conv"):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(
                self.dbb_avg.conv.weight, self.dbb_avg.bn
            )
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(
                k_1x1_avg_first,
                b_1x1_avg_first,
                k_1x1_avg_second,
                b_1x1_avg_second,
                groups=self.groups,
            )
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch(
            (k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
            (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged),
        )

    def re_parameterize(self):
        if self.is_repped:
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(
            in_channels=self.dbb_origin.conv._in_channels,
            out_channels=self.dbb_origin.conv._out_channels,
            kernel_size=self.dbb_origin.conv._kernel_size,
            stride=self.dbb_origin.conv._stride,
            padding=self.dbb_origin.conv._padding,
            dilation=self.dbb_origin.conv._dilation,
            groups=self.dbb_origin.conv._groups,
            bias=True,
        )

        self.dbb_reparam.weight.set_value(kernel)
        self.dbb_reparam.bias.set_value(bias)

        self.__delattr__("dbb_origin")
        self.__delattr__("dbb_avg")
        if hasattr(self, "dbb_1x1"):
            self.__delattr__("dbb_1x1")
        self.__delattr__("dbb_1x1_kxk")
        self.is_repped = True


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.res_dict = {}
        # self.res_name = self.full_name()
        self.res_name = self.__class__.__name__.lower()
        self.pruner = None
        self.quanter = None

        self.init_net(*args, **kwargs)

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"logits": output}
        # 'list' is needed to avoid error raised by popping self.res_dict
        for res_key in list(self.res_dict):
            # clear the res_dict because the forward process may change according to input
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def init_net(
        self,
        stages_pattern=None,
        return_patterns=None,
        return_stages=None,
        freeze_befor=None,
        stop_after=None,
        *args,
        **kwargs,
    ):
        # init the output of net
        if return_patterns or return_stages:
            if return_patterns and return_stages:
                msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."

                return_stages = None

            if return_stages is True:
                return_patterns = stages_pattern

            # return_stages is int or bool
            if type(return_stages) is int:
                return_stages = [return_stages]
            if isinstance(return_stages, list):
                if max(return_stages) > len(stages_pattern) or min(return_stages) < 0:
                    msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."

                    return_stages = [
                        val
                        for val in return_stages
                        if val >= 0 and val < len(stages_pattern)
                    ]
                return_patterns = [stages_pattern[i] for i in return_stages]

            if return_patterns:
                # call update_res function after the __init__ of the object has completed execution, that is, the constructing of layer or model has been completed.
                def update_res_hook(layer, input):
                    self.update_res(return_patterns)

                self.register_forward_pre_hook(update_res_hook)

        # freeze subnet
        if freeze_befor is not None:
            self.freeze_befor(freeze_befor)

        # set subnet to Identity
        if stop_after is not None:
            self.stop_after(stop_after)

    def init_res(self, stages_pattern, return_patterns=None, return_stages=None):

        if return_patterns and return_stages:
            return_stages = None

        if return_stages is True:
            return_patterns = stages_pattern
        # return_stages is int or bool
        if type(return_stages) is int:
            return_stages = [return_stages]
        if isinstance(return_stages, list):
            if max(return_stages) > len(stages_pattern) or min(return_stages) < 0:
                return_stages = [
                    val
                    for val in return_stages
                    if val >= 0 and val < len(stages_pattern)
                ]
            return_patterns = [stages_pattern[i] for i in return_stages]

        if return_patterns:
            self.update_res(return_patterns)

    def replace_sub(self, *args, **kwargs) -> None:
        msg = "The function 'replace_sub()' is deprecated, please use 'upgrade_sublayer()' instead."
        raise DeprecationWarning(msg)

    def upgrade_sublayer(
        self,
        layer_name_pattern: Union[str, List[str]],
        handle_func: Callable[[nn.Module, str], nn.Module],
    ) -> Dict[str, nn.Module]:
        """use 'handle_func' to modify the sub-layer(s) specified by 'layer_name_pattern'.

        Args:
            layer_name_pattern (Union[str, List[str]]): The name of layer to be modified by 'handle_func'.
            handle_func (Callable[[nn.Module, str], nn.Module]): The function to modify target layer specified by 'layer_name_pattern'. The formal params are the layer(nn.Module) and pattern(str) that is (a member of) layer_name_pattern (when layer_name_pattern is List type). And the return is the layer processed.

        Returns:
            Dict[str, nn.Module]: The key is the pattern and corresponding value is the result returned by 'handle_func()'.

        Examples:

            from paddle import nn
            import paddleclas

            def rep_func(layer: nn.Module, pattern: str):
                new_layer = nn.Conv2d(
                    in_channels=layer._in_channels,
                    out_channels=layer._out_channels,
                    kernel_size=5,
                    padding=2
                )
                return new_layer

            net = paddleclas.MobileNetV1()
            res = net.upgrade_sublayer(layer_name_pattern=["blocks[11].depthwise_conv.conv", "blocks[12].depthwise_conv.conv"], handle_func=rep_func)
            print(res)
            # {'blocks[11].depthwise_conv.conv': the corresponding new_layer, 'blocks[12].depthwise_conv.conv': the corresponding new_layer}
        """

        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]

        hit_layer_pattern_list = []
        for pattern in layer_name_pattern:
            # parse pattern to find target layer and its parent
            layer_list = parse_pattern_str(pattern=pattern, parent_layer=self)
            if not layer_list:
                continue

            sub_layer_parent = layer_list[-2]["layer"] if len(layer_list) > 1 else self
            sub_layer = layer_list[-1]["layer"]
            sub_layer_name = layer_list[-1]["name"]
            sub_layer_index_list = layer_list[-1]["index_list"]

            new_sub_layer = handle_func(sub_layer, pattern)

            if sub_layer_index_list:
                if len(sub_layer_index_list) > 1:
                    sub_layer_parent = getattr(sub_layer_parent, sub_layer_name)[
                        sub_layer_index_list[0]
                    ]
                    for sub_layer_index in sub_layer_index_list[1:-1]:
                        sub_layer_parent = sub_layer_parent[sub_layer_index]
                    sub_layer_parent[sub_layer_index_list[-1]] = new_sub_layer
                else:
                    getattr(sub_layer_parent, sub_layer_name)[
                        sub_layer_index_list[0]
                    ] = new_sub_layer
            else:
                setattr(sub_layer_parent, sub_layer_name, new_sub_layer)

            hit_layer_pattern_list.append(pattern)
        return hit_layer_pattern_list

    def stop_after(self, stop_layer_name: str) -> bool:
        """stop forward and backward after 'stop_layer_name'.

        Args:
            stop_layer_name (str): The name of layer that stop forward and backward after this layer.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        layer_list = parse_pattern_str(stop_layer_name, self)
        if not layer_list:
            return False

        parent_layer = self
        for layer_dict in layer_list:
            name, index_list = layer_dict["name"], layer_dict["index_list"]
            if not set_identity(parent_layer, name, index_list):
                msg = f"Failed to set the layers that after stop_layer_name('{stop_layer_name}') to IdentityLayer. The error layer's name is '{name}'."
                return False
            parent_layer = layer_dict["layer"]

        return True

    def freeze_befor(self, layer_name: str) -> bool:
        """freeze the layer named layer_name and its previous layer.

        Args:
            layer_name (str): The name of layer that would be freezed.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        def stop_grad(layer, pattern):
            class StopGradLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer = layer

                def forward(self, x):
                    x = self.layer(x)
                    x.stop_gradient = True
                    return x

            new_layer = StopGradLayer()
            return new_layer

        res = self.upgrade_sublayer(layer_name, stop_grad)
        if len(res) == 0:
            msg = "Failed to stop the gradient before the layer named '{layer_name}'"
            return False
        return True

    def update_res(self, return_patterns: Union[str, List[str]]) -> Dict[str, nn.Module]:
        """update the result(s) to be returned.

        Args:
            return_patterns (Union[str, List[str]]): The name of layer to return output.

        Returns:
            Dict[str, nn.Module]: The pattern(str) and corresponding layer(nn.Module) that have been set successfully.
        """

        # clear res_dict that could have been set
        self.res_dict = {}

        class Handler(object):
            def __init__(self, res_dict):
                # res_dict is a reference
                self.res_dict = res_dict

            def __call__(self, layer, pattern):
                layer.res_dict = self.res_dict
                layer.res_name = pattern
                if hasattr(layer, "hook_remove_helper"):
                    layer.hook_remove_helper.remove()
                layer.hook_remove_helper = layer.register_forward_post_hook(
                    save_sub_res_hook
                )
                return layer

        handle_func = Handler(self.res_dict)

        hit_layer_pattern_list = self.upgrade_sublayer(
            return_patterns, handle_func=handle_func
        )

        if hasattr(self, "hook_remove_helper"):
            self.hook_remove_helper.remove()
        self.hook_remove_helper = self.register_forward_post_hook(
            self._return_dict_hook
        )

        return hit_layer_pattern_list


def save_sub_res_hook(layer, input, output):
    layer.res_dict[layer.res_name] = output


def set_identity(
    parent_layer: nn.Module, layer_name: str, layer_index_list: str = None
) -> bool:
    """set the layer specified by layer_name and layer_index_list to Identity.

    Args:
        parent_layer (nn.Module): The parent layer of target layer specified by layer_name and layer_index_list.
        layer_name (str): The name of target layer to be set to Identity.
        layer_index_list (str, optional): The index of target layer to be set to Identity in parent_layer. Defaults to None.

    Returns:
        bool: True if successfully, False otherwise.
    """

    stop_after = False
    for sub_layer_name in parent_layer._sub_layers:
        if stop_after:
            parent_layer._sub_layers[sub_layer_name] = Identity()
            continue
        if sub_layer_name == layer_name:
            stop_after = True

    if layer_index_list and stop_after:
        layer_container = parent_layer._sub_layers[layer_name]
        for num, layer_index in enumerate(layer_index_list):
            stop_after = False
            for i in range(num):
                layer_container = layer_container[layer_index_list[i]]
            for sub_layer_index in layer_container._sub_layers:
                if stop_after:
                    parent_layer._sub_layers[layer_name][sub_layer_index] = Identity()
                    continue
                if layer_index == sub_layer_index:
                    stop_after = True

    return stop_after


def parse_pattern_str(
    pattern: str, parent_layer: nn.Module
) -> Union[None, List[Dict[str, Union[nn.Module, str, None]]]]:
    """parse the string type pattern.

    Args:
        pattern (str): The pattern to describe layer.
        parent_layer (nn.Module): The root layer relative to the pattern.

    Returns:
        Union[None, List[Dict[str, Union[nn.Module, str, None]]]]: None if failed. If successfully, the members are layers parsed in order:
                                                                [
                                                                    {"layer": first layer, "name": first layer's name parsed, "index": first layer's index parsed if exist},
                                                                    {"layer": second layer, "name": second layer's name parsed, "index": second layer's index parsed if exist},
                                                                    ...
                                                                ]
    """

    pattern_list = pattern.split(".")
    if not pattern_list:
        msg = f"The pattern('{pattern}') is illegal. Please check and retry."
        return None

    layer_list = []
    while len(pattern_list) > 0:
        if "[" in pattern_list[0]:
            target_layer_name = pattern_list[0].split("[")[0]
            target_layer_index_list = list(
                index.split("]")[0] for index in pattern_list[0].split("[")[1:]
            )
        else:
            target_layer_name = pattern_list[0]
            target_layer_index_list = None

        target_layer = getattr(parent_layer, target_layer_name, None)

        if target_layer is None:
            msg = f"Not found layer named('{target_layer_name}') specified in pattern('{pattern}')."
            return None

        if target_layer_index_list:
            for target_layer_index in target_layer_index_list:
                if int(target_layer_index) < 0 or int(target_layer_index) >= len(
                    target_layer
                ):
                    msg = f"Not found layer by index('{target_layer_index}') specified in pattern('{pattern}'). The index should < {len(target_layer)} and > 0."
                    return None
                target_layer = target_layer[target_layer_index]

        layer_list.append(
            {
                "layer": target_layer,
                "name": target_layer_name,
                "index_list": target_layer_index_list,
            }
        )

        pattern_list = pattern_list[1:]
        parent_layer = target_layer

    return layer_list


# class AdaptiveAvgPool2D(nn.AdaptiveAvgPool2D):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         if paddle.device.get_device().startswith("npu"):
#             self.device = "npu"
#         else:
#             self.device = None
#
#         if isinstance(self._output_size, int) and self._output_size == 1:
#             self._gap = True
#         elif (
#             isinstance(self._output_size, tuple)
#             and self._output_size[0] == 1
#             and self._output_size[1] == 1
#         ):
#             self._gap = True
#         else:
#             self._gap = False
#
#     def forward(self, x):
#         if self.device == "npu" and self._gap:
#             # Global Average Pooling
#             N, C, _, _ = x.shape
#             x_mean = torch.mean(x, dim=[2, 3])
#             x_mean = torch.reshape(x_mean, [N, C, 1, 1])
#             return x_mean
#         else:
#             return F.adaptive_avg_pool2d(
#                 x,
#                 output_size=self._output_size,
#                 data_format=self._data_format,
#                 name=self._name,
#             )


# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F
# from paddle.nn.initializer import KaimingNormal, Constant
# from paddle.nn import Conv2D, BatchNorm2D, ReLU, AdaptiveAvgPool2D, MaxPool2D
# from paddle.regularizer import L2Decay
# from paddle import ParamAttr

MODEL_URLS = {
    "PPHGNetV2_B0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B0_ssld_pretrained.pdparams",
    "PPHGNetV2_B1": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B1_ssld_pretrained.pdparams",
    "PPHGNetV2_B2": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B2_ssld_pretrained.pdparams",
    "PPHGNetV2_B3": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B3_ssld_pretrained.pdparams",
    "PPHGNetV2_B4": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B4_ssld_pretrained.pdparams",
    "PPHGNetV2_B5": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B5_ssld_pretrained.pdparams",
    "PPHGNetV2_B6": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B6_ssld_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

# kaiming_normal_ = KaimingNormal()
# zeros_ = Constant(value=0.0)
# ones_ = Constant(value=1.0)


class LearnableAffineBlock(TheseusLayer):
    """
    Create a learnable affine block module. This module can significantly improve accuracy on smaller models.

    Args:
        scale_value (float): The initial value of the scale parameter, default is 1.0.
        bias_value (float): The initial value of the bias parameter, default is 0.0.
        lr_mult (float): The learning rate multiplier, default is 1.0.
        lab_lr (float): The learning rate, default is 0.01.
    """

    def __init__(self, scale_value=1.0, bias_value=0.0, lr_mult=1.0, lab_lr=0.01):
        super().__init__()
        # self.scale = self.create_parameter(
        #     shape=[
        #         1,
        #     ],
        #     default_initializer=nn.init.Constant(value=scale_value),
        #     # attr=ParamAttr(learning_rate=lr_mult * lab_lr),
        # )
        # self.add_parameter("scale", self.scale)
        self.scale = torch.Parameter(
            nn.init.constant_(
                torch.ones(1).to(torch.float32), val=scale_value
            )
        )
        self.register_parameter("scale", self.scale)

        # self.bias = self.create_parameter(
        #     shape=[
        #         1,
        #     ],
        #     default_initializer=nn.init.Constant(value=bias_value),
        #     # attr=ParamAttr(learning_rate=lr_mult * lab_lr),
        # )
        # self.add_parameter("bias", self.bias)
        self.bias = torch.Parameter(
            nn.init.constant_(
                torch.ones(1).to(torch.float32), val=bias_value
            )
        )
        self.register_parameter("bias", self.bias)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(TheseusLayer):
    """
    ConvBNAct is a combination of convolution and batchnorm layers.
ConvBNAct: 73,728
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int/str): Padding or padding type for the convolution. Defaults to 1.
        groups (int): Number of groups for the convolution. Defaults to 1.
        use_act: (bool): Whether to use activation function. Defaults to True.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        use_act=True,
        use_lab=False,
        lr_mult=1.0,
    ):
        super().__init__()
        self.use_act = use_act # Whether to use activation function.
        self.use_lab = use_lab # Whether to use the LAB operation.
        # One big convolution that transforms directly from in_channels to out_channels
        # If in=64 and out=128, the conv has 64 × 128 × 3 × 3 = 73,728 parameters (for a 3×3 kernel)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding if isinstance(padding, str) else (kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )# it go up from 3 to 32
        self.bn = nn.BatchNorm2d(
            out_channels,
        )
        if self.use_act:
            self.act = nn.ReLU()
            if self.use_lab:
                self.lab = LearnableAffineBlock(lr_mult=lr_mult)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class LightConvBNAct(TheseusLayer):
    """
    LightConvBNAct is a combination of pw and dw layers.
8,192 + 1,152 = 9,344
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the depth-wise convolution kernel.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.
        # 8,192 + 1,152 = 9,344
    🎯 When to Use Each
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_lab=False,
        lr_mult=1.0,
        **kwargs,
    ):

        super().__init__()
        # pointwise 1 by 1: Pointwise (1×1) = Mix channels without spatial complexity
            # Takes in_channels=64 → out_channels=128
            # Uses 1×1 kernels: 64 × 128 × 1 × 1 = 8,192 parameters ✅ Much smaller!
            # No spatial complexity (no moving window, just channel mixing)
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        # Depthwise Convolution : Depthwise (k×k) = Process spatial patterns per channel independently
            # Each of the 128 channels processes itself independently
            # 128 groups × 1 × kernel_size × kernel_size parameters
            # For a 3×3 kernel: 128 × 1 × 3 × 3 = 1,152 parameters ✅ Tiny!
            # groups=out_channels means each channel gets its own 3×3 filter
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        # Instead of mixing spatial and channels at once (expensive), you separate them (cheap).
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
"""
When to Use ConvBNAct vs LightConvBNAct - Detailed Guide
Use Case                        | Pick                  | Why
--------------------------------|----------------------|---------------------------------------------
Early/initial layers            | Standard ConvBNAct    | Need rich spatial-channel mixing for raw input
Mid-range layers                | Either (flexible)     | Depends on accuracy vs speed trade-off
Deep/final layers               | LightConvBNAct        | Speed matters, features are already learned
Real-time inference             | LightConvBNAct        | Lower latency crucial
Mobile/edge devices             | LightConvBNAct        | Memory and compute severely restricted
High-accuracy tasks             | ConvBNAct             | Slight expressiveness advantage
Text recognition (like OCR)     | LightConvBNAct        | Fast inference, lower memory footprint
Detection tasks                 | Mixed (both)          | Early stages: standard, later: light
Video processing                | LightConvBNAct        | Must process frames quickly
Training speed priority         | LightConvBNAct        | Fewer parameters = faster training
Model size priority             | LightConvBNAct        | 8× fewer parameters
Accuracy priority               | ConvBNAct             | More expressive mixing
"""
class PaddingSameAsPaddleMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        """
        what does this Class Does?
            Applies MaxPool2d while preserving input dimensions (like padding="same" in Keras). The input shape stays the same after the operation.
            Why? Without this, MaxPool shrinks the feature map. This class prevents that shrinkage by padding with zeros first
            PyTorch's built-in MaxPool2d doesn't have a "same" mode. This custom class reimplements it by:
                1. Manually calculating the exact padding needed
                2. Pre-padding with torch.nn.functional.pad()
                3. Running MaxPool2d with padding=0 (no double-padding)
        In Your StemBlock Context
        When you call self.pool = PaddingSameAsPaddleMaxPool2d(kernel_size=2, stride=1), it:
            Takes input of shape e.g., (batch, 32, H, W)
            Pads it (possibly asymmetrically) to ensure output = input size
        Outputs (batch, 32, H, W) — same spatial dimensions
            This is why the comment says "Size preserved" — the padding class ensures MaxPool doesn't shrink the feature map, which is critical when you're concatenating it with other paths that also preserve size.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = torch.nn.MaxPool2d(kernel_size, stride, padding=0, ceil_mode=True)
        # This stores the kernel size and stride,
        # then creates a MaxPool2d with padding=0.
        # The key is we'll manually add padding before calling this pool.

    def forward(self, x):
        _, _, h, w = x.shape
        # Extract height and width from the input tensor. The first two dims are batch and channels (ignored).
        # STEP 1: How many output positions will MaxPool create?
        out_h = math.ceil(h / self.stride)
        # STEP 2: Where is the last output position?
        last_pos_h = (out_h - 1) * self.stride
        # STEP 3: Extend by kernel size (to see where the kernel touches the edge)
        extended_h = last_pos_h + self.kernel_size
        # STEP 4: How much padding is needed?
        pad_needed_h = extended_h - h
        # STEP 5: Make sure it's not negative
        pad_h_total = max(0, pad_needed_h)
        # STEP 1: How many output positions will MaxPool create?
        out_w = math.ceil(w / self.stride)

        # STEP 2: Where is the last output position?
        last_pos_w = (out_w - 1) * self.stride

        # STEP 3: Extend by kernel size (where does the kernel reach?)
        extended_w = last_pos_w + self.kernel_size

        # STEP 4: How much padding is needed?
        pad_needed_w = extended_w - w

        # STEP 5: Make sure it's not negative
        pad_w_total = max(0, pad_needed_w)
        pad_h = pad_h_total // 2
        pad_w = pad_w_total // 2
        # The padding format is [left, right, top, bottom] in PyTorch.
        x = torch.nn.functional.pad(x, [pad_w, pad_w_total - pad_w, pad_h, pad_h_total - pad_h])
        return self.pool(x)

class StemBlock(TheseusLayer):
    """
    StemBlock for PP-HGNetV2.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of middle channels.
        out_channels (int): Number of output channels.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        use_lab=False,
        lr_mult=1.0,
        text_rec=False,
    ):
        """
        Understanding Each Layer in Your StemBlock
            Kernel Size determines the receptive field — how much context each output value "sees". A 3×3 kernel looks at 9 input values, a 2×2 kernel looks at 4.
            Stride controls how much the kernel moves each step:
                Stride = 1: kernel moves 1 pixel, output is nearly the same size as input (minus edges)
                Stride = 2: kernel moves 2 pixels, output is roughly half the input size
            Padding adds zero borders around the input:
                Padding = 1 (or calculated): edges are padded, so output can be larger
                Padding = "same": PyTorch automatically calculates padding to preserve spatial dimensions
        you have to understand the big picture
        The Big Picture
            Your StemBlock is a multi-path feature extractor:
            * One path uses small kernels (2×2) with stride=1 — captures fine detail at full resolution
            * Another path uses MaxPool — captures dominant features
            * Both paths concatenate and merge through stem3/stem4
        """
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        # This is aggressive downsampling. Takes a large input image and cuts it to half size while computing features
        # Comments say "go from 3 to 32" — meaning 3 color channels → 32 feature maps
        # go from 3 to 32
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding="same",
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        # Small kernel, no downsampling. It extracts details at the current resolution
        # Reduces channels: 32 → 16 (the comment "from 32 to 16" is correct here)
        # got from 32 to 16
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding="same",
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        # Same size kernel as stem2a but expands channels back: 16 → 32
        # This creates a bottleneck pathway: you compress features then expand them
        # got from 16 to 32
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1 if text_rec else 2,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        # Combines the two paths (stem2a→2b and pool were concatenated on channels)
        # The comment "combine both together" is the key — two different feature extraction paths are merged
        # combine
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        # "1 by 1 kernel" — this looks at single pixels and mixes channels
        # Often used to adjust channels or add learnable nonlinearity without changing spatial dimensions
        # 1 by 1 kernel
        self.pool = PaddingSameAsPaddleMaxPool2d(
            kernel_size=2, stride=1,
        )
        # Takes the maximum value in each 2×2 window (a different operation than convolution)
        # Preserves important features, roughly the same output size

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], 1)
        # combine both togather
        x = self.stem3(x)
        # Combines the two paths (stem2a→2b and pool were concatenated on channels)
        # The comment "combine both together" is the key — two different feature extraction paths are merged
        x = self.stem4(x)

        return x


class HGV2_Block(TheseusLayer):
    """
    HGV2_Block, the basic unit that constitutes the HGV2_Stage.
    Dense feature aggregation: input is fed through `layer_num` conv layers,
    all intermediate outputs are concatenated with the input, then squeezed
    back to out_channels via two 1×1 convs (squeeze + excitation).
    When identity=True a residual shortcut is added (same as ResNet skip).
    For text_rec B4: stages 1-2 use ConvBNAct (light_block=False),
    stages 3-4 use LightConvBNAct (pw+dw, light_block=True).

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of middle channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        layer_num (int): Number of layers in the HGV2 block. Defaults to 6.
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int/str): Padding or padding type for the convolution. Defaults to 1.
        groups (int): Number of groups for the convolution. Defaults to 1.
        use_act (bool): Whether to use activation function. Defaults to True.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.
    """
    """
    Component: Why?????
    Dense aggregation :Early layers capture fine details; later layers add context. Concatenating all means the output benefits from all levels of abstraction
    Intermediate features: Without saving them, you lose information. With 6 layers, you lose a lot. Dense connections preserve it.
    Squeeze-Excitation: The concatenated tensor is huge. Squeeze-Excitation learns which concatenated features are useful, mixing and filtering them
    1×1 convolutions: Cheap, spatial-invariant, perfect for channel mixing. No kernel weight overhead like 3×3
    Bottleneck (2-step compression): Saves computation. A 256→64 step is expensive; 256→32→64 is a third the cost. Also forces a learned compression (information bottleneck).
    Optional residual: If identity=True, gradients can skip the entire block (easier backprop). Useful in very deep networks.
    light_block parameter: For stages 1-2, use standard convs (heavy but powerful). For stages 3-4, use depthwise-separable (light but still good). Balances cost and expressiveness.
    """
    def __init__(
        self,
        in_channels,# in_channels: Size of input (e.g., 16 channels)
        mid_channels,# mid_channels: Size of intermediate features (e.g., 40 channels) — all internal layers use this
        out_channels,# out_channels: Final output size (e.g., 64 channels) — after squeeze-excitation
        kernel_size=3,# kernel_size: Conv kernel size (default 3×3)
        layer_num=6, # layer_num: How many conv layers to stack (default 6) — more layers = more feature aggregation
        identity=False,# identity: Whether to use residual shortcut (like ResNet)
        light_block=True,# light_block: Whether to use depthwise-separable convs (lighter) or standard convs (heavier)
        use_lab=False,# use_lab: Special parameter for some training techniques
        lr_mult=1.0,# lr_mult: Learning rate multiplier for this block's weights
    ):
        #  This block is a "super-layer" that densely reuses all intermediate features, intelligently squeezes them down, and outputs refined features.
        #  It's a powerful building block for text recognition because it forces the network to preserve and reuse information at multiple scales.
        super().__init__()
        self.identity = identity

        self.layers = nn.ModuleList()
        block_type = "LightConvBNAct" if light_block else "ConvBNAct"
        # Create a list of 6 conv blocks (by default)
        for i in range(layer_num):
            # First layer takes in_channels (original input size) and outputs mid_channels
            self.layers.append(
                eval(block_type)(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    stride=1,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                    lr_mult=lr_mult,
                )
            )
            # All remaining layers take mid_channels as input and output mid_channels
            # This creates a consistent pipeline: all internal features are the same size
            # All 6 blocks do the same work — process features of size mid_channels
            # Why this matters:
            # LEFT (Standard): Each layer overwrites the previous. By Layer 6, you've only got Layer 6's
            # refined features. Layer 1's original details are GONE. The network can't use them anymore.
            # RIGHT (Dense): All 6 layer outputs PLUS the original input are saved. When you concatenate
            # them, the final layer receives BOTH early simple features AND late complex features at the
            # same time. It can combine them however it wants via the squeeze-excitation bottlenec
        # Why?
            # Standardizes internal processing
            # Each layer can focus on refining the same-sized feature space
            # Information flows smoothly through all layers
        # feature aggregation
        # All intermediate outputs get concatenated (we'll see this in forward)
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        # Why the name "squeeze"?
            # It squeezes (compresses) the fat concatenated tensor.
        # Why 1×1 convolution?
            # 1×1 convs are cheap (no spatial weight sharing, just channel mixing)
            # They learn which concatenated features are important
            # They combine information across channels
        self.aggregation_excitation_conv = ConvBNAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        # What's happening:
            # Takes the squeezed features (out_channels // 2) and expands back to out_channels
            # This is the final refinement step
            # Why two steps instead of one?
                # Computational efficiency: (256→32→64) is cheaper than (256→64) directly
                    # Squeeze cuts down channels, reducing computation
                    # Excitation is light work to expand back
                # Bottleneck design: Forces the network to learn a compressed representation
                    # Like an information bottleneck — features must pass through a narrow gate
                    # Only the most important features survive the squeeze
                    # This is inspired by SE-Net (Squeeze-and-Excitation networks)

    def forward(self, x):
        identity = x #  # Step 1: Save original input
        output = [] #    # Step 2: Collect all outputs
        output.append(x) #  # Step 3: Add original input to collection
        for layer in self.layers:# # Step 4: Process through 6 layers
            x = layer(x) # Apply conv
            output.append(x) # Save this layer's output
        x = torch.cat(output, dim=1) # # Step 5: Concatenate all along channel dim
        x = self.aggregation_squeeze_conv(x) #  # Step 6: Squeeze
        x = self.aggregation_excitation_conv(x) #  Step 7: Excitation
        if self.identity: #  Step 8: Optional residual
            x = x + identity        # was: x += identity
        return x

# Simple pipeline:
    # Optionally shrink the image resolution
    # Pass through all the blocks sequentially
class HGV2_Stage(TheseusLayer):
    """
    HGV2_Stage, the basic unit that constitutes the PPHGNetV2.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of middle channels.
        out_channels (int): Number of output channels.
        block_num (int): Number of blocks in the HGV2 stage.
        layer_num (int): Number of layers in the HGV2 block. Defaults to 6.
        is_downsample (bool): Whether to use downsampling operation. Defaults to False.
        light_block (bool): Whether to use light block. Defaults to True.
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        use_lab (bool, optional): Whether to use the LAB operation. Defaults to False.
        lr_mult (float, optional): Learning rate multiplier for the layer. Defaults to 1.0.
    """
    # This is a building block for a neural network called PPHGNetV2 (a pose/hand estimation model).
    # Think of it as a "stage" that processes image features through multiple processing steps.
    def __init__(
        self,
        in_channels,# in_channels: Size of input (e.g., 16 channels)
        mid_channels,# mid_channels: Size of intermediate features (e.g., 40 channels) — all internal layers use this
        out_channels,# How many channels come OUT (e.g., 64)
        block_num,# How many "blocks" to stack (more = more processing)
        layer_num=6,# How many conv layers per block (default 6 — more = deeper feature extraction)
        is_downsample=True,# Whether to shrink the image resolution (e.g., 256×256 → 128×128)
        light_block=True, # Use a lighter/faster version of the block (for efficiency)
        kernel_size=3,# kernel_size: Conv kernel size (default 3×3)
        use_lab=False, #
        stride=2, # If downsampling, how much to shrink (stride=2 means half the size)
        lr_mult=1.0,
    ):

        super().__init__()
        self.is_downsample = is_downsample
        if self.is_downsample:
            # If downsampling is enabled, it creates a depthwise convolution (groups=in_channels) that shrinks the spatial dimensions
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab,
                lr_mult=lr_mult,
            )

        blocks_list = []
        # The first block has identity=False (no skip connection), the rest have identity=True (they reuse features)
        for i in range(block_num):
            # Stacks multiple HGV2_Block units
            # The first block takes in_channels as input
            # All other blocks take out_channels (they pass data between themselves)
            blocks_list.append(
                HGV2_Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=False if i == 0 else True,
                    light_block=light_block,
                    use_lab=use_lab,
                    lr_mult=lr_mult,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.is_downsample:
            x = self.downsample(x)# # Shrink the image
        x = self.blocks(x) #  # Process through all blocks
        return x
'''
Input Image
    ↓
[Stem Block] (initial processing)
    ↓
[Stage 1] (process features)
    ↓
[Stage 2] (process features)
    ↓
[Stage 3] (process features)
    ↓
[Stage 4] (process features)
    ↓
[Output] (classification or detection)
'''

class PPHGNetV2(TheseusLayer):
    """
    PPHGNetV2

    Args:
        stage_config (dict): Config for PPHGNetV2 stages. such as the number of channels, stride, etc.
        stem_channels: (list): Number of channels of the stem of the PPHGNetV2.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        use_last_conv (bool): Whether to use the last conv layer as the output channel. Defaults to True.
        class_expand (int): Number of channels for the last 1x1 convolutional layer.
        drop_prob (float): Dropout probability for the last 1x1 convolutional layer. Defaults to 0.0.
        class_num (int): The number of classes for the classification layer. Defaults to 1000.
        lr_mult_list (list): Learning rate multiplier for the stages. Defaults to [1.0, 1.0, 1.0, 1.0, 1.0].
    Returns:
        model: nn.Module. Specific PPHGNetV2 model depends on args.
    """
    """
        Mode 2: Text Recognition (text_rec=True, det=False)
        Input Image
            ↓
        [Stem + 4 Stages]
            ↓
        [Special Pooling] → Convert to 40-step sequence
            ↓
        [Classification] → Predict which character at each step
        Outputs a sequence of class predictions (for OCR/text recognition)
        During training: adaptive pool to [1, 40] (1 height, 40 width = 40 steps)
        During inference: simpler [3, 2] pooling
    """
    def __init__(
        self,
        stage_config,# A dictionary that defines each stage (channels, stride, block count, etc.)
        stem_channels=[3, 32, 64], # Three numbers: [input_channels, mid_channels, output_channels] for the initial stem block
        use_lab=False,
        use_last_conv=True, # Whether to add a final 1×1 convolution layer before classification
        class_expand=2048,# Number of channels in that final conv layer (e.g., 2048)
        dropout_prob=0.0, # Dropout rate to prevent overfitting
        class_num=1000,# How many output classes (e.g., 1000 for classification, or character classes for text recognition)
        lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],#  Learning rate multiplier for the stages.
        det=False,# If True → detection mode; if False → text recognition mode
        text_rec=False,# If True → use special pooling for text recognition sequences
        out_indices=None,# Which stages to output from (for multi-scale features)
        **kwargs,
    ):
        super().__init__()
        self.det = det # set to false
        self.text_rec = text_rec # set to true
        self.use_lab = use_lab # set to false
        self.use_last_conv = use_last_conv # Whether to use the last conv layer as the output channel
        self.class_expand = class_expand ## # Number of channels for the last 1x1 convolutional layer
        self.class_num = class_num # number of classes that are going to be used
        self.out_indices = out_indices if out_indices is not None else [0, 1, 2, 3]
        self.out_channels = []

        # stem
        # What happens: Initial feature extraction from raw RGB image. Converts colors → learned features.
        self.stem = StemBlock(
            in_channels=stem_channels[0],#
            mid_channels=stem_channels[1],#
            out_channels=stem_channels[2],#
            use_lab=use_lab,
            lr_mult=lr_mult_list[0],
            text_rec=text_rec,
        )
        '''
        Input: [B, 3, 48, 320]
        
        StemBlock processes:
          - in_channels: 3 → mid_channels: 32 → out_channels: 64
          - Typically applies some downsampling
        
        Output: [B, 64, 24, 160]  (roughly H÷2, W÷2)
        '''
        # stages
        self.stages = nn.ModuleList()
        """
        STEM BLOCK
                Purpose: Initial feature extraction from raw RGB image. Converts colors → learned features.
                        in_channels = 3,  # RGB input
                        mid_channels= 32, # internal working size
                        out_channels= 64,
                        text_rec = True
                
        
                Input: [B, 3, 48, 320]
                        ↓
                StemBlock processes (downsamples ~H÷2, W÷2)
                        ↓
                Output: [B, 64, 24, 160]  (height and width halved)
        
        
        Stage 1
                Purpose: First pass through features. Reduce vertical dimension (height) quickly. Start expanding channels.
                        in_channels = 64,   # (from stem)
                        mid_channels= 48,   # (internal working size)
                        out_channels= 128,
                        block_num= 1,       # (just 1 block)
                        is_downsample= True,
                        light_block= False, # (full power block)
                        kernel_size= 3,
                        layer_num= 6,
                        stride= [2, 1],     # (height÷2, width stays same)
                
        
                Input: [B, 64, 24, 160]
                        ↓
                Downsample (stride [2,1]): shrink height by 2
                        ↓
                1 HGV2_Block: 64→128 channels
                  - 6 conv layers with mid_channels=48
                  - squeeze-excitation attention
                        ↓
                Output: [B, 128, 12, 160]  (height÷2 = 24→12, width same)
        
        
        Stage 2
                Purpose: Now compress the horizontal dimension. Massively expand channels to 512 (learning richer features).
                        in_channels = 128,
                        mid_channels= 96,
                        out_channels= 512,
                        block_num= 1,
                        is_downsample= True,
                        light_block= False,
                        kernel_size= 3,
                        layer_num= 6,
                        stride= [1, 2],     # (height stays same, width÷2)
                
        
                Input: [B, 128, 12, 160]
                        ↓
                Downsample (stride [1,2]): shrink width by 2
                        ↓
                1 HGV2_Block: 128→512 channels
                  - internal working with mid_channels=96
                        ↓
                Output: [B, 512, 12, 80]  (height same, width÷2 = 160→80)
        
        
        Stage 3
        Purpose:
        - Multiple blocks = deeper feature learning
        - Light blocks = more efficient
        - Bigger kernel (5×5) = sees larger context
        - This is where semantic understanding builds up        
                        in_channels = 512,
                        mid_channels= 192,
                        out_channels= 1024,
                        block_num= 3,       # 3 sequential blocks for deep processing
                        is_downsample= True,
                        light_block= True,  # Lighter/faster version
                        kernel_size= 5,     # Bigger kernel for wider context
                        layer_num= 6,
                        stride= [2, 1],     # (height÷2, width stays same)
                
                Input: [B, 512, 12, 80]
                        ↓
                Downsample (stride [2,1]): shrink height by 2
                        ↓
                Block 1: 512→1024 (heavy processing)
                  ├─ 6 conv layers with mid_channels=192
                  └─ squeeze-excitation
                        ↓
                Block 2: 1024→1024 (skip connection helps)
                  ├─ 6 conv layers with mid_channels=192
                  └─ squeeze-excitation
                        ↓
                Block 3: 1024→1024 (more feature refinement)
                  ├─ 6 conv layers with mid_channels=192
                  └─ squeeze-excitation
                        ↓
                Output: [B, 1024, 6, 80]  (height÷2 = 12→6, width same)
        Stage 4
                Purpose: Final high-level feature extraction. Output channels = 2048 (very rich features).
                        in_channels = 1024,
                        mid_channels= 384,
                        out_channels= 2048,
                        block_num= 1,
                        is_downsample= True,
                        light_block= True,
                        kernel_size= 5,
                        layer_num= 6,
                        stride= [2, 1],     # (height÷2, width stays same)
                
        
                Input: [B, 1024, 6, 80]
                        ↓
                Downsample (stride [2,1]): shrink height by 2
                        ↓
                Block 1: 1024→2048 (final feature expansion)
                  - 6 conv layers with mid_channels=384
                  - squeeze-excitation
                        ↓
                Output: [B, 2048, 3, 80]  (height÷2 = 6→3, width same)
        
        
        TEXT RECOGNITION POOLING
                Purpose: Convert [B, 2048, 3, 80] feature map into a 40-position sequence for CTC loss (character recognition)
                
        
                Input: [B, 2048, 3, 80]
                        ↓
                if self.text_rec:
                    if self.training:
                        x = F.adaptive_avg_pool2d(x, [1, 40])
                        # Adaptive pooling to exact size [1, 40]
                        # Height: 3 → 1
                        # Width: 80 → 40
                    else:
                        x = F.avg_pool2d(x, [3, 2])
                        # Average pooling with kernel [3, 2]
                        # Height: 3 ÷ 3 = 1
                        # Width: 80 ÷ 2 = 40
                        ↓
                Output: [B, 2048, 1, 40]
                
                This creates 40 positions, each with 2048 rich feature channels
                Perfect for CTC: predicts one character at each of the 40 positions!
        COMPLETE SIZE PROGRESSION
                
        
                Input:         [B, 3,    48,   320]   ← Raw image
                After Stem:    [B, 64,   24,   160]   ← Initial features
                After Stage 1: [B, 128,  12,   160]   ← More features, shorter height
                After Stage 2: [B, 512,  12,   80]    ← Rich features, narrower width
                After Stage 3: [B, 1024, 6,    80]    ← Deep understanding, half height again
                After Stage 4: [B, 2048, 3,    80]    ← Maximum channel richness
                After Pool:    [B, 2048, 1,    40]    ← 40-character CTC sequence!
        KEY INSIGHTS
        
                ✓ Height compression (48 → 3): Aggressive reduction because text doesn't need much vertical info
                
                ✓ Width preservation (320 → 40): Moderate reduction to maintain character positions
                  (320 input width → 40 output positions = ~8 pixels per character on average)
                
                ✓ Channel expansion (3 → 2048): Massive expansion for rich feature learning
                
                ✓ Stage 3 has 3 blocks: Deepest processing here for semantic understanding
                
                ✓ Light blocks in Stage 3 & 4: Balance between feature richness and computational efficiency
                
                ✓ Kernel size 5×5 in Stage 3 & 4: Larger receptive field for context understanding
                
                ✓ Final output [1, 40]: Perfect for CTC loss - predict 1 of N characters at each of 40 positions!
        """
        for i, k in enumerate(stage_config):
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                is_downsample,
                light_block,
                kernel_size,
                layer_num,
                stride,
            ) = stage_config[k]
            self.stages.append(
                HGV2_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    is_downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    stride,
                    lr_mult=lr_mult_list[i + 1],
                )
            )
            if i in self.out_indices:
                self.out_channels.append(out_channels)
        if not self.det:
            self.out_channels = stage_config["stage4"][2]

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if self.use_last_conv:
            self.last_conv = nn.Conv2d(
                in_channels=out_channels,
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.act = nn.ReLU()
            if self.use_lab:
                self.lab = LearnableAffineBlock()
            # self.dropout = nn.Dropout(p=dropout_prob, mode="downscale_in_infer")
            self.dropout = nn.Dropout(p=dropout_prob)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        if not self.det:
            self.fc = nn.Linear(
                self.class_expand if self.use_last_conv else out_channels,
                self.class_num,
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        out = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.det and i in self.out_indices:
                out.append(x)
        if self.det:
            return out

        # After stage4 x is [B, 2048, 3, 80].
        # Pool height=3→1, width 80→40 to get the 40-step CTC sequence.
        if self.text_rec:
            if self.training:
                x = F.adaptive_avg_pool2d(x, [1, 40])
            else:
                x = F.avg_pool2d(x, [3, 2])
        return x


def PPHGNetV2_B0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B0
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Module. Specific `PPHGNetV2_B0` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 256, 1, True, False, 3, 3],
        "stage3": [256, 64, 512, 2, True, True, 5, 3],
        "stage4": [512, 128, 1024, 1, True, True, 5, 3],
    }

    model = PPHGNetV2(
        stem_channels=[3, 16, 16], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B1(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B1
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Module. Specific `PPHGNetV2_B1` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, False, 3, 3],
        "stage2": [64, 48, 256, 1, True, False, 3, 3],
        "stage3": [256, 96, 512, 2, True, True, 5, 3],
        "stage4": [512, 192, 1024, 1, True, True, 5, 3],
    }

    model = PPHGNetV2(
        stem_channels=[3, 24, 32], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B2(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B2
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Module. Specific `PPHGNetV2_B2` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 96, 1, False, False, 3, 4],
        "stage2": [96, 64, 384, 1, True, False, 3, 4],
        "stage3": [384, 128, 768, 3, True, True, 5, 4],
        "stage4": [768, 256, 1536, 1, True, True, 5, 4],
    }

    model = PPHGNetV2(
        stem_channels=[3, 24, 32], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B3(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B3
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Module. Specific `PPHGNetV2_B3` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 128, 1, False, False, 3, 5],
        "stage2": [128, 64, 512, 1, True, False, 3, 5],
        "stage3": [512, 128, 1024, 3, True, True, 5, 5],
        "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
    }

    model = PPHGNetV2(
        stem_channels=[3, 24, 32], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B4(pretrained=False, use_ssld=False, det=False, text_rec=False, **kwargs):
    """
    PPHGNetV2_B4
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Module. Specific `PPHGNetV2_B4` model depends on args.
    """
    # For text recognition (text_rec=True) the stride pattern preserves the width
    # dimension longer so the model can read character sequences left-to-right.
    # Stride format: [height_stride, width_stride] per stage.
    # After StemBlock (text_rec → stem3 stride=1): [B, 48, 24, 160]
    # Stage1 [2,1]: [B,128,12,160]  Stage2 [1,2]: [B,512,12,80]
    # Stage3 [2,1]: [B,1024,6,80]   Stage4 [2,1]: [B,2048,3,80]
    # Final avg_pool([3,2]) → [B,2048,1,40]  (40 = the sequence length for CTC)
    stage_config_rec = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num, stride
        "stage1": [48, 48, 128, 1, True, False, 3, 6, [2, 1]],
        "stage2": [128, 96, 512, 1, True, False, 3, 6, [1, 2]],
        "stage3": [512, 192, 1024, 3, True, True, 5, 6, [2, 1]],
        "stage4": [1024, 384, 2048, 1, True, True, 5, 6, [2, 1]],
    }

    stage_config_det = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [48, 48, 128, 1, False, False, 3, 6, 2],
        "stage2": [128, 96, 512, 1, True, False, 3, 6, 2],
        "stage3": [512, 192, 1024, 3, True, True, 5, 6, 2],
        "stage4": [1024, 384, 2048, 1, True, True, 5, 6, 2],
    }
    model = PPHGNetV2(
        stem_channels=[3, 32, 48],
        stage_config=stage_config_det if det else stage_config_rec,
        use_lab=False,
        det=det,
        text_rec=text_rec,
        **kwargs,
    )
    return model


def PPHGNetV2_B5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B5
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Module. Specific `PPHGNetV2_B5` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [64, 64, 128, 1, False, False, 3, 6],
        "stage2": [128, 128, 512, 2, True, False, 3, 6],
        "stage3": [512, 256, 1024, 5, True, True, 5, 6],
        "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
    }

    model = PPHGNetV2(
        stem_channels=[3, 32, 64], stage_config=stage_config, use_lab=False, **kwargs
    )
    return model


def PPHGNetV2_B6(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B6
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Module. Specific `PPHGNetV2_B6` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [96, 96, 192, 2, False, False, 3, 6],
        "stage2": [192, 192, 512, 3, True, False, 3, 6],
        "stage3": [512, 384, 1024, 6, True, True, 5, 6],
        "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
    }

    model = PPHGNetV2(
        stem_channels=[3, 48, 96], stage_config=stage_config, use_lab=False, **kwargs
    )
    return model


class PPHGNetV2_B4_Formula(nn.Module):
    """
    PPHGNetV2_B4_Formula
    Args:
        in_channels (int): Number of input channels. Default is 3 (for RGB images).
        class_num (int): Number of classes for classification. Default is 1000.
    Returns:
        model: nn.Module. Specific `PPHGNetV2_B4` model with defined architecture.
    """

    def __init__(self, in_channels=3, class_num=1000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2048
        stage_config = {
            # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
            "stage1": [48, 48, 128, 1, False, False, 3, 6, 2],
            "stage2": [128, 96, 512, 1, True, False, 3, 6, 2],
            "stage3": [512, 192, 1024, 3, True, True, 5, 6, 2],
            "stage4": [1024, 384, 2048, 1, True, True, 5, 6, 2],
        }

        self.pphgnet_b4 = PPHGNetV2(
            stem_channels=[3, 32, 48],
            stage_config=stage_config,
            class_num=class_num,
            use_lab=False,
        )

    def forward(self, input_data):
        if self.training:
            pixel_values, label, attention_mask = input_data
        else:
            if isinstance(input_data, list):
                pixel_values = input_data[0]
            else:
                pixel_values = input_data
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = torch.repeat_interleave(pixel_values, repeats=3, dim=1)
        pphgnet_b4_output = self.pphgnet_b4(pixel_values)
        b, c, h, w = pphgnet_b4_output.shape
        pphgnet_b4_output = pphgnet_b4_output.reshape([b, c, h * w]).transpose(
            [0, 2, 1]
        )
        pphgnet_b4_output = DonutSwinModelOutput(
            last_hidden_state=pphgnet_b4_output,
            pooler_output=None,
            hidden_states=None,
            attentions=False,
            reshaped_hidden_states=None,
        )
        if self.training:
            return pphgnet_b4_output, label, attention_mask
        else:
            return pphgnet_b4_output


class PPHGNetV2_B6_Formula(nn.Module):
    """
    PPHGNetV2_B6_Formula
    Args:
        in_channels (int): Number of input channels. Default is 3 (for RGB images).
        class_num (int): Number of classes for classification. Default is 1000.
    Returns:
        model: nn.Module. Specific `PPHGNetV2_B6` model with defined architecture.
    """

    def __init__(self, in_channels=3, class_num=1000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2048
        stage_config = {
            # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
            "stage1": [96, 96, 192, 2, False, False, 3, 6, 2],
            "stage2": [192, 192, 512, 3, True, False, 3, 6, 2],
            "stage3": [512, 384, 1024, 6, True, True, 5, 6, 2],
            "stage4": [1024, 768, 2048, 3, True, True, 5, 6, 2],
        }

        self.pphgnet_b6 = PPHGNetV2(
            stem_channels=[3, 48, 96],
            class_num=class_num,
            stage_config=stage_config,
            use_lab=False,
        )

    def forward(self, input_data):
        if self.training:
            pixel_values, label, attention_mask = input_data
        else:
            if isinstance(input_data, list):
                pixel_values = input_data[0]
            else:
                pixel_values = input_data
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = torch.repeat_interleave(pixel_values, repeats=3, dim=1)
        pphgnet_b6_output = self.pphgnet_b6(pixel_values)
        b, c, h, w = pphgnet_b6_output.shape
        pphgnet_b6_output = pphgnet_b6_output.reshape([b, c, h * w]).transpose(
            [0, 2, 1]
        )
        pphgnet_b6_output = DonutSwinModelOutput(
            last_hidden_state=pphgnet_b6_output,
            pooler_output=None,
            hidden_states=None,
            attentions=False,
            reshaped_hidden_states=None,
        )
        if self.training:
            return pphgnet_b6_output, label, attention_mask
        else:
            return pphgnet_b6_output
