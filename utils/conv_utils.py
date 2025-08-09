# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import torch
import torch.nn.functional as F


def calculate_samesize_padding(size, kernel_size=(7, 1), stride=(1, 1)):
    ph = (stride[0] * (size[0] - 1) + kernel_size[0] - size[0]) // 2
    pw = (stride[1] * (size[1] - 1) + kernel_size[1] - size[1]) // 2
    return (ph, pw)

def batch_convolution_loop(input_tensor, weight_tensor, stride=(1, 1), padding=(0, 0)):
    batch_size, in_channels, h, w = input_tensor.shape
    _, out_channels, _, k_h, k_w = weight_tensor.shape

    output_h = (h - k_h + 2 * padding[0]) // stride[0] + 1
    output_w = (w - k_w + 2 * padding[1]) // stride[1] + 1
    output_tensor = torch.zeros((batch_size, out_channels, output_h, output_w))

    for i in range(batch_size):
        input_i = input_tensor[i].unsqueeze(0)  # [1, in_channels, h, w]
        weight_i = weight_tensor[i]  # [out_channels, in_channels, k_h, k_w]
        output_tensor[i] = F.conv2d(input_i, weight_i, stride=stride, padding=padding)

    return output_tensor.to(input_tensor.device)

def batch_convolution_group(input_tensor, weight_tensor, stride=(1, 1), padding=(0, 0)):
    batch_size, in_channels, h, w = input_tensor.shape
    _, out_channels, _, k_h, k_w = weight_tensor.shape

    input_tensor = input_tensor.view(1, batch_size * in_channels, h, w)
    weight_tensor = weight_tensor.view(batch_size * out_channels, in_channels, k_h, k_w)

    output_tensor = F.conv2d(input_tensor, weight_tensor, stride=stride, padding=padding, groups=batch_size)

    output_h = (h - k_h + 2 * padding[0]) // stride[0] + 1
    output_w = (w - k_w + 2 * padding[1]) // stride[1] + 1
    output_tensor = output_tensor.view(batch_size, out_channels, output_h, output_w)

    return output_tensor