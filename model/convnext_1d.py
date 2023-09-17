"""
title:
doi:

unofficial implementation for 1d spectral data
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channel: int, dropout: float = 0.2, kernel_size: int = 16):
        super().__init__()
        self.conv_d = nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding='same',
                                groups=in_channel)
        self.conv_1 = nn.Conv1d(in_channel, in_channel * 4, kernel_size=1, stride=1, padding='same')
        self.conv_2 = nn.Conv1d(in_channel * 4, in_channel, kernel_size=1, stride=1, padding='same')
        self.gelu = nn.GELU()
        self.ln = LayerNorm(in_channel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res_x = x
        x = self.conv_d(x)
        x = self.ln(x)
        x = self.conv_1(x)
        x = self.gelu(x)
        x = self.conv_2(x)
        x = self.dropout(x)
        return x + res_x


class ConvNeXtStage(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, block_num: int, dropout: float = 0.2,
                 down_sample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([ConvNeXtBlock(in_channel, dropout) for _ in range(block_num)])
        self.ln = LayerNorm(in_channel)
        self.down_sample = nn.Conv1d(in_channel, out_channel, kernel_size=2,
                                     stride=2) if down_sample else nn.Identity()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        x = self.down_sample(x)
        return x


class CONVNEXT1D(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, spectrum_size: int):
        super().__init__()
        channel_list = [64, 128, 32, 16]
        block_num_list = [3, 3, 3, 3]
        stem_patch = 32
        assert len(channel_list) == len(
            block_num_list), "[Error] channel_list and block_num_list must have the same length"
        self.stem = nn.Conv1d(in_channel, channel_list[0], kernel_size=stem_patch, stride=stem_patch)
        self.extractor = nn.ModuleList([
            ConvNeXtStage(channel_list[i], channel_list[i + 1] if i != len(channel_list) - 1 else None,
                          block_num_list[i], down_sample=i != len(channel_list) - 1) for i in range(len(channel_list))
        ])
        self.classifier = nn.Linear(
            channel_list[-1] * spectrum_size // stem_patch // (2 ** (len(channel_list) - 1)),
            out_channel)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.extractor:
            x = stage(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
