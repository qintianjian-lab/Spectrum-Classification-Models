"""
title: Celestial Spectra Classification Network Based on Residual and Attention Mechanisms
doi: 10.1088/1538-3873/ab7548

RC-Net unofficial implementation
"""
import torch.nn as nn


class RCNetBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv_blk = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding='same'),
        )
        self.residual_conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding='same')
        self.residual_relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv_blk(x)
        x = x + residual
        x = self.residual_relu(x)
        return x


class RCNET(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, spectrum_size: int, logistic: bool = False):
        """
        RC-Net model constructor
        :param in_channel: input spectrum channel, the input shape should be (batch_size, in_channel, spectrum_size)
        :param out_channel: output channel, as well as the number of classes
        :param spectrum_size: spectrum size, as well as the length of the spectrum
        :param logistic: whether to use logistic function as the last layer, default: False
        """
        super().__init__()
        conv_out_channel_list = [16, 16, 32, 32, 64, 64, 128, 128]
        self.conv_structure = nn.ModuleList([
            nn.Sequential(
                RCNetBlock(in_channel if _index == 0 else conv_out_channel_list[_index - 1],
                           conv_out_channel_list[_index]),
                nn.MaxPool1d(kernel_size=2, stride=2))
            for _index in range(len(conv_out_channel_list))
        ])
        self.fc_structure = nn.Sequential(
            nn.Linear(conv_out_channel_list[-1] * (spectrum_size // (2 ** len(conv_out_channel_list))), 128),
            nn.ReLU(),
            nn.Linear(128, out_channel),
            nn.Softmax(dim=-1) if logistic else nn.Identity(),
        )

    def forward(self, x):
        for layer in self.conv_structure:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_structure(x)
        return x
