"""
title: Classification of large-scale stellar spectra based on deep convolutional neural network
doi: 10.1093/mnras/sty3020

1D SSCNN unofficial implementation
"""

import torch.nn as nn


class SSCNN(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, spectrum_size: int, logistic: bool = False):
        """
        SCCNN model constructor
        :param in_channel: input spectrum channel, the input shape should be (batch_size, in_channel, spectrum_size)
        :param out_channel: output channel, as well as the number of classes
        :param spectrum_size: spectrum size, as well as the length of the spectrum
        :param logistic: whether to use logistic function as the last layer, default: False
        """
        super().__init__()
        self.conv_structure = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=16, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=16, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=16, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.fc_structure = nn.Sequential(
            nn.Linear(32 * (spectrum_size // 4 // 4), 1024),
            nn.ReLU(),
            nn.Linear(1024, out_channel),
            nn.Softmax(dim=-1) if logistic else nn.Identity(),
        )

    def forward(self, x):
        x = self.conv_structure(x)
        x = x.view(x.size(0), -1)
        x = self.fc_structure(x)
        return x
