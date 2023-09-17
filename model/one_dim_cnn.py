import torch
import torch.nn as nn
from timm.models.layers import DropPath


class LayerNormChannel1d(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, D]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels))
        self.bias = nn.Parameter(torch.zeros(1, num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1) * x + self.bias.unsqueeze(-1)
        return x


class ONEDIMCNN(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, spectrum_size: int):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=16, stride=1, padding='same'),
            LayerNormChannel1d(64),
            nn.Conv1d(64, 64, kernel_size=16, stride=1, padding='same'),
            nn.GELU(),
            # 4x down-sample conv
            nn.MaxPool1d(kernel_size=4, stride=4),

            DropPath(0.2),
            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding='same'),
            LayerNormChannel1d(32),
            nn.Conv1d(32, 32, kernel_size=16, stride=1, padding='same'),
            nn.GELU(),
            # 4x down-sample conv
            nn.MaxPool1d(kernel_size=4, stride=4),

            DropPath(0.2),
            nn.Conv1d(32, 16, kernel_size=7, stride=1, padding='same'),
            LayerNormChannel1d(16),
            nn.Conv1d(16, 16, kernel_size=7, stride=1, padding='same'),
            # 4x down-sample conv
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.fc_structure = nn.Sequential(
            nn.Linear(16 * (spectrum_size // 4 // 4 // 4), 1024),
            nn.ReLU(),
            nn.Linear(1024, out_channel),
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc_structure(x)
        return x
