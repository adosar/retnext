r"""
This module provides PyTorch modules for voxel-based processing.
"""

import torch
from torch import nn


def conv3d_block(in_channels: int, out_channels: int, **kwargs):
    r"""
    Return a block of the form Conv -> BatchNorm -> ReLU.

    Examples
    --------
    >>> block = conv3d_block(4, 8, kernel_size=2, bias=False, padding_mode='circular')
    >>> block[0].kernel_size
    (2, 2, 2)
    >>> block[0].padding_mode
    'circular'
    >>> block[0].bias
    """
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, **kwargs),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
            )


class RetNeXt(nn.Module):
    r"""
    Architecture from the paper: <Paper>

    * If ``avg_global_pool=True`` use average pooling, else max pooling.
    * If ``pretrained=True``, use pretrained weights for the backbone.

    Examples
    --------
    >>> x = torch.randn(8, 3, 32, 32, 32)
    >>> model = RetNeXt(3, 100)
    >>> model(x).shape
    torch.Size([8, 100])
    >>> model.backbone(x).shape
    torch.Size([8, 128])

    >>> # Works with different grid sizes (adaptive pooling).
    >>> x = torch.randn(16, 3, 25, 25, 25)
    >>> model(x).shape
    torch.Size([16, 100])

    >>> # Global average pooling by default.
    >>> isinstance(model.backbone[-2], nn.AdaptiveAvgPool3d)
    True
    >>> model = RetNeXt(3, 100, avg_global_pool=False)
    >>> isinstance(model.backbone[-2], nn.AdaptiveMaxPool3d)
    True

    >>> # Uses pretrained weights for the backbone.
    >>> model = RetNeXt(pretrained=True)
    >>> model.backbone[0].running_mean.item != 0.
    True
    """
    def __init__(
            self,
            in_channels: int = 1,
            n_outputs: int = 1,
            *,
            avg_global_pool: bool = True,
            pretrained: bool = False,
            ):
        super().__init__()

        if avg_global_pool:
            global_pool_layer = nn.AdaptiveAvgPool3d(1)
        else:
            global_pool_layer = nn.AdaptiveMaxPool3d(1)

        self.backbone = nn.Sequential(
                nn.BatchNorm3d(in_channels, affine=False, momentum=None),
                conv3d_block(in_channels, 32, kernel_size=3, bias=False, padding='same'),
                conv3d_block(32, 32, kernel_size=3, bias=False, padding='same'),
                nn.MaxPool3d(kernel_size=2), # 1st pooling layer
                conv3d_block(32, 64, kernel_size=3, bias=False, padding='same'),
                conv3d_block(64, 64, kernel_size=3, bias=False, padding='same'),
                nn.MaxPool3d(kernel_size=2),  # 2nd pooling layer
                conv3d_block(64, 128, kernel_size=3, bias=False),
                conv3d_block(128, 128, kernel_size=3, bias=False),
                global_pool_layer,
                nn.Flatten()
                )

        if pretrained:
            self.backbone.load_state_dict(self.get_pretrained_weights())

        self.fc = torch.nn.Linear(128, n_outputs)

    def forward(self, x):
        return self.fc(self.backbone(x))

    def get_pretrained_weights(self):
        url = 'https://raw.githubusercontent.com/adosar/retnext-paper/master/pretrained_weights/retnext_cubic_boltzmann_final_all.pt'
        return torch.hub.load_state_dict_from_url(url)
