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
    Add docstring and paper later.

    Examples
    --------
    >>> x = torch.randn(8, 3, 32, 32, 32)
    >>> model = RetNeXt(3, 100, max_global_pool=False)

    >>> model(x).shape
    torch.Size([8, 100])
    >>> model.backbone(x).shape
    torch.Size([8, 128])
    >>> isinstance(model.backbone[-2], nn.AdaptiveAvgPool3d)
    True

    >>> x = torch.randn(16, 3, 25, 25, 25)
    >>> model(x).shape
    torch.Size([16, 100])
    """
    def __init__(
            self,
            in_channels: int = 1,
            n_outputs: int = 1,
            *,
            max_global_pool: bool = True,
            ):
        super().__init__()

        if max_global_pool:
            global_pool_layer = nn.AdaptiveMaxPool3d(1)
        else:
            global_pool_layer = nn.AdaptiveAvgPool3d(1)

        self.backbone = nn.Sequential(
                nn.BatchNorm3d(in_channels, affine=False, momentum=None),
                conv3d_block(in_channels, 32, kernel_size=3, bias=False, padding='same'),
                conv3d_block(32, 32, kernel_size=3, bias=False, padding='same'),
                nn.MaxPool3d(kernel_size=2),  # 1st pooling layer.
                conv3d_block(32, 64, kernel_size=3, bias=False, padding='same'),
                conv3d_block(64, 64, kernel_size=3, bias=False, padding='same'),
                nn.MaxPool3d(kernel_size=2),  # 2nd pooling layer.
                conv3d_block(64, 128, kernel_size=3, bias=False),
                conv3d_block(128, 128, kernel_size=3, bias=False),
                global_pool_layer,
                nn.Flatten()
                )

        self.fc = torch.nn.Linear(128, n_outputs)

    def forward(self, x):
        return self.fc(self.backbone(x))
