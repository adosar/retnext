r"""
This module provides voxel transformations.

.. note::
    * All randomness under the hood is controlled with PyTorch.
    * All geometric transformations assume input of shape ``(C, D, H, W)``.
"""

from itertools import combinations

import torch


class RandomRotate90:
    r"""
    Rotate voxels around a randomly chosen axis by 90 degrees.

    Examples
    --------
    >>> x = torch.randn(3, 10, 10, 10)
    >>> out = RandomRotate90()(x)
    >>> torch.equal(x, out)
    False
    """
    def __init__(self):
        self.planes = list(combinations([1, 2, 3], 2))
        self.directions = torch.tensor([-1, 1])

    def __call__(self, x):
        p_choice = torch.randint(len(self.planes), ()).item()
        plane = self.planes[p_choice]

        d_choice = torch.randint(len(self.directions), ()).item()
        direction = self.directions[d_choice]

        return torch.rot90(x, k=direction, dims=plane)


class RandomFlip:
    r"""
    Flip voxels along a randomly chosen axis.

    Examples
    --------
    >>> x = torch.randn(3, 10, 10, 10)
    >>> out = RandomFlip()(x)
    >>> torch.equal(x, out)
    False
    """
    def __call__(self, x):
        dim = torch.randint(1, 4, ()).item()

        return torch.flip(x, [dim])


class RandomReflect:
    r"""
    Reflect voxels along a randomly chosen plane.

    Examples
    --------
    >>> x = torch.randn(3, 10, 10, 10)
    >>> out = RandomReflect()(x)
    >>> torch.equal(x, out)
    False
    """
    def __init__(self):
        self.planes = list(combinations([1, 2, 3], 2))

    def __call__(self, x):
        p_choice = torch.randint(len(self.planes), ()).item()
        plane = self.planes[p_choice]

        return torch.transpose(x, *plane)


class RandomRoll:
    r"""
    Roll voxels along a randomly chosen axis and shift value.

    .. note::
        This transormation is meaningful only for voxels satisfying PBC.

    Examples
    --------
    >>> x = torch.randn(3, 10, 10, 10)
    >>> out = RandomRoll([2, 4, 6])(x)
    >>> torch.equal(x, out)
    False
    """
    def __init__(self, shift_values: list[int]):
        self.shift_values = shift_values
    def __call__(self, x):
        dim = torch.randint(1, 4, ()).item()

        s_choice = torch.randint(len(self.shift_values), ()).item()
        shift = self.shift_values[s_choice]

        return torch.roll(x, shifts=shift, dims=dim)


class AddChannelDim:
    r"""
    Prepend a dimension to the input tensor.

    Examples
    --------
    >>> x = torch.randn(32, 32, 32)
    >>> AddChannelDim()(x).shape
    torch.Size([1, 32, 32, 32])
    """
    def __call__(self, x):
        return x.unsqueeze(0)


class ClipVoxels:
    r"""
    Clip voxels within ``[vmin, vmax]``.

    Examples
    --------
    >>> x = torch.randn(100) * 100
    >>> out = ClipVoxels(-1, 1)(x)
    >>> out.min()
    tensor(-1.)
    >>> out.max()
    tensor(1.)
    """
    def __init__(self, vmin: float, vmax: float):
        self.vmin = vmin
        self.vmax = vmax
    def __call__(self, x):
        return torch.clip(x, self.vmin, self.vmax)


class BoltzmannFactor:
    r"""
    Fill voxels with the Boltzmann factor.

    Examples
    --------
    >>> x = torch.tensor([0., torch.inf])
    >>> BoltzmannFactor()(x)
    tensor([1., 0.])
    """
    def __init__(self, temperature: float = 298):
        self.temperature = temperature
    def __call__(self, x):
        return torch.exp((-1 / self.temperature) * x)
