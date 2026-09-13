"""Convert token-aligned values into fixed-length Style-LOO waves."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _coerce_values(
    values: Sequence[float | None] | NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    finite = np.isfinite(array)
    if not np.any(finite):
        raise ValueError("values contain no finite entries")
    observed = array[finite]
    if np.any(observed < 0.0) or np.any(observed > 1.0):
        raise ValueError("finite values must lie in [0, 1]")
    if not np.all(finite):
        positions = np.arange(array.size, dtype=np.float64)
        array[~finite] = np.interp(positions[~finite], positions[finite], array[finite])
    return array


def gaussian_kernel1d(sigma: float, *, truncate: float = 4.0) -> NDArray[np.float64]:
    """Return a normalized one-dimensional Gaussian kernel."""

    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise TypeError("sigma must be a real number")
    if not math.isfinite(float(sigma)) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")
    if not math.isfinite(float(truncate)) or truncate <= 0:
        raise ValueError("truncate must be finite and positive")
    if sigma == 0:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(float(truncate) * float(sigma))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    return kernel / kernel.sum()


def fixed_length_wave(
    values: Sequence[float | None] | NDArray[np.floating[Any]],
    *,
    output_length: int,
    sigma: float,
    gaussian_truncate: float = 4.0,
) -> NDArray[np.float32]:
    """Interpolate gaps, smooth on the output-grid scale, and resample."""

    if isinstance(output_length, bool) or not isinstance(output_length, int):
        raise TypeError("output_length must be an integer")
    if output_length < 2:
        raise ValueError("output_length must be at least 2")
    array = _coerce_values(values)
    input_sigma = (
        0.0
        if array.size == 1
        else float(sigma) * float(array.size - 1) / float(output_length - 1)
    )
    kernel = gaussian_kernel1d(input_sigma, truncate=gaussian_truncate)
    if kernel.size > 1:
        radius = kernel.size // 2
        array = np.convolve(np.pad(array, radius, mode="edge"), kernel, mode="valid")
    if array.size == 1:
        result = np.full(output_length, array[0], dtype=np.float64)
    else:
        source = np.linspace(0.0, 1.0, num=array.size, dtype=np.float64)
        target = np.linspace(0.0, 1.0, num=output_length, dtype=np.float64)
        result = np.interp(target, source, array)
    return result.astype(np.float32, copy=False)


__all__ = ["fixed_length_wave", "gaussian_kernel1d"]

