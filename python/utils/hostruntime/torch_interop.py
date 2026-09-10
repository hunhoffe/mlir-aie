# torch_interop.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Zero-copy bridging between numpy dtypes and their torch equivalents."""

import numpy as np

# Mapping from ml_dtypes (non-native numpy) types to their torch equivalents.
# Native numpy dtypes (float32, int32, …) are handled directly by torch.from_numpy
# and do not need an entry here.
# Populated lazily at first use to avoid importing torch/ml_dtypes at module load.
_ML_DTYPE_TO_TORCH: dict | None = None
# The same mapping keyed the other way, for the torch -> numpy direction.
_TORCH_TO_ML_DTYPE: dict | None = None


def _ml_dtype_to_torch_map():
    global _ML_DTYPE_TO_TORCH
    if _ML_DTYPE_TO_TORCH is None:
        import ml_dtypes
        import torch  # pyright: ignore[reportMissingImports]

        _candidates = {
            ml_dtypes.bfloat16: torch.bfloat16,
        }
        for attr in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
        ):
            ml_dt = getattr(ml_dtypes, attr, None)
            torch_dt = getattr(torch, attr, None)
            if ml_dt is not None and torch_dt is not None:
                _candidates[ml_dt] = torch_dt
        _ML_DTYPE_TO_TORCH = {
            np.dtype(ml_dt): torch_dt for ml_dt, torch_dt in _candidates.items()
        }
    return _ML_DTYPE_TO_TORCH


def _torch_to_ml_dtype_map():
    """Return the forward map inverted: torch dtype -> ml_dtypes numpy dtype.

    Derived from :func:`_ml_dtype_to_torch_map` rather than written out again,
    so the two directions cannot describe different sets of types.
    """
    global _TORCH_TO_ML_DTYPE
    if _TORCH_TO_ML_DTYPE is None:
        _TORCH_TO_ML_DTYPE = {
            torch_dt: np_dt for np_dt, torch_dt in _ml_dtype_to_torch_map().items()
        }
    return _TORCH_TO_ML_DTYPE


# Same-width unsigned integer dtype for the ND reinterpret-view trick.
_UINT_VIEW_DTYPE = {
    1: np.uint8,
    2: np.uint16,
    4: np.uint32,
    8: np.uint64,
}

# The same widths named on the torch side, for the reverse trick.  Looked up by
# name so this module still imports where torch lacks one of the unsigned types.
_TORCH_UINT_VIEW_NAMES = {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}


def _array_to_torch(array: np.ndarray):
    """Convert a numpy array to a torch tensor, zero-copy.

    For native numpy dtypes (float32, float16, int32, …) torch.from_numpy is used directly
    (fastest path for these types).

    For ml_dtypes types (bfloat16, float8_*) that torch cannot consume via from_numpy:
    reinterpret as a same-width unsigned integer numpy view, wrap with from_numpy,
    then view as the target torch dtype.  This is guaranteed zero-copy for all ranks.

    Raises:
        ImportError: If torch is not installed.
    """
    # _ml_dtype_to_torch_map() imports torch (raising ImportError with a helpful message
    # if absent) and returns the ml_dtype -> torch dtype mapping.
    torch_dtype = _ml_dtype_to_torch_map().get(array.dtype)
    import torch  # pyright: ignore[reportMissingImports]  # already imported by _ml_dtype_to_torch_map(); cached by Python

    if torch_dtype is None:
        # Native numpy dtype: torch.from_numpy handles it directly and fastest.
        return torch.from_numpy(array)

    # ml_dtype: reinterpret memory as a same-width uint, then view as the torch dtype.
    uint_dtype = _UINT_VIEW_DTYPE[array.dtype.itemsize]
    return torch.from_numpy(array.view(uint_dtype)).view(torch_dtype)


def torch_to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array, zero-copy where possible.

    The inverse of :func:`_array_to_torch`, and the counterpart every caller
    that reads a torch tensor back into numpy needs: it detaches from autograd,
    brings the tensor to the host, and makes it contiguous before handing back
    an array, so callers do not each re-derive that sequence.

    Types torch and numpy agree on convert through ``Tensor.numpy()``.  For the
    ml_dtypes types torch cannot hand to numpy directly (bfloat16, float8_*),
    the memory is reinterpreted through a same-width unsigned integer view, the
    mirror of the trick :func:`_array_to_torch` uses going the other way.

    Non-tensor input is passed to ``np.asarray``, so a caller holding "a torch
    tensor or something array-like" does not have to branch on which it has.

    Args:
        tensor: A torch tensor, or anything ``np.asarray`` accepts.

    Returns:
        np.ndarray: The tensor's data.  Shares memory with ``tensor`` unless
        detaching, moving to the host, or making it contiguous had to copy.

    Raises:
        ImportError: If torch is not installed and ``tensor`` is a torch tensor.
    """
    if not hasattr(tensor, "detach"):
        # Not a torch tensor: nothing to unwrap, and numpy already knows how to
        # read array-likes.  Checked by duck-typing so this path costs nothing
        # when torch is absent.
        return np.asarray(tensor)

    import torch  # pyright: ignore[reportMissingImports]

    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_contiguous():
        # .numpy() and the reinterpret view below both require contiguity.
        t = t.contiguous()

    np_dtype = _torch_to_ml_dtype_map().get(t.dtype)
    if np_dtype is None:
        # A dtype numpy shares with torch: hand it over directly.
        return t.numpy()

    # An ml_dtype: view the same bytes as a same-width unsigned integer (which
    # both libraries have) and reinterpret on the numpy side.
    uint_name = _TORCH_UINT_VIEW_NAMES[np_dtype.itemsize]
    uint_dtype = getattr(torch, uint_name, None)
    if uint_dtype is None:
        raise TypeError(
            f"Cannot convert a torch tensor of dtype {t.dtype} to numpy: this "
            f"torch build has no torch.{uint_name} to reinterpret it through."
        )
    return t.view(uint_dtype).numpy().view(np_dtype)
