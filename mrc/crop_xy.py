import os
from typing import Optional, Sequence, Union

import imageio.v2 as imageio
import numpy as np
import tifffile

from mrc.io import get_tomo, save_tomo


def save_tiff(data, path, dtype=None):
    """
    Save a numpy array as a TIFF image.

    Parameters
    ----------
    data : array_like
        Image data to be written. Can be 2D or 3D.
    path : str
        Destination file path.
    dtype : data-type, optional
        If provided, the data is cast to this dtype before writing.
    """
    array = np.asarray(data)
    if dtype is not None:
        array = array.astype(dtype)

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    tifffile.imwrite(path, array)


def save_jpg(
    data,
    path,
    normalize: bool = True,
    clip_percentile: Optional[Union[float, Sequence[float]]] = None,
    quality: int = 95,
):
    """
    Save an array as a JPEG file.

    Parameters
    ----------
    data : array_like
        2D grayscale or 3D image data.
    path : str
        Destination file path.
    normalize : bool, default True
        When True, rescale values to 0-255 before writing.
    clip_percentile : float or pair of float, optional
        If provided, clip data based on lower/upper percentiles prior to
        normalization. A single value is treated as (value, 100 - value).
    quality : int, default 95
        JPEG compression quality passed to the writer.
    """
    array = np.asarray(data).squeeze()

    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)

    if array.ndim == 2:
        processed = array
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        processed = array
    else:
        raise ValueError("save_jpg expects 2D grayscale or 3D RGB/RGBA input.")

    if clip_percentile is not None:
        perc = np.asarray(clip_percentile, dtype=float)
        if perc.ndim == 0 or perc.size == 1:
            lower = float(np.squeeze(perc))
            upper = 100.0 - lower
        elif perc.size == 2:
            lower, upper = perc
        else:
            raise ValueError("clip_percentile must be a scalar or contain two values.")
        processed = processed.astype(np.float32, copy=False)
        lower_val, upper_val = np.percentile(processed, [lower, upper])
        if upper_val > lower_val:
            processed = np.clip(processed, lower_val, upper_val)

    if normalize:
        processed = processed.astype(np.float32, copy=False)
        processed -= processed.min()
        max_val = processed.max()
        if max_val > 0:
            processed /= max_val
        processed = (processed * 255.0).round()

    processed = processed.astype(np.uint8)

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    imageio.imwrite(path, processed, quality=quality)

ori_tomo = get_tomo(f'/media/liushuo/data1/data/synapse_seg/pp676/pp676.mrc')
# image_for_emdb = ori_tomo[178, 150:850, 0:700]
# save_jpg(image_for_emdb, f'/media/liushuo/data1/data/synapse_seg/temp_upload/pp472-bin4-wbp_xy_slice.jpg')
crop_data = ori_tomo[100:200, 300:400, 300:400]
save_tomo(crop_data, f'/media/liushuo/data1/data/synapse_seg/pp676/pp676_crop.mrc', datetype=crop_data.dtype)
