import os
import io
from pathlib import Path
from enum import Enum
from typing import Type, Union, Optional

import numpy as np

from PIL import Image
from PIL.Image import Image as PILImage

import onnxruntime as ort

from rembg.session_base import BaseSession

from Utils.simple_session import SimpleSession
from Utils.bg import post_process, naive_cutout, alpha_matting_cutout, get_concat_v_multi


class ReturnType(Enum):
    BYTES = 0
    PILLOW = 1
    NDARRAY = 2


def remove(
    data: Union[bytes, PILImage, np.ndarray],
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    session: Optional[BaseSession] = None,
    only_mask: bool = False,
    post_process_mask: bool = False,
    model_name: str = 'u2net',
    size: int = 320,
) -> Union[bytes, PILImage, np.ndarray]:

    if isinstance(data, PILImage):
        return_type = ReturnType.PILLOW
        img = data
    elif isinstance(data, bytes):
        return_type = ReturnType.BYTES
        img = Image.open(io.BytesIO(data))
    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = Image.fromarray(data)
    else:
        raise ValueError("Input type {} is not supported.".format(type(data)))

    if session is None:
        session = new_session(model_name)

    masks = session.predict(img, size)
    cutouts = []

    for mask in masks:
        if post_process_mask:
            mask = Image.fromarray(post_process(np.array(mask)))

        if only_mask:
            cutout = mask

        elif alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                )
            except ValueError:
                cutout = naive_cutout(img, mask)

        else:
            cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    cutout = img
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)

    if ReturnType.PILLOW == return_type:
        return cutout

    if ReturnType.NDARRAY == return_type:
        return np.asarray(cutout)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)

    return bio.read()


def new_session(model_name: str = "u2net") -> BaseSession:
    session_class: Type[BaseSession]
    md5 = "60024c5c889badc19c04ad937298a77b"
    url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
    session_class = SimpleSession

    if model_name == "u2netp":
        md5 = "8e83ca70e441ab06c318d82300c84806"
        url = (
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
        )
        session_class = SimpleSession

    u2net_home = os.getenv(
        "U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
    )

    fname = f"{model_name}.onnx"
    path = Path(u2net_home).expanduser()
    full_path = Path(u2net_home).expanduser() / fname

    # pooch.retrieve(
    #     url,
    #     f"md5:{md5}",
    #     fname=fname,
    #     path=Path(u2net_home).expanduser(),
    #     progressbar=True,
    # )

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    return session_class(
        model_name,
        ort.InferenceSession(
            str(full_path),
            providers=ort.get_available_providers(),
            sess_options=sess_opts,
        ),
    )