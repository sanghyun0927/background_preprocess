import os
from pathlib import Path
from typing import Type

import onnxruntime as ort

from typing import List

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from rembg.session_base import BaseSession


class SimpleSession(BaseSession):
    def predict(self, img: PILImage, size: int) -> List[PILImage]:
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (size, size)
            ),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        return [mask]

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