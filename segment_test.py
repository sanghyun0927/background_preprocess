import argparse
import os
from glob import glob

import numpy as np

from rembg import remove
from PIL import Image
from tqdm import tqdm


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Car segmentation")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=False,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    args = parse_args()

    if args.image_path:
        image_dirs = glob(os.path.join(args.image_path, '*'))
    else:
        image_dirs = glob(".\\images\\*\\*", recursive=True)

    if not args.model_name:
        model_name = 'u2car_v2.1'
    else:
        model_name = args.model_name

    for i in tqdm(image_dirs):
        mask_path = "." + i.split(".")[1] + '_mask.png'
        final_path = "." + i.split(".")[1] + '_car.png'

        img = Image.open(i)
        # x, y = img.size
        # size = max(x, y)
        # black = np.zeros((size, size), dtype='uint8')
        # full_gray = np.ones((size, size, 4), dtype='uint8') * 170

        car = remove(img, alpha_matting=True, post_process_mask=True, model_name=model_name, size=1024)  #, model_name='InSPyReNet', size=1024  #, model_name='u2car_v2.1', size=320
        car.save(mask_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)