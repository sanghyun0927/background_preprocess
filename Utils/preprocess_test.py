import numpy as np

import cv2
from PIL import Image

from typing import List, Tuple


def rescale_and_paste(old_fore: Image, old_back: Image, only_rescale: bool, scale_factor: int = 1) -> Tuple[
    Image.Image, Image.Image]:
    # config
    fore_x_ratio = 0.5
    fore_y_ratio = 0.65
    fore_ratio = 0.6

    # define scale and resized image size
    if old_fore.width > old_back.width:
        # define new image size
        new_fore_width = old_back.width * fore_ratio * scale_factor
        new_fore_height = old_fore.height * new_fore_width / old_fore.width
        new_width = np.rint(new_fore_width).astype('uint16')
        new_height = np.rint(new_fore_height).astype('uint16')

        # resizing
        new_fore = old_fore.resize((new_width, new_height))
        new_back = old_back
    else:
        # define new image size
        new_back_width = old_fore.width / fore_ratio / scale_factor
        new_back_height = old_back.height * new_back_width / old_back.width
        new_width = np.rint(new_back_width).astype('uint16')
        new_height = np.rint(new_back_height).astype('uint16')

        # resizing
        new_fore = old_fore
        new_back = old_back.resize((new_width, new_height))

    # Foreground(차량)의 높이가 background(배경)의 높이를 초과하는 경우 에러를 발생시킴
    assert new_fore.height < new_back.height, "입력 이미지의 세로 길이가 제한 공간을 초과 하였습니다. 입력 이미지의 세로 여백을 잘라내고 다시 시도하세요."

    # define the location of foreground
    y_center = int(new_back.height * fore_y_ratio)
    x_center = int(new_back.width * fore_x_ratio)
    y = y_center - new_fore.height // 2
    x = x_center - new_fore.width // 2

    # 새 alpha 채널 직사각형 이미지 생성
    new_im = Image.new(mode='RGBA', size=new_back.size)

    # paste foreground on transparent background
    try:
        new_im.paste(new_fore, (x, y))
    except:
        print('check preprocess.py')
        new_im.paste(new_fore, (x, new_back.height - new_fore.height))

    if only_rescale:
        return new_im
    else:
        # paste foreground to background
        result = Image.alpha_composite(new_back, new_im)
        return result, new_im


def create_manipulated_mask(mask_np: np.ndarray, alpha_np: np.ndarray, contour_np: np.ndarray) -> Image.Image:
    """Create a manipulated RGBA mask image.

        Args:
            mask_np (numpy.ndarray): The input RGBA mask array.
            alpha_np (numpy.ndarray): The input alpha array.
            contour_np (numpy.ndarray): The input contour array.

        Returns:
            PIL.Image.Image: The output RGBA mask image.
    """
    # config
    top_y_ratio = 1
    mask_y_ratio = 0.525

    # Extract the shape of the mask_np array
    mask_height, mask_width = mask_np.shape[:2]

    # Find indices where the non-masked area located
    mask_indices = np.where(mask_np[:, :, 3] == 0)
    # Find indices where the foreground, car, located
    alpha_indices = np.where(alpha_np[:, :] == 0)
    # Find indices modified contour and foreground located
    contour_indices = np.where(contour_np[:, :] == 255)

    # Initialize manipulated_mask array with the same shape as the input mask
    manipulated_mask = np.zeros((mask_height, mask_width), dtype='uint8')

    # Value of non-masked area equal 255
    manipulated_mask[mask_indices] = 255
    # Make the value of the area below a specified height equal 0
    manipulated_mask[int(mask_height * mask_y_ratio):, :] = 0
    # Value of foreground located area equal 255
    manipulated_mask[alpha_indices] = 255
    # Make the value of the area, above the foreground, equal 255
    manipulated_mask[:alpha_indices[0].min() * top_y_ratio, :] = 255
    # Make the value of modified contour and foreground located area equal 255
    manipulated_mask[contour_indices] = 255

    # Create a binary mask with values below 128 set to 255 and others set to 0
    binary_mask = np.where(manipulated_mask < 128, 255, 0).astype('uint8')

    # Expand the binary mask dimensions to create a reverse mask
    mask_reverse = np.expand_dims(binary_mask, axis=2)

    # Expand the dimensions of the manipulated mask and duplicate it along the color channels
    mask_rgb = np.tile(np.expand_dims(manipulated_mask, axis=2), reps=[1, 1, 3])

    # Combine the mask_rgb and mask_reverse arrays along the last axis to create an RGBA mask
    mask_rgba = np.concatenate((mask_rgb, mask_reverse), axis=-1)

    # Convert the mask_rgba array to an RGBA image
    mask_image = Image.fromarray(mask_rgba, mode='RGBA')

    return mask_image
