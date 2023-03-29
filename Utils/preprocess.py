import numpy as np

import cv2
from PIL import Image

import xgboost as xgb

from typing import List, Tuple

from typing import Tuple, Union
from PIL import Image
import numpy as np


def rescale_and_paste(
        old_foreground: Image.Image,
        old_background: Image.Image,
        only_rescale: bool,
        scale_factor: int = 1
) -> Union[Tuple[Image.Image, Image.Image], Image.Image]:
    """
    Rescale and paste the foreground image onto the background image.

    Args:
        old_foreground (Image.Image): The original foreground image.
        old_background (Image.Image): The original background image.
        only_rescale (bool): Whether to return only the rescaled foreground image.
        scale_factor (int): An optional scale factor to apply.

    Returns:
        Union[Tuple[Image.Image, Image.Image], Image.Image]: Either a tuple containing the
        resulting image and the new foreground image, or only the new foreground image.
    """
    # Configuration
    fore_x_ratio = 0.5
    fore_y_ratio = 0.65
    fore_ratio = 0.6

    # Define scale and resized image size
    if old_foreground.width > old_background.width:
        new_fore_width = old_background.width * fore_ratio * scale_factor
        new_fore_height = old_foreground.height * new_fore_width / old_foreground.width

        new_width = np.rint(new_fore_width).astype('uint16')
        new_height = np.rint(new_fore_height).astype('uint16')

        # Resizing
        new_foreground = old_foreground.resize((new_width, new_height))
        new_background = old_background
    else:
        new_back_width = old_foreground.width / fore_ratio / scale_factor
        new_back_height = old_background.height * new_back_width / old_background.width

        new_width = np.rint(new_back_width).astype('uint16')
        new_height = np.rint(new_back_height).astype('uint16')

        # Resizing
        new_foreground = old_foreground
        new_background = old_background.resize((new_width, new_height))

    # Foreground(차량)의 높이가 background(배경)의 높이를 초과하는 경우 에러를 발생시킴
    assert new_foreground.height < new_background.height, (
        "입력 이미지의 세로 길이가 제한 공간을 초과 하였습니다. 입력 이미지의 세로 여백을 잘라내고 다시 시도하세요."
    )

    # Define the location of the foreground
    y_center = int(new_background.height * fore_y_ratio)
    x_center = int(new_background.width * fore_x_ratio)
    y = y_center - new_foreground.height // 2
    x = x_center - new_foreground.width // 2

    # 새 alpha 채널 직사각형 이미지 생성
    new_image = Image.new(mode='RGBA', size=new_background.size)

    # Paste foreground on transparent background
    try:
        new_image.paste(new_foreground, (x, y))
    except:
        print('Check preprocess.py')
        new_image.paste(new_foreground, (x, new_background.height - new_foreground.height))

    if only_rescale:
        return new_image
    else:
        # Paste foreground onto background
        result = Image.alpha_composite(new_background, new_image)
        return result, new_image


def stroke_mask(img_array: np.array, mask_size: int):
    """
    세그멘테이션된 차량 이미지 외곽에 매우 굵은 윤곽선을 생성하여 Stable diffusion inpaint 모델의 마스크로 사용함

    Args:
        image_array (np.array): Pillow image, 'RGBA'
        stroke_size (str): 윤곽석 굵이, 픽셀 단위
        colors ((int,int,int)): 십진수 색상, 'RGB', 0-255

    Returns:
        result (PIL.Image.Image): 윤곽선을 포함한 차량 이미지, 배경 없음
    """

    # Add padding to the image
    padding = mask_size
    alpha = img_array[:, :, 3]
    rgb_img = img_array[:, :, 0:3]
    padded_img = cv2.copyMakeBorder(rgb_img, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    alpha = cv2.copyMakeBorder(alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    # Merge the padded image and alpha channel
    padded_img = cv2.merge((padded_img, alpha))

    # Apply threshold to the alpha channel
    _, alpha_without_shadow = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)

    # Calculate distance transform
    alpha_without_shadow = 255 - alpha_without_shadow
    dist = cv2.distanceTransform(alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3)  # dist l1 : L1 , dist l2 : l2

    # Modify distance matrix based on mask size
    masked = change_matrix(dist, mask_size)
    masked_alpha = (masked * 255).astype(np.uint8)

    # Create stroke image
    h, w, _ = padded_img.shape
    colors = (255, 255, 255)
    stroke_b = np.full((h, w), colors[2], np.uint8)
    stroke_g = np.full((h, w), colors[1], np.uint8)
    stroke_r = np.full((h, w), colors[0], np.uint8)
    mask = cv2.merge((stroke_r, stroke_g, stroke_b, masked_alpha))

    # Modify distance matrix based on contour size
    mask = Image.fromarray(mask)
    padded_img = Image.fromarray(padded_img)

    # Combine stroke and padded_img using alpha composite
    result = Image.alpha_composite(mask, padded_img)
    alpha = Image.fromarray(alpha_without_shadow)

    return padded_img, result, alpha


def stroke_contour(img_array: np.array, mask_size: int, contour_size: int):
    """
    세그멘테이션된 차량 이미지에 윤곽선을 생성

    Args:
        image (Image.Image): Pillow image, 'RGBA'
        threshold (str): Alpha 차원 (:,:,3)의 threshold값
        stroke_size (str): 윤곽석 굵이, 픽셀 단위
        colors ((int,int,int)): 십진수 색상, 'RGB', 0-255

    Returns:
        result (PIL.Image.Image): 윤곽선을 포함한 차량 이미지, 배경 없음
        bigger_image (PIL.Image.Image): 윤곽석을 제외한 차량 이미지, 배경 없음
    """
    if contour_size < 2:
        contour_size = 2

    # Add padding to the image
    padding = mask_size
    alpha = img_array[:, :, 3]
    rgb_img = img_array[:, :, 0:3]
    padded_img = cv2.copyMakeBorder(rgb_img, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    alpha = cv2.copyMakeBorder(alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    # Merge the padded image and alpha channel
    padded_img = cv2.merge((padded_img, alpha))

    # Apply threshold to the alpha channel
    _, alpha_without_shadow = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)

    # Calculate distance transform
    alpha_without_shadow = 255 - alpha_without_shadow
    dist = cv2.distanceTransform(alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3)  # dist l1 : L1 , dist l2 : l2

    # Modify distance matrix based on contour size
    stroked = change_matrix(dist, contour_size)
    stroke_alpha = (stroked * 255).astype(np.uint8)

    # Create stroke image
    h, w, _ = padded_img.shape
    colors = (255, 255, 255)
    stroke_b = np.full((h, w), colors[2], np.uint8)
    stroke_g = np.full((h, w), colors[1], np.uint8)
    stroke_r = np.full((h, w), colors[0], np.uint8)
    stroke = cv2.merge((stroke_r, stroke_g, stroke_b, stroke_alpha))

    # Convert arrays to PIL images
    stroke = Image.fromarray(stroke)
    padded_img = Image.fromarray(padded_img)

    # Combine stroke and padded_img using alpha composite
    result = Image.alpha_composite(stroke, padded_img)

    return result


def change_matrix(input_mat, stroke_size):
    """
        Modify input matrix based on the stroke size.

        Args:
            input_mat (np.array): Input matrix to be modified.
            stroke_size (int): Width of the stroke in pixels.

        Returns:
            mat (np.array): Modified matrix.
    """

    stroke_size = stroke_size - 1
    mat = np.ones(input_mat.shape)
    check_size = stroke_size + 1.0
    mat[input_mat > check_size] = 0
    border = (input_mat > stroke_size) & (input_mat <= check_size)
    mat[border] = 1.0 - (input_mat[border] - stroke_size)
    return mat


def adjust_contour_height(mask, contour_alpha):
    """
    1. Compute the left and the right height of contour
    2. Based on computed height, make manipulated alpha mask of contour and foreground
    Args:
        mask (np.array): A binary mask of the object.
        contour_alpha (np.array): The alpha mask of the object's contour and foreground.

    Returns:
        np.array: The modified contour alpha mask.

    """

    h, w = contour_alpha.shape
    _, bw = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find edges in the alpha mask
    edges, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Get the number of edges and sort them in descending order
    edge_lengths = [len(edge) for edge in edges]
    sorted_index = [edge_lengths.index(x) for x in sorted(edge_lengths, reverse=True)]

    # Select the edge to adjust
    try:
        first_idx, second_idx = sorted_index[:2]
        if (edges[first_idx][0, 0, 0] == 0) | (edges[first_idx][0, 0, 1] == 0):
            edge = edges[second_idx]
        else:
            edge = edges[first_idx]
    except ValueError:
        edge = edges[0]

    model_left = xgb.XGBRegressor()
    model_right = xgb.XGBRegressor()
    model_left.load_model('./xgb_model/outline_left.model')
    model_right.load_model('./xgb_model/outline_right.model')

    # Preprocess contour data for XGBoost models
    hundred_idx = np.rint(np.linspace(0, len(edge) - 1, 200)).astype('uint16')
    x_array = (edge[hundred_idx, 0, 0].flatten() - edge[:, 0, 0].min()) / (
            edge[:, 0, 0].max() - edge[:, 0, 0].min())
    y_array = edge[hundred_idx, 0, 1].flatten() / (edge[:, 0, 0].max() - edge[:, 0, 0].min())

    # Predict the heights to adjust for left and right sides
    features = np.append(x_array, y_array).reshape((1, 400))
    y_left = np.rint(model_left.predict(features) * (edge[:, 0, 0].max() - edge[:, 0, 0].min()))
    y_right = np.rint(model_right.predict(features) * (edge[:, 0, 0].max() - edge[:, 0, 0].min()))

    # Find non-zero pixels in the contour alpha mask
    y, x = np.where(contour_alpha != 0)

    # Calculate height and width offsets
    height_offset = (y.max() - y.min()) * 2 // 3
    width_offset = (x.max() - x.min()) // 20

    # Adjust the contour alpha mask
    contour_alpha[y_left[0].astype('int'):, :w // 2] = 0
    contour_alpha[y_right[0].astype('int'):, w // 2:] = 0
    contour_alpha[y.min() + height_offset:, x.min() + width_offset:x.max() - width_offset] = 0

    return contour_alpha


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

# def draw_axis(img: np.ndarray, start_point: Tuple[float, float], end_point: Tuple[float, float], color: Tuple[int, int, int], scale: float) -> Tuple[List[float], float]:
#     """
#     Draws an arrow representing an axis from a start point to an end point in the given image.
#
#     Args:
#         img: The image in which to draw the arrow.
#         start_point: A tuple representing the (x, y) coordinates of the start point of the axis.
#         end_point: A tuple representing the (x, y) coordinates of the end point of the axis.
#         color: A tuple representing the (R, G, B) values of the color of the arrow.
#         scale: A float representing the length of the arrow in relation to the length of the axis.
#
#     Returns:
#         A tuple containing a list representing the end point of the arrow and a float representing the length of the axis.
#
#     """
#     start = list(start_point)
#     end = list(end_point)
#
#     angle = np.arctan2(start[1] - end[1], start[0] - end[0])  # angle in radians
#     hypotenuse = np.sqrt((start[1] - end[1]) ** 2 + (start[0] - end[0]) ** 2)
#     # Here we lengthen the arrow by a factor of scale
#     end[0] = start[0] - scale * hypotenuse * np.cos(angle)
#     end[1] = start[1] - scale * hypotenuse * np.sin(angle)
#     # cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, 1, cv2.LINE_AA)
#     return end, hypotenuse
#
#
# def colour_gradient(img1):
#     array_1d = np.append(np.linspace(0.5, 1, img1.shape[0]//3*2), np.linspace(0.1, 1, img1.shape[0]//3))
#     mask1 = np.repeat(np.tile(array_1d, (img1.shape[1], 1))[:, :, np.newaxis], 3, axis=2)
#     mask1 = np.rot90(mask1, 3)
#     result = mask1 * img1[:,:,:3]
#     # cv2.imshow('', result)
#     # cv2.waitKey(0)
#     result = np.append(result.astype('uint8'), np.ones((img1.shape[0], img1.shape[1], 1), dtype='uint8')*255, axis=2)
#     return result
#

# def get_orientation(pts: np.ndarray, img: np.ndarray, scale: float) -> Tuple[Tuple[int, int], Tuple[List[float], float], Tuple[List[float], float], List[float]]:
#     """
#     Determines the orientation of an object based on its points in an image.
#
#     Args:
#         pts: A NumPy array of shape (N, 1, 2) representing the points of the object.
#         img: The image in which the object is located.
#         scale: A float representing the length of the arrow in relation to the length of the axis.
#
#     Returns:
#         A tuple containing the center of the object as a tuple of (x, y) coordinates, the end points of the two axes as tuples of (x, y) coordinates, and the radii of the axes as a list of floats.
#
#     """
#     sz = len(pts)
#     data_points = np.empty((sz, 2), dtype=np.float64)
#     for i in range(data_points.shape[0]):
#         data_points[i, 0] = pts[i, 0, 0]
#         data_points[i, 1] = pts[i, 0, 1]
#     # Perform PCA analysis
#     mean = np.empty((0))
#     mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_points, mean)
#     # Store the center of the object
#     center = (int(mean[0, 0]), int(mean[0, 1]))
#
#     cv2.circle(img, center, 3, (255, 0, 255), 2)
#     end1 = (center[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], center[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
#     end2 = (center[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], center[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
#     end_point1, radius1 = draw_axis(img, center, end1, (0, 255, 0), scale)
#     end_point2, radius2 = draw_axis(img, center, end2, (255, 255, 0), scale)
#     angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) # orientation in radians
#
#     return center, end_point1, end_point2, angle, [radius1, radius2]
