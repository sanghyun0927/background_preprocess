import numpy as np

import cv2
from PIL import Image

import xgboost as xgb

from typing import List, Tuple


def scale_for_paste(background: np.ndarray, foreground: np.ndarray, channel_dimension: int, alpha_value: int, car_y_ratio: int, small: bool = False) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Resize and position the foreground image to match the background image, and return the modified images.

    If the width of the background image is larger than the width of the foreground image:
        1. reduce the size of the background image to match the foreground image
        2. paste the foreground image at the appropriate height in an array of the same size as the reduced background image

    If the width of the background image is smaller than the width of the foreground image:
        1. reduce the size of the foreground image to fit the background image
        2. paste the foreground image at the appropriate height in an array of the same size as the background image

    Args:
        background: A NumPy array representing the background image.
        foreground: A NumPy array representing the foreground car image.
        channel_dimension: An integer representing the dimension of the color channel.
        alpha_value: An integer representing the alpha channel value.

    Returns:
        A tuple of two NumPy arrays. The first array is the modified foreground image, and the second array is the
        modified background image.
    """

    # Scale the images to match each other's width
    if not small:
        foreground_width = foreground.shape[1]
        background_width = background.shape[1]
        if foreground_width < background_width:
            scale_factor = foreground_width / background_width
            background = cv2.resize(background, dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                                    interpolation=cv2.INTER_AREA)
        else:
            scale_factor = background_width / foreground_width
            foreground = cv2.resize(foreground, dsize=(0, 0), fx=scale_factor, fy=scale_factor,interpolation=cv2.INTER_AREA)

            # If the foreground image is grayscale, add a third dimension for compatibility with the background image
            if len(foreground.shape) == 2:
                foreground = np.expand_dims(foreground, axis=2)
            foreground[:, :, channel_dimension - 1] = np.where(foreground[:, :, channel_dimension - 1] < 128, 0, 255)

        # Create a new array to hold the resized foreground image
        foreground_for_pasting = np.ones((background.shape[0], background.shape[1], channel_dimension),
                                         dtype='uint8') * alpha_value
        foreground_for_pasting[:, :, channel_dimension - 1] = 255 - alpha_value

        # If the foreground image is grayscale, add a third dimension for compatibility with the background image
        if len(foreground.shape) == 2:
            foreground = np.expand_dims(foreground, axis=2)

        # Find the y-position and height of the car in the foreground image
        i, _ = np.where(foreground[:, :, channel_dimension - 1] == alpha_value)
        car_height = (i.max() - i.min()) // 2
        car_y_pos = int(car_y_ratio * background.shape[0])
        assert car_y_pos >= car_height, print('입력 이미지의 세로 길이가 제한 공간을 초과 하였습니다. 입력 이미지의 세로 여백을 잘라내고 다시 시도하세요.')

        # Copy the car from the foreground image to the foreground_for_pasting array, locate to specified y coordinate
        if car_y_pos + car_height > foreground_for_pasting.shape[0]:
            y_low_limit_idx = foreground_for_pasting.shape[0] * 99 // 100
            y_high_limit_idx = y_low_limit_idx - i.max() + i.min()
            foreground_for_pasting[y_high_limit_idx:y_low_limit_idx, :, :] = foreground[i.min():i.max(), :, :]
        else:
            if (i.max() - i.min()) % 2 == 0:
                foreground_for_pasting[(car_y_pos - car_height):(car_y_pos + car_height), :, :] = foreground[i.min():i.max(), :, :]
            else:
                foreground_for_pasting[(car_y_pos - car_height - 1):(car_y_pos + car_height), :, :] = foreground[i.min():i.max(), :, :]

        white_channel = np.ones((background.shape[0], background.shape[1], 1), dtype='uint8') * 255
        background = np.concatenate([background, white_channel], axis=2)

    else:
        foreground_width = foreground.shape[1]
        background_width = background.shape[1]
        if foreground_width < background_width:
            scale_factor = foreground_width / background_width * (7/6)
            background = cv2.resize(background, dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                                    interpolation=cv2.INTER_AREA)
        else:
            scale_factor = background_width / foreground_width * (6/7)
            foreground = cv2.resize(foreground, dsize=(0, 0), fx=scale_factor, fy=scale_factor,interpolation=cv2.INTER_AREA)

            # If the foreground image is grayscale, add a third dimension for compatibility with the background image
            if len(foreground.shape) == 2:
                foreground = np.expand_dims(foreground, axis=2)
            foreground[:, :, channel_dimension - 1] = np.where(foreground[:, :, channel_dimension - 1] < 128, 0, 255)

        # Create a new array to hold the resized foreground image
        foreground_for_pasting = np.ones((background.shape[0], background.shape[1], channel_dimension),
                                         dtype='uint8') * alpha_value
        foreground_for_pasting[:, :, channel_dimension - 1] = 255 - alpha_value

        # Find the y-position and height of the car in the foreground image
        i, j = np.where(foreground[:, :, channel_dimension - 1] == alpha_value)
        X_width = np.rint(foreground_for_pasting.shape[1] / 14).astype('uint16')
        # X_offset = foreground_for_pasting.shape[1] % 14
        car_height = (i.max() - i.min()) // 2
        car_y_pos = int(car_y_ratio * background.shape[0])
        assert car_y_pos >= car_height, print('입력 이미지의 세로 길이가 제한 공간을 초과 하였습니다. 입력 이미지의 세로 여백을 잘라내고 다시 시도하세요.')
        # Copy the car from the foreground image to the foreground_for_pasting array, locate to specified y coordinate

        if car_y_pos + car_height > foreground_for_pasting.shape[0]:
            y_low_limit_idx = foreground_for_pasting.shape[0] * 99 // 100
            if y_low_limit_idx > (i.max() - i.min()):
                y_high_limit_idx = y_low_limit_idx - i.max() + i.min()
                foreground_for_pasting[y_high_limit_idx:y_low_limit_idx, X_width:(X_width+foreground.shape[1]), :] = foreground[i.min():i.max(), :, :]
            else:
                foreground_for_pasting[0:y_low_limit_idx, X_width:(X_width + foreground.shape[1]),:] = foreground[(i.max()-y_low_limit_idx):i.max(), :, :]

        elif (i.max() - i.min()) % 2 == 0:
            foreground_for_pasting[(car_y_pos - car_height):(car_y_pos + car_height), X_width:(X_width+foreground.shape[1]), :] = \
                foreground[i.min():i.max(), :, :]
        else:
            foreground_for_pasting[(car_y_pos - car_height - 1):(car_y_pos + car_height), X_width:(X_width+foreground.shape[1]), :] = \
                foreground[i.min():i.max(), :, :]
        white_channel = np.ones((background.shape[0], background.shape[1], 1), dtype='uint8') * 255
        background = np.concatenate([background, white_channel], axis=2)

    return foreground_for_pasting, background


def stroke_mask(img_array: np.array, threshold: int, mask_size: int, colors: (int, int, int)):
    """
    세그멘테이션된 차량 이미지 외곽에 매우 굵은 윤곽선을 생성하여 Stable diffusion inpaint 모델의 마스크로 사용함

    Args:
        image (Image.Image): Pillow image, 'RGBA'
        threshold (str): Alpha 차원 (:,:,3)의 threshold값
        stroke_size (str): 윤곽석 굵이, 픽셀 단위
        colors ((int,int,int)): 십진수 색상, 'RGB', 0-255

    Returns:
        result (PIL.Image.Image): 윤곽선을 포함한 차량 이미지, 배경 없음
        bigger_image (PIL.Image.Image): 윤곽석을 제외한 차량 이미지, 배경 없음
    """

    h, w, _ = img_array.shape
    pad_int = int(0)
    padding = mask_size + pad_int
    alpha = img_array[:, :, 3]
    rgb_img = img_array[:, :, 0:3]
    bigger_img = cv2.copyMakeBorder(rgb_img, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    alpha = cv2.copyMakeBorder(alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    bigger_img = cv2.merge((bigger_img, alpha))
    h, w, _ = bigger_img.shape

    _, alpha_without_shadow = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)  # threshold=0 in photoshop
    alpha_without_shadow = 255 - alpha_without_shadow
    dist = cv2.distanceTransform(alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3)  # dist l1 : L1 , dist l2 : l2
    stroked = change_matrix(dist, mask_size)
    stroke_alpha = (stroked * 255).astype(np.uint8)

    stroke_b = np.full((h, w), colors[2], np.uint8)
    stroke_g = np.full((h, w), colors[1], np.uint8)
    stroke_r = np.full((h, w), colors[0], np.uint8)
    stroke = cv2.merge((stroke_r, stroke_g, stroke_b, stroke_alpha))

    stroke = Image.fromarray(stroke)
    bigger_img = Image.fromarray(bigger_img)
    result = Image.alpha_composite(stroke, bigger_img)
    return result, bigger_img, alpha_without_shadow


def stroke_contour(img_array: np.array, threshold: int, mask_size: int, colors: (int, int, int), contour_size: int):
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

    h, w = img_array.shape[:2]
    pad_int = int(0)
    padding = mask_size + pad_int
    alpha = img_array[:, :, 3]
    rgb_img = img_array[:, :, 0:3]
    bigger_img = cv2.copyMakeBorder(rgb_img, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    alpha = cv2.copyMakeBorder(alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    bigger_img = cv2.merge((bigger_img, alpha))
    h, w, _ = bigger_img.shape

    _, alpha_without_shadow = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)  # threshold=0 in photoshop
    alpha_without_shadow = 255 - alpha_without_shadow
    dist = cv2.distanceTransform(alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3)  # dist l1 : L1 , dist l2 : l2
    stroked = change_matrix(dist, contour_size)
    stroke_alpha = (stroked * 255).astype(np.uint8)

    stroke_b = np.full((h, w), colors[2], np.uint8)
    stroke_g = np.full((h, w), colors[1], np.uint8)
    stroke_r = np.full((h, w), colors[0], np.uint8)
    stroke = cv2.merge((stroke_r, stroke_g, stroke_b, stroke_alpha))

    stroke = Image.fromarray(stroke)
    bigger_img = Image.fromarray(bigger_img)
    result = Image.alpha_composite(stroke, bigger_img)
    return result


def change_matrix(input_mat, stroke_size):
    stroke_size = stroke_size - 1
    mat = np.ones(input_mat.shape)
    check_size = stroke_size + 1.0
    mat[input_mat > check_size] = 0
    border = (input_mat > stroke_size) & (input_mat <= check_size)
    mat[border] = 1.0 - (input_mat[border] - stroke_size)
    return mat


def adjust_outline_height(mask, contour_alpha):
    h, w, _ = contour_alpha.shape
    _, bw = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contour_lengths = [len(contour) for contour in contours]
    sorted_index = [contour_lengths.index(x) for x in sorted(contour_lengths, reverse=True)]

    try:
        first_idx, second_idx = sorted_index[:2]
        if (contours[first_idx][0, 0, 0] == 0) | (contours[first_idx][0, 0, 1] == 0):
            contour = contours[second_idx]
        else:
            contour = contours[first_idx]
    except ValueError:
        contour = contours[0]

    model_left = xgb.XGBRegressor()
    model_right = xgb.XGBRegressor()
    model_left.load_model('./xgb_model/outline_left.model')
    model_right.load_model('./xgb_model/outline_right.model')

    hundred_idx = np.rint(np.linspace(0, len(contour) - 1, 200)).astype('uint16')
    x_array = (contour[hundred_idx, 0, 0].flatten() - contour[:, 0, 0].min()) / (
            contour[:, 0, 0].max() - contour[:, 0, 0].min())
    y_array = contour[hundred_idx, 0, 1].flatten() / (contour[:, 0, 0].max() - contour[:, 0, 0].min())

    features = np.append(x_array, y_array).reshape((1, 400))
    y_left = np.rint(model_left.predict(features) * (contour[:, 0, 0].max() - contour[:, 0, 0].min()))
    y_right = np.rint(model_right.predict(features) * (contour[:, 0, 0].max() - contour[:, 0, 0].min()))

    i, j, _ = np.where(contour_alpha != 0)
    height_offset = (i.max() - i.min()) // 2
    width_offset = (j.max() - j.min()) // 20

    contour_alpha[y_left[0].astype('int'):, :w // 2] = 0
    contour_alpha[y_right[0].astype('int'):, w // 2:] = 0
    contour_alpha[i.min() + height_offset:, j.min() + width_offset:j.max() - width_offset] = 0

    return contour_alpha

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
