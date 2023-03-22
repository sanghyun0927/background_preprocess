import os
from glob import glob
import shutil

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from Utils.u2net_bg import remove
from Utils.preprocess import stroke_contour, stroke_mask, scale_for_paste, adjust_outline_height

top_y_ratio = 1
mask_y_ratio = 0.525
car_y_ratio = 0.65
keyword = './background/background.png'

cwd = os.getcwd()
image_dirs = glob(cwd + "\\bakcha2\\vancar\\*\\", recursive = True)
img_paths = []


for dir in image_dirs:
    img_path = os.listdir(dir)
    for path in img_path:
        if 'mask' in path:
            continue
        if 'orig' in path:
            continue
        if 'process' in path:
            continue
        if 'visual' in path:
            continue
        img_paths.append(dir + path)

for i in tqdm(img_paths):
    print(i)
    img_path = i
    process_path = i.split(".")[0] + '_process.png'
    mask_path = i.split(".")[0] + '_mask.png'
    orig_path = i.split(".")[0] + '_orig.png'
    visual_path = i.split(".")[0] + '_visualize.png'

    # 세그멘테이션 및 정사각형 이미지 생성
    car = np.array(remove(Image.open(i), alpha_matting=True, post_process_mask=True, model_name='InSPyReNet', size=1024))    #model_name='u2car_v2.1', size=320 model_name='InSPyReNet', size=1024

    i, j = np.where(car[:, :, 3] > 0)
    ratio = ((j.max() - j.min()) / (i.max() - i.min())) ** 2 * 0.35
    width_offset = np.rint((j.max() - j.min()) * 0.15 / ratio).astype('uint16')
    height_offset = np.rint((i.max() - i.min()) * 0.15 / ratio).astype('uint16')

    y_min = i.min()-height_offset if i.min()-height_offset > 0 else 0
    y_max = i.max()+height_offset if i.max()+height_offset < car.shape[0] else car.shape[0]
    x_min = j.min()-width_offset if j.min()-width_offset > 0 else 0
    x_max = j.max()+width_offset if j.max()+width_offset < car.shape[1] else car.shape[1]
    car = car[y_min:y_max, x_min:x_max, :]

    ## 마스크 및 윤곽선 생성
    contour_size = np.max(np.shape(car)[:2]) // 100 if np.max(np.shape(car)[:2]) < 2048 else 2048//108
    foreground_mask, foreground, foreground_alpha = stroke_mask(car, threshold=0, mask_size=car.shape[0] // 8,
                                                                colors=(230, 230, 230))
    foreground_contour = stroke_contour(car, threshold=0, mask_size=np.shape(car)[0] // 8, colors=(230, 230, 230),
                                        contour_size=contour_size)  # 윤곽선을 포함한 차량 이미지, 배경 없음 (N x N x 4), 'RGBA'

    ## 배경 생성
    img = Image.open(keyword).convert('RGB')   # + img_path.split('\\')[-1]
    img_array = np.array(img)

    foreground_resized, background_resized = scale_for_paste(img_array, np.array(foreground), 4, 255, car_y_ratio)
    image = Image.alpha_composite(Image.fromarray(background_resized), Image.fromarray(
        foreground_resized))  # 합성된 차량 이미지, 디지털 이미지 배경 (N x N x 4), N = argmax(H,W)+stroke_size, 'RGBA'

    ## 마스크 array 크기 조정
    mask_resized, _ = scale_for_paste(img_array, np.array(foreground_mask), 4, 255, car_y_ratio)

    ## 윤곽선 array 및 foreground alpha array 크기 조정
    white_image = np.ones(img_array.shape, dtype='uint8') * 255
    contour_resized, _ = scale_for_paste(white_image, np.expand_dims(np.array(foreground_contour)[:, :, 3], axis=2), 1,
                                         255, car_y_ratio)
    alpha_resized, _ = scale_for_paste(white_image, np.expand_dims(foreground_alpha, axis=2), 1, 0, car_y_ratio)   # (N x N x 1)

    ## 윤곽선 array에서 좌우 윤곽선 높이 지정
    contour_resized = adjust_outline_height(alpha_resized, contour_resized)

    ## 최초 마스크 및 윤곽선, Alpha 채널 값을 고려한 최종 마스크 생성
    i, j = np.where(mask_resized[:, :, 3] == 0)
    k, l = np.where(alpha_resized[:, :, 0] == 0)
    m, n = np.where(contour_resized[:, :, 0] == 255)

    mask_ = np.ones(mask_resized.shape[:2], dtype='uint8') * 0
    mask_[i, j] = 255
    mask_[int(mask_resized.shape[0] * mask_y_ratio):, :] = 0
    mask_[k, l] = 255
    mask_[:k.min()*top_y_ratio, :] = 255
    mask_[m, n] = 255

    mask = Image.fromarray(mask_)

    mask_reverse = np.expand_dims(np.where(mask_< 128, 255, 0).astype('uint8'), axis=2)
    mask_ = np.append(np.tile(np.expand_dims(mask_, axis=2), reps=[1, 1, 3]), mask_reverse, axis=-1) # , np.ones((mask_.shape[0], mask_.shape[1], 1), dtype='uint8') * 255, axis = 2) // ,
    mask = Image.fromarray(mask_, mode='RGBA')


    image.save(process_path)
    mask.save(mask_path)
    Image.fromarray(foreground_resized).save(orig_path)
    Image.alpha_composite(image, mask).save(visual_path)
    # Image.alpha_composite(image, Image.fromarray(foreground_resized)).save(visual_path)