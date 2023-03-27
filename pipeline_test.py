import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from Utils.u2net_bg import remove
from Utils.preprocess import stroke_contour, stroke_mask, adjust_outline_height

top_y_ratio = 1
mask_y_ratio = 0.525
car_y_ratio = 0.65
background_path = './background/background.png'

for i in tqdm(range(1)):
    i = './bakcha2/bakcha/117_8600/1.png'
    img_path = i
    process_path = i.split(".")[0] + '_process.png'
    mask_path = i.split(".")[0] + '_mask.png'
    orig_path = i.split(".")[0] + '_orig.png'
    visual_path = i.split(".")[0] + '_visualize.png'

    # 세그멘테이션 및 정사각형 이미지 생성
    car = remove(Image.open(i), alpha_matting=True, post_process_mask=True, model_name='InSPyReNet',
                 size=1024)  # model_name='u2car_v2.1', size=320 model_name='InSPyReNet', size=1024
    car_array = np.array(car)

    # Foreground(자동차) 이미지에서 상하, 좌우에 위치한 여백을 지운다.
    # 결과물인 car_full_array는 직사각형 이미지에 자동차가 여백없이 채워진 형태이다.
    y, x = np.where(car_array[:, :, 3] > 0)
    car_full_array = car_array[y.min():y.max(), x.min():x.max(), :]

    # 마스크 및 윤곽선 생성
    image_size = car_full_array.shape[:2]
    mask_size = image_size[0] // 7
    contour_size = np.max(image_size) // 80 if np.max(image_size) < 2048 else 2048 // 108
    foreground_mask, foreground, foreground_alpha = stroke_mask(car_array, threshold=0,
                                                                mask_size=car_array.shape[0] // 8,
                                                                colors=(230, 230, 230))
    foreground_contour = stroke_contour(car_array, threshold=0, mask_size=mask_size,
                                        colors=(230, 230, 230),
                                        contour_size=contour_size)  # 윤곽선을 포함한 차량 이미지, 배경 없음 (N x N x 4), 'RGBA'

    ## 배경 생성
    img = Image.open(background_path).convert('RGB')  # + img_path.split('\\')[-1]
    img_array = np.array(img)

    foreground_resized, background_resized = scale_for_paste(img_array, np.array(foreground), 4, 255, car_y_ratio, True)
    image = Image.alpha_composite(Image.fromarray(background_resized), Image.fromarray(
        foreground_resized))  # 합성된 차량 이미지, 디지털 이미지 배경 (N x N x 4), N = argmax(H,W)+stroke_size, 'RGBA'

    ## 마스크 array 크기 조정
    mask_resized, _ = scale_for_paste(img_array, np.array(foreground_mask), 4, 255, car_y_ratio, True)

    ## 윤곽선 array 및 foreground alpha array 크기 조정
    white_image = np.ones(img_array.shape, dtype='uint8') * 255
    contour_resized, _ = scale_for_paste(white_image, np.expand_dims(np.array(foreground_contour)[:, :, 3], axis=2), 1,
                                         255, car_y_ratio, True)
    alpha_resized, _ = scale_for_paste(white_image, np.expand_dims(foreground_alpha, axis=2), 1, 0, car_y_ratio,
                                       True)  # (N x N x 1)

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
    mask_[:k.min() * top_y_ratio, :] = 255
    mask_[m, n] = 255

    mask = Image.fromarray(mask_)

    mask_reverse = np.expand_dims(np.where(mask_ < 128, 255, 0).astype('uint8'), axis=2)
    mask_ = np.append(np.tile(np.expand_dims(mask_, axis=2), reps=[1, 1, 3]), mask_reverse,
                      axis=-1)  # , np.ones((mask_.shape[0], mask_.shape[1], 1), dtype='uint8') * 255, axis = 2) // ,
    mask = Image.fromarray(mask_, mode='RGBA')

    image.save(process_path)
    mask.save(mask_path)
    Image.fromarray(foreground_resized).save(orig_path)
    Image.alpha_composite(image, mask).save(visual_path)
