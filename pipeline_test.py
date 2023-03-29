import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from Utils.u2net_bg import remove
from Utils.preprocess import rescale_and_paste, stroke_contour, stroke_mask, adjust_contour_height, \
    create_manipulated_mask

cwd = os.getcwd()
image_dirs = glob(cwd + "\\bakcha2\\*\\*\\", recursive=True)
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

background_path = './background/background.png'

for i in tqdm(img_paths):
    print(i)
    process_path = i.split(".")[0] + '_process.png'
    mask_path = i.split(".")[0] + '_mask.png'
    orig_path = i.split(".")[0] + '_orig.png'
    visual_path = i.split(".")[0] + '_visualize.png'

    # 세그멘테이션 , (model_name='u2car_v2.1', size=320 / model_name='InSPyReNet', size=1024)
    car = remove(Image.open(i), post_process_mask=True, model_name='InSPyReNet', size=1024)
    car_array = np.array(car)

    # Foreground(자동차) 이미지에서 상하, 좌우에 위치한 여백을 지운다.
    # 결과물인 car_full_array는 직사각형 이미지에 자동차가 여백없이 채워진 형태이다.
    y, x = np.where(car_array[:, :, 3] > 0)
    car_full_array = car_array[y.min():y.max(), x.min():x.max(), :]

    # 마스크 및 윤곽선 생성
    image_size = car_full_array.shape[:2]
    mask_size = image_size[0] // 7
    contour_size = np.max(image_size) // 80 if np.max(image_size) < 2048 else 2048 // 108
    foreground, foreground_mask, foreground_alpha = stroke_mask(car_full_array, mask_size=mask_size)
    foreground_contour = stroke_contour(car_full_array, mask_size=mask_size, contour_size=contour_size)

    ## 배경 생성
    background = Image.open(background_path).convert('RGBA')  # + img_path.split('\\')[-1]
    # img_array = np.array(img)

    # 합성된 차량 이미지, 디지털 이미지 배경 (N x N x 4), N = argmax(H,W)+stroke_size, 'RGBA'
    img_pasted, new_foreground = rescale_and_paste(foreground, background, only_rescale=False)

    # 마스크 array 크기 조정
    mask_resized = rescale_and_paste(foreground_mask, background, only_rescale=True)
    mask_resized_np = np.array(mask_resized)

    # 윤곽선 array 크기 조정 및 윤곽선 array의 alpha 채널 추출
    contour_resized = rescale_and_paste(foreground_contour, background, only_rescale=True)
    contour_alpha_np = np.array(contour_resized)[:, :, 3]

    # foreground alpha array 크기 조정
    alpha_resized = rescale_and_paste(foreground_alpha, background, only_rescale=True)
    alpha_resized_np = np.array(alpha_resized)

    # rescale_and_paste 함수를 통일하여 사용하는 과정에서 binary image 인 alpha_resized는 추가적 코딩이 필요하게 되었다.
    y, x = np.where(alpha_resized_np[:, :, 3] == 0)
    alpha_resized_np[y, x, :] = 255
    alpha_resized_np = alpha_resized_np[:, :, 0]

    # 윤곽선 array에서 좌우 윤곽선 높이 지정
    contour_modified = adjust_contour_height(alpha_resized_np, contour_alpha_np)

    # 최종 입력 마스크 이미지 생성
    mask = create_manipulated_mask(mask_resized_np, alpha_resized_np, contour_modified)

    img_pasted.save(process_path)
    mask.save(mask_path)
    new_foreground.save(orig_path)
    Image.alpha_composite(img_pasted, mask).save(visual_path)
