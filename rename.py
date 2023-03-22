import os

folder = './bakcha2/bakcha/'

image_dir = os.listdir(folder)

for dir in image_dir:
    for idx, path in enumerate(os.listdir(folder + dir)):
        os.rename(os.path.join(folder, dir, path), os.path.join(folder, dir, f'{idx+1}.png'))