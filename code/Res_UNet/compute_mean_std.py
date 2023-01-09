import os
import numpy as np
from PIL import Image


def get_filelist(path):
    filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".png"):
                filelist.append(os.path.join(home, filename))
    return filelist


def main(img_dir, img_channels):
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    img_name_list = get_filelist(img_dir)
    print(f"total files: {len(img_name_list)}\n")
    accumulated_mean = np.zeros(img_channels)
    accumulated_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img = np.array(Image.open(img_name))
        img = np.reshape(img, (-1, img.shape[-1]))
        accumulated_mean += img.mean(axis=0)
        accumulated_std += img.std(axis=0)
    mean = accumulated_mean / len(img_name_list)
    std = accumulated_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    img_dir = r"..\..\dataset\leftImg8bit"
    img_channels = 3
    main(img_dir, img_channels)


# mean: (121.90957298, 130.36595824, 117.22788076)
# std: (38.19792449, 35.57061731, 41.42008869)
