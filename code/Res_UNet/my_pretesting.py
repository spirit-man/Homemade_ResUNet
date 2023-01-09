import numpy as np
import cv2
import os
from distutils.log import error
from PIL import Image


def count_labels(data_path = r"..\..\dataset\gtFine\test"):
    assert os.path.exists(data_path), f"path '{data_path}' does not exists."
    img_names = [i for i in os.listdir(data_path) if i.endswith(".png")]
    img_list = [os.path.join(data_path, i)for i in img_names]
    mp = dict()
    for img in img_list:
        original_img = np.array(Image.open(img).convert('L'))
        for i in range(original_img.shape[0]):
            for j in range(original_img.shape[1]):
                if original_img[i, j] in mp.keys():
                    mp[original_img[i, j]] += 1
                else:
                    mp[original_img[i, j]] = 1
    for x in mp:
        print("label ", x, ": ", mp[x])
    print("total_labels: {}".format(len(mp)))


def find_min_max_pixel(data_path = r"..\..\dataset\gtFine\test"):
    assert os.path.exists(data_path), f"path '{data_path}' does not exists."
    img_names = [i for i in os.listdir(data_path) if i.endswith(".png")]
    img_list = [os.path.join(data_path, i)for i in img_names]
    total_pixel = set()
    for img in img_list:
        original_img = np.array(Image.open(img))
        total_pixel.add(np.min(original_img))
        total_pixel.add(np.max(original_img))
    print(f"min_pixel:{min(total_pixel)}\nmax_pixel:{max(total_pixel)}")


def read_img(img, data_path = r"..\..\dataset\gtFine\train"):
    img_path = os.path.join(data_path, img)
    assert os.path.exists(img_path), f"path '{img_path}' does not exists."
    original_img = np.array(Image.open(img_path))
    # original_img = cv2.cvtColor(
    #     original_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    print(f"shape:{original_img.shape}")
    print(f"min:{np.min(original_img)}")
    print(f"max:{np.max(original_img)}")
    cv2.imshow(str(img), original_img)
    cv2.waitKey(0)


def main():
    while True:
        f = input("count_labels or find_min_max_pixel or read_img or exit:\n")
        if f == "count_labels":
            count_labels()
        elif f == "find_min_max_pixel":
            find_min_max_pixel()
        elif f == "read_img":
            read_img(input("image name:\n"))
        elif f == "exit":
            break
        else:
            raise error("illegal input!")


if __name__ == '__main__':
    main()
