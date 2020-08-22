import os

import cv2 as cv
import numpy as np

PATH = '../data/train/annotations'
if __name__ == '__main__':
    names = os.listdir(PATH)
    class_1 = 0
    class_2 = 0
    class_3 = 0
    class_4 = 0
    class_5 = 0
    class_6 = 0
    class_7 = 0
    class_8 = 0
    for name in names:
        ann = cv.imread(os.path.join(PATH, name), 0)
        class_1 += np.sum(ann == 1)
        class_2 += np.sum(ann == 2)
        class_3 += np.sum(ann == 3)
        class_4 += np.sum(ann == 4)
        class_5 += np.sum(ann == 5)
        class_6 += np.sum(ann == 6)
        class_7 += np.sum(ann == 7)
        class_8 += np.sum(ann == 8)

    print(f'inflammation: {class_1}')
    print(f'low: {class_2}')
    print(f'high: {class_3}')
    print(f'carcinoma: {class_4}')
    print(f'indefinite1: {class_5}')
    print(f'indefinite2: {class_6}')
    print(f'indefinite3: {class_7}')
    print(f'squash: {class_8}')
