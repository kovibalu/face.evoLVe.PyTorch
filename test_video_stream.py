import cv2
import numpy as np
from tqdm import tqdm


def main():
    cap = cv2.VideoCapture(0)

    for i in tqdm(range(100)):
        ret, image_np = cap.read()
        cv2.imwrite('{}-stream.jpg'.format(i), image_np)


if __name__ == '__main__':
    main()
