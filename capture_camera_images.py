import cv2
import os
"""
pip install opencv-python==3.4.5.20
captures 100 images from front facing camera on MAC and saves the images
"""
def main():
    cap = cv2.VideoCapture(0)
    for i in range(10):
        ret, image_np = cap.read()
        filename = 'img-{}.jpg'.format(i)
        cv2.imwrite(filename,image_np)


if __name__ == '__main__':
        main()

