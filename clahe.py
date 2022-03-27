import cv2
import numpy as np


def applyClahe(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(1, 1))
    v = clahe.apply(v)
    merged = cv2.merge([h, s, v])
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


if __name__ == '__main__':
    img = cv2.imread("out/simplestColorBalance.jpg")

    out = applyClahe(img)
    cv2.imwrite("out/clahe.jpg", out)
