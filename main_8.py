# coding:utf-8
# describe: 灰度共生矩阵
import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalization(img):
    max = np.max(img)
    min = np.min(img)
    long, width = img.shape
    for i in range(long):
        for ii in range(width):
            img[i][ii] = (img[i][ii] - min) / (max - min) * 255
    img = img.astype(np.uint8)
    return img


img_path = 'flower.jpg'
img = cv2.imread(img_path, 0)
cv2.imshow("original image", img)

GLCM = np.zeros([256, 256])

a, b = img.shape
for i in range(a):
    for j in range(b - 1):
        x = img[i][j]
        y = img[i][j + 1]
        GLCM[x][y] += 1

GLCM = GLCM / np.sum(GLCM)

np.savetxt('GLMC.txt', GLCM, fmt='%.3f')

GLCM = normalization(GLCM)

cv2.imshow("GLMC", GLCM)


cv2.waitKey(0)
cv2.destroyAllWindows()