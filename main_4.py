# coding:utf-8
# 生成一幅带噪声的运动模糊图像，并采用逆滤波进行恢复

import cv2
import numpy as np
import matplotlib.pyplot as plt

def move(img):
    long, width = img.shape
    img_fre = np.fft.fftshift(np.fft.fft2(img))
    a = 0.05
    b = 0.05
    T = 0.5
    for i in range(long):
        for ii in range(width):
            if i == long / 2 and ii == width / 2:
                img_fre[i, ii] = img_fre[i, ii]
            else:
                img_fre[i, ii] = img_fre[i, ii] * (T / np.pi / ((i - long / 2) * a + (ii - width / 2) * b) * np.sin(np.pi * ((i - long / 2) * a + (ii - width / 2) * b)) * np.exp(-np.pi * 1j * ((i - long / 2) * a + (ii - width / 2) * b)))
    return img_fre

def noise(img):
    long, width = img.shape
    noisy = np.random.rand(long, width)
    noisy = noisy * (50 / 255 * np.max(img)) ** 0.5
    img = img + noisy
    return img


def inverse_filter(img):
    long, width = img.shape
    img_fre = np.fft.fftshift(np.fft.fft2(img))
    a = 0.05
    b = 0.05
    T = 0.5
    for i in range(long):
        for ii in range(width):
            if i == long / 2 and ii == width / 2:
                img_fre[i, ii] = img_fre[i, ii]
            else:
                img_fre[i, ii] = img_fre[i, ii] / (T / np.pi / ((i - long / 2) * a + (ii - width / 2) * b) * np.sin(np.pi * ((i - long / 2) * a + (ii - width / 2) * b)) * np.exp(-np.pi * 1j * ((i - long / 2) * a + (ii - width / 2) * b)))
    return img_fre

def normalization(img):
    max = np.max(img)
    min = np.min(img)
    long, width = img.shape
    for i in range(long):
        for ii in range(width):
            img[i][ii] = (img[i][ii] - min) / (max - min) * 255
    return img

imagepath = "/Users/mac/Desktop/flower.jpg"
img = cv2.imread(imagepath, 0)
img_move = np.fft.ifft2(np.fft.ifftshift(move(img)))
img_inverse = np.fft.ifft2(np.fft.ifftshift(inverse_filter(img_move)))
img_move = noise(img_move)
img_inverse_n = np.fft.ifft2(np.fft.ifftshift(inverse_filter(img_move)))
img_move = normalization(np.abs(img_move)).astype(np.uint8)
img_inverse = normalization(np.abs(img_inverse)).astype(np.uint8)
img_inverse_n = normalization(np.abs(img_inverse_n)).astype(np.uint8)

cv2.imshow("move_img", img_move)
cv2.imshow("origin_img", img)
cv2.imshow("inverse_img_without_noisy", img_inverse)
cv2.imshow("inverse_img", img_inverse_n)

cv2.waitKey(0)
cv2.destroyAllWindows()

