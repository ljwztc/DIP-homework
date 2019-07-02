#!/usr/bin/python

import numpy as np
import cv2
import time

imagepath = "/Users/mac/Downloads/220px-Lenna_(test_image).png"


def func_nearestneighbor(img, processed_height, processed_width):
    height, width = img.shape
    new_img = np.zeros((processed_height, processed_width, 1), np.uint8)
    for i in range(processed_height):
        for j in range(processed_width):
            raw_row = i * height / processed_height
            raw_column = j * width / processed_width
            new_img[i, j] = img[int(raw_row), int(raw_column)]
    return new_img


def func_bilinear(img, processed_height, processed_width):
    height, width = img.shape
    new_img = np.zeros((processed_height, processed_width, 1), np.uint8)
    for i in range(processed_height):
        for j in range(processed_width):
            raw_row = i * 1.0 * height / processed_height
            raw_column = j * 1.0 * width / processed_width
            column = int(raw_column)
            row = int(raw_row)
            a = raw_column - column
            b = raw_row - row
            if column < (width - 1) and row < (height - 1):
                new_img[i, j] = (1 - b) * a * img[row, column + 1] + (1 - b) * (1 - a) * img[row, column] +\
                                b * a * img[row + 1, column + 1] + b * (1 - a) * img[row + 1, column]
            else:
                new_img[i, j] = img[row, column]
    return new_img


def func_bicubic(img, processed_height, processed_width):
    height, width = img.shape
    new_img = np.zeros((processed_height, processed_width, 1), np.uint8)
    xy = np.mat(np.zeros((16, 16)))
    xy[0, 0] = 1
    for l in range(4):
        xy[1, l] = 1
        xy[2, l] = 2 ** l
        xy[3, l] = 3 ** l
        xy[4, l * 4] = 1
        xy[8, l * 4] = 2 ** l
        xy[12, l * 4] = 3 ** l
    for l in range(16):
        xy[5, l] = 1
        xy[6, l] = 2 ** (l % 4)
        xy[7, l] = 3 ** (l % 4)
        xy[9, l] = 2 ** (l / 4)
        xy[10, l] = 2 ** (l / 4) * 2 ** (l % 4)
        xy[11, l] = 2 ** (l / 4) * 3 ** (l % 4)
        xy[13, l] = 3 ** (l / 4)
        xy[14, l] = 3 ** (l / 4) * 2 ** (l % 4)
        xy[15, l] = 3 ** (l / 4) * 3 ** (l % 4)
    for i in range(processed_height):
        for j in range(processed_width):
            raw_row = i * 1.0 * height / processed_height
            raw_column = j * 1.0 * width / processed_width
            x = int(raw_row)
            y = int(raw_column)
            if x < (width - 2) and x >= 1 and y < (height - 2) and y >= 1:
                v = np.mat([
                    img[x - 1, y - 1], img[x - 1, y], img[x - 1, y + 1], img[x -1, y +2],
                    img[x, y - 1], img[x, y], img[x, y + 1], img[x, y +2],
                    img[x + 1, y - 1], img[x + 1, y], img[x + 1, y + 1], img[x + 1, y + 2],
                    img[x + 2, y - 1], img[x + 2, y], img[x + 2, y + 1], img[x + 2, y + 2]])
                a = xy.I * v.T
                realx = raw_row - x + 1
                realy = raw_column - y + 1
                real_location = np.mat([1, realy, realy ** 2, realy ** 3,
                                        realx, realx * realy, realx * realy ** 2, realx * realy ** 3,
                                        realx ** 2, realx ** 2 * realy, realx ** 2 * realy ** 2, realx ** 2 * realy ** 3,
                                        realx ** 3, realx ** 3 * realy, realx ** 3 * realy ** 2, realx ** 3 * realy ** 3])
                new_img[i, j] = real_location * a
            else:
                new_img[i, j] = img[x, y]

    return new_img


def func_sample(img):
    height, width = img.shape
    new_img = np.zeros((height / 2, width / 2), np.uint8)
    for i in range(height / 2):
        for j in range(width / 2):
            raw_row = i * 2
            raw_column = j * 2
            if raw_column >= width:
                new_img[i, j] = img[raw_row, raw_column - 1]
            elif raw_row >= height:
                new_img[i, j] = img[raw_row - 1, raw_column]
            else:
                new_img[i, j] = img[raw_row, raw_column]
    return new_img


img = cv2.imread(imagepath, 0)  # read the image as the grey picture.
image_sample = func_sample(img)
image_sample2 = func_sample(image_sample)

time_start=time.time()
image_nearestneighbor = func_nearestneighbor(image_sample2, 550, 550)
time_end=time.time()
time_nearestneighbor = time_end - time_start

time_start=time.time()
image_bilinear = func_bilinear(image_sample2, 550, 550)
time_end=time.time()
time_bilinear = time_end - time_start

time_start=time.time()
image_bicubic= func_bicubic(image_sample2, 550, 550)
time_end=time.time()
time_bicubic = time_end - time_start

image_bilinear2 = func_bilinear(image_sample, 550, 550)

cv2.imshow("raw_image", img)
cv2.imshow("simple", image_sample)
cv2.imshow("simple2", image_sample2)
cv2.imshow("nearest_neightbor", image_nearestneighbor)
cv2.imshow("bilinear", image_bilinear)
cv2.imshow("bicubic", image_bicubic)
cv2.imshow("bilinear2", image_bilinear2)

print 'time_nearestneighbor = ' + str(time_nearestneighbor)
print 'time_bilinear = ' + str(time_bilinear)
print 'time_bicubic = ' + str(time_bicubic)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
file=open('/Users/mac/Downloads/image_nearestneighbor.txt', 'w')
for i in range(image_nearestneighbor.shape[0]):
    for j in range(image_nearestneighbor.shape[1] - 190):
        file.write(str(image_nearestneighbor[i][j]))
        file.write(' ')
    file.write('\n')
file.close()
file=open('/Users/mac/Downloads/image_bilinear.txt', 'w')
for i in range(image_bilinear.shape[0]):
    for j in range(image_bilinear.shape[1] - 190):
        file.write(str(image_bilinear[i][j]))
        file.write(' ')
    file.write('\n')
file.close()
'''


