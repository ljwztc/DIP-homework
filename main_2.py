#!/usr/bin/python
######################################################
#  this is a code for digital image process lab_2
#  this program generate a image with salt and pepper noise
#  and filter it
######################################################

import cv2
import numpy as np


def salt_pepper_noise(jay_chou, percent):
    height, width = jay_chou.shape
    jay_chou_noise = np.random.rand(height, width)
    jay_chou_noise[jay_chou_noise > 1 - percent/2] = 1
    jay_chou_noise[jay_chou_noise < percent/2] = 2
    processed_image = jay_chou
    processed_image[jay_chou_noise == 1] = 255  # salt
    processed_image[jay_chou_noise == 2] = 0    # pepper
    return processed_image


def average_filter(jay_chou, kernel_size):
    filter_average = np.ones([kernel_size, kernel_size])
    filter_average = filter_average / kernel_size / kernel_size
    processed_image = cv2.filter2D(jay_chou, -1, filter_average)
    return processed_image


def median_filter(jay_chou, kernel_sieze):
    height, width = jay_chou.shape
    processed_image = np.ones([height, width])
    for i in range(height):
        for ii in range(width):
            if(i < kernel_sieze / 2 or i > height - kernel_sieze / 2
                    or ii < kernel_sieze / 2 or ii > width - kernel_sieze / 2):
                processed_image[i, ii] = jay_chou[i, ii]
            else:
                middle = jay_chou[i - kernel_sieze / 2: i + kernel_sieze / 2,\
                        ii - kernel_sieze / 2: ii + kernel_sieze / 2]
                row, column = middle.shape
                series = []
                for j in range(row):
                    for jj in range(column):
                        series.append(middle[j, jj])
                series.sort()
                median = series[row * column / 2]
                processed_image[i, ii] = median
    return processed_image


filepath = "/Users/mac/Downloads/jay_chou.jpg"
jay_chou = cv2.imread(filepath, 0)

cv2.imshow("initial image", jay_chou)

jay_chou_noisy = salt_pepper_noise(jay_chou, 0.30)
cv2.imshow("noisy image", jay_chou_noisy)

jay_chou_filtered_average_4 = average_filter(jay_chou_noisy, 4)
jay_chou_filtered_average_8 = average_filter(jay_chou_noisy, 8)
cv2.imshow("average filtered_4 image", jay_chou_filtered_average_4)
cv2.imshow("average filtered_8 image", jay_chou_filtered_average_8)

jay_chou_filtered_median_2 = median_filter(jay_chou, 2).astype(np.uint8)
jay_chou_filtered_median_4 = median_filter(jay_chou, 4).astype(np.uint8)
cv2.imshow("median filtered_2 image", jay_chou_filtered_median_2)
cv2.imshow("median filtered_4 image", jay_chou_filtered_median_4)

cv2.waitKey(0)
cv2.destroyAllWindows()