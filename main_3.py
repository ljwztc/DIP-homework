#!/usr/bin/python
######################################################
#  this is a code for digital image process lab_3
#  this program improve high frequency of a blurred picture
######################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# # this function generate the template of sobel
# # The Input
# # long: long of template, so is width
# # The Output
# # filter_fre: the frequency domain of sobel template
# # filter_log: log
# # filter: the space domain of sobel template
# def Sobel_filter(long, width):
#     filter = np.zeros([long, width])
#
#     center_l = int(long / 2)
#     center_h = int(width / 2)
#     filter[center_l - 1][center_h - 1] = -1
#     filter[center_l][center_h - 1] = -2
#     filter[center_l + 1][center_h - 1] = -1
#     filter[center_l - 1][center_h + 1] = 1
#     filter[center_l][center_h + 1] = 2
#     filter[center_l + 1][center_h + 1] = 1
#     # filter[0][0] = -1
#     # filter[0][2] = 1
#     # filter[1][0] = -2
#     # filter[1][2] = 2
#     # filter[2][0] = -1
#     # filter[3][2] = 1
#     filter_shift = np.zeros([long, width])
#     for i in range(long):
#         for ii in range(width):
#             filter_shift[i][ii] = filter[i][ii] * (-1) ** (i + ii)
#     filter_fre = np.fft.fft2(filter_shift)
#     filter_log = np.log(1 + np.abs(filter_fre))
#     return filter_fre, filter_log, filter


# this function generate the template of ideal_hp_filter
# The Input
# long: long of template, so is width
# d0: the boundary of filter
# The Output
# filter: the frequency domain of sobel template
def ideal_hpfilter(long, width, d0):
    filter = np.zeros([long, width])
    center_l = int(long / 2)
    center_h = int(width / 2)
    for i in range(long):
        for ii in range(width):
            if ((i - center_h)**2 + (ii - center_l)**2)**0.5 <= d0:
                filter[i][ii] = 0
            else:
                filter[i][ii] = 1
    return filter


# this function generate the template of butterworth_hp_filter
# The Input
# long: long of template, so is width
# d0: the boundary of filter
# The Output
# filter: the frequency domain of sobel template
def butterworth_hpfilter(long, width, d0):
    filter = np.zeros([long, width])
    center_l = int(long / 2)
    center_h = int(width / 2)
    for i in range(long):
        for ii in range(width):
            if i != center_l or ii != center_h:
                filter[i][ii] = (1 / (1 + (d0 / ((i - center_l)**2 + (ii - center_h)**2)**0.5)**4))**0.5
            else:
                filter[i][ii] = 0
    return filter

# this function generate the template of gaussian_hp_filter
# The Input
# long: long of template, so is width
# d0: the boundary of filter
# The Output
# filter: the frequency domain of sobel template
def gaussian_hpfilter(long, width, d0):
    filter = np.zeros([long, width])
    center_l = int(long / 2)
    center_h = int(width / 2)
    for i in range(long):
        for ii in range(width):
                filter[i][ii] = 1 - math.exp(-((i - center_l)**2 + (ii - center_h)**2)**0.5 / (2 * d0**2))
    return filter


# this function apply high pass filter to img
# The Input
# long: long of output, so is width
# img_fre: the frequency domain of img
# filter_fre: the frequency domain of filter
# The Output
# img_ideal_high: filter result
def hp_filter(long, width, img_fre, filter_fre):
    img_ideal_fre = img_fre * filter_fre
    img_ideal_raw = np.fft.ifft2(img_ideal_fre)
    img_ideal_high = np.zeros([long, width])

    for i in range(long):
        for ii in range(width):
            img_ideal_high[i][ii] = np.real(img_ideal_raw[i][ii])
            img_ideal_high[i][ii] = img_ideal_high[i][ii] * (-1) ** (i + ii)
    return img_ideal_high


# this function calculate the energy inside d0
# The Input
# d0: the boundary of filter
# img_fre: the frequency domain of img
# The Output: result
def cal_energy_inside(img_fre, d0):
    sum = 0
    inside = 0
    long, width = img_fre.shape
    center_l = int(long / 2)
    center_h = int(width / 2)
    for i in range(long):
        for ii in range(width):
            sum += abs(img_fre[i][ii])
            if(((i - center_h)**2 + (ii - center_l)**2)**0.5 <= d0):
                inside += abs(img_fre[i][ii])
    return inside / sum


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
long, width = img.shape
img_processed = np.zeros([long * 2, width * 2])
img_shift = np.zeros([long * 2, width * 2])

# this part is used to expand the image and shift the center of frequency domain
for i in range(long):
    for ii in range(width):
        img_processed[i][ii] = img[i][ii]
        img_shift[i][ii] = img_processed[i][ii] * (-1) ** (i + ii)

# this part is used to calculate the DFT of picture
img_fre = np.fft.fft2(img_shift)
img_log = np.log(1 + np.abs(img_fre))

plt.figure(1)
plt.subplot(221), plt.imshow(img, "gray"), plt.title("original image")
plt.subplot(222), plt.imshow(img_shift, "gray"), plt.title("processed image")
plt.subplot(223), plt.imshow(abs(img_fre), "gray"), plt.title("frequency domain")
plt.subplot(224), plt.imshow(img_log, "gray"), plt.title("log_frequency domain")

# # this part is used to calculate the energy inside d0
# energy_array = np.zeros([2, 5])
# count = 0
# for i in range(130, 200, 15):
#     energy = cal_energy_inside(img_fre, i)
#     if energy > 0.5 and energy < 0.9:
#         energy_array[0][count] = i
#         energy_array[1][count] = round(energy, 2)
#         count += 1

# # this part use sobel to filter
# sobel, sobel_log, sobel_space = Sobel_filter(long * 2, width * 2)
# img_sobel_fre = img_fre * sobel
# img_sobel_raw = np.fft.ifft2(img_sobel_fre)
# img_sobel_high = np.zeros([long, width])
#
# for i in range(long):
#     for ii in range(width):
#         img_sobel_high[i][ii] = np.real(img_sobel_raw[i][ii])
#         img_sobel_high[i][ii] = img_sobel_high[i][ii] * (-1) ** (i + ii)
#
# img_sobel_space_raw = cv2.filter2D(img_processed, 1, sobel_space)
# img_sobel_space = np.zeros([long, width])
# for i in range(long):
#     for ii in range(width):
#         img_sobel_space[i][ii] = img_sobel_space_raw[i][ii]
#
# plt.figure(2)
# plt.subplot(221), plt.imshow(abs(sobel), "gray"), plt.title("sobel filter")
# plt.subplot(222), plt.imshow(img_sobel_high, "gray"), plt.title("sobel frequency")
# plt.subplot(223), plt.imshow(img_sobel_space, "gray"), plt.title("sobel space")
# plt.subplot(224), plt.imshow(0.75 * img_sobel_space + 0.5 * img, "gray"), plt.title("k=1.8")

# # ideal HPF
# plt.figure(3)
# high_ideal = []
# for i in range(3):
#     high_ideal.append(hp_filter(
#         long,
#         width,
#         img_fre,
#         ideal_hpfilter(long * 2, width * 2, 130 + 30 * i)))
#     number = 230 + i + 1
#     plt.subplot(number), plt.imshow(high_ideal[i], "gray"), plt.title(
#         "idealhp d0: {} energy: {}".format(130 + 30 * i, round(cal_energy_inside(img_fre, 130 + 30 * i), 2)))
# for i in range(3):
#     number = 230 + i + 4
#     img_ideal = normalization(1.8 * high_ideal[i] + img)
#     plt.subplot(number), plt.imshow(img_ideal, "gray"), plt.title("corresponding img")
#
# # Butterworth HPF
# plt.figure(5)
# high_bw = hp_filter(long, width, img_fre, butterworth_hpfilter(long * 2, width * 2, 190))
# high_bw_fre = np.fft.fft2(high_bw)
# img_bw = normalization(1.8 * high_bw + img)
# plt.subplot(121), plt.imshow(high_bw, "gray"), plt.title("bwhp d0: 190")
# plt.subplot(122), plt.imshow(img_bw, "gray"), plt.title("bwhp d0: 190 k: 1.8")


# Gaussian HPF
plt.figure(6)
high_bw = hp_filter(long, width, img_fre, gaussian_hpfilter(long * 2, width * 2, 190))
img_bw = 1.8 * high_bw + img
for i in range(long ):
    for ii in range(width ):
        img_bw[i][ii] = max(img_bw[i][ii], 0)
plt.subplot(121), plt.imshow(high_bw, "gray"), plt.title("gaussian_hp d0: 190")
plt.subplot(122), plt.imshow(img_bw, "gray"), plt.title("gaussian_hp d0: 190 k: 1.8")

plt.pause(0)