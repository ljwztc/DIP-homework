# coding:utf-8
# describe:用同频率不同相位的三个正弦函数对灰度图像进行伪彩色增强

import cv2
import numpy as np

def trans_rgb(img):
    long, width = img.shape
    img_r = np.zeros([long,width])
    img_g = np.zeros([long, width])
    img_b = np.zeros([long, width])
    for i in range(long):
        for ii in range (width):
            img_r[i, ii] = np.sin(np.pi * img[i, ii] / 50)
            img_g[i, ii] = np.sin(np.pi * img[i, ii] / 50 + np.pi / 5)
            img_b[i, ii] = np.sin(np.pi * img[i, ii] / 50 + np.pi * 3 / 5)

    return img_r, img_g, img_b



imagepath = "/Users/mac/Desktop/jay_chou.jpg"
img = cv2.imread(imagepath, 0)
channel_r, channel_g, channel_b = trans_rgb(img)
img_color = cv2.merge([channel_r, channel_g, channel_b])
cv2.imshow("origin", img)
cv2.imshow("pseudocolor", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()