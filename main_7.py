# coding:utf-8
# describe: 用霍夫变换提取车道线
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 用sobel算子对边界进行提取
def edge_extraction(img):
    long, width = img.shape
    img_x = np.zeros([long, width])
    img_y = np.zeros([long, width])
    img = np.array(img, dtype=np.float32)
    for i in range(long):
        for ii in range(width):
            if i == 0 or i == long - 1 or ii == 0 or ii == width - 1:
                img_x[i][ii] = 0
                img_y[i][ii] = 0
            else:
                # print(np.array(img[i - 1: i + 2][:, ii - 1: ii + 2]), sobel_model_x)
                # print(np.array(img[i - 1: i + 2][:, ii - 1: ii + 2]).shape)
                # print(sobel_model_x.shape)
                img_x[i][ii] = img[i - 1, ii - 1] * -1 + img[i - 1, ii] * -2 + img[i - 1, ii + 1] * -1\
                                + img[i + 1, ii - 1] * 1 + img[i + 1, ii] * 2 + img[i + 1, ii + 1] * 1
                img_y[i][ii] = img[i - 1, ii - 1] * -1 + img[i, ii - 1] * -2 + img[i + 1, ii - 1] * -1\
                                + img[i - 1, ii + 1] * 1 + img[i, ii + 1] * 2 + img[i + 1, ii + 1] * 1
    processed_img = np.abs(img_x) + np.abs(img_y)
    return processed_img

def normalization(img):
    max = np.max(img)
    min = np.min(img)
    long, width = img.shape
    for i in range(long):
        for ii in range(width):
            img[i][ii] = (img[i][ii] - min) / (max - min) * 255
    img = img.astype(np.uint8)
    return img

def threshold(img, num):
    long, width = img.shape
    for i in range(long):
        for ii in range(width):
            if img[i][ii] > num:
                img[i][ii] = 255
            else:
                img[i][ii] = 0
    return img

def plot_his(img):
    img = img.reshape([-1, 1])
    plt.hist(img, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.show()

def huofu_transform(img):
    long, width = img.shape
    huofu_map = np.zeros([int((long**2 + width**2)**0.5), 180])
    for i in range(long):
        for ii in range(width):
            if img[i][ii] != 0:
                for j in range(180):
                    #print(j)
                    ro = np.abs(ii - np.tan(j * np.pi / 180) * i) / (1 + np.tan(j * np.pi / 180)**2)**0.5
                    huofu_map[int(ro), j] += 1
    sort = np.sort(huofu_map.reshape([1, -1]))
    for i in range(huofu_map.shape[0]):
        for j in range(huofu_map.shape[1]):
            if huofu_map[i][j] == sort[0][-1]:
                index = [i, j]
            elif huofu_map[i][j] == sort[0][-100]:
                index2 = [i, j]
    line = np.zeros([long, width])
    for i in range(90, long):
        j = (i * np.sin(index[1] * np.pi / 180) - index[0]) / np.cos(index[1] * np.pi / 180)
        j2 = (i * np.sin(index2[1] * np.pi / 180) - index2[0]) / np.cos(index2[1] * np.pi / 180)
        try:
            line[i][int(np.round(j))] = 255
            line[i][int(np.round(j2))] = 255
        except IndexError:
            continue
    return line



img_path = '/Users/mac/Desktop/lane_detection.png'
img = cv2.imread(img_path, 0)
cv2.imshow("original image", img)
plot_his(img)
img = threshold(img, 200)
cv2.imshow("line image", img)
img = edge_extraction(img)
img = normalization(img)
cv2.imshow("primary edge image", img)
plot_his(img)
img = threshold(img, 150)
cv2.imshow("edge image", img)
line = huofu_transform(img)
cv2.imshow("line", line)

cv2.waitKey(0)
cv2.destroyAllWindows()